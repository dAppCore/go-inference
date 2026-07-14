// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
)

// hipLoRATrainer is HIP's retained output-head LoRA session. The quantized
// Gemma base remains frozen; only the packed A/B factors are updated.
type hipLoRATrainer struct {
	loaded *hipLoadedModel
	state  *NativeAdamWState
	rows   int
	cols   int
	rank   int
	alpha  float32
	closed bool
}

var _ engine.Trainer = (*hipLoRATrainer)(nil)

func newHIPLoRATrainer(loaded *hipLoadedModel, cfg inference.TrainingConfig) (*hipLoRATrainer, error) {
	if loaded == nil || !hipLoadedGemma4Q4GenerateLinked(loaded) {
		return nil, core.NewError("hip.LoRATrainer: trainer needs a Gemma4-Q4 linked runtime")
	}
	rows, cols := loaded.modelInfo.VocabSize, loaded.modelInfo.HiddenSize
	if rows <= 0 || cols <= 0 {
		return nil, core.NewError("hip.LoRATrainer: model reports a zero vocabulary or hidden size")
	}
	rank := cfg.LoRA.Rank
	if rank <= 0 {
		rank = 8
	}
	alpha := cfg.LoRA.Alpha
	if alpha == 0 {
		alpha = 16
	}
	optimizer := DefaultNativeAdamWConfig()
	if cfg.LearningRate > 0 {
		optimizer.LearningRate = cfg.LearningRate
		optimizer.LearningRateSet = true
	}
	state, err := NewNativeLoRAAdamWState(hipInitializeLoRAFactorA(rank*cols, cols), make([]float32, rows*rank), rows, cols, rank, optimizer)
	if err != nil {
		return nil, core.E("hip.LoRATrainer", "initialise LoRA AdamW state", err)
	}
	return &hipLoRATrainer{loaded: loaded, state: state, rows: rows, cols: cols, rank: rank, alpha: alpha}, nil
}

func (t *hipLoRATrainer) Step(batch inference.Batch) (float64, error) {
	loss, dA, dB, count, err := t.gradients([]inference.Batch{batch})
	if err != nil {
		return 0, err
	}
	if count == 0 {
		return 0, core.NewError("hip.LoRATrainer.Step: batch has no trainable tokens")
	}
	if err := t.update(dA, dB, count); err != nil {
		return 0, err
	}
	return loss / float64(count), nil
}

func (t *hipLoRATrainer) StepAccumulated(batches []inference.Batch) (float64, error) {
	loss, dA, dB, count, err := t.gradients(batches)
	if err != nil {
		return 0, err
	}
	if count == 0 {
		return 0, core.NewError("hip.LoRATrainer.StepAccumulated: batches have no trainable tokens")
	}
	if err := t.update(dA, dB, count); err != nil {
		return 0, err
	}
	return loss / float64(count), nil
}

func (t *hipLoRATrainer) Loss(batch inference.Batch) (float64, error) {
	loss, _, _, count, err := t.gradients([]inference.Batch{batch})
	if err != nil {
		return 0, err
	}
	if count == 0 {
		return 0, core.NewError("hip.LoRATrainer.Loss: batch has no trainable tokens")
	}
	return loss / float64(count), nil
}

func (t *hipLoRATrainer) Save(path string) error {
	if err := t.available("Save"); err != nil {
		return err
	}
	a, b, err := nativeLoRAAdamWStateViews(t.state, t.rows, t.cols, t.rank)
	if err != nil {
		return core.E("hip.LoRATrainer.Save", "read LoRA state", err)
	}
	return saveMetalHeadLoRAAdapter(path, a, b, t.rows, t.cols, t.rank, t.alpha)
}

func (t *hipLoRATrainer) Close() error {
	if t != nil {
		t.closed = true
	}
	return nil
}

func (t *hipLoRATrainer) gradients(batches []inference.Batch) (float64, []float32, []float32, int, error) {
	if err := t.available("gradients"); err != nil {
		return 0, nil, nil, 0, err
	}
	if len(batches) == 0 {
		return 0, nil, nil, 0, core.NewError("hip.LoRATrainer: batches are required")
	}
	a, b, err := nativeLoRAAdamWStateViews(t.state, t.rows, t.cols, t.rank)
	if err != nil {
		return 0, nil, nil, 0, err
	}
	sumA, sumB := make([]float32, len(a)), make([]float32, len(b))
	var loss float64
	count := 0
	for _, batch := range batches {
		for sample, ids := range batch.TokenIDs {
			if len(ids) < 2 {
				return 0, nil, nil, 0, core.NewError("hip.LoRATrainer: a training sequence needs at least 2 tokens")
			}
			sequenceLoss, sequenceA, sequenceB, sequenceCount, err := t.sequenceGradients(ids, batch.LossMask, sample, a, b)
			if err != nil {
				return 0, nil, nil, 0, err
			}
			loss += sequenceLoss
			count += sequenceCount
			for i := range sumA {
				sumA[i] += sequenceA[i]
			}
			for i := range sumB {
				sumB[i] += sequenceB[i]
			}
		}
	}
	return loss, sumA, sumB, count, nil
}

func (t *hipLoRATrainer) sequenceGradients(ids []int32, mask inference.LossMask, sample int, loraA, loraB []float32) (float64, []float32, []float32, int, error) {
	cfg, err := t.loaded.cachedGemma4Q4ForwardConfig(t.loaded.modelInfo.NumLayers)
	if err != nil {
		return 0, nil, nil, 0, err
	}
	state := hipGemma4Q4DecodeState{}
	sumA, sumB := make([]float32, len(loraA)), make([]float32, len(loraB))
	var loss float64
	count := 0
	last := cfg.Layers[len(cfg.Layers)-1]
	for position := 0; position < len(ids)-1; position++ {
		forward, next, err := hipRunGemma4Q4SingleTokenForwardWithState(context.Background(), t.loaded.driver, cfg, state, hipGemma4Q4ForwardRequest{
			TokenID: ids[position], Position: position, Epsilon: 1e-6, SkipFinalSample: true, OmitLabels: true,
		})
		if err != nil {
			return 0, nil, nil, 0, err
		}
		state = next
		if !hipTrainerIncludesTarget(mask, sample, position, len(ids)) {
			continue
		}
		if len(forward.FinalHidden) != t.cols {
			return 0, nil, nil, 0, core.NewError("hip.LoRATrainer: Gemma4 forward did not return final hidden state")
		}
		norm := last.FinalNorm
		norm.Epsilon = 1e-6
		input, err := hipRunRMSNormKernelWithDeviceWeightConfig(context.Background(), t.loaded.driver, forward.FinalHidden, norm)
		if err != nil {
			return 0, nil, nil, 0, err
		}
		baseLogits, err := hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(context.Background(), t.loaded.driver, input, last.LMHeadProjection)
		if err != nil {
			return 0, nil, nil, 0, err
		}
		rawLogits, err := hipApplyHeadLoRA(input, baseLogits, loraA, loraB, t.rows, t.cols, t.rank, t.alpha)
		if err != nil {
			return 0, nil, nil, 0, err
		}
		logits, err := hipGemma4Q4SoftcapLogits(append([]float32(nil), rawLogits...), last.FinalLogitSoftcap)
		if err != nil {
			return 0, nil, nil, 0, err
		}
		stepLoss, upstream, err := hipTrainerCrossEntropyGradient(logits, ids[position+1])
		if err != nil {
			return 0, nil, nil, 0, err
		}
		if last.FinalLogitSoftcap > 0 {
			for index, raw := range rawLogits {
				tanh := float32(math.Tanh(float64(raw / last.FinalLogitSoftcap)))
				upstream[index] *= 1 - tanh*tanh
			}
		}
		grads, err := RunNativeLoRABackwardPass(input, loraA, loraB, upstream, t.rows, t.cols, t.rank, t.alpha)
		if err != nil {
			return 0, nil, nil, 0, err
		}
		for i := range sumA {
			sumA[i] += grads[0][i]
		}
		for i := range sumB {
			sumB[i] += grads[1][i]
		}
		loss += stepLoss
		count++
	}
	return loss, sumA, sumB, count, nil
}

func (t *hipLoRATrainer) update(dA, dB []float32, count int) error {
	for i := range dA {
		dA[i] /= float32(count)
	}
	for i := range dB {
		dB[i] /= float32(count)
	}
	if handled, err := t.loaded.RunAdamWUpdate(context.Background(), t.state, [][]float32{dA, dB}); err != nil {
		return err
	} else if handled {
		return nil
	}
	return t.state.StepInPlace([][]float32{dA, dB})
}

func (t *hipLoRATrainer) available(operation string) error {
	if t == nil || t.closed {
		return core.NewError("hip.LoRATrainer." + operation + ": trainer is closed")
	}
	return nil
}

func hipTrainerIncludesTarget(mask inference.LossMask, sample, position, tokens int) bool {
	if len(mask.Values) == 0 {
		return true
	}
	if sample >= len(mask.Values) || len(mask.Values[sample]) != tokens {
		return false
	}
	return mask.Values[sample][position+1] > 0
}

func hipTrainerCrossEntropyGradient(logits []float32, target int32) (float64, []float32, error) {
	if target < 0 || int(target) >= len(logits) || !rocmFloat32SliceFinite(logits) {
		return 0, nil, core.NewError("hip.LoRATrainer: invalid head logits or target")
	}
	maxLogit := float64(logits[0])
	for _, logit := range logits[1:] {
		maxLogit = math.Max(maxLogit, float64(logit))
	}
	sum := 0.0
	gradient := make([]float32, len(logits))
	for i, logit := range logits {
		p := math.Exp(float64(logit) - maxLogit)
		sum += p
		gradient[i] = float32(p)
	}
	for i := range gradient {
		gradient[i] /= float32(sum)
	}
	gradient[target]--
	return maxLogit + math.Log(sum) - float64(logits[target]), gradient, nil
}

func hipInitializeLoRAFactorA(count, cols int) []float32 {
	values := make([]float32, count)
	if cols <= 0 {
		return values
	}
	stddev := float32(1 / math.Sqrt(float64(cols)))
	state := uint32(0x9E3779B9)
	for index := range values {
		state = state*1664525 + 1013904223
		unit := float32(state) / float32(1<<32)
		values[index] = (unit*2 - 1) * stddev
	}
	return values
}

func hipApplyHeadLoRA(input, baseLogits, loraA, loraB []float32, rows, cols, rank int, alpha float32) ([]float32, error) {
	if rows <= 0 || cols <= 0 || rank <= 0 {
		return nil, core.NewError("hip.LoRATrainer: head LoRA geometry must be positive")
	}
	if len(input) != cols || len(baseLogits) != rows || len(loraA) != rank*cols || len(loraB) != rows*rank {
		return nil, core.NewError("hip.LoRATrainer: head LoRA tensor shapes do not match")
	}
	if !hipQ8ScaleIsPositiveFinite(alpha) || !rocmFloat32SliceFinite(input) || !rocmFloat32SliceFinite(baseLogits) || !rocmFloat32SliceFinite(loraA) || !rocmFloat32SliceFinite(loraB) {
		return nil, core.NewError("hip.LoRATrainer: head LoRA tensors and alpha must be finite")
	}
	down := make([]float32, rank)
	for r := 0; r < rank; r++ {
		for col := 0; col < cols; col++ {
			down[r] += loraA[r*cols+col] * input[col]
		}
	}
	out := append([]float32(nil), baseLogits...)
	scale := alpha / float32(rank)
	for row := 0; row < rows; row++ {
		for r := 0; r < rank; r++ {
			out[row] += scale * loraB[row*rank+r] * down[r]
		}
	}
	return out, nil
}
