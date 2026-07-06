// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// train_trainer.go promotes the real-session SFT proof (TestRealSessionHeadLoRASFT) into an exported,
// reusable trainer: a LoRA adapter on the output head, trained over the engine's OWN forward. The base
// stays frozen and is run by the real ArchSession (ForwardCaptureHiddens), so the trainer is
// architecture-agnostic on the base — it drives a real gemma-4 E2B/E4B (PLE) session exactly, because
// the whole PLE forward is the engine's, not a host re-derivation. Only the head LoRA (A [rank,dModel],
// B [vocab,rank]) and its two AdamW states live here; each Step captures the frozen final hidden, adds
// the LoRA delta to the frozen head logits, and steps A/B down the cross-entropy of the sequence's own
// next token. This is the LoRATrainer half of the engine.Trainer seam (go/engine/trainer.go); the shape
// mirrors go-mlx pkg/metal.LoRAAdapter (Step / StepAccumulated / Loss / Save), with the no-cgo gradient
// kernels (train_lora.go / train_optim.go) standing in for mlx autodiff.
//
// Scope note (the honest boundary): this trains the HEAD LoRA, the seam that is EXACT over any real base
// via the frozen capture. A LoRA on the per-LAYER projections would need a backward through the engine's
// real (PLE / QK-norm / post-norm) forward — the host block backwards (train_backward.go) model a
// simplified gemma layer only, so that is a separate engine train-step follow-up, not this seam.

// adapterConfigJSON is the go-mlx on-disk adapter_config.json: rank/alpha, the number of decoder layers
// the adapter spans (0 for a head-only adapter), and the target projection names. Written by
// LoRATrainer.Save and read back by the load path (lora_apply.go).
type adapterConfigJSON struct {
	Rank       int      `json:"rank"`
	Alpha      float32  `json:"alpha"`
	NumLayers  int      `json:"num_layers"`
	LoRALayers []string `json:"lora_layers"`
}

// LoRATrainer is a retained head-LoRA SFT session over a loaded native model. It wraps a fresh frozen
// base decode session, the trainable head LoRA factors, and their AdamW state — the exported promotion
// of the real-session SFT proof. Single-goroutine (the ArchSession contract).
type LoRATrainer struct {
	sess      *ArchSession
	finalNorm []float32 // [dModel] bf16→f32, the frozen final RMSNorm weight
	lmHead    []float32 // [vocab,dModel] bf16→f32, the frozen output head
	a, b      []float32 // trainable: A [rank,dModel], B [vocab,rank] (B starts at zero → adapter is a no-op)
	optA      *AdamW
	optB      *AdamW
	dModel    int
	vocab     int
	rank      int
	alpha     float32
	scaling   float32 // alpha/rank
	eps       float32
}

// NewLoRATrainer opens a head-LoRA trainer over tm: it takes a fresh frozen base session, widens the
// final-norm + head weights to f32, and initialises the LoRA factors (A small-random, B zero, so the
// adapter starts as the identity). cfg.LoRA supplies rank/alpha (defaults 8/16); cfg.LearningRate the
// AdamW step (default 0.05). The trainer OWNS the base session — Close releases it.
func NewLoRATrainer(tm *NativeTokenModel, cfg inference.TrainingConfig) (*LoRATrainer, error) {
	if tm == nil || tm.bf16 == nil {
		return nil, core.NewError("native.NewLoRATrainer: trainer needs a loaded bf16 model")
	}
	if len(tm.bf16.FinalNorm) == 0 || len(tm.bf16.LMHead) == 0 {
		return nil, core.NewError("native.NewLoRATrainer: model is missing the final norm or head weight")
	}
	dModel, vocab, eps := tm.arch.Hidden, tm.arch.Vocab, tm.arch.Eps
	if dModel <= 0 || vocab <= 0 {
		return nil, core.NewError("native.NewLoRATrainer: model reports a zero hidden or vocab size")
	}
	rank := cfg.LoRA.Rank
	if rank <= 0 {
		rank = 8
	}
	alpha := cfg.LoRA.Alpha
	if alpha == 0 {
		alpha = 16
	}
	lr := float32(cfg.LearningRate)
	if lr <= 0 {
		lr = 0.05
	}

	stepper, err := tm.OpenSession()
	if err != nil {
		return nil, err
	}
	sess, ok := stepper.(*ArchSession)
	if !ok {
		if closer, closeOK := stepper.(interface{ Close() error }); closeOK {
			_ = closer.Close()
		}
		return nil, core.NewError("native.NewLoRATrainer: token model does not open an ArchSession")
	}

	return &LoRATrainer{
		sess:      sess,
		finalNorm: bf16ToF32Slice(tm.bf16.FinalNorm),
		lmHead:    bf16ToF32Slice(tm.bf16.LMHead),
		a:         initLoRAFactorA(rank*dModel, dModel),
		b:         make([]float32, vocab*rank),
		optA:      NewAdamW(rank*dModel, lr, 0),
		optB:      NewAdamW(vocab*rank, lr, 0),
		dModel:    dModel,
		vocab:     vocab,
		rank:      rank,
		alpha:     alpha,
		scaling:   alpha / float32(rank),
		eps:       eps,
	}, nil
}

// initLoRAFactorA fills the LoRA A factor with small deterministic pseudo-random values (Kaiming-style
// stddev 1/√dModel), so training is reproducible and starts from a non-degenerate A while B=0 keeps the
// adapter an initial no-op. Deterministic LCG — no RNG dependency, no per-run drift.
func initLoRAFactorA(n, dModel int) []float32 {
	a := make([]float32, n)
	stddev := float32(1.0 / math.Sqrt(float64(dModel)))
	state := uint32(0x9E3779B9)
	for i := range a {
		state = state*1664525 + 1013904223
		u := float32(state) / float32(1<<32) // [0,1)
		a[i] = (u*2 - 1) * stddev            // [-stddev, stddev)
	}
	return a
}

// forwardFrozen runs the engine's real forward over ids and returns the post-final-norm hidden
// (normed [T,dModel]) and the frozen base head logits (baseLogits [T,vocab]) — the frozen half of every
// step. The base weights never change, so this is the model's own (PLE-correct) forward; the LoRA delta
// is added on top of baseLogits.
func (t *LoRATrainer) forwardFrozen(ids []int32) (normed, baseLogits []float32, rows int, err error) {
	_, perLayer, err := t.sess.ForwardCaptureHiddens(ids)
	if err != nil {
		return nil, nil, 0, err
	}
	if len(perLayer) == 0 {
		return nil, nil, 0, core.NewError("native.LoRATrainer: ForwardCaptureHiddens returned no layers")
	}
	tokens := len(ids)
	hLast := bf16ToF32Slice(perLayer[len(perLayer)-1]) // [T,dModel]
	normed = rmsNormForwardF32(hLast, t.finalNorm, tokens, t.dModel, t.eps)
	baseLogits, err = MatMulF32NT(normed, t.lmHead, tokens, t.dModel, t.vocab)
	if err != nil {
		return nil, nil, 0, err
	}
	return normed, baseLogits, tokens, nil
}

// seqGrads runs one sequence's head-LoRA forward+backward under the current A/B and returns its loss and
// the gradients of A and B. Targets are the sequence's own next token (causal SFT): hidden t predicts
// token t+1, so the trainable rows are 0..T-2. No optimiser step (the caller accumulates then steps).
func (t *LoRATrainer) seqGrads(ids []int32) (loss float32, dA, dB []float32, err error) {
	if len(ids) < 2 {
		return 0, nil, nil, core.NewError("native.LoRATrainer: a training sequence needs at least 2 tokens")
	}
	normed, baseLogits, tokens, err := t.forwardFrozen(ids)
	if err != nil {
		return 0, nil, nil, err
	}
	rows := tokens - 1
	normedPred := normed[:rows*t.dModel]
	targets := make([]int32, rows)
	for i := range rows {
		targets[i] = ids[i+1]
	}
	xA, delta, err := LoRAForwardF32(normedPred, t.a, t.b, rows, t.dModel, t.vocab, t.rank, t.scaling)
	if err != nil {
		return 0, nil, nil, err
	}
	logits := make([]float32, rows*t.vocab)
	for i := range logits {
		logits[i] = baseLogits[i] + delta[i]
	}
	loss, dLogits, err := CrossEntropyBackwardF32(logits, targets, rows, t.vocab)
	if err != nil {
		return 0, nil, nil, err
	}
	dA, dB, _, err = LoRABackwardF32(dLogits, normedPred, t.a, t.b, xA, rows, t.dModel, t.vocab, t.rank, t.scaling)
	if err != nil {
		return 0, nil, nil, err
	}
	return loss, dA, dB, nil
}

// accumulate sums the per-sequence loss and A/B gradients across every sequence in batch (no step).
func (t *LoRATrainer) accumulate(batch inference.Batch) (lossSum float64, sumDA, sumDB []float32, n int, err error) {
	sumDA = make([]float32, len(t.a))
	sumDB = make([]float32, len(t.b))
	for _, ids := range batch.TokenIDs {
		loss, dA, dB, e := t.seqGrads(ids)
		if e != nil {
			return 0, nil, nil, 0, e
		}
		for i := range sumDA {
			sumDA[i] += dA[i]
		}
		for i := range sumDB {
			sumDB[i] += dB[i]
		}
		lossSum += float64(loss)
		n++
	}
	return lossSum, sumDA, sumDB, n, nil
}

// applyMeanStep scales the summed gradients by 1/count and applies one AdamW update to A and B.
func (t *LoRATrainer) applyMeanStep(sumDA, sumDB []float32, count int) error {
	inv := float32(1.0 / float64(count))
	for i := range sumDA {
		sumDA[i] *= inv
	}
	for i := range sumDB {
		sumDB[i] *= inv
	}
	if err := t.optA.Step(t.a, sumDA); err != nil {
		return err
	}
	return t.optB.Step(t.b, sumDB)
}

// Step runs one SFT gradient step over batch (one AdamW update from the batch-mean gradient) and returns
// the mean cross-entropy loss. Implements engine.Trainer.
func (t *LoRATrainer) Step(batch inference.Batch) (float64, error) {
	if len(batch.TokenIDs) == 0 {
		return 0, core.NewError("native.LoRATrainer.Step: empty batch")
	}
	lossSum, sumDA, sumDB, n, err := t.accumulate(batch)
	if err != nil {
		return 0, err
	}
	if n == 0 {
		return 0, core.NewError("native.LoRATrainer.Step: batch produced no trainable sequences")
	}
	if err := t.applyMeanStep(sumDA, sumDB, n); err != nil {
		return 0, err
	}
	return lossSum / float64(n), nil
}

// StepAccumulated accumulates the gradients of every micro-batch and applies ONE AdamW update from their
// combined mean. Returns the mean loss across all sequences. Implements engine.Trainer.
func (t *LoRATrainer) StepAccumulated(batches []inference.Batch) (float64, error) {
	if len(batches) == 0 {
		return 0, core.NewError("native.LoRATrainer.StepAccumulated: no batches")
	}
	totalDA := make([]float32, len(t.a))
	totalDB := make([]float32, len(t.b))
	var lossSum float64
	total := 0
	for _, batch := range batches {
		ls, sumDA, sumDB, n, err := t.accumulate(batch)
		if err != nil {
			return 0, err
		}
		for i := range totalDA {
			totalDA[i] += sumDA[i]
		}
		for i := range totalDB {
			totalDB[i] += sumDB[i]
		}
		lossSum += ls
		total += n
	}
	if total == 0 {
		return 0, core.NewError("native.LoRATrainer.StepAccumulated: batches produced no trainable sequences")
	}
	if err := t.applyMeanStep(totalDA, totalDB, total); err != nil {
		return 0, err
	}
	return lossSum / float64(total), nil
}

// Loss is the forward-only mean cross-entropy over batch under the current adapter weights: no
// gradients, no optimiser update — the validation lane. Implements engine.Trainer.
func (t *LoRATrainer) Loss(batch inference.Batch) (float64, error) {
	if len(batch.TokenIDs) == 0 {
		return 0, core.NewError("native.LoRATrainer.Loss: empty batch")
	}
	var lossSum float64
	n := 0
	for _, ids := range batch.TokenIDs {
		if len(ids) < 2 {
			return 0, core.NewError("native.LoRATrainer.Loss: a sequence needs at least 2 tokens")
		}
		normed, baseLogits, tokens, err := t.forwardFrozen(ids)
		if err != nil {
			return 0, err
		}
		rows := tokens - 1
		normedPred := normed[:rows*t.dModel]
		targets := make([]int32, rows)
		for i := range rows {
			targets[i] = ids[i+1]
		}
		_, delta, err := LoRAForwardF32(normedPred, t.a, t.b, rows, t.dModel, t.vocab, t.rank, t.scaling)
		if err != nil {
			return 0, err
		}
		logits := make([]float32, rows*t.vocab)
		for i := range logits {
			logits[i] = baseLogits[i] + delta[i]
		}
		loss, _, err := CrossEntropyBackwardF32(logits, targets, rows, t.vocab)
		if err != nil {
			return 0, err
		}
		lossSum += float64(loss)
		n++
	}
	return lossSum / float64(n), nil
}

// Save writes the trained head LoRA as a reloadable adapter package — adapter.safetensors (the A/B
// factors as F32) + adapter_config.json — in the go-mlx on-disk format. The head is named "lm_head"
// (lm_head.lora_a / lm_head.lora_b); the native load path honours it via AdapterPath (lora_apply.go).
// Implements engine.Trainer.
func (t *LoRATrainer) Save(path string) error {
	if path == "" {
		return core.NewError("native.LoRATrainer.Save: path is required")
	}
	if res := core.MkdirAll(path, core.FileMode(0o755)); !res.OK {
		return core.E("native.LoRATrainer.Save", "ensure adapter dir", resultErr(res))
	}
	tensors := map[string]safetensors.Tensor{
		"lm_head.lora_a": {Dtype: "F32", Shape: []int{t.rank, t.dModel}, Data: safetensors.EncodeFloat32(t.a)},
		"lm_head.lora_b": {Dtype: "F32", Shape: []int{t.vocab, t.rank}, Data: safetensors.EncodeFloat32(t.b)},
	}
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		return core.E("native.LoRATrainer.Save", "encode adapter safetensors", err)
	}
	if werr := coreio.Local.Write(core.PathJoin(path, "adapter.safetensors"), string(blob)); werr != nil {
		return core.E("native.LoRATrainer.Save", "write adapter.safetensors", werr)
	}
	cfg := adapterConfigJSON{Rank: t.rank, Alpha: t.alpha, NumLayers: 0, LoRALayers: []string{"lm_head"}}
	cj := core.JSONMarshal(cfg)
	if !cj.OK {
		return core.E("native.LoRATrainer.Save", "marshal adapter_config.json", nil)
	}
	if werr := coreio.Local.Write(core.PathJoin(path, "adapter_config.json"), string(cj.Value.([]byte))); werr != nil {
		return core.E("native.LoRATrainer.Save", "write adapter_config.json", werr)
	}
	return nil
}

// Close releases the retained frozen base session. Implements engine.Trainer.
func (t *LoRATrainer) Close() error {
	if t == nil || t.sess == nil {
		return nil
	}
	err := t.sess.Close()
	t.sess = nil
	return err
}

// resultErr unwraps a core.Result's error value (nil when the Result carried no error).
func resultErr(res core.Result) error {
	if err, ok := res.Value.(error); ok {
		return err
	}
	return nil
}
