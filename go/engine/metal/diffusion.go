// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"math"
	"sort"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

const (
	DefaultCanvasLength = 64
	DefaultMaxSteps     = 16
)

type DiffusionStepConfig struct {
	EntropyBound   float32
	MaxTemperature float32
	MinTemperature float32
	Exponent       float32
	TextVocabSize  int32
	Seed           uint64
}

func DefaultDiffusionStepConfig(textVocabSize int32) DiffusionStepConfig {
	return DiffusionStepConfig{
		EntropyBound:   0.3,
		MaxTemperature: 0.8,
		MinTemperature: 0.4,
		Exponent:       1.0,
		TextVocabSize:  textVocabSize,
	}
}

type DiffusionStepResult struct {
	Canvas      []int32
	Greedy      []int32
	SCEmb       []byte
	Accepted    int
	Changed     int
	MeanEntropy float32
}

type DiffusionGenerateConfig struct {
	Step                DiffusionStepConfig
	CanvasLength        int32
	MaxSteps            int
	StabilityThreshold  int
	ConfidenceThreshold float32
	MaxCanvases         int
	StopTokens          []int32
	OnStep              func(canvasIdx, step int, res DiffusionStepResult, d time.Duration)
	OnCanvas            func(canvasIdx int, kept []int32, steps int, d time.Duration)
}

type DiffusionMetrics struct {
	Canvases       int
	TotalSteps     int
	EmittedTokens  int
	PrefillTokens  int
	PrefillDur     time.Duration
	DenoiseDur     time.Duration
	CommitDur      time.Duration
	TotalDur       time.Duration
	StoppedOnToken bool
}

type BlockDiffusionOptions struct {
	MaxTokens   int
	Temperature float32
	Seed        uint64
	SeedSet     bool
	StopTokens  []int32
}

type BlockDiffusionTokenGenerator interface {
	GenerateBlockDiffusionTokens(context.Context, []int32, BlockDiffusionOptions, func(int32) bool) (DiffusionMetrics, error)
}

type DiffusionDenoiseRequest struct {
	Canvas          []int32
	SCEmb           []byte
	CanvasIndex     int
	Step            int
	NoiseProportion float32
	Prefix          int
	GlobalMask      []float32
	GlobalMaskShape []int
	LocalMask       []float32
	LocalMaskShape  []int
	StepConfig      DiffusionStepConfig
}

type DiffusionGenerateOps struct {
	Prefill     func(context.Context) (int, error)
	CacheOffset func() int
	Denoise     func(context.Context, DiffusionDenoiseRequest) (DiffusionStepResult, error)
	TruncateTo  func(int) error
	Commit      func(context.Context, []int32) error
}

func RunDiffusionGenerate(ctx context.Context, cfg DiffusionGenerateConfig, eosTokens []int32, textVocabSize int32, slidingWindow int, ops DiffusionGenerateOps) ([]int32, DiffusionMetrics, error) {
	const op = "native.RunDiffusionGenerate"
	if ctx == nil {
		ctx = context.Background()
	}
	var metrics DiffusionMetrics
	start := time.Now()
	if ops.Prefill == nil {
		return nil, metrics, core.NewError(op + ": prefill callback is nil")
	}
	if ops.Denoise == nil {
		return nil, metrics, core.NewError(op + ": denoise callback is nil")
	}
	cfg = resolveDiffusionGenerateConfig(cfg, eosTokens, textVocabSize)
	if cfg.Step.TextVocabSize <= 0 {
		return nil, metrics, core.NewError(op + ": TextVocabSize must be positive")
	}

	prefillStart := time.Now()
	promptTokens, err := ops.Prefill(ctx)
	if err != nil {
		return nil, metrics, core.E(op, "prompt prefill", err)
	}
	metrics.PrefillDur = time.Since(prefillStart)
	if promptTokens <= 0 {
		return nil, metrics, core.NewError(op + ": prompt encoded to zero tokens")
	}
	metrics.PrefillTokens = promptTokens

	canvasLen := int(cfg.CanvasLength)
	emitted := make([]int32, 0, canvasLen*cfg.MaxCanvases)
	for canvasIdx := 0; canvasIdx < cfg.MaxCanvases; canvasIdx++ {
		if err := ctx.Err(); err != nil {
			return emitted, metrics, err
		}
		prefix := promptTokens + len(emitted)
		if ops.CacheOffset != nil {
			prefix = ops.CacheOffset()
		}
		canvasStart := time.Now()
		canvas := diffusionInitialCanvas(cfg.CanvasLength, cfg.Step.TextVocabSize, cfg.Step.Seed, canvasIdx)
		if len(canvas) != canvasLen {
			return emitted, metrics, core.E(op, core.Sprintf("initial canvas length = %d, want %d", len(canvas), canvasLen), nil)
		}
		canvasStepCfg := cfg.Step
		canvasStepCfg.Seed = cfg.Step.Seed + uint64(canvasIdx)*0x9E3779B97F4A7C15
		keyLen := prefix + canvasLen
		globalMask, globalShape := diffusionGlobalCanvasMaskData(1, canvasLen, keyLen)
		localMask, localShape := diffusionBlockLocalCanvasMaskData(1, canvasLen, keyLen, prefix, slidingWindow)

		var scEmb []byte
		var prevGreedy []int32
		var lastGreedy []int32
		stableRun := 0
		steps := 0
		for step := 0; step < cfg.MaxSteps; step++ {
			if err := ctx.Err(); err != nil {
				return emitted, metrics, err
			}
			stepStart := time.Now()
			noise := 1.0 - float32(step)/float32(cfg.MaxSteps)
			res, err := ops.Denoise(ctx, DiffusionDenoiseRequest{
				Canvas:          canvas,
				SCEmb:           scEmb,
				CanvasIndex:     canvasIdx,
				Step:            step,
				NoiseProportion: noise,
				Prefix:          prefix,
				GlobalMask:      globalMask,
				GlobalMaskShape: append([]int(nil), globalShape...),
				LocalMask:       localMask,
				LocalMaskShape:  append([]int(nil), localShape...),
				StepConfig:      canvasStepCfg,
			})
			if err != nil {
				return emitted, metrics, err
			}
			if ops.TruncateTo != nil {
				if err := ops.TruncateTo(prefix); err != nil {
					return emitted, metrics, core.E(op, core.Sprintf("cache declined TruncateTo(%d)", prefix), err)
				}
			}
			steps++
			metrics.TotalSteps++
			if cfg.OnStep != nil {
				cfg.OnStep(canvasIdx, step, res, time.Since(stepStart))
			}

			if prevGreedy != nil && core.SliceEqual(res.Greedy, prevGreedy) {
				stableRun++
			} else {
				stableRun = 0
			}
			prevGreedy = append(prevGreedy[:0], res.Greedy...)
			lastGreedy = append(lastGreedy[:0], res.Greedy...)
			scEmb = res.SCEmb
			if stableRun >= cfg.StabilityThreshold && res.MeanEntropy < cfg.ConfidenceThreshold {
				break
			}
			canvas = res.Canvas
		}
		if lastGreedy != nil {
			canvas = lastGreedy
		}
		metrics.DenoiseDur += time.Since(canvasStart)

		kept, stopped := diffusionKeepUntilStop(canvas, cfg.StopTokens)
		if len(kept) > 0 {
			if ops.Commit == nil {
				return emitted, metrics, core.NewError(op + ": commit callback is nil")
			}
			commitStart := time.Now()
			if err := ops.Commit(ctx, kept); err != nil {
				return emitted, metrics, core.E(op, "canvas commit", err)
			}
			metrics.CommitDur += time.Since(commitStart)
		}
		emitted = append(emitted, kept...)
		metrics.Canvases++
		metrics.EmittedTokens = len(emitted)
		if cfg.OnCanvas != nil {
			cfg.OnCanvas(canvasIdx, kept, steps, time.Since(canvasStart))
		}
		if stopped {
			metrics.StoppedOnToken = true
			break
		}
	}
	metrics.TotalDur = time.Since(start)
	return emitted, metrics, nil
}

func resolveDiffusionGenerateConfig(cfg DiffusionGenerateConfig, eosTokens []int32, textVocabSize int32) DiffusionGenerateConfig {
	if cfg.CanvasLength <= 0 {
		cfg.CanvasLength = DefaultCanvasLength
	}
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = DefaultMaxSteps
	}
	if cfg.StabilityThreshold <= 0 {
		cfg.StabilityThreshold = 1
	}
	if cfg.ConfidenceThreshold <= 0 {
		cfg.ConfidenceThreshold = 0.005
	}
	if cfg.MaxCanvases <= 0 {
		cfg.MaxCanvases = 1
	}
	if len(cfg.StopTokens) == 0 {
		cfg.StopTokens = append([]int32(nil), eosTokens...)
	}
	if cfg.Step.TextVocabSize <= 0 {
		cfg.Step.TextVocabSize = textVocabSize
	}
	return cfg
}

func diffusionInitialCanvas(canvasLen, textVocabSize int32, seed uint64, canvasIdx int) []int32 {
	if canvasLen <= 0 || textVocabSize <= 0 {
		return nil
	}
	sampler := model.NewSampler(seed ^ (uint64(canvasIdx+1) << 32))
	canvas := make([]int32, canvasLen)
	for i := range canvas {
		id := int32(sampler.Draw() * float32(textVocabSize))
		if id >= textVocabSize {
			id = textVocabSize - 1
		}
		canvas[i] = id
	}
	return canvas
}

func diffusionKeepUntilStop(canvas, stops []int32) ([]int32, bool) {
	for i, id := range canvas {
		if tokenInSet(id, stops) {
			return canvas[:i], true
		}
	}
	return canvas, false
}

func tokenInSet(id int32, set []int32) bool {
	for _, s := range set {
		if id == s {
			return true
		}
	}
	return false
}

func diffusionGlobalCanvasMaskData(batch, canvasLen, keyLen int) ([]float32, []int) {
	shape := []int{batch, 1, canvasLen, keyLen}
	if batch <= 0 || canvasLen <= 0 || keyLen <= 0 {
		return nil, shape
	}
	return make([]float32, batch*canvasLen*keyLen), shape
}

func diffusionBlockLocalCanvasMaskData(batch, canvasLen, keyLen, offset, window int) ([]float32, []int) {
	shape := []int{batch, 1, canvasLen, keyLen}
	if batch <= 0 || canvasLen <= 0 || keyLen <= 0 {
		return nil, shape
	}
	negInf := float32(math.Inf(-1))
	contextStart := offset - window
	if contextStart < 0 {
		contextStart = 0
	}
	data := make([]float32, batch*canvasLen*keyLen)
	for b := 0; b < batch; b++ {
		base := b * canvasLen * keyLen
		for i := 0; i < canvasLen; i++ {
			row := base + i*keyLen
			for j := 0; j < keyLen; j++ {
				inContext := j >= contextStart && j < offset
				inCanvas := j >= offset && j < offset+canvasLen
				if !inContext && !inCanvas {
					data[row+j] = negInf
				}
			}
		}
	}
	return data, shape
}

func DiffusionSelfConditionBF16(h, scEmb, preNormW, wGate, wUp, wDown []byte, rows, dModel, dFF int, eps float32) ([]byte, error) {
	if rows < 0 || dModel < 0 || dFF < 0 {
		return nil, core.NewError("native.DiffusionSelfConditionBF16: dimensions must be non-negative")
	}
	if len(h) != rows*dModel*bf16Size {
		return nil, core.NewError("native.DiffusionSelfConditionBF16: h must be rows*dModel bf16 bytes")
	}
	ones := diffusionOnesBF16(dModel)
	combined := h
	if len(scEmb) > 0 {
		if len(scEmb) != len(h) {
			return nil, core.NewError("native.DiffusionSelfConditionBF16: scEmb must match h length")
		}
		if len(preNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DiffusionSelfConditionBF16: preNormW must be dModel bf16 bytes")
		}
		if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size {
			return nil, core.NewError("native.DiffusionSelfConditionBF16: gate/up weights must be dFF*dModel bf16 bytes")
		}
		if len(wDown) != dModel*dFF*bf16Size {
			return nil, core.NewError("native.DiffusionSelfConditionBF16: down weight must be dModel*dFF bf16 bytes")
		}
		normed, err := RMSNormBF16(scEmb, preNormW, rows, dModel, eps)
		if err != nil {
			return nil, err
		}
		gate, err := MatRowsBF16(wGate, normed, rows, dFF, dModel)
		if err != nil {
			return nil, err
		}
		up, err := MatRowsBF16(wUp, normed, rows, dFF, dModel)
		if err != nil {
			return nil, err
		}
		gated, err := GeluGateMulBF16(gate, up)
		if err != nil {
			return nil, err
		}
		ffw, err := MatRowsBF16(wDown, gated, rows, dModel, dFF)
		if err != nil {
			return nil, err
		}
		combined, err = AddBF16(h, ffw)
		if err != nil {
			return nil, err
		}
	}
	return RMSNormBF16(combined, ones, rows, dModel, eps)
}

func DiffusionEncodeLogitsBF16(logits, embed []byte, rows, vocab, dModel int) ([]byte, error) {
	if rows < 0 || vocab < 0 || dModel < 0 {
		return nil, core.NewError("native.DiffusionEncodeLogitsBF16: dimensions must be non-negative")
	}
	if len(logits) != rows*vocab*bf16Size {
		return nil, core.NewError("native.DiffusionEncodeLogitsBF16: logits must be rows*vocab bf16 bytes")
	}
	if len(embed) != vocab*dModel*bf16Size {
		return nil, core.NewError("native.DiffusionEncodeLogitsBF16: embed must be vocab*dModel bf16 bytes")
	}
	if rows == 0 || vocab == 0 || dModel == 0 {
		return make([]byte, rows*dModel*bf16Size), nil
	}
	probs, err := SoftmaxF32(bf16ToF32Slice(logits), vocab)
	if err != nil {
		return nil, err
	}
	encoded, err := MatMulF32(probs, bf16ToF32Slice(embed), rows, vocab, dModel)
	if err != nil {
		return nil, err
	}
	scale := float32(math.Sqrt(float64(dModel)))
	for i := range encoded {
		encoded[i] *= scale
	}
	return f32ToBf16Slice(encoded), nil
}

func DiffusionEncodeLogitsQuant(logits, packed, scales, biases []byte, rows, vocab, dModel, groupSize, bits int) ([]byte, error) {
	if rows < 0 || vocab < 0 || dModel < 0 {
		return nil, core.NewError("native.DiffusionEncodeLogitsQuant: dimensions must be non-negative")
	}
	if len(logits) != rows*vocab*bf16Size {
		return nil, core.NewError("native.DiffusionEncodeLogitsQuant: logits must be rows*vocab bf16 bytes")
	}
	if rows == 0 || vocab == 0 || dModel == 0 {
		return make([]byte, rows*dModel*bf16Size), nil
	}
	dense, err := dequantizeAffineRowsF32(packed, scales, biases, vocab, dModel, groupSize, bits)
	if err != nil {
		return nil, err
	}
	probs, err := SoftmaxF32(bf16ToF32Slice(logits), vocab)
	if err != nil {
		return nil, err
	}
	encoded, err := MatMulF32(probs, dense, rows, vocab, dModel)
	if err != nil {
		return nil, err
	}
	scale := float32(math.Sqrt(float64(dModel)))
	for i := range encoded {
		encoded[i] *= scale
	}
	return f32ToBf16Slice(encoded), nil
}

func dequantizeAffineRowsF32(packed, scales, biases []byte, rows, cols, groupSize, bits int) ([]float32, error) {
	if bits <= 0 || bits > 8 {
		return nil, core.NewError("native.dequantizeAffineRowsF32: bits must be in 1..8")
	}
	if groupSize <= 0 || cols%groupSize != 0 {
		return nil, core.NewError("native.dequantizeAffineRowsF32: groupSize must be > 0 and divide cols")
	}
	if cols*bits%8 != 0 {
		return nil, core.NewError("native.dequantizeAffineRowsF32: cols*bits must be byte-aligned")
	}
	rowPacked := cols * bits / 8
	rowSB := (cols / groupSize) * bf16Size
	if len(packed) != rows*rowPacked || len(scales) != rows*rowSB || len(biases) != rows*rowSB {
		return nil, core.NewError("native.dequantizeAffineRowsF32: packed/scales/biases size mismatch")
	}
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		pRow := packed[r*rowPacked : (r+1)*rowPacked]
		sRow := scales[r*rowSB : (r+1)*rowSB]
		bRow := biases[r*rowSB : (r+1)*rowSB]
		for c := 0; c < cols; c++ {
			g := c / groupSize
			scale := bf16ToF32(sRow[g*bf16Size], sRow[g*bf16Size+1])
			bias := bf16ToF32(bRow[g*bf16Size], bRow[g*bf16Size+1])
			code := extractAffineCode(pRow, c*bits, bits)
			out[r*cols+c] = scale*float32(code) + bias
		}
	}
	return out, nil
}

func DiffusionSampleDenoiseStepBF16(logits, embed []byte, canvas []int32, vocab, dModel int, step int, noiseProportion float32, cfg DiffusionStepConfig) (DiffusionStepResult, error) {
	const op = "native.DiffusionSampleDenoiseStepBF16"
	L := len(canvas)
	if len(embed) != vocab*dModel*bf16Size {
		return DiffusionStepResult{}, core.NewError(op + ": embed must be vocab*dModel bf16 bytes")
	}
	return diffusionSampleDenoiseStep(logits, canvas, vocab, dModel, step, noiseProportion, cfg, op, func(shaped []byte) ([]byte, error) {
		return DiffusionEncodeLogitsBF16(shaped, embed, L, vocab, dModel)
	})
}

func DiffusionSampleDenoiseStepQuant(logits, packed, scales, biases []byte, canvas []int32, vocab, dModel, groupSize, bits int, step int, noiseProportion float32, cfg DiffusionStepConfig) (DiffusionStepResult, error) {
	const op = "native.DiffusionSampleDenoiseStepQuant"
	L := len(canvas)
	return diffusionSampleDenoiseStep(logits, canvas, vocab, dModel, step, noiseProportion, cfg, op, func(shaped []byte) ([]byte, error) {
		return DiffusionEncodeLogitsQuant(shaped, packed, scales, biases, L, vocab, dModel, groupSize, bits)
	})
}

func diffusionSampleDenoiseStep(logits []byte, canvas []int32, vocab, dModel int, step int, noiseProportion float32, cfg DiffusionStepConfig, op string, encode func([]byte) ([]byte, error)) (DiffusionStepResult, error) {
	L := len(canvas)
	if vocab <= 0 || dModel < 0 {
		return DiffusionStepResult{}, core.NewError(op + ": vocab must be positive and dModel must be non-negative")
	}
	if cfg.TextVocabSize <= 0 {
		return DiffusionStepResult{}, core.NewError(op + ": TextVocabSize must be positive")
	}
	if len(logits) != L*vocab*bf16Size {
		return DiffusionStepResult{}, core.NewError(op + ": logits must be len(canvas)*vocab bf16 bytes")
	}
	if encode == nil {
		return DiffusionStepResult{}, core.NewError(op + ": encode callback is nil")
	}
	if L == 0 {
		return DiffusionStepResult{Canvas: []int32{}, Greedy: []int32{}, SCEmb: []byte{}}, nil
	}

	frac := 1.0 - float32(math.Pow(float64(1.0-noiseProportion), float64(cfg.Exponent)))
	temp := cfg.MinTemperature + frac*(cfg.MaxTemperature-cfg.MinTemperature)
	if temp <= 0 {
		temp = 1e-6
	}
	shapedF := bf16ToF32Slice(logits)
	for i := range shapedF {
		shapedF[i] /= temp
	}
	shaped := f32ToBf16Slice(shapedF)
	shapedF = bf16ToF32Slice(shaped)

	categoricalSampler := model.NewSampler(cfg.Seed ^ (uint64(step)*2 + 1))
	renoiseSampler := model.NewSampler(cfg.Seed ^ (uint64(step)*2 + 2))
	sampledIDs := make([]int32, L)
	greedyIDs := make([]int32, L)
	entropies := make([]float32, L)
	var entropySum float32
	for i := 0; i < L; i++ {
		rowBytes := shaped[i*vocab*bf16Size : (i+1)*vocab*bf16Size]
		id, err := categoricalSampler.Sample(rowBytes, vocab, model.SampleParams{Temperature: 1})
		if err != nil {
			return DiffusionStepResult{}, err
		}
		sampledIDs[i] = id
		greedy, err := model.Greedy(rowBytes, vocab)
		if err != nil {
			return DiffusionStepResult{}, err
		}
		greedyIDs[i] = greedy
		entropies[i] = diffusionEntropyF32(shapedF[i*vocab : (i+1)*vocab])
		entropySum += entropies[i]
	}

	scEmb, err := encode(shaped)
	if err != nil {
		return DiffusionStepResult{}, err
	}

	renoise := make([]int32, L)
	for i := range renoise {
		id := int32(renoiseSampler.Draw() * float32(cfg.TextVocabSize))
		if id >= cfg.TextVocabSize {
			id = cfg.TextVocabSize - 1
		}
		renoise[i] = id
	}

	order := make([]int, L)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return entropies[order[a]] < entropies[order[b]] })
	accept := make([]bool, L)
	accepted := 0
	var accumulated float32
	for _, idx := range order {
		if accumulated > cfg.EntropyBound {
			break
		}
		accept[idx] = true
		accepted++
		accumulated += entropies[idx]
	}

	next := make([]int32, L)
	changed := 0
	for i := range next {
		if accept[i] {
			next[i] = sampledIDs[i]
		} else {
			next[i] = renoise[i]
		}
		if next[i] != canvas[i] {
			changed++
		}
	}

	return DiffusionStepResult{
		Canvas:      next,
		Greedy:      greedyIDs,
		SCEmb:       scEmb,
		Accepted:    accepted,
		Changed:     changed,
		MeanEntropy: entropySum / float32(L),
	}, nil
}

func diffusionEntropyF32(row []float32) float32 {
	if len(row) == 0 {
		return 0
	}
	maxLogit := row[0]
	for _, v := range row[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var sum, weighted float32
	for _, v := range row {
		e := float32(math.Exp(float64(v - maxLogit)))
		sum += e
		weighted += e * v
	}
	return maxLogit + float32(math.Log(float64(sum))) - weighted/sum
}

func withDiffusionEncoderScalarsBF16(g *BF16Model, diffusion *model.LoadedDiffusion, fn func()) {
	if fn == nil {
		return
	}
	if g == nil || diffusion == nil || len(diffusion.EncoderLayerScalars) != len(g.Layers) {
		fn()
		return
	}
	for i := range g.Layers {
		g.Layers[i].LayerScalarW, diffusion.EncoderLayerScalars[i] = diffusion.EncoderLayerScalars[i], g.Layers[i].LayerScalarW
	}
	defer func() {
		for i := range g.Layers {
			g.Layers[i].LayerScalarW, diffusion.EncoderLayerScalars[i] = diffusion.EncoderLayerScalars[i], g.Layers[i].LayerScalarW
		}
	}()
	fn()
}

func withDiffusionEncoderScalarsQuant(g *QuantModel, diffusion *model.LoadedDiffusion, fn func()) {
	if fn == nil {
		return
	}
	if g == nil || diffusion == nil || len(diffusion.EncoderLayerScalars) != len(g.Layers) {
		fn()
		return
	}
	for i := range g.Layers {
		g.Layers[i].LayerScalarW, diffusion.EncoderLayerScalars[i] = diffusion.EncoderLayerScalars[i], g.Layers[i].LayerScalarW
	}
	defer func() {
		for i := range g.Layers {
			g.Layers[i].LayerScalarW, diffusion.EncoderLayerScalars[i] = diffusion.EncoderLayerScalars[i], g.Layers[i].LayerScalarW
		}
	}()
	fn()
}

func diffusionOnesBF16(n int) []byte {
	if n <= 0 {
		return nil
	}
	return f32ToBf16Slice(fillConst(n, 1))
}
