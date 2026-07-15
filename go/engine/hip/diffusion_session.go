// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"sort"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

const (
	rocmDiffusionDefaultCanvasLength = 64
	rocmDiffusionDefaultMaxSteps     = 16
	rocmDiffusionCanvasSeedStride    = uint64(0x9E3779B97F4A7C15)
)

// ROCmBlockDiffusionOptions configures a DiffusionGemma token stream.
type ROCmBlockDiffusionOptions struct {
	MaxTokens   int
	Temperature float32
	Seed        uint64
	SeedSet     bool
	StopTokens  []int32
}

// ROCmDiffusionStepConfig carries sampler parameters owned by the model route.
type ROCmDiffusionStepConfig struct {
	EntropyBound   float32
	MaxTemperature float32
	MinTemperature float32
	Exponent       float32
	TextVocabSize  int32
	Seed           uint64
}

// ROCmDiffusionStepResult is the result of one canvas denoise pass.
type ROCmDiffusionStepResult struct {
	Canvas      []int32
	Greedy      []int32
	SCEmb       []byte
	Accepted    int
	Changed     int
	MeanEntropy float32
}

// ROCmDiffusionDenoiseRequest contains the canvas and exact resident-prefix
// masks the HIP denoiser must evaluate for one diffusion step.
type ROCmDiffusionDenoiseRequest struct {
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
	StepConfig      ROCmDiffusionStepConfig
}

// ROCmDiffusionGenerateConfig is the model-owned policy used to drive a
// block-diffusion session. Zero values resolve to DiffusionGemma defaults.
type ROCmDiffusionGenerateConfig struct {
	Step                ROCmDiffusionStepConfig
	CanvasLength        int
	MaxTokens           int
	MaxSteps            int
	StabilityThreshold  int
	ConfidenceThreshold float32
	MaxCanvases         int
	StopTokens          []int32
	TextVocabSize       int
	Seed                uint64
	SlidingWindow       int
}

// ROCmDiffusionMetrics records one block-diffusion generation call.
type ROCmDiffusionMetrics struct {
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

// ROCmDiffusionSession is the runtime-owned state needed by the portable
// scheduler. The implementation owns device K/V, denoising, and rollback;
// this layer preserves the Metal-compatible generation order.
type ROCmDiffusionSession interface {
	PrefillTokens([]int32) (int, error)
	CacheOffset() int
	Denoise(context.Context, ROCmDiffusionDenoiseRequest) (ROCmDiffusionStepResult, error)
	TruncateTo(int) error
	CommitTokens([]int32) error
}

// RunROCmDiffusionGenerate runs the portable DiffusionGemma session contract.
func RunROCmDiffusionGenerate(ctx context.Context, cfg ROCmDiffusionGenerateConfig, session ROCmDiffusionSession, prompt []int32, yield func(int32) bool) (ROCmDiffusionMetrics, error) {
	const op = "hip.RunROCmDiffusionGenerate"
	var metrics ROCmDiffusionMetrics
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return metrics, err
	}
	if session == nil {
		return metrics, core.NewError(op + ": diffusion session is nil")
	}
	if len(prompt) == 0 {
		return metrics, core.NewError(op + ": empty prompt")
	}
	cfg = resolveROCmDiffusionGenerateConfig(cfg)
	if cfg.Step.TextVocabSize <= 0 {
		return metrics, core.NewError(op + ": TextVocabSize must be positive")
	}

	started := time.Now()
	prefillStarted := time.Now()
	promptTokens, err := session.PrefillTokens(prompt)
	if err != nil {
		return metrics, core.E(op, "prompt prefill", err)
	}
	metrics.PrefillDur = time.Since(prefillStarted)
	if promptTokens <= 0 {
		return metrics, core.NewError(op + ": prompt encoded to zero tokens")
	}
	metrics.PrefillTokens = promptTokens

	canvasLength := cfg.CanvasLength
	emitted := make([]int32, 0, canvasLength*cfg.MaxCanvases)
	for canvasIndex := 0; canvasIndex < cfg.MaxCanvases; canvasIndex++ {
		if err := ctx.Err(); err != nil {
			return metrics, err
		}
		prefix := promptTokens + len(emitted)
		if offset := session.CacheOffset(); offset >= 0 {
			prefix = offset
		}
		canvas := rocmDiffusionInitialCanvas(canvasLength, cfg.Step.TextVocabSize, cfg.Step.Seed, canvasIndex)
		canvasStep := cfg.Step
		canvasStep.Seed += uint64(canvasIndex) * rocmDiffusionCanvasSeedStride
		keyLength := prefix + canvasLength
		globalMask, globalShape := rocmDiffusionGlobalCanvasMask(canvasLength, keyLength)
		localMask, localShape := rocmDiffusionLocalCanvasMask(canvasLength, keyLength, prefix, cfg.SlidingWindow)

		canvasStarted := time.Now()
		var scEmb []byte
		var previousGreedy []int32
		var lastGreedy []int32
		stableRun := 0
		for step := 0; step < cfg.MaxSteps; step++ {
			if err := ctx.Err(); err != nil {
				return metrics, err
			}
			noise := 1 - float32(step)/float32(cfg.MaxSteps)
			result, err := session.Denoise(ctx, ROCmDiffusionDenoiseRequest{
				Canvas:          append([]int32(nil), canvas...),
				SCEmb:           scEmb,
				CanvasIndex:     canvasIndex,
				Step:            step,
				NoiseProportion: noise,
				Prefix:          prefix,
				GlobalMask:      append([]float32(nil), globalMask...),
				GlobalMaskShape: append([]int(nil), globalShape...),
				LocalMask:       append([]float32(nil), localMask...),
				LocalMaskShape:  append([]int(nil), localShape...),
				StepConfig:      canvasStep,
			})
			if err != nil {
				return metrics, core.E(op, "canvas denoise", err)
			}
			if err := session.TruncateTo(prefix); err != nil {
				return metrics, core.E(op, core.Sprintf("cache declined TruncateTo(%d)", prefix), err)
			}
			metrics.TotalSteps++
			if len(result.Greedy) == 0 {
				return metrics, core.NewError(op + ": denoise returned an empty greedy canvas")
			}
			if rocmDiffusionTokensEqual(result.Greedy, previousGreedy) {
				stableRun++
			} else {
				stableRun = 0
			}
			previousGreedy = append(previousGreedy[:0], result.Greedy...)
			lastGreedy = append(lastGreedy[:0], result.Greedy...)
			scEmb = append(scEmb[:0], result.SCEmb...)
			if stableRun >= cfg.StabilityThreshold && result.MeanEntropy < cfg.ConfidenceThreshold {
				break
			}
			if len(result.Canvas) == 0 {
				return metrics, core.NewError(op + ": denoise returned an empty canvas")
			}
			canvas = append(canvas[:0], result.Canvas...)
		}
		metrics.DenoiseDur += time.Since(canvasStarted)
		kept, stopped := rocmDiffusionKeepUntilStop(lastGreedy, cfg.StopTokens)
		if cfg.MaxTokens > 0 {
			remaining := cfg.MaxTokens - len(emitted)
			if remaining <= 0 {
				break
			}
			if len(kept) > remaining {
				kept = kept[:remaining]
			}
		}
		if len(kept) > 0 {
			commitStarted := time.Now()
			if err := session.CommitTokens(kept); err != nil {
				return metrics, core.E(op, "canvas commit", err)
			}
			metrics.CommitDur += time.Since(commitStarted)
			for _, id := range kept {
				emitted = append(emitted, id)
				metrics.EmittedTokens = len(emitted)
				if yield != nil && !yield(id) {
					return metrics, core.NewError(op + ": yield stopped")
				}
			}
		}
		metrics.Canvases++
		if stopped {
			metrics.StoppedOnToken = true
			break
		}
	}
	metrics.TotalDur = time.Since(started)
	return metrics, nil
}

func resolveROCmDiffusionGenerateConfig(cfg ROCmDiffusionGenerateConfig) ROCmDiffusionGenerateConfig {
	if cfg.CanvasLength <= 0 {
		cfg.CanvasLength = rocmDiffusionDefaultCanvasLength
	}
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = rocmDiffusionDefaultMaxSteps
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
	if cfg.Step.TextVocabSize <= 0 {
		cfg.Step.TextVocabSize = int32(cfg.TextVocabSize)
	}
	if cfg.Step.Seed == 0 {
		cfg.Step.Seed = cfg.Seed
	}
	if cfg.Step.EntropyBound <= 0 {
		cfg.Step.EntropyBound = 0.3
	}
	if cfg.Step.MaxTemperature <= 0 {
		cfg.Step.MaxTemperature = 0.8
	}
	if cfg.Step.MinTemperature <= 0 {
		cfg.Step.MinTemperature = 0.4
	}
	if cfg.Step.Exponent <= 0 {
		cfg.Step.Exponent = 1
	}
	return cfg
}

func rocmDiffusionInitialCanvas(length int, vocab int32, seed uint64, canvasIndex int) []int32 {
	canvas := make([]int32, length)
	if vocab <= 0 {
		return canvas
	}
	sampler := model.NewSampler(seed ^ (uint64(canvasIndex+1) << 32))
	for index := range canvas {
		canvas[index] = int32(sampler.Draw() * float32(vocab))
		if canvas[index] >= vocab {
			canvas[index] = vocab - 1
		}
	}
	return canvas
}

func rocmDiffusionGlobalCanvasMask(canvasLength, keyLength int) ([]float32, []int) {
	return make([]float32, canvasLength*keyLength), []int{1, 1, canvasLength, keyLength}
}

func rocmDiffusionLocalCanvasMask(canvasLength, keyLength, prefix, window int) ([]float32, []int) {
	mask := make([]float32, canvasLength*keyLength)
	if window <= 0 {
		return mask, []int{1, 1, canvasLength, keyLength}
	}
	blocked := float32(math.Inf(-1))
	contextStart := prefix - window
	if contextStart < 0 {
		contextStart = 0
	}
	for query := 0; query < canvasLength; query++ {
		for key := 0; key < keyLength; key++ {
			if (key >= contextStart && key < prefix) || (key >= prefix && key < prefix+canvasLength) {
				continue
			}
			mask[query*keyLength+key] = blocked
		}
	}
	return mask, []int{1, 1, canvasLength, keyLength}
}

func rocmDiffusionKeepUntilStop(canvas, stopTokens []int32) ([]int32, bool) {
	for index, token := range canvas {
		for _, stop := range stopTokens {
			if token == stop {
				return append([]int32(nil), canvas[:index]...), true
			}
		}
	}
	return append([]int32(nil), canvas...), false
}

func rocmDiffusionTokensEqual(left, right []int32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func rocmDiffusionSampleDenoiseStep(logits []float32, canvas []int32, vocab, hidden, step int, noiseProportion float32, cfg ROCmDiffusionStepConfig, encode func([]float32) ([]byte, error)) (ROCmDiffusionStepResult, error) {
	const op = "hip.rocmDiffusionSampleDenoiseStep"
	length := len(canvas)
	if vocab <= 0 || hidden <= 0 {
		return ROCmDiffusionStepResult{}, core.NewError(op + ": vocab and hidden sizes must be positive")
	}
	if cfg.TextVocabSize <= 0 {
		return ROCmDiffusionStepResult{}, core.NewError(op + ": TextVocabSize must be positive")
	}
	if len(logits) != length*vocab {
		return ROCmDiffusionStepResult{}, core.NewError(op + ": logits must be len(canvas)*vocab float32 values")
	}
	if encode == nil {
		return ROCmDiffusionStepResult{}, core.NewError(op + ": encode callback is nil")
	}
	if length == 0 {
		return ROCmDiffusionStepResult{Canvas: []int32{}, Greedy: []int32{}, SCEmb: []byte{}}, nil
	}

	fraction := 1 - float32(math.Pow(float64(1-noiseProportion), float64(cfg.Exponent)))
	temperature := cfg.MinTemperature + fraction*(cfg.MaxTemperature-cfg.MinTemperature)
	if temperature <= 0 {
		temperature = 1e-6
	}
	shaped := make([]float32, len(logits))
	shapedBF16 := make([]byte, len(logits)*2)
	for index, logit := range logits {
		value := logit / temperature
		bf16 := hipFloat32ToBFloat16(value)
		binary.LittleEndian.PutUint16(shapedBF16[index*2:], bf16)
		shaped[index] = hipBFloat16ToFloat32(bf16)
	}

	probabilities := make([]float32, len(shaped))
	categorical := model.NewSampler(cfg.Seed ^ (uint64(step)*2 + 1))
	renoiseSampler := model.NewSampler(cfg.Seed ^ (uint64(step)*2 + 2))
	sampled := make([]int32, length)
	greedy := make([]int32, length)
	entropies := make([]float32, length)
	var entropySum float32
	for row := 0; row < length; row++ {
		rowBytes := shapedBF16[row*vocab*2 : (row+1)*vocab*2]
		id, err := categorical.Sample(rowBytes, vocab, model.SampleParams{Temperature: 1})
		if err != nil {
			return ROCmDiffusionStepResult{}, err
		}
		sampled[row] = id
		id, err = model.Greedy(rowBytes, vocab)
		if err != nil {
			return ROCmDiffusionStepResult{}, err
		}
		greedy[row] = id
		rowLogits := shaped[row*vocab : (row+1)*vocab]
		rowProbabilities := probabilities[row*vocab : (row+1)*vocab]
		entropy, err := rocmDiffusionSoftmaxEntropy(rowLogits, rowProbabilities)
		if err != nil {
			return ROCmDiffusionStepResult{}, err
		}
		entropies[row] = entropy
		entropySum += entropy
	}

	scEmb, err := encode(probabilities)
	if err != nil {
		return ROCmDiffusionStepResult{}, err
	}
	if len(scEmb) != length*hidden*2 {
		return ROCmDiffusionStepResult{}, core.NewError(op + ": self-conditioning embedding byte count mismatch")
	}

	renoise := make([]int32, length)
	for index := range renoise {
		id := int32(renoiseSampler.Draw() * float32(cfg.TextVocabSize))
		if id >= cfg.TextVocabSize {
			id = cfg.TextVocabSize - 1
		}
		renoise[index] = id
	}
	order := make([]int, length)
	for index := range order {
		order[index] = index
	}
	sort.Slice(order, func(left, right int) bool { return entropies[order[left]] < entropies[order[right]] })
	accepted := make([]bool, length)
	acceptedCount := 0
	var accumulated float32
	for _, index := range order {
		if accumulated > cfg.EntropyBound {
			break
		}
		accepted[index] = true
		acceptedCount++
		accumulated += entropies[index]
	}
	next := make([]int32, length)
	changed := 0
	for index := range next {
		if accepted[index] {
			next[index] = sampled[index]
		} else {
			next[index] = renoise[index]
		}
		if next[index] != canvas[index] {
			changed++
		}
	}
	return ROCmDiffusionStepResult{
		Canvas:      next,
		Greedy:      greedy,
		SCEmb:       scEmb,
		Accepted:    acceptedCount,
		Changed:     changed,
		MeanEntropy: entropySum / float32(length),
	}, nil
}

func rocmDiffusionSoftmaxEntropy(logits, probabilities []float32) (float32, error) {
	if len(logits) == 0 || len(probabilities) != len(logits) {
		return 0, core.NewError("hip.rocmDiffusionSoftmaxEntropy: row shape mismatch")
	}
	maximum := logits[0]
	for _, value := range logits[1:] {
		if value > maximum {
			maximum = value
		}
	}
	var sum, weighted float32
	for index, value := range logits {
		probability := float32(math.Exp(float64(value - maximum)))
		probabilities[index] = probability
		sum += probability
		weighted += probability * value
	}
	if sum <= 0 || math.IsNaN(float64(sum)) || math.IsInf(float64(sum), 0) {
		return 0, core.NewError("hip.rocmDiffusionSoftmaxEntropy: probability sum is not finite")
	}
	for index := range probabilities {
		probabilities[index] /= sum
	}
	return maximum + float32(math.Log(float64(sum))) - weighted/sum, nil
}

func (model *hipLoadedModel) loadedDiffusionGemmaEncoderLayerScalar(layer int) (float32, error) {
	if model == nil || model.driver == nil {
		return 0, core.E("rocm.hip.DiffusionGemma", "loaded model is required", nil)
	}
	base := core.Sprintf("model.encoder.language_model.layers.%d.layer_scalar", layer)
	var tensor hipTensor
	var ok bool
	for _, name := range []string{base, base + ".weight"} {
		if tensor, ok = model.tensors[name]; ok {
			break
		}
	}
	if !ok {
		return 0, core.E("rocm.hip.DiffusionGemma", core.Sprintf("encoder layer %d scalar tensor is required", layer), nil)
	}
	bytes, err := hipGemma4LayerScalarBytes(tensor.info, "encoder layer scalar tensor")
	if err != nil {
		return 0, err
	}
	if tensor.pointer == 0 {
		return 0, core.E("rocm.hip.DiffusionGemma", "encoder layer scalar tensor pointer is required", nil)
	}
	payload := make([]byte, bytes)
	if err := model.driver.CopyDeviceToHost(tensor.pointer, payload); err != nil {
		return 0, core.E("rocm.hip.DiffusionGemma", "copy encoder layer scalar", err)
	}
	return hipGemma4LayerScalarValue(tensor.info, payload)
}

func hipDiffusionDenoiseForwardConfig(base hipGemma4Q4ForwardConfig, encoderScalars []float32, canvasLength int) (hipGemma4Q4ForwardConfig, error) {
	if canvasLength <= 0 {
		return hipGemma4Q4ForwardConfig{}, core.E("rocm.hip.DiffusionGemma", "canvas length must be positive", nil)
	}
	if len(base.Layers) == 0 || len(encoderScalars) != len(base.Layers) {
		return hipGemma4Q4ForwardConfig{}, core.E("rocm.hip.DiffusionGemma", "encoder layer scalar count mismatch", nil)
	}
	out := base
	out.Layers = append([]hipGemma4Q4Layer0Config(nil), base.Layers...)
	for index := range out.Layers {
		out.Layers[index].LayerScalar = encoderScalars[index]
		if out.Layers[index].SlidingWindow > 0 {
			out.Layers[index].SlidingWindow += canvasLength
		}
	}
	return out, nil
}

type hipROCmDiffusionSessionProvider interface {
	OpenROCmDiffusionSession(context.Context, *hipLoadedModel) (ROCmDiffusionSession, error)
}

// OpenROCmDiffusionSession binds a loaded DiffusionGemma payload to the
// portable scheduler through the runtime's canvas-denoiser implementation.
func (model *hipLoadedModel) OpenROCmDiffusionSession(ctx context.Context) (ROCmDiffusionSession, error) {
	const op = "hip.LoadedModel.OpenROCmDiffusionSession"
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if model == nil {
		return nil, core.NewError(op + ": loaded model is nil")
	}
	if model.modelIdentity().Architecture != "diffusion_gemma" {
		return nil, core.NewError(op + ": model is not DiffusionGemma")
	}
	provider, ok := model.kernelSet().(hipROCmDiffusionSessionProvider)
	if !ok {
		return nil, core.NewError(op + ": HIP diffusion denoiser is not linked")
	}
	session, err := provider.OpenROCmDiffusionSession(ctx, model)
	if err != nil {
		return nil, core.E(op, "open runtime session", err)
	}
	if session == nil {
		return nil, core.NewError(op + ": runtime returned a nil diffusion session")
	}
	return session, nil
}

// GenerateBlockDiffusionTokens reaches the runtime-owned DiffusionGemma
// session directly, without falling back to autoregressive generation.
func (model *rocmModel) GenerateBlockDiffusionTokens(ctx context.Context, prompt []int32, opts ROCmBlockDiffusionOptions, yield func(int32) bool) (ROCmDiffusionMetrics, error) {
	const op = "rocm.Model.GenerateBlockDiffusionTokens"
	var metrics ROCmDiffusionMetrics
	if model == nil || model.native == nil {
		return metrics, core.NewError(op + ": native model is nil")
	}
	if opts.MaxTokens <= 0 {
		return metrics, core.NewError(op + ": MaxTokens must be > 0")
	}
	loaded, ok := model.native.(*hipLoadedModel)
	if !ok {
		return metrics, core.NewError(op + ": native model has no HIP diffusion session")
	}
	session, err := loaded.OpenROCmDiffusionSession(ctx)
	if err != nil {
		return metrics, err
	}
	if closer, ok := session.(interface{ Close() error }); ok {
		defer func() { _ = closer.Close() }()
	}
	route, _ := ROCmDiffusionSamplerRouteForInfo(loaded.modelPath, model.modelInfo, model.modelLabels)
	canvasLength := rocmDiffusionLabelPositiveInt(model.modelLabels["diffusion_canvas_length"])
	if canvasLength <= 0 {
		canvasLength = route.DefaultCanvasLength
	}
	if canvasLength <= 0 {
		canvasLength = rocmDiffusionDefaultCanvasLength
	}
	if canvasLength > opts.MaxTokens {
		canvasLength = opts.MaxTokens
	}
	maxCanvases := (opts.MaxTokens + canvasLength - 1) / canvasLength
	cfg := ROCmDiffusionGenerateConfig{
		Step: ROCmDiffusionStepConfig{
			EntropyBound:   float32(route.EntropyBound),
			MaxTemperature: float32(route.MaxTemperature),
			MinTemperature: float32(route.MinTemperature),
			Exponent:       float32(route.TemperatureExponent),
		},
		CanvasLength:        canvasLength,
		MaxTokens:           opts.MaxTokens,
		MaxSteps:            route.DefaultMaxSteps,
		StabilityThreshold:  route.StabilityThreshold,
		ConfidenceThreshold: float32(route.ConfidenceThreshold),
		MaxCanvases:         maxCanvases,
		StopTokens:          append([]int32(nil), opts.StopTokens...),
		TextVocabSize:       model.modelInfo.VocabSize,
		SlidingWindow: rocmDiffusionFirstPositiveLabel(model.modelLabels,
			"attention_sliding_window",
			"sliding_window",
			"gemma4_sliding_window",
		),
	}
	if opts.SeedSet {
		cfg.Seed = opts.Seed
	}
	if opts.Temperature > 0 {
		cfg.Step.MinTemperature = opts.Temperature
		cfg.Step.MaxTemperature = opts.Temperature
	}
	return RunROCmDiffusionGenerate(ctx, cfg, session, prompt, yield)
}

func rocmDiffusionLabelPositiveInt(value string) int {
	parsed := core.ParseInt(value, 10, 64)
	if !parsed.OK {
		return 0
	}
	number, ok := parsed.Value.(int64)
	if !ok || number <= 0 {
		return 0
	}
	return int(number)
}

func rocmDiffusionFirstPositiveLabel(labels map[string]string, keys ...string) int {
	for _, key := range keys {
		if value := rocmDiffusionLabelPositiveInt(labels[key]); value > 0 {
			return value
		}
	}
	return 0
}
