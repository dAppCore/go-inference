// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "math"

const (
	DiffusionDefaultCanvasLength     = 64
	DiffusionReferenceCanvasLength   = 256
	DiffusionDefaultMaxSteps         = 16
	DiffusionReferenceMaxSteps       = 48
	DiffusionDefaultStabilitySteps   = 1
	DiffusionDefaultConfidence       = 0.005
	DiffusionDefaultEntropyBound     = 0.3
	DiffusionReferenceEntropyBound   = 0.1
	DiffusionDefaultMaxTemperature   = 0.8
	DiffusionDefaultMinTemperature   = 0.4
	DiffusionDefaultTempExponent     = 1.0
	diffusionStepTemperatureFloor    = 1e-6
	diffusionCanvasStepSeedIncrement = 0x9E3779B97F4A7C15
)

// DiffusionStepPolicy is the backend-neutral denoising-step sampler contract
// used by DiffusionGemma runtimes.
type DiffusionStepPolicy struct {
	EntropyBound     float64
	MaxTemperature   float64
	MinTemperature   float64
	Exponent         float64
	TextVocabSize    int
	Seed             uint64
	ReferenceEntropy float64
}

// DiffusionGeneratePolicy is the model-owned block-diffusion generation
// contract. ROCm runtimes can consume it without importing the MLX reference.
type DiffusionGeneratePolicy struct {
	Step                  DiffusionStepPolicy
	CanvasLength          int
	MaxSteps              int
	StabilityThreshold    int
	ConfidenceThreshold   float64
	MaxCanvases           int
	StopTokens            []int32
	ReferenceCanvasLength int
	ReferenceMaxSteps     int
}

type DiffusionPolicyConfig struct {
	CanvasLength          int
	TextVocabSize         int
	VocabSize             int
	MaxSteps              int
	StabilityThreshold    int
	ConfidenceThreshold   float64
	MaxCanvases           int
	StopTokens            []int32
	Seed                  uint64
	EntropyBound          float64
	MaxTemperature        float64
	MinTemperature        float64
	TemperatureExponent   float64
	ReferenceCanvasLength int
	ReferenceMaxSteps     int
}

func DefaultDiffusionStepPolicy(textVocabSize int) DiffusionStepPolicy {
	return DiffusionStepPolicy{
		EntropyBound:     DiffusionDefaultEntropyBound,
		MaxTemperature:   DiffusionDefaultMaxTemperature,
		MinTemperature:   DiffusionDefaultMinTemperature,
		Exponent:         DiffusionDefaultTempExponent,
		TextVocabSize:    positiveInt(textVocabSize),
		ReferenceEntropy: DiffusionReferenceEntropyBound,
	}
}

func DiffusionGeneratePolicyOf(cfg DiffusionPolicyConfig) DiffusionGeneratePolicy {
	textVocabSize := firstPositiveIntValue(cfg.TextVocabSize, cfg.VocabSize)
	step := DefaultDiffusionStepPolicy(textVocabSize)
	step.Seed = cfg.Seed
	if cfg.EntropyBound > 0 {
		step.EntropyBound = cfg.EntropyBound
	}
	if cfg.MaxTemperature > 0 {
		step.MaxTemperature = cfg.MaxTemperature
	}
	if cfg.MinTemperature > 0 {
		step.MinTemperature = cfg.MinTemperature
	}
	if cfg.TemperatureExponent > 0 {
		step.Exponent = cfg.TemperatureExponent
	}
	return DiffusionGeneratePolicy{
		Step:                  step,
		CanvasLength:          firstPositiveIntValue(cfg.CanvasLength, DiffusionDefaultCanvasLength),
		MaxSteps:              firstPositiveIntValue(cfg.MaxSteps, DiffusionDefaultMaxSteps),
		StabilityThreshold:    firstPositiveIntValue(cfg.StabilityThreshold, DiffusionDefaultStabilitySteps),
		ConfidenceThreshold:   firstPositiveFloatValue(cfg.ConfidenceThreshold, DiffusionDefaultConfidence),
		MaxCanvases:           firstPositiveIntValue(cfg.MaxCanvases, 1),
		StopTokens:            append([]int32(nil), cfg.StopTokens...),
		ReferenceCanvasLength: firstPositiveIntValue(cfg.ReferenceCanvasLength, DiffusionReferenceCanvasLength),
		ReferenceMaxSteps:     firstPositiveIntValue(cfg.ReferenceMaxSteps, DiffusionReferenceMaxSteps),
	}
}

func DiffusionNoiseAtStep(step, maxSteps int) float64 {
	maxSteps = firstPositiveIntValue(maxSteps, DiffusionDefaultMaxSteps)
	if step < 0 {
		step = 0
	}
	return 1.0 - float64(step)/float64(maxSteps)
}

func DiffusionTemperature(noiseProportion float64, step DiffusionStepPolicy) float64 {
	if step.MaxTemperature <= 0 {
		step.MaxTemperature = DiffusionDefaultMaxTemperature
	}
	if step.MinTemperature <= 0 {
		step.MinTemperature = DiffusionDefaultMinTemperature
	}
	if step.Exponent <= 0 {
		step.Exponent = DiffusionDefaultTempExponent
	}
	frac := 1.0 - math.Pow(1.0-noiseProportion, step.Exponent)
	temp := step.MinTemperature + frac*(step.MaxTemperature-step.MinTemperature)
	if temp <= 0 {
		return diffusionStepTemperatureFloor
	}
	return temp
}

func DiffusionInitialCanvasSeed(base uint64, canvasIndex int) uint64 {
	if canvasIndex < 0 {
		canvasIndex = 0
	}
	return base ^ (uint64(canvasIndex+1) << 32)
}

func DiffusionCanvasStepSeed(base uint64, canvasIndex int) uint64 {
	if canvasIndex < 0 {
		canvasIndex = 0
	}
	return base + uint64(canvasIndex)*diffusionCanvasStepSeedIncrement
}

func DiffusionConverged(stableRun int, meanEntropy float64, policy DiffusionGeneratePolicy) bool {
	stability := firstPositiveIntValue(policy.StabilityThreshold, DiffusionDefaultStabilitySteps)
	confidence := firstPositiveFloatValue(policy.ConfidenceThreshold, DiffusionDefaultConfidence)
	return stableRun >= stability && meanEntropy < confidence
}

func ApplyDiffusionPolicyLabels(labels map[string]string, policy DiffusionGeneratePolicy) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	policy = DiffusionGeneratePolicyOf(DiffusionPolicyConfig{
		CanvasLength:          policy.CanvasLength,
		TextVocabSize:         policy.Step.TextVocabSize,
		MaxSteps:              policy.MaxSteps,
		StabilityThreshold:    policy.StabilityThreshold,
		ConfidenceThreshold:   policy.ConfidenceThreshold,
		MaxCanvases:           policy.MaxCanvases,
		StopTokens:            policy.StopTokens,
		Seed:                  policy.Step.Seed,
		EntropyBound:          policy.Step.EntropyBound,
		MaxTemperature:        policy.Step.MaxTemperature,
		MinTemperature:        policy.Step.MinTemperature,
		TemperatureExponent:   policy.Step.Exponent,
		ReferenceCanvasLength: policy.ReferenceCanvasLength,
		ReferenceMaxSteps:     policy.ReferenceMaxSteps,
	})
	setDiffusionIntLabel(labels, "default_canvas_length", policy.CanvasLength)
	setDiffusionIntLabel(labels, "reference_canvas_length", policy.ReferenceCanvasLength)
	setDiffusionIntLabel(labels, "default_max_steps", policy.MaxSteps)
	setDiffusionIntLabel(labels, "reference_max_steps", policy.ReferenceMaxSteps)
	setDiffusionIntLabel(labels, "stability_threshold", policy.StabilityThreshold)
	setDiffusionIntLabel(labels, "max_canvases", policy.MaxCanvases)
	setDiffusionIntLabel(labels, "text_vocab_size", policy.Step.TextVocabSize)
	setDiffusionFloatLabel(labels, "confidence_threshold", policy.ConfidenceThreshold)
	setDiffusionFloatLabel(labels, "entropy_bound", policy.Step.EntropyBound)
	setDiffusionFloatLabel(labels, "reference_entropy_bound", policy.Step.ReferenceEntropy)
	setDiffusionFloatLabel(labels, "max_temperature", policy.Step.MaxTemperature)
	setDiffusionFloatLabel(labels, "min_temperature", policy.Step.MinTemperature)
	setDiffusionFloatLabel(labels, "temperature_exponent", policy.Step.Exponent)
	return labels
}

func setDiffusionIntLabel(labels map[string]string, suffix string, value int) {
	if value <= 0 {
		return
	}
	setPositiveIntLabel(labels, "diffusion_"+suffix, value)
	setPositiveIntLabel(labels, "gemma4_diffusion_"+suffix, value)
}

func setDiffusionFloatLabel(labels map[string]string, suffix string, value float64) {
	if value <= 0 {
		return
	}
	setPositiveFloatLabel(labels, "diffusion_"+suffix, value)
	setPositiveFloatLabel(labels, "gemma4_diffusion_"+suffix, value)
}

func firstPositiveFloatValue(values ...float64) float64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}
