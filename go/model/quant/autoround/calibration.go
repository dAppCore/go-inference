// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import core "dappco.re/go"

type CalibrationSample struct {
	ID      string `json:"id,omitempty"`
	Text    string `json:"text,omitempty"`
	TokenN  int    `json:"token_count,omitempty"`
	Skipped bool   `json:"skipped,omitempty"`
}

type CalibrationConfig struct {
	Scheme       Scheme  `json:"scheme,omitempty"`
	Bits         int     `json:"bits,omitempty"`
	GroupSize    int     `json:"group_size,omitempty"`
	Symmetric    bool    `json:"sym,omitempty"`
	Iters        int     `json:"iters,omitempty"`
	LearningRate float32 `json:"lr,omitempty"`
	NSamples     int     `json:"nsamples,omitempty"`
	SeqLen       int     `json:"seqlen,omitempty"`
}

type CalibrationPlan struct {
	Config          CalibrationConfig   `json:"config"`
	Samples         []CalibrationSample `json:"samples,omitempty"`
	SelectedSamples int                 `json:"selected_samples"`
	InputSamples    int                 `json:"input_samples"`
	TokenCount      int                 `json:"token_count,omitempty"`
	Truncated       bool                `json:"truncated,omitempty"`
	Notes           []string            `json:"notes,omitempty"`
}

type QuantizeRun struct {
	Plan    CalibrationPlan  `json:"plan"`
	Weights QuantizedWeights `json:"weights"`
}

func CalibrationConfigFromProfile(profile Profile) CalibrationConfig {
	cfg := ConfigFromProfile(profile)
	return CalibrationConfig{
		Scheme:       cfg.Scheme,
		Bits:         cfg.Bits,
		GroupSize:    cfg.GroupSize,
		Symmetric:    cfg.Symmetric,
		Iters:        cfg.Iters,
		LearningRate: cfg.LearningRate,
		NSamples:     profile.NSamples,
		SeqLen:       profile.SeqLen,
	}
}

func BuildCalibrationPlan(samples []CalibrationSample, cfg CalibrationConfig) (CalibrationPlan, error) {
	cfg, err := normaliseCalibrationConfig(cfg)
	if err != nil {
		return CalibrationPlan{}, err
	}
	plan := CalibrationPlan{
		Config:       cfg,
		InputSamples: len(samples),
		Notes: []string{
			"Calibration planning is native metadata only; model-gradient capture is supplied by the caller before SignRound quantization.",
		},
	}
	if len(samples) == 0 {
		return plan, nil
	}
	limit := min(cfg.NSamples, len(samples))
	plan.Truncated = limit < len(samples)
	plan.Samples = make([]CalibrationSample, 0, limit)
	for _, sample := range samples[:limit] {
		sample.TokenN = boundedCalibrationTokenCount(sample, cfg.SeqLen)
		if sample.TokenN == 0 {
			sample.Skipped = true
		}
		plan.TokenCount += sample.TokenN
		plan.Samples = append(plan.Samples, sample)
	}
	plan.SelectedSamples = len(plan.Samples)
	return plan, nil
}

func QuantizeWithCalibration(weights []float32, gradients []float32, samples []CalibrationSample, cfg CalibrationConfig) (QuantizeRun, error) {
	plan, err := BuildCalibrationPlan(samples, cfg)
	if err != nil {
		return QuantizeRun{}, err
	}
	quantCfg := QuantizeConfig{
		Scheme:       plan.Config.Scheme,
		Bits:         plan.Config.Bits,
		GroupSize:    plan.Config.GroupSize,
		Symmetric:    plan.Config.Symmetric,
		Iters:        plan.Config.Iters,
		LearningRate: plan.Config.LearningRate,
		Gradients:    gradients,
	}
	quantized, err := QuantizeWeights(weights, quantCfg)
	if err != nil {
		return QuantizeRun{}, err
	}
	return QuantizeRun{Plan: plan, Weights: quantized}, nil
}

func normaliseCalibrationConfig(cfg CalibrationConfig) (CalibrationConfig, error) {
	quantCfg, err := normaliseQuantizeConfig(QuantizeConfig{
		Scheme:       cfg.Scheme,
		Bits:         cfg.Bits,
		GroupSize:    cfg.GroupSize,
		Symmetric:    cfg.Symmetric,
		Iters:        cfg.Iters,
		LearningRate: cfg.LearningRate,
	})
	if err != nil {
		return cfg, err
	}
	cfg.Scheme = quantCfg.Scheme
	cfg.Bits = quantCfg.Bits
	cfg.GroupSize = quantCfg.GroupSize
	cfg.Symmetric = quantCfg.Symmetric
	cfg.Iters = quantCfg.Iters
	cfg.LearningRate = quantCfg.LearningRate
	if cfg.NSamples == 0 {
		cfg.NSamples = 128
	}
	if cfg.SeqLen == 0 {
		cfg.SeqLen = 2048
	}
	if cfg.NSamples < 0 {
		return cfg, core.NewError("autoround: nsamples must be non-negative")
	}
	if cfg.SeqLen < 0 {
		return cfg, core.NewError("autoround: seqlen must be non-negative")
	}
	return cfg, nil
}

func boundedCalibrationTokenCount(sample CalibrationSample, seqLen int) int {
	count := sample.TokenN
	if count == 0 && sample.Text != "" {
		count = countCalibrationTextFields(sample.Text)
	}
	if count < 0 {
		count = 0
	}
	if seqLen > 0 && count > seqLen {
		return seqLen
	}
	return count
}

func countCalibrationTextFields(text string) int {
	count := 0
	inField := false
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case ' ', '\t', '\n', '\r':
			inField = false
		default:
			if !inField {
				count++
				inField = true
			}
		}
	}
	return count
}
