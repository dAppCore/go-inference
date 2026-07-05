// SPDX-Licence-Identifier: EUPL-1.2

// Package autoround contains native AutoRound quantisation profiles and the
// weight-only rounding primitive used by pack-level quantisers.
package autoround

import (
	"math"

	core "dappco.re/go"
)

type ProfileID string
type Scheme string
type ExportFormat string

const (
	ProfileAutoRound      ProfileID = "auto-round"
	ProfileAutoRoundBest  ProfileID = "auto-round-best"
	ProfileAutoRoundLight ProfileID = "auto-round-light"

	SchemeW2A16     Scheme = "W2A16"
	SchemeW4A16     Scheme = "W4A16"
	SchemeW8A16     Scheme = "W8A16"
	SchemeMXFP4     Scheme = "MXFP4"
	SchemeNVFP4     Scheme = "NVFP4"
	SchemeFP8Static Scheme = "FP8_STATIC"
	SchemeGGUFQ4KM  Scheme = "GGUF:Q4_K_M"

	FormatAutoRound ExportFormat = "auto_round"
	FormatGGUFQ4KM  ExportFormat = "gguf:q4_k_m"
)

type SchemeInfo struct {
	Scheme         Scheme       `json:"scheme"`
	Bits           int          `json:"bits"`
	ActivationBits int          `json:"activation_bits"`
	GroupSize      int          `json:"group_size,omitempty"`
	Symmetric      bool         `json:"symmetric,omitempty"`
	ExportFormat   ExportFormat `json:"export_format"`
	Family         string       `json:"family,omitempty"`
}

type Profile struct {
	ID           ProfileID    `json:"id"`
	Scheme       Scheme       `json:"scheme"`
	ExportFormat ExportFormat `json:"export_format"`
	Iters        int          `json:"iters"`
	NSamples     int          `json:"nsamples"`
	SeqLen       int          `json:"seqlen"`
	GroupSize    int          `json:"group_size"`
	Symmetric    bool         `json:"sym"`
	LearningRate float32      `json:"lr,omitempty"`
	Notes        []string     `json:"notes,omitempty"`
}

type QuantizeConfig struct {
	Scheme       Scheme    `json:"scheme,omitempty"`
	Bits         int       `json:"bits,omitempty"`
	GroupSize    int       `json:"group_size,omitempty"`
	Symmetric    bool      `json:"sym,omitempty"`
	Iters        int       `json:"iters,omitempty"`
	LearningRate float32   `json:"lr,omitempty"`
	Gradients    []float32 `json:"-"`
}

type QuantizedWeights struct {
	Scheme      Scheme    `json:"scheme,omitempty"`
	Bits        int       `json:"bits"`
	GroupSize   int       `json:"group_size"`
	Symmetric   bool      `json:"sym"`
	Iters       int       `json:"iters,omitempty"`
	QValues     []int16   `json:"qvalues,omitempty"`
	Dequantized []float32 `json:"dequantized,omitempty"`
	Scales      []float32 `json:"scales,omitempty"`
	ZeroPoints  []float32 `json:"zero_points,omitempty"`
}

func BuiltinProfiles() []Profile {
	profiles := []Profile{
		{
			ID:           ProfileAutoRound,
			Scheme:       SchemeW4A16,
			ExportFormat: FormatAutoRound,
			Iters:        200,
			NSamples:     128,
			SeqLen:       2048,
			GroupSize:    128,
			Symmetric:    true,
			LearningRate: 5e-3,
			Notes:        []string{"default W4A16 SignRound profile"},
		},
		{
			ID:           ProfileAutoRoundBest,
			Scheme:       SchemeW2A16,
			ExportFormat: FormatAutoRound,
			Iters:        1000,
			NSamples:     512,
			SeqLen:       2048,
			GroupSize:    32,
			Symmetric:    true,
			LearningRate: 5e-3,
			Notes:        []string{"accuracy-first W2A16 profile; enables longer SignRound optimisation"},
		},
		{
			ID:           ProfileAutoRoundLight,
			Scheme:       SchemeW4A16,
			ExportFormat: FormatAutoRound,
			Iters:        50,
			NSamples:     128,
			SeqLen:       2048,
			GroupSize:    128,
			Symmetric:    true,
			LearningRate: 5e-3,
			Notes:        []string{"faster W4A16 profile for local calibration smoke runs"},
		},
	}
	out := make([]Profile, len(profiles))
	for i, profile := range profiles {
		out[i] = cloneProfile(profile)
	}
	return out
}

func LookupProfile(id ProfileID) (Profile, bool) {
	for _, profile := range BuiltinProfiles() {
		if profile.ID == id {
			return profile, true
		}
	}
	return Profile{}, false
}

func ConfigFromProfile(profile Profile) QuantizeConfig {
	return QuantizeConfig{
		Scheme:       profile.Scheme,
		Bits:         profile.GroupScheme().Bits,
		GroupSize:    profile.GroupSize,
		Symmetric:    profile.Symmetric,
		Iters:        profile.Iters,
		LearningRate: profile.LearningRate,
	}
}

func (profile Profile) GroupScheme() SchemeInfo {
	info, ok := ResolveScheme(profile.Scheme)
	if !ok {
		return SchemeInfo{Scheme: profile.Scheme}
	}
	if profile.GroupSize > 0 {
		info.GroupSize = profile.GroupSize
	}
	info.Symmetric = profile.Symmetric
	info.ExportFormat = profile.ExportFormat
	return info
}

func ResolveScheme(scheme Scheme) (SchemeInfo, bool) {
	normal := normaliseScheme(scheme)
	switch normal {
	case SchemeW2A16:
		return SchemeInfo{Scheme: SchemeW2A16, Bits: 2, ActivationBits: 16, GroupSize: 128, Symmetric: true, ExportFormat: FormatAutoRound, Family: "int_woq"}, true
	case SchemeW4A16:
		return SchemeInfo{Scheme: SchemeW4A16, Bits: 4, ActivationBits: 16, GroupSize: 128, Symmetric: true, ExportFormat: FormatAutoRound, Family: "int_woq"}, true
	case SchemeW8A16:
		return SchemeInfo{Scheme: SchemeW8A16, Bits: 8, ActivationBits: 16, GroupSize: 128, Symmetric: true, ExportFormat: FormatAutoRound, Family: "int_woq"}, true
	case SchemeMXFP4:
		return SchemeInfo{Scheme: SchemeMXFP4, Bits: 4, ActivationBits: 16, GroupSize: 32, ExportFormat: FormatAutoRound, Family: "mx_fp"}, true
	case SchemeNVFP4:
		return SchemeInfo{Scheme: SchemeNVFP4, Bits: 4, ActivationBits: 16, GroupSize: 16, ExportFormat: FormatAutoRound, Family: "nv_fp"}, true
	case SchemeFP8Static:
		return SchemeInfo{Scheme: SchemeFP8Static, Bits: 8, ActivationBits: 16, ExportFormat: FormatAutoRound, Family: "fp8"}, true
	case SchemeGGUFQ4KM:
		return SchemeInfo{Scheme: SchemeGGUFQ4KM, Bits: 4, ActivationBits: 16, GroupSize: 256, ExportFormat: FormatGGUFQ4KM, Family: "gguf"}, true
	default:
		return SchemeInfo{}, false
	}
}

func QuantizeWeights(weights []float32, cfg QuantizeConfig) (QuantizedWeights, error) {
	if len(weights) == 0 {
		return QuantizedWeights{}, core.NewError("autoround: weights are required")
	}
	cfg, err := normaliseQuantizeConfig(cfg)
	if err != nil {
		return QuantizedWeights{}, err
	}
	if cfg.Iters > 0 && len(cfg.Gradients) != 0 && len(cfg.Gradients) != len(weights) {
		return QuantizedWeights{}, core.NewError("autoround: gradient count must match weights")
	}
	groups := (len(weights) + cfg.GroupSize - 1) / cfg.GroupSize
	out := QuantizedWeights{
		Scheme:      cfg.Scheme,
		Bits:        cfg.Bits,
		GroupSize:   cfg.GroupSize,
		Symmetric:   cfg.Symmetric,
		Iters:       cfg.Iters,
		QValues:     make([]int16, len(weights)),
		Dequantized: make([]float32, len(weights)),
		Scales:      make([]float32, groups),
		ZeroPoints:  make([]float32, groups),
	}
	for group := 0; group < groups; group++ {
		start := group * cfg.GroupSize
		end := start + cfg.GroupSize
		if end > len(weights) {
			end = len(weights)
		}
		scale, zero := quantParams(weights[start:end], cfg)
		out.Scales[group] = scale
		out.ZeroPoints[group] = zero
		for i := start; i < end; i++ {
			q := quantizeOne(weights[i], scale, zero, cfg)
			if cfg.Iters > 0 && len(cfg.Gradients) == len(weights) {
				q = signRoundAdjust(q, weights[i], cfg.Gradients[i], scale, zero, cfg)
			}
			out.QValues[i] = int16(q)
			out.Dequantized[i] = (float32(q) - zero) * scale
		}
	}
	return out, nil
}

func normaliseQuantizeConfig(cfg QuantizeConfig) (QuantizeConfig, error) {
	if cfg.Scheme != "" {
		info, ok := ResolveScheme(cfg.Scheme)
		if !ok {
			return cfg, core.NewError("autoround: unsupported scheme: " + string(cfg.Scheme))
		}
		if cfg.Bits == 0 {
			cfg.Bits = info.Bits
		}
		if cfg.GroupSize == 0 {
			cfg.GroupSize = info.GroupSize
		}
		if !cfg.Symmetric {
			cfg.Symmetric = info.Symmetric
		}
	}
	if cfg.Bits == 0 {
		cfg.Bits = 4
	}
	if cfg.GroupSize == 0 {
		cfg.GroupSize = 128
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 5e-3
	}
	if cfg.Bits != 2 && cfg.Bits != 3 && cfg.Bits != 4 && cfg.Bits != 8 {
		return cfg, core.NewError("autoround: bits must be one of 2, 3, 4, or 8")
	}
	if cfg.GroupSize != 32 && cfg.GroupSize != 64 && cfg.GroupSize != 128 && cfg.GroupSize != 256 {
		return cfg, core.NewError("autoround: group size must be one of 32, 64, 128, or 256")
	}
	if cfg.Iters < 0 {
		return cfg, core.NewError("autoround: iters must be non-negative")
	}
	if cfg.LearningRate < 0 || math.IsNaN(float64(cfg.LearningRate)) || math.IsInf(float64(cfg.LearningRate), 0) {
		return cfg, core.NewError("autoround: learning rate must be finite and non-negative")
	}
	return cfg, nil
}

func quantParams(values []float32, cfg QuantizeConfig) (float32, float32) {
	minValue, maxValue := values[0], values[0]
	for _, value := range values[1:] {
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
	}
	if cfg.Symmetric {
		qmaxInt := (1 << (cfg.Bits - 1)) - 1
		qmax := float32(qmaxInt)
		maxAbs := float32(math.Max(math.Abs(float64(minValue)), math.Abs(float64(maxValue))))
		if maxAbs == 0 {
			return 1, 0
		}
		return maxAbs / qmax, 0
	}
	qmaxInt := (1 << cfg.Bits) - 1
	qmax := float32(qmaxInt)
	if maxValue == minValue {
		return 1, 0
	}
	scale := (maxValue - minValue) / qmax
	zero := float32(math.Round(float64(-minValue / scale)))
	return scale, zero
}

func quantizeOne(value, scale, zero float32, cfg QuantizeConfig) int {
	q := int(math.Round(float64(value/scale + zero)))
	qmin, qmax := quantRange(cfg)
	return clampInt(q, qmin, qmax)
}

func signRoundAdjust(q int, value, gradient, scale, zero float32, cfg QuantizeConfig) int {
	if gradient == 0 || scale == 0 {
		return q
	}
	position := value/scale + zero
	floorQ := int(math.Floor(float64(position)))
	ceilQ := floorQ + 1
	qmin, qmax := quantRange(cfg)
	floorQ = clampInt(floorQ, qmin, qmax)
	ceilQ = clampInt(ceilQ, qmin, qmax)
	if floorQ == ceilQ {
		return floorQ
	}
	if gradient > 0 {
		return floorQ
	}
	return ceilQ
}

func quantRange(cfg QuantizeConfig) (int, int) {
	if cfg.Symmetric {
		return -(1 << (cfg.Bits - 1)), (1 << (cfg.Bits - 1)) - 1
	}
	return 0, (1 << cfg.Bits) - 1
}

func clampInt(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func normaliseScheme(scheme Scheme) Scheme {
	value := core.Replace(core.Replace(core.Trim(string(scheme)), "-", "_"), "gguf:", "GGUF:")
	value = core.Replace(value, "gguf_", "GGUF:")
	upper := core.Upper(value)
	if core.HasPrefix(upper, "GGUF:") {
		return Scheme("GGUF:" + core.Upper(value[5:]))
	}
	return Scheme(upper)
}

func cloneProfile(profile Profile) Profile {
	profile.Notes = core.SliceClone(profile.Notes)
	return profile
}
