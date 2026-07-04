// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"

	core "dappco.re/go"
)

// NativeAdamWConfig records the optimizer hyperparameters used by the packed
// ROCm training-state path. Packed is a layout marker for future HIP kernels;
// this reference implementation always stores parameter/m/v slabs contiguously.
type NativeAdamWConfig struct {
	LearningRate float64 `json:"learning_rate"`
	Beta1        float64 `json:"beta1"`
	Beta2        float64 `json:"beta2"`
	Eps          float64 `json:"eps"`
	WeightDecay  float64 `json:"weight_decay"`
	Packed       bool    `json:"packed"`

	LearningRateSet bool `json:"-"`
	Beta1Set        bool `json:"-"`
	Beta2Set        bool `json:"-"`
	EpsSet          bool `json:"-"`
	WeightDecaySet  bool `json:"-"`
}

// NativeAdamWParam is one trainable tensor copied into a packed AdamW state.
type NativeAdamWParam struct {
	Name   string    `json:"name,omitempty"`
	Shape  []int     `json:"shape,omitempty"`
	Values []float32 `json:"values,omitempty"`
}

// NativeAdamWParamLayout identifies one tensor view inside the packed slabs.
type NativeAdamWParamLayout struct {
	Name   string `json:"name,omitempty"`
	Offset int    `json:"offset"`
	Length int    `json:"length"`
	Shape  []int  `json:"shape,omitempty"`
}

// NativeAdamWState stores trainable parameters and AdamW moments as one
// contiguous slab: [parameters | first moments | second moments].
type NativeAdamWState struct {
	Config NativeAdamWConfig        `json:"config"`
	Step   int                      `json:"step"`
	Layout []NativeAdamWParamLayout `json:"layout,omitempty"`
	Slab   []float32                `json:"slab,omitempty"`
}

// DefaultNativeAdamWConfig returns go-mlx-compatible AdamW defaults.
func DefaultNativeAdamWConfig() NativeAdamWConfig {
	return NativeAdamWConfig{
		LearningRate: 1e-5,
		Beta1:        0.9,
		Beta2:        0.999,
		Eps:          1e-8,
		WeightDecay:  0.01,
		Packed:       true,
	}
}

// NewNativeAdamWState packs parameters into contiguous parameter/m/v slabs.
func NewNativeAdamWState(params []NativeAdamWParam, cfg NativeAdamWConfig) (*NativeAdamWState, error) {
	cfg = normalizeNativeAdamWConfig(cfg)
	if err := validateNativeAdamWConfig(cfg); err != nil {
		return nil, err
	}
	if len(params) == 0 {
		return nil, core.NewError("rocm: AdamW parameters are required")
	}
	total := 0
	layout := make([]NativeAdamWParamLayout, len(params))
	for i, param := range params {
		if len(param.Values) == 0 {
			return nil, core.Errorf("rocm: AdamW parameter %d values are required", i)
		}
		if !rocmFloat32SliceFinite(param.Values) {
			return nil, core.Errorf("rocm: AdamW parameter %d values must be finite", i)
		}
		if err := validateNativeAdamWShape(param.Shape, len(param.Values)); err != nil {
			return nil, core.E("rocm.AdamW.State", "parameter shape", err)
		}
		layout[i] = NativeAdamWParamLayout{
			Name:   param.Name,
			Offset: total,
			Length: len(param.Values),
			Shape:  append([]int(nil), param.Shape...),
		}
		total += len(param.Values)
	}
	slab := make([]float32, total*3)
	for i, param := range params {
		desc := layout[i]
		copy(slab[desc.Offset:desc.Offset+desc.Length], param.Values)
	}
	return &NativeAdamWState{Config: cfg, Layout: layout, Slab: slab}, nil
}

// NewNativeLoRAAdamWState packs LoRA A/B tensors using stable target names.
func NewNativeLoRAAdamWState(loraA, loraB []float32, rows, cols, rank int, cfg NativeAdamWConfig) (*NativeAdamWState, error) {
	if rank <= 0 || rows <= 0 || cols <= 0 {
		return nil, core.NewError("rocm: LoRA AdamW rows, cols, and rank must be positive")
	}
	if len(loraA) != rank*cols {
		return nil, core.Errorf("rocm: LoRA AdamW A length %d does not match rank*cols %d", len(loraA), rank*cols)
	}
	if len(loraB) != rows*rank {
		return nil, core.Errorf("rocm: LoRA AdamW B length %d does not match rows*rank %d", len(loraB), rows*rank)
	}
	return NewNativeAdamWState([]NativeAdamWParam{
		{Name: "lora_a", Shape: []int{rank, cols}, Values: loraA},
		{Name: "lora_b", Shape: []int{rows, rank}, Values: loraB},
	}, cfg)
}

// Parameters returns the mutable packed parameter slab.
func (state *NativeAdamWState) Parameters() []float32 {
	total := stateTotalLen(state)
	if total == 0 {
		return nil
	}
	return state.Slab[:total]
}

// FirstMoment returns the mutable packed first-moment slab.
func (state *NativeAdamWState) FirstMoment() []float32 {
	total := stateTotalLen(state)
	if total == 0 {
		return nil
	}
	return state.Slab[total : 2*total]
}

// SecondMoment returns the mutable packed second-moment slab.
func (state *NativeAdamWState) SecondMoment() []float32 {
	total := stateTotalLen(state)
	if total == 0 {
		return nil
	}
	return state.Slab[2*total : 3*total]
}

// ParamView returns the mutable parameter view for layout index.
func (state *NativeAdamWState) ParamView(index int) ([]float32, bool) {
	if state == nil || index < 0 || index >= len(state.Layout) {
		return nil, false
	}
	desc := state.Layout[index]
	params := state.Parameters()
	return params[desc.Offset : desc.Offset+desc.Length], true
}

// StepInPlace applies one AdamW step using gradients parallel to Layout.
func (state *NativeAdamWState) StepInPlace(gradients [][]float32) error {
	if state == nil {
		return core.NewError("rocm: AdamW state is nil")
	}
	if len(gradients) != len(state.Layout) {
		return core.Errorf("rocm: AdamW gradients length %d does not match parameter count %d", len(gradients), len(state.Layout))
	}
	if err := validateNativeAdamWConfig(state.Config); err != nil {
		return err
	}
	params := state.Parameters()
	momentsM := state.FirstMoment()
	momentsV := state.SecondMoment()
	if len(params) == 0 || len(state.Slab) != len(params)*3 {
		return core.NewError("rocm: AdamW packed slab shape is invalid")
	}
	step := state.Step + 1
	biasCorrection1 := 1 - math.Pow(state.Config.Beta1, float64(step))
	biasCorrection2 := 1 - math.Pow(state.Config.Beta2, float64(step))
	for i, gradient := range gradients {
		desc := state.Layout[i]
		if len(gradient) != desc.Length {
			return core.Errorf("rocm: AdamW gradient %d length %d does not match parameter length %d", i, len(gradient), desc.Length)
		}
		if !rocmFloat32SliceFinite(gradient) {
			return core.Errorf("rocm: AdamW gradient %d values must be finite", i)
		}
		for j, grad32 := range gradient {
			offset := desc.Offset + j
			param := float64(params[offset])
			grad := float64(grad32)
			m := state.Config.Beta1*float64(momentsM[offset]) + (1-state.Config.Beta1)*grad
			v := state.Config.Beta2*float64(momentsV[offset]) + (1-state.Config.Beta2)*grad*grad
			mHat := m / biasCorrection1
			vHat := v / biasCorrection2
			decayed := param * (1 - state.Config.LearningRate*state.Config.WeightDecay)
			next := decayed - state.Config.LearningRate*mHat/(math.Sqrt(vHat)+state.Config.Eps)
			if math.IsNaN(next) || math.IsInf(next, 0) {
				return core.Errorf("rocm: AdamW update %d produced non-finite parameter", i)
			}
			params[offset] = float32(next)
			momentsM[offset] = float32(m)
			momentsV[offset] = float32(v)
		}
	}
	state.Step = step
	return nil
}

func normalizeNativeAdamWConfig(cfg NativeAdamWConfig) NativeAdamWConfig {
	defaults := DefaultNativeAdamWConfig()
	if cfg.LearningRate == 0 && !cfg.LearningRateSet {
		cfg.LearningRate = defaults.LearningRate
	}
	if cfg.Beta1 == 0 && !cfg.Beta1Set {
		cfg.Beta1 = defaults.Beta1
	}
	if cfg.Beta2 == 0 && !cfg.Beta2Set {
		cfg.Beta2 = defaults.Beta2
	}
	if cfg.Eps == 0 && !cfg.EpsSet {
		cfg.Eps = defaults.Eps
	}
	if cfg.WeightDecay == 0 && !cfg.WeightDecaySet {
		cfg.WeightDecay = defaults.WeightDecay
	}
	cfg.Packed = true
	return cfg
}

func validateNativeAdamWConfig(cfg NativeAdamWConfig) error {
	if cfg.LearningRate < 0 || math.IsNaN(cfg.LearningRate) || math.IsInf(cfg.LearningRate, 0) {
		return core.NewError("rocm: AdamW learning rate must be non-negative and finite")
	}
	if cfg.Beta1 < 0 || cfg.Beta1 >= 1 || math.IsNaN(cfg.Beta1) || math.IsInf(cfg.Beta1, 0) {
		return core.NewError("rocm: AdamW beta1 must be finite and within [0,1)")
	}
	if cfg.Beta2 < 0 || cfg.Beta2 >= 1 || math.IsNaN(cfg.Beta2) || math.IsInf(cfg.Beta2, 0) {
		return core.NewError("rocm: AdamW beta2 must be finite and within [0,1)")
	}
	if cfg.Eps <= 0 || math.IsNaN(cfg.Eps) || math.IsInf(cfg.Eps, 0) {
		return core.NewError("rocm: AdamW epsilon must be positive and finite")
	}
	if cfg.WeightDecay < 0 || math.IsNaN(cfg.WeightDecay) || math.IsInf(cfg.WeightDecay, 0) {
		return core.NewError("rocm: AdamW weight decay must be non-negative and finite")
	}
	return nil
}

func validateNativeAdamWShape(shape []int, values int) error {
	if len(shape) == 0 {
		return nil
	}
	product := 1
	for _, dim := range shape {
		if dim <= 0 {
			return core.NewError("shape dimensions must be positive")
		}
		product *= dim
	}
	if product != values {
		return core.Errorf("shape product %d does not match values length %d", product, values)
	}
	return nil
}

func stateTotalLen(state *NativeAdamWState) int {
	if state == nil || len(state.Layout) == 0 || len(state.Slab)%3 != 0 {
		return 0
	}
	return len(state.Slab) / 3
}
