// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// SequenceMixerConfigInput is the model-owned subset of config.json metadata
// needed to plan go-mlx-style config-composed and hybrid mixer stacks.
type SequenceMixerConfigInput struct {
	ModelType           string
	TextModelType       string
	LayerTypes          []string
	TextLayerTypes      []string
	NumHiddenLayers     int
	NumLayers           int
	TextNumHiddenLayers int
	TextNumLayers       int
}

// SequenceMixerConfigProbe is the model-owned config-composed/hybrid planning
// result. Runtime packages can bind the returned layer/cache plan to HIP/CUDA/CPU
// tensors after model-pack inspection discovers concrete checkpoint leaves.
type SequenceMixerConfigProbe struct {
	LayerTypes  []string                 `json:"layer_types,omitempty"`
	LayerSource string                   `json:"layer_source,omitempty"`
	PlanStatus  string                   `json:"plan_status,omitempty"`
	PlanError   string                   `json:"plan_error,omitempty"`
	Composed    bool                     `json:"composed,omitempty"`
	Layers      []SequenceMixerLayerPlan `json:"layers,omitempty"`
	Cache       SequenceMixerCachePlan   `json:"cache"`
}

func (probe SequenceMixerConfigProbe) Clone() SequenceMixerConfigProbe {
	probe.LayerTypes = append([]string(nil), probe.LayerTypes...)
	probe.Layers = CloneSequenceMixerLayerPlans(probe.Layers)
	probe.Cache = probe.Cache.Clone()
	return probe
}

// ProbeSequenceMixerConfig applies the same declared-kind rules as go-mlx's
// composed loader: explicit layer_types win, otherwise a registered mixer
// model_type becomes a uniform stack, while composed/hybrid without layer_types
// refuses loudly.
func ProbeSequenceMixerConfig(input SequenceMixerConfigInput) SequenceMixerConfigProbe {
	if SequenceMixerConfigComposedModelType(input) == "" && SequenceMixerConfigUniformKind(input) == "" {
		return SequenceMixerConfigProbe{}
	}
	layerTypes, source := SequenceMixerConfigPlanLayerTypes(input)
	if len(layerTypes) == 0 {
		if source, err := SequenceMixerConfigPlanError(input); err != nil {
			return SequenceMixerConfigProbe{
				LayerSource: source,
				PlanStatus:  "invalid",
				PlanError:   err.Error(),
				Composed:    SequenceMixerConfigComposedModelType(input) != "",
			}
		}
		return SequenceMixerConfigProbe{}
	}

	probe := SequenceMixerConfigProbe{
		LayerTypes:  append([]string(nil), layerTypes...),
		LayerSource: source,
		Composed:    source != "" || SequenceMixerConfigComposedModelType(input) != "",
	}
	numLayers := sequenceMixerConfigNumLayers(input)
	if numLayers <= 0 {
		numLayers = len(layerTypes)
	}
	if len(layerTypes) != numLayers {
		probe.PlanStatus = "invalid"
		probe.PlanError = core.Sprintf("layer_types length %d != num_hidden_layers %d", len(layerTypes), numLayers)
		return probe.Clone()
	}

	layers := make([]SequenceMixerLayerPlan, 0, len(layerTypes))
	for layer, raw := range layerTypes {
		kind := NormalizeSequenceMixerKind(raw)
		family, ok := SequenceMixerFamilyByKind(kind)
		if !ok {
			probe.PlanStatus = "invalid"
			probe.PlanError = core.Sprintf("layer %d: unregistered mixer kind %q", layer, kind)
			return probe.Clone()
		}
		layers = append(layers, SequenceMixerLayerPlan{
			Layer:      layer,
			Kind:       family.Kind,
			State:      family.State,
			StateSlots: append([]string(nil), family.StateSlots...),
			Source:     family.Source,
			Runtime:    family.Runtime,
		})
	}
	cache, err := BuildSequenceMixerCachePlan(layers)
	if err != nil {
		probe.PlanStatus = "invalid"
		probe.PlanError = err.Error()
		return probe.Clone()
	}
	probe.Layers = layers
	probe.Cache = cache
	probe.PlanStatus = "valid"
	return probe.Clone()
}

func SequenceMixerConfigPlanLayerTypes(input SequenceMixerConfigInput) ([]string, string) {
	numLayers := sequenceMixerConfigNumLayers(input)
	if numLayers <= 0 {
		return nil, ""
	}
	switch {
	case len(input.LayerTypes) > 0:
		return NormalizeSequenceMixerLayerTypes(input.LayerTypes), "layer_types"
	case len(input.TextLayerTypes) > 0:
		return NormalizeSequenceMixerLayerTypes(input.TextLayerTypes), "text_config.layer_types"
	default:
		uniform := SequenceMixerConfigUniformKind(input)
		if uniform == "" {
			return nil, ""
		}
		layerTypes := make([]string, numLayers)
		for index := range layerTypes {
			layerTypes[index] = uniform
		}
		return layerTypes, "model_type"
	}
}

func SequenceMixerConfigPlanError(input SequenceMixerConfigInput) (string, error) {
	if sequenceMixerConfigNumLayers(input) <= 0 ||
		len(input.LayerTypes) > 0 ||
		len(input.TextLayerTypes) > 0 ||
		SequenceMixerConfigUniformKind(input) != "" ||
		SequenceMixerConfigComposedModelType(input) == "" {
		return "", nil
	}
	return "model_type", core.NewError("needs per-layer layer_types or a mixer model_type")
}

func SequenceMixerConfigUniformKind(input SequenceMixerConfigInput) string {
	for _, value := range []string{input.ModelType, input.TextModelType} {
		kind := NormalizeSequenceMixerKind(value)
		if _, ok := SequenceMixerFamilyByKind(kind); ok {
			return kind
		}
	}
	return ""
}

func SequenceMixerConfigComposedModelType(input SequenceMixerConfigInput) string {
	for _, value := range []string{input.ModelType, input.TextModelType} {
		switch kind := NormalizeSequenceMixerKind(value); kind {
		case "composed", "hybrid":
			return kind
		}
	}
	return ""
}

func sequenceMixerConfigNumLayers(input SequenceMixerConfigInput) int {
	return firstPositiveInt(input.NumHiddenLayers, input.NumLayers, input.TextNumHiddenLayers, input.TextNumLayers)
}
