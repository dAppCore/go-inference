// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func AssembleDiffusion(weights map[string]safetensors.Tensor, cfg *Gemma4TextConfig) (*model.LoadedDiffusion, error) {
	if cfg == nil {
		return nil, nil
	}
	preNorm := diffusionWeight(weights, "self_conditioning.pre_norm.weight", "self_conditioning.pre_norm")
	gate := model.LoadLinear(weights, "self_conditioning.gate_proj", int(cfg.HiddenSize), "affine")
	up := model.LoadLinear(weights, "self_conditioning.up_proj", int(cfg.HiddenSize), "affine")
	down := model.LoadLinear(weights, "self_conditioning.down_proj", int(cfg.IntermediateSize), "affine")
	if len(preNorm) == 0 || gate == nil || up == nil || down == nil {
		return nil, core.NewError("gemma4.AssembleDiffusion: self-conditioning block incomplete in checkpoint")
	}
	scalars := diffusionEncoderScalars(weights, int(cfg.NumHiddenLayers))
	if len(scalars) != int(cfg.NumHiddenLayers) {
		return nil, core.NewError(core.Sprintf("gemma4.AssembleDiffusion: encoder layer scalars: %d of %d", len(scalars), cfg.NumHiddenLayers))
	}
	return &model.LoadedDiffusion{
		SelfCondPreNorm:     preNorm,
		SelfCondGate:        gate,
		SelfCondUp:          up,
		SelfCondDown:        down,
		EncoderLayerScalars: scalars,
		CanvasLength:        cfg.CanvasLength,
		EOSTokens:           diffusionEOSTokens(cfg.EOSTokenID),
	}, nil
}

func diffusionWeight(weights map[string]safetensors.Tensor, names ...string) []byte {
	if t, ok := model.WeightAny(weights, names...); ok {
		return t.Data
	}
	return nil
}

func diffusionEncoderScalars(weights map[string]safetensors.Tensor, numLayers int) [][]byte {
	if numLayers <= 0 {
		return nil
	}
	scalars := make([][]byte, numLayers)
	for i := range numLayers {
		base := core.Sprintf("model.encoder.language_model.layers.%d.layer_scalar", i)
		scalars[i] = diffusionWeight(weights, base, base+".weight")
	}
	out := scalars[:0]
	for _, scalar := range scalars {
		if len(scalar) > 0 {
			out = append(out, scalar)
		}
	}
	return out
}

func diffusionEOSTokens(value any) []int32 {
	switch v := value.(type) {
	case float64:
		return []int32{int32(v)}
	case []any:
		out := make([]int32, 0, len(v))
		for _, elem := range v {
			if f, ok := elem.(float64); ok {
				out = append(out, int32(f))
			}
		}
		return out
	default:
		return nil
	}
}
