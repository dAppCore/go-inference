// SPDX-Licence-Identifier: EUPL-1.2

package granitemoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// NormalizeWeights exposes GraniteMoE's packed expert tensors through the
// neutral composed MoE roles without copying their backing bytes.
func NormalizeWeights(in map[string]safetensors.Tensor, cfg *Config) (map[string]safetensors.Tensor, error) {
	out := make(map[string]safetensors.Tensor, len(in)+cfg.NumHiddenLayers*(3*cfg.NumLocalExperts+1))
	for name, tensor := range in {
		out[name] = tensor
	}
	for layer := range cfg.NumHiddenLayers {
		base := core.Sprintf("model.layers.%d.block_sparse_moe.", layer)
		input, inputOK := in[base+"input_linear.weight"]
		output, outputOK := in[base+"output_linear.weight"]
		router, routerOK := in[base+"router.layer.weight"]
		if !inputOK || !outputOK || !routerOK || len(input.Shape) != 3 || len(output.Shape) != 3 || input.Shape[0] != cfg.NumLocalExperts || input.Shape[1] != 2*cfg.IntermediateSize || input.Shape[2] != cfg.HiddenSize || output.Shape[0] != cfg.NumLocalExperts || output.Shape[1] != cfg.HiddenSize || output.Shape[2] != cfg.IntermediateSize {
			return nil, core.E("granitemoe.NormalizeWeights", core.Sprintf("layer %d packed expert geometry mismatch", layer), nil)
		}
		out[core.Sprintf("model.layers.%d.mlp.gate.weight", layer)] = router
		inputStride := 2 * cfg.IntermediateSize * cfg.HiddenSize * len(input.Data) / (input.Shape[0] * input.Shape[1] * input.Shape[2])
		outputStride := cfg.HiddenSize * cfg.IntermediateSize * len(output.Data) / (output.Shape[0] * output.Shape[1] * output.Shape[2])
		gateBytes := inputStride / 2
		for expert := range cfg.NumLocalExperts {
			prefix := core.Sprintf("model.layers.%d.mlp.experts.%d.", layer, expert)
			ib := input.Data[expert*inputStride : (expert+1)*inputStride]
			ob := output.Data[expert*outputStride : (expert+1)*outputStride]
			out[prefix+"gate_proj.weight"] = safetensors.Tensor{Dtype: input.Dtype, Shape: []int{cfg.IntermediateSize, cfg.HiddenSize}, Data: ib[:gateBytes]}
			out[prefix+"up_proj.weight"] = safetensors.Tensor{Dtype: input.Dtype, Shape: []int{cfg.IntermediateSize, cfg.HiddenSize}, Data: ib[gateBytes:]}
			out[prefix+"down_proj.weight"] = safetensors.Tensor{Dtype: output.Dtype, Shape: []int{cfg.HiddenSize, cfg.IntermediateSize}, Data: ob}
		}
	}
	return out, nil
}

func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"granitemoe"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			r := ParseConfig(data)
			if !r.OK {
				return nil, core.E("granitemoe.Parse", r.Error(), nil)
			}
			return r.Value.(*Config), nil
		},
		// Weights gives GraniteMoE the factory route (model.Assemble + arch_session — the #18
		// unification target), dual-registered alongside the Composed hook below exactly as
		// mixtral/qwen35 carry both: Composed stays the A/B reference + the route a caller that
		// deliberately bypasses model.Load still reaches, while model.Load now succeeds instead of
		// rejecting GraniteMoE as composed-only. No NormalizeConfig needed — see FactoryWeightNames'
		// doc: the checkpoint already ships its packed expert tensors in the exact byte layout the
		// factory MoE loader expects.
		Weights: FactoryWeightNames(),
	})
}
