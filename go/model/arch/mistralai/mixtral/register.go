// SPDX-Licence-Identifier: EUPL-1.2

package mixtral

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
)

// Names documents the Hugging Face Mixtral sparse-expert tensor layout.
type Names struct {
	Router, ExpertGate, ExpertDown, ExpertUp string
}

// WeightNames returns the exact Mixtral router and expert tensor suffixes.
func WeightNames() Names {
	return Names{
		Router: ".block_sparse_moe.gate.weight", ExpertGate: ".block_sparse_moe.experts.%d.w1.weight",
		ExpertDown: ".block_sparse_moe.experts.%d.w2.weight", ExpertUp: ".block_sparse_moe.experts.%d.w3.weight",
	}
}

// NormalizeWeights aliases Mixtral sparse-expert names to the composed MoE roles.
func NormalizeWeights(in map[string]safetensors.Tensor) map[string]safetensors.Tensor {
	out := make(map[string]safetensors.Tensor, len(in)*2)
	for name, tensor := range in {
		out[name] = tensor
		alias := core.Replace(name, ".block_sparse_moe.", ".mlp.")
		alias = core.Replace(alias, ".w1.weight", ".gate_proj.weight")
		alias = core.Replace(alias, ".w2.weight", ".down_proj.weight")
		alias = core.Replace(alias, ".w3.weight", ".up_proj.weight")
		out[alias] = tensor
	}
	return out
}

func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"mixtral"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("mixtral.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		// Weights + NormalizeConfig give Mixtral the factory route (model.Assemble + arch_session — the
		// #18 unification target), dual-registered alongside the Composed hook below exactly as qwen35
		// carries both for the Qwen 3.6 hybrid: Composed stays the A/B reference + the route a caller that
		// deliberately bypasses model.Load still reaches, while model.Load now succeeds instead of
		// rejecting Mixtral as composed-only.
		Weights: FactoryWeightNames(),
		NormalizeConfig: func(tensors map[string]safetensors.Tensor, ac model.ArchConfig) map[string]safetensors.Tensor {
			cfg := ac.(*Config)
			if packed, err := packExperts(tensors, cfg.NumHiddenLayers, cfg.NumLocalExperts); err == nil {
				return packed
			}
			return tensors // malformed/absent experts — Assemble's nil-safe load surfaces the gap downstream
		},
		Composed: func(tensors map[string]safetensors.Tensor, configJSON []byte) (model.TokenModel, error) {
			cm, err := composed.LoadComposed(NormalizeWeights(tensors), configJSON)
			if err != nil {
				return nil, core.E("mixtral.Load", "assemble composed model", err)
			}
			return composed.NewTokenModel(cm), nil
		},
	})
}
