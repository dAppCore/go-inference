// SPDX-Licence-Identifier: EUPL-1.2

package qwenmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
)

func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"qwen2_moe", "qwen3_moe"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("qwenmoe.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		// Weights + NormalizeConfig give Qwen2-MoE/Qwen3-MoE the factory route (model.Assemble +
		// arch_session — the #18 unification target), dual-registered alongside the Composed hook below
		// exactly as mixtral/qwen35/granitemoe carry both: Composed stays the A/B reference + the route a
		// caller that deliberately bypasses model.Load still reaches, while model.Load now succeeds
		// instead of rejecting Qwen2-MoE/Qwen3-MoE as composed-only.
		Weights: FactoryWeightNames(),
		NormalizeConfig: func(tensors map[string]safetensors.Tensor, ac model.ArchConfig) map[string]safetensors.Tensor {
			cfg := ac.(*Config)
			if packed, err := packExperts(tensors, cfg.NumHiddenLayers, cfg.NumExperts); err == nil {
				return packed
			}
			return tensors // malformed/absent experts — Assemble's nil-safe load surfaces the gap downstream
		},
		Composed: func(tensors map[string]safetensors.Tensor, configJSON []byte) (model.TokenModel, error) {
			var cfg Config
			if r := core.JSONUnmarshal(configJSON, &cfg); !r.OK {
				return nil, core.NewError("qwenmoe.Load: config.json parse failed")
			}
			arch, err := cfg.Arch()
			if err != nil {
				return nil, core.E("qwenmoe.Load", "resolve architecture", err)
			}
			// Zero-copy: the packed quant projection weights (attention q/k/v/o, embed, lm_head) VIEW the
			// mapped checkpoint rather than being copied to the heap. model.LoadComposedDir hands the model
			// the mapping via RetainMmap, so it stays alive for the aliasing weights' lifetime.
			assembled, err := composed.LoadComposedWithArchMmap(tensors, configJSON, arch)
			if err != nil {
				return nil, core.E("qwenmoe.Load", "assemble composed model", err)
			}
			return composed.NewTokenModel(assembled), nil
		},
	})
}
