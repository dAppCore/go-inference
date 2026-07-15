// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"jetmoe"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("jetmoe.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		Composed: func(tensors map[string]safetensors.Tensor, configJSON []byte) (model.TokenModel, error) {
			var cfg Config
			if r := core.JSONUnmarshal(configJSON, &cfg); !r.OK {
				return nil, core.NewError("jetmoe.Load: config.json parse failed")
			}
			if _, err := cfg.Arch(); err != nil {
				return nil, core.E("jetmoe.Load", "resolve architecture", err)
			}
			if _, err := adaptFFNWeights(tensors, cfg); err != nil {
				return nil, core.E("jetmoe.Load", "adapt packed FFN experts", err)
			}
			if _, ok := tensors["model.layers.0.self_attention.experts.input_linear.weight"]; ok {
				return nil, core.NewError("jetmoe.Load: MoA requires routed query/output attention projections with shared KV; composed attention does not yet expose that primitive")
			}
			return nil, core.NewError("jetmoe.Load: checkpoint lacks JetMoE routed-attention weights")
		},
	})
}
