// SPDX-Licence-Identifier: EUPL-1.2

package olmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
)

func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"olmoe"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("olmoe.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		Composed: func(tensors map[string]safetensors.Tensor, configJSON []byte) (model.TokenModel, error) {
			var cfg Config
			if r := core.JSONUnmarshal(configJSON, &cfg); !r.OK {
				return nil, core.NewError("olmoe.Load: config.json parse failed")
			}
			arch, err := cfg.Arch()
			if err != nil {
				return nil, core.E("olmoe.Load", "resolve architecture", err)
			}
			// Zero-copy: the packed quant projection weights (attention q/k/v/o, embed, lm_head) VIEW the
			// mapped checkpoint rather than being copied to the heap. model.LoadComposedDir hands the model
			// the mapping via RetainMmap, so it stays alive for the aliasing weights' lifetime.
			cm, err := composed.LoadComposedWithArchMmap(tensors, configJSON, arch)
			if err != nil {
				return nil, core.E("olmoe.Load", "assemble composed model", err)
			}
			return composed.NewTokenModel(cm), nil
		},
	})
}
