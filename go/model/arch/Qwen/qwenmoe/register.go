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
		Composed: func(tensors map[string]safetensors.Tensor, configJSON []byte) (model.TokenModel, error) {
			var cfg Config
			if r := core.JSONUnmarshal(configJSON, &cfg); !r.OK {
				return nil, core.NewError("qwenmoe.Load: config.json parse failed")
			}
			arch, err := cfg.Arch()
			if err != nil {
				return nil, core.E("qwenmoe.Load", "resolve architecture", err)
			}
			assembled, err := composed.LoadComposedWithArch(tensors, configJSON, arch)
			if err != nil {
				return nil, core.E("qwenmoe.Load", "assemble composed model", err)
			}
			return composed.NewTokenModel(assembled), nil
		},
	})
}
