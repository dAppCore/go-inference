// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
)

// register.go declares the Qwen 3.6 hybrid (qwen3_5 / qwen3_5_moe + their nested text_config aliases) to the
// reactive loader with BOTH routes on one ArchSpec: Parse + Weights for the factory (model.Assemble +
// arch_session — the #18 unification target) AND a Composed hook that delegates to model/composed (the
// current default + the A/B reference while the factory decode lands). model.LoadComposedDir sees the
// Composed hook and keeps serving through composed by default; the engine opts INTO the factory route with
// the factory route by DEFAULT (engine/metal/load.go; LTHN_QWEN_COMPOSED=1 reverts), landed once the factory decode was
// validated against composed.
//
// This registration REPLACES composed's own (composed/register.go no longer lists the qwen3_5* ids).
// RegisterArch is last-wins, but carrying both hooks on one spec here makes the ownership explicit rather
// than init-order-dependent. qwen35 may import composed (composed does not import qwen35), so the delegating
// hook sits cleanly on this side of that edge.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("qwen35.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		Weights: WeightNames(),
		Composed: func(tensors map[string]safetensors.Tensor, configJSON []byte) (model.TokenModel, error) {
			// Delegate to the composed loader (the current default + A/B reference). Zero-copy mmap build,
			// matching the behaviour composed's own qwen3_5 hook had before this package took the registration.
			cm, err := composed.LoadComposedMmap(tensors, configJSON)
			if err != nil {
				return nil, core.E("qwen35.Load", "assemble composed model", err)
			}
			return composed.NewTokenModel(cm), nil
		},
	})
}
