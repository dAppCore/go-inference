// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
)

// register.go declares the Qwen 3.6 hybrid family to the reactive loader with BOTH routes on one ArchSpec:
// Parse + Weights for the factory (model.Assemble + arch_session — the #18 unification target) AND a
// Composed hook that delegates to model/composed (the current default + the A/B reference while the
// factory decode lands). model.LoadComposedDir sees the Composed hook and keeps serving through composed by
// default; the engine opts INTO the factory route with the factory route by DEFAULT (engine/metal/load.go;
// LTHN_QWEN_COMPOSED=1 reverts), landed once the factory decode was validated against composed.
//
// The family is ONE architecture released under three model_type strings: qwen3_5 / qwen3_5_moe (+ their
// nested text_config aliases), qwen3_6 / qwen3_6_moe (the same hybrid under its other released name — see
// model/composed/register.go's prior comment, and loader.go's case that has always built qwen3_5 and
// qwen3_6 identically), and qwen3_next (Qwen 3.6's predecessor, the same gated-delta/full-attention
// hybrid). All five ids share this one Parse/Weights/Composed declaration — there is no per-id behaviour
// difference; the geometry (gated-delta key/value heads, conv kernel) is DERIVED from weight shapes at
// assemble time (assembleGatedDelta below), not trusted from config field names that drift release to
// release, which is exactly what makes one declaration safe for all three released names (#50 archzoo).
//
// This registration REPLACES composed's own for all five ids (composed/register.go no longer lists any of
// them). RegisterArch is last-wins, but carrying both hooks on one spec here makes the ownership explicit
// rather than init-order-dependent. qwen35 may import composed (composed does not import qwen35), so the
// delegating hook sits cleanly on this side of that edge.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{
			"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
			"qwen3_6", "qwen3_6_moe", "qwen3_next",
		},
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
