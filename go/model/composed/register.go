// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// register.go declares the Qwen 3.6 hybrid to the engine's reactive loader, so model.LoadComposedDir
// reaches the composed loader through the ArchSpec registry — the neutral (backend-agnostic) routing that
// supersedes engine/metal's hardcoded model_type switch. The composed stack is NOT the reactive transformer
// Assemble (its linear_attention layers have no q/k/v to assemble), so it registers a Composed hook rather
// than a Parse + Weights layout; model.Load routes a Composed arch to LoadComposedDir instead of Assemble.
//
// Registered ids: the wrapper model_types (qwen3_5, qwen3_5_moe) and their nested text_config aliases
// (qwen3_5_text, qwen3_5_moe_text) so probeModelTypes resolves either, plus qwen3_next — Qwen 3.6's
// predecessor, the same gated-delta/full-attention hybrid the composed loader already builds.
//
// The registration lives here, not in model/qwen3: composed imports qwen3 for the gated-delta block, so
// qwen3 cannot import composed — the hook that reaches LoadComposed must sit on the composed side of that
// edge. (A serve composition still needs to import this package for the init() to run; the engine/metal
// switch covers the current serve path, so this is the forward-looking neutral wiring.)
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"},
		Composed: func(tensors map[string]safetensors.Tensor, configJSON []byte) (model.TokenModel, error) {
			cm, err := LoadComposed(tensors, configJSON)
			if err != nil {
				return nil, err
			}
			return NewTokenModel(cm), nil
		},
	})
}
