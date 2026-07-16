// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"strings"

	core "dappco.re/go"
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
// (qwen3_5_text, qwen3_5_moe_text), qwen3_6 / qwen3_6_moe (the same hybrid under its other released name),
// qwen3_next (Qwen 3.6's predecessor, the same gated-delta/full-attention hybrid), and the generic
// "composed"/"hybrid" ids a checkpoint without a qwen-family model_type may carry. This is every id
// engine/metal's hardcoded model_type switch used to name directly — registering them here is what let
// that switch be deleted rather than merely bypassed.
//
// The registration lives here, not in model/qwen3: composed imports qwen3 for the gated-delta block, so
// qwen3 cannot import composed — the hook that reaches LoadComposed must sit on the composed side of that
// edge.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{
			"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
			"qwen3_6", "qwen3_6_moe", "qwen3_next",
			"composed", "hybrid",
		},
		Composed: func(tensors map[string]safetensors.Tensor, configJSON []byte) (model.TokenModel, error) {
			// Zero-copy build: the packed quant weights VIEW the mapped checkpoint rather than being
			// copied to the heap (the RSS win for a low-end Bonsai load). model.LoadComposedDir hands the
			// resulting model the mapping via RetainMmap, so it stays alive for the weights' lifetime.
			cm, err := loadComposed(tensors, configJSON, nil, true)
			if err != nil {
				return nil, err
			}
			return NewTokenModel(cm), nil
		},
	})

	// The multi-token-prediction drafter (qwen3_5_mtp) is a REGISTERED model_type that PAIRS with its base
	// (assistant.go declares it to the reactive assistant registry; mtp.go realises the head + the verify
	// loop) but still REFUSES a STANDALONE load: it is the small speculative head trained alongside a Qwen
	// 3.6 base, served as that base's drafter, and it has no standalone forward (it shares the base's
	// embedding + LM head and projects from the base's last hidden state — there is no base hidden to
	// project from on its own). So a user who points lem at the MTP submodule ALONE gets direction toward
	// pairing rather than a mystery; a paired load goes through LoadSpeculativePairDirs, not this hook.
	// Registered separately from the base hybrid so the refusal message is distinct from a real load failure.
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"qwen3_5_mtp", "qwen3_5_mtp_text", "qwen3_6_mtp"},
		Composed: func(map[string]safetensors.Tensor, []byte) (model.TokenModel, error) {
			return nil, core.NewError("composed: qwen3_5_mtp is an MTP drafter with no standalone forward — serve it paired with its base model (lem pair <base> <mtp>)")
		},
	})
}

// ChatMLDialect reports whether a composed model_type is served with the ChatML chat dialect
// (<|im_start|>role\n…<|im_end|> turns, an "assistant" generation cue and a <think> reasoning block)
// rather than the gemma turn template. Every Qwen hybrid this package builds — qwen3_5 and its
// text/MoE aliases, qwen3_next, and any future qwenX — speaks ChatML; a non-qwen composed arch (the
// generic "composed"/"hybrid" ids, or a later family) keeps the gemma fallback. The serve wrap consults
// this to DECLARE its chat template, so the dialect follows config.json's model_type rather than being
// hardcoded per checkpoint. Matching on the "qwen" prefix keeps a new qwen model_type ChatML with zero
// edits, mirroring the registry's zero-edit routing.
func ChatMLDialect(modelType string) bool {
	return strings.HasPrefix(strings.ToLower(modelType), "qwen")
}
