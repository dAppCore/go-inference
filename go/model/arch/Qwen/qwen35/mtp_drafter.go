// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// mtp_drafter.go gives the qwen3_5_mtp / qwen3_6_mtp multi-token-prediction drafter checkpoint a
// real, tested config parse + Arch derivation + weight-name mapping — the v1 slice of #59 item 2 (see
// docs/design-qwen-mtp-pair.md for the full study, the historical composed reference, and why a live
// speculative pair is explicitly OUT of this slice).
//
// This file's exports are DELIBERATELY not wired into any registry:
//   - model.RegisterArch's existing entry in register.go keeps its friendly standalone-load refusal
//     unchanged (protected by TestMTPDrafterRefusal_Bad) — replacing it would trade a clear, named
//     "serve it paired with its base model" message for a generic "model.embed_tokens absent" one
//     (confirmed non-crashing — see mtp_drafter_test.go — but strictly less helpful), for no
//     functional gain, since nothing here can serve the checkpoint standalone either way.
//   - mtp.RegisterAssistant is deliberately not called: its AssistantSpec.Method defaults to
//     mtp.MTPDraftModel, which is documented as sharing the TARGET's KV streams — gemma4's shape, not
//     qwen's (the drafter keeps its own separate attention state; see design doc part 1-2). Claiming
//     that method would be a coherent-but-wrong declaration.
//
// ParseDrafterConfig / Config.DrafterArch / DrafterWeightNames / DrafterTensorNames are plain,
// directly-callable, directly-tested package API — the proven-correct starting point a future lane
// wiring a real pair (a new mtp.MTPMethod, or a bespoke own-KV engine loader) can build on instead of
// re-deriving the checkpoint's shape from the historical composed source under time pressure.

// ParseDrafterConfig parses a qwen3_5_mtp / qwen3_6_mtp drafter checkpoint's config.json into a
// validated Config — the SAME type the base parses into, because a real drafter checkpoint's
// mtp_num_hidden_layers is nested inside ITS OWN text_config (the base's text_config, reused
// verbatim), and only Config sees that nesting via effective(). Returns an error when the declared
// shape does not look like an MTP drafter (mtp_num_hidden_layers absent or <= 0) or the dims
// DrafterArch itself requires are missing.
func ParseDrafterConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("qwen35.ParseDrafterConfig: config.json parse failed")
	}
	if cfg.effective().MTPNumHiddenLayers <= 0 {
		return nil, core.NewError("qwen35.ParseDrafterConfig: mtp_num_hidden_layers absent or <= 0 — not an MTP drafter config")
	}
	if _, err := cfg.DrafterArch(); err != nil {
		return nil, core.E("qwen35.ParseDrafterConfig", "derive drafter arch", err)
	}
	return &cfg, nil
}

// DrafterArch derives the qwen3_5_mtp / qwen3_6_mtp drafter head's OWN decode architecture: one (or
// MTPNumHiddenLayers, for a checkpoint that declares more) full-attention transformer layer that
// projects from a paired base's hidden state, sharing the base's embedding + LM head
// (mtp_use_dedicated_embeddings / tie_word_embeddings both false on the real checkpoint — see the
// design doc). It reuses Config.Arch() COMPLETELY UNCHANGED on a shallow copy that overrides only the
// layer schedule — every head layer is full_attention (the real checkpoint's layers.N.self_attn.*
// tensors are plain gated attention regardless of the base's hybrid schedule) and the layer count
// (MTPNumHiddenLayers, the head's OWN depth — distinct from NumHiddenLayers, which a real drafter
// config still carries as an artefact of nesting the base's own text_config verbatim, but the head
// never uses as its depth). Every other dimension (hidden, heads, kv_heads, head_dim, rope, eps, FF,
// AttnOutputGate, MoE) is the base's own, because the real checkpoint's text_config IS the base's
// text_config. See the design doc's closing note on the untested MoE-base case.
func (c *Config) DrafterArch() (model.Arch, error) {
	eff := c.effective()
	if eff.MTPNumHiddenLayers <= 0 {
		return model.Arch{}, core.NewError("qwen35.Config.DrafterArch: mtp_num_hidden_layers must be > 0 — this config does not declare an MTP drafter head")
	}
	head := *eff
	head.NumHiddenLayers = eff.MTPNumHiddenLayers
	head.LayerTypes = nil          // clear the base's parsed hybrid schedule
	head.FullAttentionInterval = 1 // layerTypes() then synthesises every layer full_attention
	return head.Arch()
}

// DrafterWeightNames documents the qwen3_5_mtp / qwen3_6_mtp drafter checkpoint's tensor layout for
// its OWN transformer stack — FLAT (no "model." prefix), unlike the base, because the real checkpoint
// (mlx-community/Qwen3.6-27B-MTP-4bit, reconciled by mtp_drafter_real_test.go) ships the head
// standalone: "layers.<i>...." at the top level, not "model.layers.<i>....". The per-layer norm/
// attention/FFN suffix conventions are otherwise unchanged from the base's own WeightNames() — the
// head IS structurally one of the base's own full-attention layers. Embed and LMHead are
// DELIBERATELY left empty: the drafter shares the base's token embedding and LM head and carries
// neither tensor itself — model.Assemble refuses a nil Embed with a clean, named error (never a
// crash; see TestDrafterAssembleWithoutEmbed_Bad), which is the safety net behind leaving this
// mapping unregistered (see the file doc above).
func DrafterWeightNames() model.WeightNames {
	w := WeightNames()
	w.LayerPrefix = "layers.%d"
	w.FinalNorm = "norm.weight"
	w.Embed, w.LMHead = "", ""
	return w
}

// Drafter-only tensors with no home in model.WeightNames (the fc input combiner + its two pre-norms).
// Literal names from the real checkpoint (mtp_drafter_real_test.go), unchanged from the historical
// composed loader (LoadMTPHead, b1f6c21a^:go/model/composed/mtp.go).
const (
	DrafterFCWeight         = "fc.weight"
	DrafterEmbedNormWeight  = "pre_fc_norm_embedding.weight"
	DrafterHiddenNormWeight = "pre_fc_norm_hidden.weight"
)

// DrafterTensorNames returns every tensor name a qwen3_5_mtp / qwen3_6_mtp drafter checkpoint of
// nLayers head layers carries — DrafterWeightNames' per-layer suffixes instantiated at each layer
// index, the three extra combiner tensors, and the head's own final norm. It is the single source of
// truth mtp_drafter_test.go's synthetic fixture and mtp_drafter_real_test.go's real-checkpoint
// reconciliation both build from, so the two receipts can never silently drift apart. Names are
// returned WITHOUT a quantised checkpoint's .scales/.biases sidecars (callers that need those derive
// them from the returned .weight names, matching how model.Assemble's own LoadLinear treats them as
// optional companions of the same logical tensor, not separate required entries).
func DrafterTensorNames(nLayers int) []string {
	w := DrafterWeightNames()
	names := []string{DrafterFCWeight, DrafterEmbedNormWeight, DrafterHiddenNormWeight, w.FinalNorm}
	for i := 0; i < nLayers; i++ {
		prefix := core.Sprintf(w.LayerPrefix, i)
		for _, suffix := range []string{
			w.AttnNorm, w.MLPNorm, // already carry ".weight"
			w.QNorm, w.KNorm, // already carry ".weight"
			w.Q + ".weight", w.K + ".weight", w.V + ".weight", w.O + ".weight",
			w.Gate + ".weight", w.Up + ".weight", w.Down + ".weight",
		} {
			if suffix == "" {
				continue
			}
			names = append(names, prefix+suffix)
		}
	}
	return names
}
