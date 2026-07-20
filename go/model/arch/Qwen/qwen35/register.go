// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// register.go declares the Qwen 3.6 hybrid family to the reactive loader: Parse + Weights for the
// factory route (model.Assemble + arch_session — the #18 unification target, and since #50 the ONLY
// route; the parallel composed engine that used to carry the A/B reference is retired).
//
// The family is ONE architecture released under three model_type strings: qwen3_5 / qwen3_5_moe (+
// their nested text_config aliases), qwen3_6 / qwen3_6_moe (the same hybrid under its other released
// name), and qwen3_next (Qwen 3.6's predecessor, the same gated-delta/full-attention hybrid). All
// seven ids share this one Parse/Weights declaration — there is no per-id behaviour difference; the
// geometry (gated-delta key/value heads, conv kernel) is DERIVED from weight shapes at assemble time
// (assembleGatedDelta), not trusted from config field names that drift release to release, which is
// exactly what makes one declaration safe for all released names (#50 archzoo).
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
	})

	// The multi-token-prediction drafter (qwen3_5_mtp) is a REGISTERED model_type that PAIRS with
	// its base (assistant-side declarations; the MTP head + verify loop) but REFUSES a STANDALONE
	// load: it is the small speculative head trained alongside a Qwen 3.6 base, it shares the
	// base's embedding + LM head and projects from the base's last hidden state — there is no base
	// hidden to project from on its own. A user who points lem at the MTP submodule ALONE gets
	// direction toward pairing rather than a mystery. Registered separately from the base hybrid
	// so the refusal message is distinct from a real load failure. (The refusal moved here from
	// the retired composed engine's register — #50; note the pairing itself currently declines
	// too: the composed pair loader retired with it, the factory pair route is pending.)
	//
	// This refusal is DELIBERATELY left unchanged (#59 item 2 — see docs/design-qwen-mtp-pair.md):
	// mtp_drafter.go gives the checkpoint a real, tested Parse/Arch/weight-name route
	// (ParseDrafterConfig/Config.DrafterArch/DrafterWeightNames), but as plain, uncalled package
	// API — not wired here — because trading this named "serve it paired" message for
	// model.Assemble's generic "absent" error would be a UX regression for zero functional gain
	// (nothing can serve the checkpoint standalone either way; see TestMTPDrafterRefusal_Bad).
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"qwen3_5_mtp", "qwen3_5_mtp_text", "qwen3_6_mtp"},
		Parse: func([]byte) (model.ArchConfig, error) {
			return nil, core.NewError("qwen35: qwen3_5_mtp is an MTP drafter with no standalone forward — serve it paired with its base model (lem pair <base> <mtp>)")
		},
	})
}
