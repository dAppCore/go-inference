// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	core "dappco.re/go"
)

// TestArchitecture_IsROCmDenseQuickWinArchitecture_DenseFamilies_Good asserts
// every genuinely dense, single-attention-type architecture still qualifies
// for the experimental small-decode quick-win route.
func TestArchitecture_IsROCmDenseQuickWinArchitecture_DenseFamilies_Good(t *testing.T) {
	for _, architecture := range []string{
		"gemma3", "gemma3_text", "qwen3", "mistral", "phi", "glm", "glm4", "hermes", "granite",
	} {
		core.AssertTrue(t, isROCmDenseQuickWinArchitecture(architecture), architecture)
	}
}

// TestArchitecture_IsROCmDenseQuickWinArchitecture_Qwen36Hybrid_Bad asserts
// qwen3_6 (and its normalisation aliases) is NOT a dense quick-win candidate
// post-#50: dense_config.go's IsQwen36Hybrid names it a gated-delta hybrid,
// not a plain dense transformer, and its only attempted end-to-end ROCm
// runtime — the composed-engine detour — is retired outright
// (loadHIPComposedTextModel declines it; profile.SupportedNativeArchitecture
// excludes it). Before this fix the whitelist claimed otherwise.
func TestArchitecture_IsROCmDenseQuickWinArchitecture_Qwen36Hybrid_Bad(t *testing.T) {
	for _, architecture := range []string{"qwen3_6", "Qwen3_5ForConditionalGeneration", "qwen3.6"} {
		core.AssertFalse(t, isROCmDenseQuickWinArchitecture(architecture), architecture)
	}
}

// TestArchitecture_IsROCmDenseQuickWinArchitecture_NonDenseFamilies_Ugly
// asserts MoE, embedding, and other non-dense families surprisingly-but-
// correctly stay out of the whitelist too — the "quick win" label is
// specific to plain dense transformers, not a general "recognised" bucket.
func TestArchitecture_IsROCmDenseQuickWinArchitecture_NonDenseFamilies_Ugly(t *testing.T) {
	for _, architecture := range []string{"mixtral", "qwen3_moe", "deepseek", "deepseek_r1", "bert", "gemma4", ""} {
		core.AssertFalse(t, isROCmDenseQuickWinArchitecture(architecture), architecture)
	}
}
