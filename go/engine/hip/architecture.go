// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
)

func normalizeROCmArchitecture(architecture string) string {
	return rocmprofile.NormalizeArchitecture(architecture)
}

func isROCmGemma4Architecture(architecture string) bool {
	switch normalizeROCmArchitecture(architecture) {
	case "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text":
		return true
	default:
		return false
	}
}

func isROCmGemma4BackboneArchitecture(architecture string) bool {
	return isROCmGemma4Architecture(architecture) || normalizeROCmArchitecture(architecture) == "diffusion_gemma"
}

// isROCmDenseQuickWinArchitecture reports whether architecture is a plain dense
// transformer (single attention type, no MoE routing, no hybrid/recurrent
// layers) eligible for HIP's experimental loader-neutral small-decode route
// (hip_small_decode.go) — a synthetic-buffer kernel smoke path, not a full
// model loader. "qwen3_6" does NOT belong here: dense_config.go's
// IsQwen36Hybrid names it a gated-delta hybrid (interleaved linear/full
// attention), and its only attempt at real end-to-end ROCm serving was the
// composed-engine detour that #50 retired outright (loadHIPComposedTextModel
// declines it, profile.SupportedNativeArchitecture excludes it) — there is no
// working route left for it to be a "quick win" candidate for.
func isROCmDenseQuickWinArchitecture(architecture string) bool {
	switch normalizeROCmArchitecture(architecture) {
	case "gemma3", "gemma3_text", "qwen3", "mistral", "phi", "glm", "glm4", "hermes", "granite":
		return true
	default:
		return false
	}
}

func isROCmGemma4AssistantArchitecture(architecture string) bool {
	switch normalizeROCmArchitecture(architecture) {
	case "gemma4_assistant", "gemma4_unified_assistant":
		return true
	default:
		return false
	}
}

func supportedNativeArchitecture(architecture string) bool {
	return rocmprofile.SupportedNativeArchitecture(architecture)
}

// recognisedArchitecture is the inspection-surface predicate: known to the registry, runnable or
// not (see rocmprofile.RecognisedArchitecture — the #50 recognised-vs-runnable split).
func recognisedArchitecture(architecture string) bool {
	return rocmprofile.RecognisedArchitecture(architecture)
}

func supportedNativeQuantization(bits int, quantType string) bool {
	if bits == 0 && quantType == "" {
		return true
	}
	if bits > 0 && bits <= 8 {
		return true
	}
	quantType = core.Lower(quantType)
	if quantType == "f16" || quantType == "f32" || quantType == "bf16" {
		return true
	}
	return core.Contains(quantType, "q2") ||
		core.Contains(quantType, "q3") ||
		core.Contains(quantType, "q4") ||
		core.Contains(quantType, "q5") ||
		core.Contains(quantType, "q6") ||
		core.Contains(quantType, "q8") ||
		isROCmMetadataQuantization(quantType)
}

func isROCmMetadataQuantization(quantType string) bool {
	quantType = core.Lower(quantType)
	return core.Contains(quantType, "jang") ||
		core.Contains(quantType, "mxtq") ||
		core.Contains(quantType, "codebook") ||
		core.Contains(quantType, "vq") ||
		core.Contains(quantType, "iq") ||
		core.Contains(quantType, "mxfp4") ||
		core.Contains(quantType, "nvfp4")
}

func isROCmMoEArchitecture(architecture string) bool {
	return rocmprofile.IsMoEArchitecture(architecture)
}
