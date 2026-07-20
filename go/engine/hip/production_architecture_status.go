// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
)

type ProductionArchitectureStatusReport struct {
	TotalArchitectures        int
	NativeArchitectures       int
	MetadataOnlyArchitectures int
	NativeIDs                 []string
	MetadataOnlyIDs           []string
	RemainingGaps             []ProductionArchitectureGap
}

type ProductionArchitectureGap struct {
	ID            string
	Family        string
	Generation    bool
	Chat          bool
	Embeddings    bool
	Rerank        bool
	MoE           bool
	ParserID      string
	ToolParserID  string
	MissingNative string
	NextWork      []string
	Notes         []string
}

// DefaultProductionArchitectureStatus reports ROCm native/staged coverage for
// every architecture advertised by the backend capability report.
func DefaultProductionArchitectureStatus() ProductionArchitectureStatusReport {
	report := ProductionArchitectureStatusReport{
		TotalArchitectures: len(rocmCapabilityArchitectures),
		NativeIDs:          make([]string, 0, len(rocmCapabilityArchitectures)),
		MetadataOnlyIDs:    make([]string, 0),
		RemainingGaps:      make([]ProductionArchitectureGap, 0),
	}
	for _, architecture := range rocmCapabilityArchitectures {
		id := normalizeROCmArchitecture(architecture)
		if supportedNativeArchitecture(id) {
			report.NativeArchitectures++
			report.NativeIDs = append(report.NativeIDs, id)
			continue
		}
		report.MetadataOnlyArchitectures++
		report.MetadataOnlyIDs = append(report.MetadataOnlyIDs, id)
		report.RemainingGaps = append(report.RemainingGaps, productionArchitectureGap(id))
	}
	return report
}

func productionArchitectureGap(id string) ProductionArchitectureGap {
	return ProductionArchitectureGap{
		ID:            id,
		Family:        productionArchitectureFamily(id),
		Generation:    productionArchitectureGeneration(id),
		Chat:          productionArchitectureGeneration(id),
		Embeddings:    id == "bert",
		Rerank:        id == "bert_rerank",
		MoE:           isROCmMoEArchitecture(id),
		MissingNative: productionArchitectureMissingNative(id),
		NextWork:      productionArchitectureNextWork(id),
		Notes:         productionArchitectureGapNotes(id),
	}
}

// productionArchitectureGapNotes reports the truthful, per-id explanation for
// why id has no native ROCm runtime yet. It defers to
// profile.ArchitectureProfileNotes — the single source of truth c73a9377
// corrected post-#50 — rather than duplicating its judgement here, so this
// report cannot drift back out of sync with the profile registry the way it
// did between the composed-engine retirement and this fix (this file was
// last touched at the original hip-engine landing and was never revisited by
// either the composed-strip or the truth-pass merges). For every id currently
// reachable here (deepseek, deepseek_r1, mixtral, qwen3_6, qwen3_6_moe,
// qwen3_moe, qwen3_next — the full set produced by
// DefaultProductionArchitectureStatus today), that source names the retired
// composed detour by number and states plainly that no native execution path
// replaced it.
func productionArchitectureGapNotes(id string) []string {
	return rocmprofile.ArchitectureProfileNotes(id)
}

func productionArchitectureFamily(id string) string {
	switch id {
	case "bert", "bert_rerank":
		return "bert"
	case "qwen2", "qwen3", "qwen3_6", "qwen3_6_moe", "qwen3_moe", "qwen3_next":
		return "qwen"
	case "gemma", "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text", "gemma4_assistant", "gemma4_unified", "gemma4_unified_text":
		return "gemma"
	case "deepseek", "deepseek_r1":
		return "deepseek"
	case "minimax", "minimax_m2":
		return "minimax"
	default:
		return id
	}
}

func productionArchitectureGeneration(id string) bool {
	return id != "bert" && id != "bert_rerank"
}

func productionArchitectureMissingNative(id string) string {
	if id == "bert" {
		return "embedding encoder"
	}
	if id == "bert_rerank" {
		return "rerank scorer"
	}
	if isROCmMoEArchitecture(id) {
		if id == "qwen3_6_moe" {
			return "hybrid linear attention plus sparse expert router"
		}
		if id == "deepseek" || id == "deepseek_r1" {
			return "MoE router plus MLA attention variants"
		}
		if id == "gpt-oss" {
			return "MoE router plus channel parser validation"
		}
		return "sparse expert router"
	}
	if id == "qwen3_6" {
		return "hybrid linear attention"
	}
	return "native loader"
}

func productionArchitectureNextWork(id string) []string {
	switch id {
	case "qwen3_6":
		return []string{"linear_attention_kernel", "native_load_generate_smoke", "retained_state_smoke"}
	case "qwen3_6_moe":
		return []string{"linear_attention_kernel", "sparse_expert_router", "native_load_generate_smoke"}
	case "qwen3_moe", "mixtral", "kimi":
		return []string{"sparse_expert_router", "selected_expert_matvec", "native_load_generate_smoke"}
	case "deepseek", "deepseek_r1":
		return []string{"sparse_expert_router", "mla_attention_variant", "native_load_generate_smoke"}
	case "gpt-oss":
		return []string{"channel_parser_validation", "sparse_expert_router", "native_load_generate_smoke"}
	case "bert":
		return []string{"encoder_loader", "pooled_embedding_output", "no_generation_kv_smoke"}
	case "bert_rerank":
		return []string{"cross_encoder_loader", "score_head_output", "no_generation_kv_smoke"}
	default:
		return []string{"native_loader", "native_smoke"}
	}
}
