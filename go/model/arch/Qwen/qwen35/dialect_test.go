// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import "testing"

// TestChatMLDialect_Good proves every qwen-family model_type (any casing) declares ChatML.
func TestChatMLDialect_Good(t *testing.T) {
	for _, mt := range []string{"qwen3_5", "qwen3_5_moe_text", "qwen3_next", "Qwen3", "qwen2", "QWEN3_6_MOE"} {
		if !ChatMLDialect(mt) {
			t.Fatalf("ChatMLDialect(%q) = false, want true — every qwen model_type speaks ChatML", mt)
		}
	}
}

// TestChatMLDialect_Bad proves non-qwen model_types keep the gemma fallback (false).
func TestChatMLDialect_Bad(t *testing.T) {
	for _, mt := range []string{"gemma4", "mamba2", "llama", "", "mixtral"} {
		if ChatMLDialect(mt) {
			t.Fatalf("ChatMLDialect(%q) = true, want false — non-qwen archs keep the gemma template", mt)
		}
	}
}

// TestHybridModelType_Good proves all seven released hybrid ids are recognised.
func TestHybridModelType_Good(t *testing.T) {
	for _, mt := range []string{"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_6", "qwen3_6_moe", "qwen3_next"} {
		if !HybridModelType(mt) {
			t.Fatalf("HybridModelType(%q) = false, want true — a released hybrid id", mt)
		}
	}
}

// TestHybridModelType_Bad proves plain-transformer qwen ids and the MTP drafter ids are NOT
// hybrids — the drafter refuses standalone loads via its own registration, not this predicate,
// and qwen2/qwen3 are ordinary transformers the ArchSession serves.
func TestHybridModelType_Bad(t *testing.T) {
	for _, mt := range []string{"qwen2", "qwen3", "qwen3_5_mtp", "qwen3_6_mtp", "gemma4", ""} {
		if HybridModelType(mt) {
			t.Fatalf("HybridModelType(%q) = true, want false", mt)
		}
	}
}
