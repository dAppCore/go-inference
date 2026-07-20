// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import "strings"

// dialect.go — family predicates the serving engines consult. They live HERE because the arch
// package owns its released model_type ids (AX: path is documentation); the retired composed
// engine's register.go used to carry the dialect helper (#50).

// ChatMLDialect reports whether a model_type is served with the ChatML chat dialect
// (<|im_start|>role\n…<|im_end|> turns, an "assistant" generation cue and a <think> reasoning
// block) rather than the gemma turn template. Every Qwen family model speaks ChatML; matching on
// the "qwen" prefix keeps a new qwen model_type ChatML with zero edits.
//
//	qwen35.ChatMLDialect("qwen3_5_moe") // true — ChatML framing
//	qwen35.ChatMLDialect("gemma4")      // false — gemma turn template
func ChatMLDialect(modelType string) bool {
	return strings.HasPrefix(strings.ToLower(modelType), "qwen")
}

// HybridModelType reports whether a model_type is one of this package's released gated-delta
// hybrid ids — the seven names registered in register.go (qwen3_5/qwen3_5_moe + their text_config
// aliases, qwen3_6/qwen3_6_moe, and qwen3_next). Serving engines consult it where hybrid
// recurrent state changes the route (e.g. the speculative pair loader declines these targets:
// gated-delta layers thread recurrent state, not KV an AssistantPair could share).
//
//	qwen35.HybridModelType("qwen3_6_moe") // true — gated-delta hybrid
//	qwen35.HybridModelType("qwen3")       // false — plain transformer
func HybridModelType(modelType string) bool {
	switch modelType {
	case "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
		"qwen3_6", "qwen3_6_moe", "qwen3_next":
		return true
	}
	return false
}
