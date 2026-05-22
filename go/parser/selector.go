// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	core "dappco.re/go"
)

//	key := parser.NormaliseKey("Qwen-3.5")  // "qwen_3_5"
func NormaliseKey(value string) string {
	value = core.Lower(core.Trim(value))
	value = replaceAll(value, "-", "_")
	value = replaceAll(value, ".", "_")
	return value
}

//	family := parser.Family(parser.Hint{Architecture: "qwen3"})  // "qwen"
func Family(hint Hint) string {
	arch := NormaliseKey(hint.Architecture)
	adapter := NormaliseKey(hint.AdapterName)
	combined := core.Concat(arch, " ", adapter)
	switch {
	case core.Contains(combined, "qwen"):
		return "qwen"
	case core.Contains(combined, "gemma"):
		return "gemma"
	case core.Contains(combined, "minimax"):
		return "minimax"
	case core.Contains(combined, "deepseek"):
		return "deepseek_r1"
	case core.Contains(combined, "gpt_oss"), core.Contains(combined, "gptoss"):
		return "gpt_oss"
	case core.Contains(combined, "mistral"), core.Contains(combined, "mixtral"):
		return "mistral"
	case core.Contains(combined, "kimi"), core.Contains(combined, "moonshot"):
		return "kimi"
	case core.Contains(combined, "glm"), core.Contains(combined, "chatglm"):
		return "glm"
	case core.Contains(combined, "hermes"):
		return "hermes"
	case core.Contains(combined, "granite"):
		return "granite"
	default:
		return "generic"
	}
}

// replaceAll delegates to core.Replace (strings.ReplaceAll). The
// stdlib implementation pre-counts occurrences and allocates the
// result buffer exactly once — same shape as the hand-rolled loop but
// with byte-level optimisations the builder loop didn't reach. Old
// shape was already 1-2 allocs; stdlib is the same with less code to
// audit.
func replaceAll(text, old, next string) string {
	if old == "" {
		return text
	}
	return core.Replace(text, old, next)
}

// indexString delegates to stdlib via core.Index. The previous
// hand-rolled implementation was a naive O(N×M) byte-by-byte scan;
// stdlib's strings.Index uses Rabin-Karp / SIMD-accelerated byte
// search and runs O(N+M) for the multi-byte markers (`<think>`,
// `<|channel>analysis\n`, etc.) that the thinking/reasoning parsers
// scan against on every per-token Process call.
func indexString(s, substr string) int {
	return core.Index(s, substr)
}
