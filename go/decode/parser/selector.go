// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	core "dappco.re/go"
)

// key := parser.NormaliseKey("Qwen-3.5")  // "qwen_3_5"
func NormaliseKey(value string) string {
	value = core.Trim(value)
	if value == "" {
		return ""
	}
	// Fast path: scan for any byte that needs transforming (uppercase
	// letter, '-', '.'). If none found, return the trimmed string
	// directly with no allocation. Adapter sites that pass already-
	// canonical keys (e.g. "qwen3", "gemma4_text") land here on every
	// Lookup / LookupHint call. The previous shape always paid the
	// core.Lower string copy + two replaceAll string copies regardless
	// of whether substitution actually happened.
	needsTransform := false
	for i := 0; i < len(value); i++ {
		c := value[i]
		if (c >= 'A' && c <= 'Z') || c == '-' || c == '.' {
			needsTransform = true
			break
		}
	}
	if !needsTransform {
		return value
	}
	// Fused single-pass transform: lowercase ASCII letters AND replace
	// `-` and `.` with `_` in one allocation. Non-ASCII bytes pass
	// through unchanged (Lower only touches ASCII anyway — core.Lower
	// → strings.ToLower returns the input unchanged when no Unicode
	// uppercase letters are present, but otherwise allocates a new
	// string; for our wire-key inputs that's a guaranteed alloc when
	// any A-Z is present).
	buf := make([]byte, len(value))
	for i := 0; i < len(value); i++ {
		c := value[i]
		switch {
		case c >= 'A' && c <= 'Z':
			buf[i] = c + ('a' - 'A')
		case c == '-' || c == '.':
			buf[i] = '_'
		default:
			buf[i] = c
		}
	}
	return string(buf)
}

// family := parser.Family(parser.Hint{Architecture: "qwen3"})  // "qwen"
func Family(hint Hint) string {
	arch := NormaliseKey(hint.Architecture)
	adapter := NormaliseKey(hint.AdapterName)
	// Scan arch and adapter separately rather than concatenating them.
	// The old shape built `arch + " " + adapter` once per call (one
	// string alloc on the per-stream LookupHint path) purely to run
	// Contains over the pair. Because the separator is a space and no
	// family needle below contains a space, a needle can never straddle
	// the boundary — so Contains(arch+" "+adapter, n) is exactly
	// Contains(arch, n) || Contains(adapter, n). familyContains encodes
	// that, byte-identically, with zero allocation.
	switch {
	case familyContains(arch, adapter, "qwen"):
		return "qwen"
	case familyContains(arch, adapter, "gemma"):
		return "gemma"
	case familyContains(arch, adapter, "minimax"):
		return "minimax"
	case familyContains(arch, adapter, "deepseek"):
		return "deepseek_r1"
	case familyContains(arch, adapter, "gpt_oss"), familyContains(arch, adapter, "gptoss"):
		return "gpt_oss"
	case familyContains(arch, adapter, "mistral"), familyContains(arch, adapter, "mixtral"):
		return "mistral"
	case familyContains(arch, adapter, "kimi"), familyContains(arch, adapter, "moonshot"):
		return "kimi"
	case familyContains(arch, adapter, "glm"), familyContains(arch, adapter, "chatglm"):
		return "glm"
	case familyContains(arch, adapter, "hermes"):
		return "hermes"
	case familyContains(arch, adapter, "granite"):
		return "granite"
	default:
		return "generic"
	}
}

// familyContains reports whether needle occurs in either the architecture
// or adapter key. It replaces a Concat-then-Contains over the joined pair;
// the needle never contains the space separator the join would insert, so
// the two are equivalent. A plain function (not a closure over arch/adapter)
// keeps it allocation-free — a capturing closure would heap-escape.
func familyContains(arch, adapter, needle string) bool {
	return core.Contains(arch, needle) || core.Contains(adapter, needle)
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
