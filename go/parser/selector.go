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

func replaceAll(text, old, next string) string {
	if old == "" {
		return text
	}
	out := core.NewBuilder()
	for {
		idx := indexString(text, old)
		if idx < 0 {
			out.WriteString(text)
			return out.String()
		}
		out.WriteString(text[:idx])
		out.WriteString(next)
		text = text[idx+len(old):]
	}
}

func indexString(s, substr string) int {
	if substr == "" {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
