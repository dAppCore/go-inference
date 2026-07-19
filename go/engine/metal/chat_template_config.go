// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/engine"
)

// loadChatTemplateDefaultSystem reads the checkpoint's chat template and returns
// the default system prompt it injects when a conversation carries none — the
// Qwen2.5 rule — or "" when the template injects none (gemma, Qwen3.5/3.6) or no
// template ships. It prefers the standalone chat_template.jinja (newer
// checkpoints) and falls back to the "chat_template" field of
// tokenizer_config.json (older, e.g. Qwen2.5-Coder). The byte-level extraction
// lives in the shared engine package (engine.ExtractDefaultSystem, folded by
// engine/metal and engine/hip alike); this shell only locates and reads the
// file, the same soft-optional convention as loadGenerationConfigStops.
func loadChatTemplateDefaultSystem(dir string) string {
	if read := core.ReadFile(core.PathJoin(dir, "chat_template.jinja")); read.OK {
		if sys := engine.ExtractDefaultSystem(string(read.Bytes())); sys != "" {
			return sys
		}
	}
	read := core.ReadFile(core.PathJoin(dir, "tokenizer_config.json"))
	if !read.OK {
		return ""
	}
	data := read.Bytes()
	var cfg struct {
		ChatTemplate string `json:"chat_template"`
	}
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return ""
	}
	return engine.ExtractDefaultSystem(cfg.ChatTemplate)
}
