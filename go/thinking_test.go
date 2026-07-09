// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "testing"

func TestThinking_ThinkingConfig_Good(t *testing.T) {
	var got ThinkingChunk
	cfg := ThinkingConfig{Mode: ThinkingCapture, Capture: func(c ThinkingChunk) { got = c }}

	cfg.Capture(ThinkingChunk{Text: "reasoning", Channel: "thinking", Model: "gemma4"})

	checkEqual(t, ThinkingCapture, cfg.Mode)
	checkEqual(t, "reasoning", got.Text)
	checkEqual(t, "thinking", got.Channel)
	checkEqual(t, "gemma4", got.Model)
}

func TestThinking_ThinkingMode_Good(t *testing.T) {
	checkEqual(t, ThinkingMode("show"), ThinkingShow)
	checkEqual(t, ThinkingMode("hide"), ThinkingHide)
	checkEqual(t, ThinkingMode("capture"), ThinkingCapture)
}

func TestThinking_WithThinking_Good(t *testing.T) {
	cfg := ApplyGenerateOpts([]GenerateOption{WithThinking(ThinkingConfig{Mode: ThinkingHide})})

	checkEqual(t, ThinkingHide, cfg.Thinking.Mode)
}

func TestThinking_GenerateConfigEngineKnobs_Good(t *testing.T) {
	cfg := GenerateConfig{
		TraceTokenPhases:             true,
		TraceTokenText:               true,
		GenerationClearCache:         true,
		GenerationClearCacheInterval: 32,
	}

	checkTrue(t, cfg.TraceTokenPhases)
	checkTrue(t, cfg.TraceTokenText)
	checkTrue(t, cfg.GenerationClearCache)
	checkEqual(t, 32, cfg.GenerationClearCacheInterval)
}
