// SPDX-Licence-Identifier: EUPL-1.2

// Tests for request decoding, validation, and generate-option resolution.
package openai

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
)

func TestOpenAI_DecodeRequest_Good_StopStringAndDefaults(t *testing.T) {
	body := strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stop":"END"}`)

	req, err := DecodeRequest(body)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if req.Model != "qwen" || len(req.Messages) != 1 {
		t.Fatalf("DecodeRequest() = %+v", req)
	}
	stops, err := NormalizeStopSequences(req.Stop)
	if err != nil {
		t.Fatalf("NormalizeStopSequences() error = %v", err)
	}
	if len(stops) != 1 || stops[0] != "END" {
		t.Fatalf("stops = %#v, want END", stops)
	}

	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.Temperature != DefaultTemperature || cfg.TopP != DefaultTopP || cfg.TopK != DefaultTopK || cfg.MaxTokens != DefaultMaxTokens {
		t.Fatalf("defaults = %+v", cfg)
	}
}

func TestOpenAI_GenerateOptions_Good_HonoursExplicitZero(t *testing.T) {
	zeroFloat := float32(0)
	zeroInt := 0
	req := ChatCompletionRequest{
		Model:       "qwen",
		Messages:    []ChatMessage{{Role: "user", Content: "hi"}},
		Temperature: &zeroFloat,
		TopP:        &zeroFloat,
		TopK:        &zeroInt,
		MaxTokens:   &zeroInt,
	}

	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.Temperature != 0 || cfg.TopP != 0 || cfg.TopK != 0 || cfg.MaxTokens != 0 {
		t.Fatalf("explicit zero options = %+v", cfg)
	}
}

func TestOpenAI_GenerateOptions_Good_ThinkingOffViaChatTemplateKwargs(t *testing.T) {
	off := false
	req := ChatCompletionRequest{
		Model:              "m",
		Messages:           []ChatMessage{{Role: "user", Content: "hi"}},
		ChatTemplateKwargs: &ChatTemplateKwargs{EnableThinking: &off},
	}
	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.EnableThinking == nil || *cfg.EnableThinking {
		t.Fatalf("EnableThinking = %v, want &false", cfg.EnableThinking)
	}
}

func TestOpenAI_GenerateOptions_Good_ThinkingBudgetViaChatTemplateKwargs(t *testing.T) {
	// Decode the budget off the wire (exercises the hand-rolled kwargs walker)
	// then confirm it reaches the GenerateConfig.
	req, err := DecodeRequest(strings.NewReader(
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"thinking_budget":256}}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if req.ChatTemplateKwargs == nil || req.ChatTemplateKwargs.ThinkingBudget == nil || *req.ChatTemplateKwargs.ThinkingBudget != 256 {
		t.Fatalf("decoded thinking_budget = %v, want 256", req.ChatTemplateKwargs)
	}
	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	if cfg := inference.ApplyGenerateOpts(opts); cfg.ThinkingBudget != 256 {
		t.Fatalf("ThinkingBudget = %d, want 256", cfg.ThinkingBudget)
	}
}

func TestOpenAI_GenerateOptions_Good_ThinkingBudgetZeroIgnored(t *testing.T) {
	zero := 0
	req := ChatCompletionRequest{
		Model:              "m",
		Messages:           []ChatMessage{{Role: "user", Content: "hi"}},
		ChatTemplateKwargs: &ChatTemplateKwargs{ThinkingBudget: &zero},
	}
	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	if cfg := inference.ApplyGenerateOpts(opts); cfg.ThinkingBudget != 0 {
		t.Fatalf("ThinkingBudget = %d, want 0 (zero is unlimited, no option emitted)", cfg.ThinkingBudget)
	}
}

func TestOpenAI_GenerateOptions_Good_ThinkingOffViaReasoningEffortNone(t *testing.T) {
	req := ChatCompletionRequest{
		Model:           "m",
		Messages:        []ChatMessage{{Role: "user", Content: "hi"}},
		ReasoningEffort: "none",
	}
	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.EnableThinking == nil || *cfg.EnableThinking {
		t.Fatalf("reasoning_effort=none → EnableThinking = %v, want &false", cfg.EnableThinking)
	}
}

func TestOpenAI_GenerateOptions_Good_ThinkingDefaultLeavesNil(t *testing.T) {
	req := ChatCompletionRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}
	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.EnableThinking != nil {
		t.Fatalf("default → EnableThinking = %v, want nil (model default)", cfg.EnableThinking)
	}
}
