// SPDX-Licence-Identifier: EUPL-1.2

// Tests for request decoding, validation, and generate-option resolution.
package openai

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
)

// erroringReader fails every Read — exercises DecodeRequest's
// io.ReadAll error branch without any real network I/O.
type erroringReader struct{}

func (erroringReader) Read([]byte) (int, error) {
	return 0, errRead
}

// TestOpenAI_DecodeRequest_Bad covers the nil-body, read-error, and
// malformed-JSON rejections.
func TestOpenAI_DecodeRequest_Bad(t *testing.T) {
	if _, err := DecodeRequest(nil); err == nil {
		t.Fatal("DecodeRequest(nil) error = nil, want request-body-nil failure")
	}
	if _, err := DecodeRequest(erroringReader{}); err == nil {
		t.Fatal("DecodeRequest(erroring reader) error = nil, want read failure")
	}
	if _, err := DecodeRequest(strings.NewReader(`{`)); err == nil {
		t.Fatal("DecodeRequest(malformed json) error = nil, want decode failure")
	}
}

// TestOpenAI_ValidateRequest_Bad drives every rejection branch:
// missing model, empty messages, an unrecognised role, and each
// sampling-field out-of-range case.
func TestOpenAI_ValidateRequest_Bad(t *testing.T) {
	badTemp := float32(3)
	badTopP := float32(-0.1)
	badTopK := -1
	badMaxTokens := -1
	validMsgs := []ChatMessage{{Role: "user", Content: "hi"}}

	cases := []struct {
		name string
		req  ChatCompletionRequest
	}{
		{"model-empty", ChatCompletionRequest{Messages: validMsgs}},
		{"messages-empty", ChatCompletionRequest{Model: "m"}},
		{"role-invalid", ChatCompletionRequest{Model: "m", Messages: []ChatMessage{{Role: "bogus", Content: "hi"}}}},
		{"temperature-out-of-range", ChatCompletionRequest{Model: "m", Messages: validMsgs, Temperature: &badTemp}},
		{"top_p-out-of-range", ChatCompletionRequest{Model: "m", Messages: validMsgs, TopP: &badTopP}},
		{"top_k-negative", ChatCompletionRequest{Model: "m", Messages: validMsgs, TopK: &badTopK}},
		{"max_tokens-negative", ChatCompletionRequest{Model: "m", Messages: validMsgs, MaxTokens: &badMaxTokens}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateRequest(tc.req); err == nil {
				t.Fatalf("ValidateRequest(%+v) error = nil, want rejection", tc.req)
			}
		})
	}
}

// TestOpenAI_ValidateRequest_Good covers every accepted role — the
// Bad table only ever exercises "user"; the others (system, developer,
// assistant, tool) share the same switch statement but need their own
// hit to mark that case arm covered.
func TestOpenAI_ValidateRequest_Good(t *testing.T) {
	for _, role := range []string{"system", "developer", "user", "assistant", "tool"} {
		t.Run(role, func(t *testing.T) {
			req := ChatCompletionRequest{Model: "m", Messages: []ChatMessage{{Role: role, Content: "hi"}}}
			if err := ValidateRequest(req); err != nil {
				t.Fatalf("ValidateRequest(role=%s) error = %v", role, err)
			}
		})
	}
}

// TestOpenAI_NormalizeStopSequences_Bad covers the empty-after-trim
// rejection.
func TestOpenAI_NormalizeStopSequences_Bad(t *testing.T) {
	_, err := NormalizeStopSequences(StopList{"END", "   "})
	if err == nil || !strings.Contains(err.Error(), "stop") {
		t.Fatalf("NormalizeStopSequences() error = %v, want stop-sequence validation failure", err)
	}
}

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
		MinP:        &zeroFloat,
		TopK:        &zeroInt,
		MaxTokens:   &zeroInt,
	}

	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.Temperature != 0 || cfg.TopP != 0 || cfg.MinP != 0 || cfg.TopK != 0 || cfg.MaxTokens != 0 {
		t.Fatalf("explicit zero options = %+v", cfg)
	}
}

func TestOpenAI_GenerateOptions_Good_MinP(t *testing.T) {
	minP := float32(0.05)
	req := ChatCompletionRequest{
		Model:    "qwen",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
		MinP:     &minP,
	}
	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.MinP != 0.05 {
		t.Fatalf("MinP = %v, want 0.05", cfg.MinP)
	}
}

func TestOpenAI_GenerateOptions_Bad_MinP(t *testing.T) {
	minP := float32(1.1)
	req := ChatCompletionRequest{
		Model:    "qwen",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
		MinP:     &minP,
	}
	_, err := GenerateOptions(req)
	if err == nil || !strings.Contains(err.Error(), "min_p") {
		t.Fatalf("GenerateOptions() error = %v, want min_p validation", err)
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
