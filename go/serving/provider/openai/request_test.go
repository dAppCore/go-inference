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

// TestRequest_DecodeRequest_Bad covers the nil-body, read-error, and
// malformed-JSON rejections.
func TestRequest_DecodeRequest_Bad(t *testing.T) {
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

// TestRequest_DecodeRequest_Ugly covers forward-compatibility — an
// unrecognised top-level field must be skipped rather than rejected,
// so a newer client sending a field this adapter doesn't model yet
// still decodes.
func TestRequest_DecodeRequest_Ugly(t *testing.T) {
	body := strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hi"}],"unknown_field":{"nested":true}}`)

	req, err := DecodeRequest(body)
	if err != nil || req.Model != "qwen" || len(req.Messages) != 1 {
		t.Fatalf("DecodeRequest(unknown field) = %+v, err=%v, want the unknown field skipped", req, err)
	}
}

// TestRequest_ValidateRequest_Bad drives every rejection branch:
// missing model, empty messages, an unrecognised role, and each
// sampling-field out-of-range case.
func TestRequest_ValidateRequest_Bad(t *testing.T) {
	badTemp := float32(3)
	badTopP := float32(-0.1)
	badTopK := -1
	badMaxTokens := -1
	badVisionBudget := -1
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
		{"mm_processor_kwargs-max_soft_tokens-negative", ChatCompletionRequest{Model: "m", Messages: validMsgs, MMProcessorKwargs: &MMProcessorKwargs{MaxSoftTokens: &badVisionBudget}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateRequest(tc.req); err == nil {
				t.Fatalf("ValidateRequest(%+v) error = nil, want rejection", tc.req)
			}
		})
	}
}

// TestRequest_ValidateRequest_Good covers every accepted role — the
// Bad table only ever exercises "user"; the others (system, developer,
// assistant, tool) share the same switch statement but need their own
// hit to mark that case arm covered.
func TestRequest_ValidateRequest_Good(t *testing.T) {
	for _, role := range []string{"system", "developer", "user", "assistant", "tool"} {
		t.Run(role, func(t *testing.T) {
			req := ChatCompletionRequest{Model: "m", Messages: []ChatMessage{{Role: role, Content: "hi"}}}
			if err := ValidateRequest(req); err != nil {
				t.Fatalf("ValidateRequest(role=%s) error = %v", role, err)
			}
		})
	}
}

// TestRequest_ValidateRequest_Ugly covers normalisation — a padded
// model name and a padded, mixed-case role must still be accepted
// (core.Trim/core.Lower run before the required-field and role checks).
func TestRequest_ValidateRequest_Ugly(t *testing.T) {
	req := ChatCompletionRequest{Model: "  m  ", Messages: []ChatMessage{{Role: "  User  ", Content: "hi"}}}

	if err := ValidateRequest(req); err != nil {
		t.Fatalf("ValidateRequest(padded model, mixed-case role) error = %v, want normalised acceptance", err)
	}
}

// TestRequest_NormalizeStopSequences_Good covers the plain trimming
// path across multiple stop entries.
func TestRequest_NormalizeStopSequences_Good(t *testing.T) {
	stops, err := NormalizeStopSequences(StopList{" END ", "STOP"})
	if err != nil {
		t.Fatalf("NormalizeStopSequences() error = %v", err)
	}
	if len(stops) != 2 || stops[0] != "END" || stops[1] != "STOP" {
		t.Fatalf("stops = %#v, want trimmed [END STOP]", stops)
	}
}

// TestRequest_NormalizeStopSequences_Bad covers the empty-after-trim
// rejection.
func TestRequest_NormalizeStopSequences_Bad(t *testing.T) {
	_, err := NormalizeStopSequences(StopList{"END", "   "})
	if err == nil || !strings.Contains(err.Error(), "stop") {
		t.Fatalf("NormalizeStopSequences() error = %v, want stop-sequence validation failure", err)
	}
}

// TestRequest_NormalizeStopSequences_Ugly covers the empty-input fast
// path — no stop sequences configured returns nil, nil rather than an
// allocated empty slice.
func TestRequest_NormalizeStopSequences_Ugly(t *testing.T) {
	stops, err := NormalizeStopSequences(nil)
	if err != nil || stops != nil {
		t.Fatalf("NormalizeStopSequences(nil) = %#v, err=%v, want nil, nil", stops, err)
	}
}

func TestRequest_DecodeRequest_Good(t *testing.T) {
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
	// Omitted sampling scalars stay UNSET (flags false, zero values) so the
	// model's declared generation_config defaults win downstream; only
	// max_tokens carries a provider default.
	if cfg.TemperatureSet || cfg.TopPSet || cfg.TopKSet || cfg.MinPSet {
		t.Fatalf("omitted sampling fields raised set flags: %+v", cfg)
	}
	if cfg.Temperature != 0 || cfg.TopP != 0 || cfg.TopK != 0 || cfg.MaxTokens != DefaultMaxTokens {
		t.Fatalf("defaults = %+v", cfg)
	}
}

// TestRequest_GenerateOptions_Good is the canonical explicit-zero
// case; the other TestOpenAI_GenerateOptions_Good_* variants below
// cover additional sampling-field and thinking-control branches.
func TestRequest_GenerateOptions_Good(t *testing.T) {
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

// TestRequest_GenerateOptions_Bad is the canonical validation-failure
// case (out-of-range min_p); TestOpenAI_ValidateRequest_Bad already
// drives ValidateRequest's other rejection branches, all of which
// GenerateOptions delegates to before building options.
func TestRequest_GenerateOptions_Bad(t *testing.T) {
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

// TestRequest_GenerateOptions_Ugly covers thinkingOverride's own
// precedence rule — chat_template_kwargs.enable_thinking must win over
// a conflicting reasoning_effort=="none", not the other way round.
func TestRequest_GenerateOptions_Ugly(t *testing.T) {
	on := true
	req := ChatCompletionRequest{
		Model:              "m",
		Messages:           []ChatMessage{{Role: "user", Content: "hi"}},
		ReasoningEffort:    "none",
		ChatTemplateKwargs: &ChatTemplateKwargs{EnableThinking: &on},
	}

	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.EnableThinking == nil || !*cfg.EnableThinking {
		t.Fatalf("EnableThinking = %v, want chat_template_kwargs to win over reasoning_effort=none", cfg.EnableThinking)
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

// --- VisionBudget --------------------------------------------------------

// TestOpenAI_GenerateOptions_Good_VisionBudgetViaMMProcessorKwargs pins the
// vLLM mm_processor_kwargs surface: decoded off the wire (exercises the
// hand-rolled kwargs walker) then confirmed to reach GenerateConfig.
func TestOpenAI_GenerateOptions_Good_VisionBudgetViaMMProcessorKwargs(t *testing.T) {
	req, err := DecodeRequest(strings.NewReader(
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"mm_processor_kwargs":{"max_soft_tokens":1120}}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	if cfg := inference.ApplyGenerateOpts(opts); cfg.VisionBudget != 1120 {
		t.Fatalf("VisionBudget = %d, want 1120", cfg.VisionBudget)
	}
}

// TestOpenAI_GenerateOptions_Good_VisionBudgetViaImageDetail covers the
// OpenAI-native image_url.detail mapping: "low"->70, "high"->1120; "auto" and
// an absent detail leave the model's own configured default (0, no override).
func TestOpenAI_GenerateOptions_Good_VisionBudgetViaImageDetail(t *testing.T) {
	cases := []struct {
		name   string
		detail string
		want   int
	}{
		{"low", "low", 70},
		{"high", "high", 1120},
		{"auto", "auto", 0},
		{"absent", "", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := ChatCompletionRequest{
				Model:    "m",
				Messages: []ChatMessage{{Role: "user", Content: "hi", ImageDetail: tc.detail}},
			}
			opts, err := GenerateOptions(req)
			if err != nil {
				t.Fatalf("GenerateOptions() error = %v", err)
			}
			if cfg := inference.ApplyGenerateOpts(opts); cfg.VisionBudget != tc.want {
				t.Fatalf("VisionBudget = %d, want %d", cfg.VisionBudget, tc.want)
			}
		})
	}
}

// TestOpenAI_GenerateOptions_Good_VisionBudgetImageDetailLastMessageWins pins
// the multi-message resolution order: visionBudgetOverride scans from the
// LAST message backwards, so the most recent explicit detail hint governs
// when messages disagree.
func TestOpenAI_GenerateOptions_Good_VisionBudgetImageDetailLastMessageWins(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "m",
		Messages: []ChatMessage{
			{Role: "user", Content: "first", ImageDetail: "high"},
			{Role: "user", Content: "second", ImageDetail: "low"},
		},
	}
	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	if cfg := inference.ApplyGenerateOpts(opts); cfg.VisionBudget != 70 {
		t.Fatalf("VisionBudget = %d, want 70 (the LAST message's detail hint)", cfg.VisionBudget)
	}
}

// TestOpenAI_GenerateOptions_Good_VisionBudgetMMProcessorKwargsPrecedence
// pins visionBudgetOverride's own precedence rule — mm_processor_kwargs.
// max_soft_tokens must win over a conflicting image_url.detail, not the other
// way round (mirrors TestRequest_GenerateOptions_Ugly's thinkingOverride
// precedence pin).
func TestOpenAI_GenerateOptions_Good_VisionBudgetMMProcessorKwargsPrecedence(t *testing.T) {
	budget := 280
	req := ChatCompletionRequest{
		Model:             "m",
		Messages:          []ChatMessage{{Role: "user", Content: "hi", ImageDetail: "high"}}, // alone -> 1120
		MMProcessorKwargs: &MMProcessorKwargs{MaxSoftTokens: &budget},
	}
	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	if cfg := inference.ApplyGenerateOpts(opts); cfg.VisionBudget != 280 {
		t.Fatalf("VisionBudget = %d, want 280 (mm_processor_kwargs must win over image_url.detail=high)", cfg.VisionBudget)
	}
}

// TestOpenAI_GenerateOptions_Bad_VisionBudgetNegative pins the
// mm_processor_kwargs.max_soft_tokens validation surfaced through
// GenerateOptions: a negative budget is a request error, not silently
// clamped or ignored.
func TestOpenAI_GenerateOptions_Bad_VisionBudgetNegative(t *testing.T) {
	budget := -1
	req := ChatCompletionRequest{
		Model:             "m",
		Messages:          []ChatMessage{{Role: "user", Content: "hi"}},
		MMProcessorKwargs: &MMProcessorKwargs{MaxSoftTokens: &budget},
	}
	_, err := GenerateOptions(req)
	if err == nil || !strings.Contains(err.Error(), "mm_processor_kwargs") {
		t.Fatalf("GenerateOptions() error = %v, want mm_processor_kwargs validation", err)
	}
}
