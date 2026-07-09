// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"testing"

	"dappco.re/go/inference"
)

func TestResponses_ResponseMessages_Good(t *testing.T) {
	req := ResponseRequest{
		Instructions: "Be concise.",
		Input: []ResponseInputMessage{
			{Role: "user", Content: "hello"},
		},
	}

	messages := ResponseMessages(req)

	if len(messages) != 2 {
		t.Fatalf("len(messages) = %d, want 2", len(messages))
	}
	if messages[0].Role != "system" || messages[1].Content != "hello" {
		t.Fatalf("messages = %+v", messages)
	}
}

// TestResponses_ResponseMessages_Bad covers the degenerate zero-value
// request — no instructions and no input must yield an empty (not
// nil-panicking) message slice.
func TestResponses_ResponseMessages_Bad(t *testing.T) {
	messages := ResponseMessages(ResponseRequest{})

	if len(messages) != 0 {
		t.Fatalf("ResponseMessages(zero value) = %+v, want empty", messages)
	}
}

// TestResponses_ResponseMessages_Ugly covers pass-through fidelity —
// ResponseMessages performs no validation of its own, so an input
// message with a blank role/content still maps straight through
// unchanged (ValidateRequest, not this converter, is where rejection
// belongs).
func TestResponses_ResponseMessages_Ugly(t *testing.T) {
	req := ResponseRequest{Input: []ResponseInputMessage{{Role: "", Content: ""}}}

	messages := ResponseMessages(req)

	if len(messages) != 1 || messages[0].Role != "" || messages[0].Content != "" {
		t.Fatalf("ResponseMessages() = %+v, want the blank message passed through unchanged", messages)
	}
}

func TestResponses_ResponseGenerateOptions_Good(t *testing.T) {
	maxTokens := 12
	temperature := float32(0)
	minP := float32(0.04)
	req := ResponseRequest{
		Model:           "qwen",
		Input:           []ResponseInputMessage{{Role: "user", Content: "hi"}},
		MaxOutputTokens: &maxTokens,
		Temperature:     &temperature,
		MinP:            &minP,
	}

	opts, err := ResponseGenerateOptions(req)
	if err != nil {
		t.Fatalf("ResponseGenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.MaxTokens != 12 || cfg.Temperature != 0 || cfg.MinP != 0.04 {
		t.Fatalf("cfg = %+v", cfg)
	}
}

// TestResponses_ResponseGenerateOptions_Bad pins the min_p validation
// error through the Responses request path (mirrors the
// ChatCompletionRequest min_p thread-check in request_test.go).
func TestResponses_ResponseGenerateOptions_Bad(t *testing.T) {
	minP := float32(1.5)
	req := ResponseRequest{
		Model: "qwen",
		Input: []ResponseInputMessage{{Role: "user", Content: "hi"}},
		MinP:  &minP,
	}

	_, err := ResponseGenerateOptions(req)
	if err == nil {
		t.Fatal("ResponseGenerateOptions() error = nil, want min_p validation failure")
	}
}

// TestResponses_ResponseGenerateOptions_Ugly covers the fallback path
// where Input is empty but Instructions is set — ResponseGenerateOptions
// must still produce a system message so ValidateRequest's
// non-empty-messages check passes.
func TestResponses_ResponseGenerateOptions_Ugly(t *testing.T) {
	req := ResponseRequest{
		Model:        "qwen",
		Instructions: "Be concise.",
	}

	opts, err := ResponseGenerateOptions(req)
	if err != nil {
		t.Fatalf("ResponseGenerateOptions() error = %v", err)
	}
	if opts == nil {
		t.Fatal("ResponseGenerateOptions() opts = nil, want synthesised system message to satisfy validation")
	}
}

func TestResponses_NewTextResponse_Good(t *testing.T) {
	resp := NewTextResponse("resp_1", "qwen", "ok", inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 2})

	if resp.ID != "resp_1" || resp.Object != "response" || resp.Model != "qwen" {
		t.Fatalf("response identity = %+v", resp)
	}
	if resp.Usage.TotalTokens != 5 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if resp.Output[0].Content[0].Text != "ok" {
		t.Fatalf("output = %+v", resp.Output)
	}
}

// TestResponses_NewTextResponse_Bad covers the all-empty/zero-value
// inputs — id, model and text all blank, zero metrics — the response
// must still be well-formed (a populated Output entry) rather than
// leaving a nil slice a caller would need to nil-check.
func TestResponses_NewTextResponse_Bad(t *testing.T) {
	resp := NewTextResponse("", "", "", inference.GenerateMetrics{})

	if resp.Object != "response" || len(resp.Output) != 1 || resp.Output[0].Content[0].Text != "" {
		t.Fatalf("NewTextResponse(all empty) = %+v, want a well-formed response with blank fields", resp)
	}
	if resp.Usage.TotalTokens != 0 {
		t.Fatalf("Usage = %+v, want all-zero", resp.Usage)
	}
}

// TestResponses_NewTextResponse_Ugly covers independence between
// calls — each invocation must build its own Output/Content slices
// rather than sharing backing storage with a previous response.
func TestResponses_NewTextResponse_Ugly(t *testing.T) {
	first := NewTextResponse("resp_1", "qwen", "first", inference.GenerateMetrics{})
	second := NewTextResponse("resp_2", "qwen", "second", inference.GenerateMetrics{})

	if first.Output[0].Content[0].Text != "first" {
		t.Fatalf("first response mutated to %+v after building a second response", first.Output)
	}
	if second.Output[0].Content[0].Text != "second" {
		t.Fatalf("second response = %+v, want second", second.Output)
	}
}
