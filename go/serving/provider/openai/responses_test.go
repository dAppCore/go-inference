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

// TestResponses_ResponseGenerateOptions_Bad_MinP pins the min_p
// validation error through the Responses request path (mirrors the
// ChatCompletionRequest min_p thread-check in request_test.go).
func TestResponses_ResponseGenerateOptions_Bad_MinP(t *testing.T) {
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

// TestResponses_ResponseGenerateOptions_Ugly_InstructionsOnlySynthesisesSystemMessage
// covers the fallback path where Input is empty but Instructions is
// set — ResponseGenerateOptions must still produce a system message
// so ValidateRequest's non-empty-messages check passes.
func TestResponses_ResponseGenerateOptions_Ugly_InstructionsOnlySynthesisesSystemMessage(t *testing.T) {
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
