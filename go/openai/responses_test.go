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
	req := ResponseRequest{
		Model:           "qwen",
		Input:           []ResponseInputMessage{{Role: "user", Content: "hi"}},
		MaxOutputTokens: &maxTokens,
		Temperature:     &temperature,
	}

	opts, err := ResponseGenerateOptions(req)
	if err != nil {
		t.Fatalf("ResponseGenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.MaxTokens != 12 || cfg.Temperature != 0 {
		t.Fatalf("cfg = %+v", cfg)
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
