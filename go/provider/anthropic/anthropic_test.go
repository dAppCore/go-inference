// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import (
	"testing"

	"dappco.re/go/inference"
)

func TestAnthropic_InferenceMessages_Good(t *testing.T) {
	req := MessageRequest{
		System: "system",
		Messages: []Message{{
			Role:    "user",
			Content: []ContentBlock{{Type: "text", Text: "hello"}},
		}},
	}

	messages := InferenceMessages(req)

	if len(messages) != 2 {
		t.Fatalf("len(messages) = %d, want 2", len(messages))
	}
	if messages[0].Role != "system" || messages[1].Content != "hello" {
		t.Fatalf("messages = %+v", messages)
	}
}

func TestAnthropic_GenerateOptions_Good(t *testing.T) {
	temp := float32(0.2)
	minP := float32(0.04)
	topK := 4
	opts := GenerateOptions(MessageRequest{MaxTokens: 9, Temperature: &temp, MinP: &minP, TopK: &topK})

	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.MaxTokens != 9 || cfg.Temperature != 0.2 || cfg.MinP != 0.04 || cfg.TopK != 4 {
		t.Fatalf("cfg = %+v", cfg)
	}
}

func TestAnthropic_NewTextResponse_Good(t *testing.T) {
	resp := NewTextResponse("msg_1", "claude-ish", "ok", inference.GenerateMetrics{PromptTokens: 2, GeneratedTokens: 3})

	if resp.ID != "msg_1" || resp.Type != "message" || resp.Role != "assistant" {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.Content[0].Text != "ok" || resp.Usage.OutputTokens != 3 {
		t.Fatalf("resp = %+v", resp)
	}
}
