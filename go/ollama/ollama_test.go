// SPDX-Licence-Identifier: EUPL-1.2

package ollama

import (
	"testing"

	"dappco.re/go/inference"
)

func TestOllama_InferenceMessages_Good(t *testing.T) {
	messages := InferenceMessages([]Message{{Role: "user", Content: "hi"}})

	if len(messages) != 1 || messages[0].Role != "user" || messages[0].Content != "hi" {
		t.Fatalf("messages = %+v", messages)
	}
}

func TestOllama_GenerateOptions_Good(t *testing.T) {
	opts := GenerateOptions(Options{NumPredict: 12, Temperature: 0.4, TopK: 8, TopP: 0.7})

	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.MaxTokens != 12 || cfg.Temperature != 0.4 || cfg.TopK != 8 || cfg.TopP != 0.7 {
		t.Fatalf("cfg = %+v", cfg)
	}
}

func TestOllama_NewResponses_Good(t *testing.T) {
	metrics := inference.GenerateMetrics{PromptTokens: 5, GeneratedTokens: 6}
	chat := NewChatResponse("qwen", "ok", metrics)
	generate := NewGenerateResponse("qwen", "ok", metrics)

	if !chat.Done || chat.Message.Content != "ok" || chat.PromptEvalCount != 5 || chat.EvalCount != 6 {
		t.Fatalf("chat = %+v", chat)
	}
	if !generate.Done || generate.Response != "ok" || generate.PromptEvalCount != 5 || generate.EvalCount != 6 {
		t.Fatalf("generate = %+v", generate)
	}
}
