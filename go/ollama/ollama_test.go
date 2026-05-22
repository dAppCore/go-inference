// SPDX-Licence-Identifier: EUPL-1.2

package ollama

import (
	"encoding/json"
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

// TestOllama_AppendChatResponse_WireMatchesEncodingJSON pins the
// hand-rolled AppendChatResponse output byte-for-byte against
// encoding/json.Marshal across the canonical streaming and final-
// chunk shapes the server emits. Wire compatibility is load-bearing
// — ollama-compatible clients (e.g. open-webui's stream parser)
// expect field-order-stable NDJSON.
func TestOllama_AppendChatResponse_WireMatchesEncodingJSON(t *testing.T) {
	cases := []struct {
		name string
		in   ChatResponse
	}{
		{"streaming intermediate", ChatResponse{Model: "qwen3", Message: Message{Content: "tok"}, Done: false}},
		{"streaming priming", ChatResponse{Model: "qwen3", Message: Message{Role: "assistant", Content: "The"}, Done: false}},
		{"final with metrics", ChatResponse{
			Model: "qwen3", Message: Message{Role: "assistant", Content: "summary is concise."}, Done: true,
			PromptEvalCount:    200,
			EvalCount:          32,
			TotalDuration:      1_500_000_000,
			LoadDuration:       100_000_000,
			PromptEvalDuration: 200_000_000,
			EvalDuration:       1_200_000_000,
		}},
		{"escape-heavy content", ChatResponse{Model: "qwen3", Message: Message{Content: "line1\n\"q\"\tend"}, Done: false}},
		{"empty model + message", ChatResponse{Done: true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := AppendChatResponse(nil, tc.in)
			want, err := json.Marshal(tc.in)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			if string(got) != string(want) {
				t.Fatalf("wire drift:\n got = %s\nwant = %s", got, want)
			}
			// Round-trip through encoding/json decoder must yield the
			// original struct — proves the wire output is parseable by
			// downstream ollama-compat clients.
			var back ChatResponse
			if err := json.Unmarshal(got, &back); err != nil {
				t.Fatalf("Unmarshal(%s): %v", got, err)
			}
			if back != tc.in {
				t.Fatalf("round-trip:\n got = %+v\nwant = %+v", back, tc.in)
			}
		})
	}
}

// TestOllama_AppendGenerateResponse_WireMatchesEncodingJSON mirrors
// the ChatResponse pin for /api/generate.
func TestOllama_AppendGenerateResponse_WireMatchesEncodingJSON(t *testing.T) {
	cases := []struct {
		name string
		in   GenerateResponse
	}{
		{"streaming token", GenerateResponse{Model: "qwen3", Response: "tok", Done: false}},
		{"empty response", GenerateResponse{Model: "qwen3", Done: false}},
		{"final with metrics", GenerateResponse{
			Model: "qwen3", Response: "The summary is concise.", Done: true,
			PromptEvalCount:    200,
			EvalCount:          32,
			TotalDuration:      1_500_000_000,
			LoadDuration:       100_000_000,
			PromptEvalDuration: 200_000_000,
			EvalDuration:       1_200_000_000,
		}},
		{"escape-heavy", GenerateResponse{Model: "qwen3", Response: "line1\n\"q\"\tend", Done: false}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := AppendGenerateResponse(nil, tc.in)
			want, err := json.Marshal(tc.in)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			if string(got) != string(want) {
				t.Fatalf("wire drift:\n got = %s\nwant = %s", got, want)
			}
			var back GenerateResponse
			if err := json.Unmarshal(got, &back); err != nil {
				t.Fatalf("Unmarshal(%s): %v", got, err)
			}
			if back != tc.in {
				t.Fatalf("round-trip:\n got = %+v\nwant = %+v", back, tc.in)
			}
		})
	}
}

