// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"
	"reflect"
	"testing"
)

// TestUnmarshalChatCompletionRequest_ThinkingControls pins the hand-rolled
// decoder for the reasoning toggle: reasoning_effort (top-level string) and
// chat_template_kwargs.enable_thinking (nested object, vLLM/SGLang convention).
func TestUnmarshalChatCompletionRequest_ThinkingControls(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none","chat_template_kwargs":{"foo":"bar","enable_thinking":false}}`
	var req ChatCompletionRequest
	if err := json.Unmarshal([]byte(in), &req); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if req.ReasoningEffort != "none" {
		t.Fatalf("ReasoningEffort = %q, want %q", req.ReasoningEffort, "none")
	}
	if req.ChatTemplateKwargs == nil || req.ChatTemplateKwargs.EnableThinking == nil || *req.ChatTemplateKwargs.EnableThinking {
		t.Fatalf("ChatTemplateKwargs.EnableThinking = %+v, want &false", req.ChatTemplateKwargs)
	}
}

// TestUnmarshalChatCompletionRequest_DirectShapes pins the hand-rolled
// decoder against direct JSON literals. Locks the per-field dispatch
// — present / absent / null variants of every pointer field, the
// StopList variant shape (string vs array), escape-heavy strings,
// multi-turn arrays.
func TestUnmarshalChatCompletionRequest_DirectShapes(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.95)
	topK := 64
	maxTok := 1024
	cases := []struct {
		name string
		in   string
		want ChatCompletionRequest
	}{
		{
			name: "minimal",
			in:   `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`,
			want: ChatCompletionRequest{
				Model:    "gpt-4",
				Messages: []ChatMessage{{Role: "user", Content: "hi"}},
			},
		},
		{
			name: "all-optional-fields-set",
			in:   `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"temperature":0.7,"top_p":0.95,"top_k":64,"max_tokens":1024,"stream":true,"stop":["</s>"],"user":"u123"}`,
			want: ChatCompletionRequest{
				Model:       "gpt-4",
				Messages:    []ChatMessage{{Role: "user", Content: "hi"}},
				Temperature: &temp,
				TopP:        &topP,
				TopK:        &topK,
				MaxTokens:   &maxTok,
				Stream:      true,
				Stop:        StopList{"</s>"},
				User:        "u123",
			},
		},
		{
			name: "stop-as-string",
			in:   `{"model":"gpt-4","messages":[],"stop":"END"}`,
			want: ChatCompletionRequest{
				Model:    "gpt-4",
				Messages: nil,
				Stop:     StopList{"END"},
			},
		},
		{
			name: "pointer-fields-null-keeps-zero",
			in:   `{"model":"gpt-4","messages":[],"temperature":null,"top_p":null,"top_k":null,"max_tokens":null,"stream":null}`,
			want: ChatCompletionRequest{
				Model: "gpt-4",
			},
		},
		{
			name: "unknown-fields-ignored",
			in:   `{"model":"gpt-4","messages":[],"future":42,"extra":"x"}`,
			want: ChatCompletionRequest{
				Model: "gpt-4",
			},
		},
		{
			name: "whitespace-friendly",
			in: `{
				"model": "gpt-4",
				"messages": [
					{ "role": "user", "content": "hi" }
				]
			}`,
			want: ChatCompletionRequest{
				Model:    "gpt-4",
				Messages: []ChatMessage{{Role: "user", Content: "hi"}},
			},
		},
		{
			name: "escape-heavy",
			in:   `{"model":"gpt-4","messages":[{"role":"user","content":"a\nb \"c\" \\d"}]}`,
			want: ChatCompletionRequest{
				Model:    "gpt-4",
				Messages: []ChatMessage{{Role: "user", Content: "a\nb \"c\" \\d"}},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got ChatCompletionRequest
			if err := json.Unmarshal([]byte(tc.in), &got); err != nil {
				t.Fatalf("Unmarshal error = %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("Unmarshal mismatch\ngot:  %+v\nwant: %+v", got, tc.want)
			}
		})
	}
}

func TestUnmarshalResponseRequest_DirectShapes(t *testing.T) {
	temp := float32(0.7)
	maxOut := 256
	cases := []struct {
		name string
		in   string
		want ResponseRequest
	}{
		{
			name: "minimal",
			in:   `{"model":"gpt-4","input":[{"role":"user","content":"hi"}]}`,
			want: ResponseRequest{
				Model: "gpt-4",
				Input: []ResponseInputMessage{{Role: "user", Content: "hi"}},
			},
		},
		{
			name: "with-instructions-and-options",
			in:   `{"model":"gpt-4","input":[{"role":"user","content":"hi"}],"instructions":"sys","temperature":0.7,"max_output_tokens":256,"stream":true}`,
			want: ResponseRequest{
				Model:           "gpt-4",
				Input:           []ResponseInputMessage{{Role: "user", Content: "hi"}},
				Instructions:    "sys",
				Temperature:     &temp,
				MaxOutputTokens: &maxOut,
				Stream:          true,
			},
		},
		{
			name: "stop-as-array",
			in:   `{"model":"gpt-4","input":[],"stop":["</s>","x"]}`,
			want: ResponseRequest{
				Model: "gpt-4",
				Stop:  StopList{"</s>", "x"},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got ResponseRequest
			if err := json.Unmarshal([]byte(tc.in), &got); err != nil {
				t.Fatalf("Unmarshal error = %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("Unmarshal mismatch\ngot:  %+v\nwant: %+v", got, tc.want)
			}
		})
	}
}

// TestUnmarshalChatCompletionRequest_InvalidShapes asserts cleanly
// rejected error shapes — no panics, just errors.
func TestUnmarshalChatCompletionRequest_InvalidShapes(t *testing.T) {
	cases := []string{
		``,
		`{`,
		`}`,
		`{"messages":not-an-array}`,
		`{"temperature":"hot"}`,
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			var req ChatCompletionRequest
			if err := json.Unmarshal([]byte(in), &req); err == nil {
				t.Fatalf("Unmarshal(%q) returned nil error", in)
			}
		})
	}
}
