// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import (
	"encoding/json"
	"reflect"
	"testing"
)

// TestUnmarshalMessageRequest_DirectShapes pins the hand-rolled
// MessageRequest decoder against direct JSON literals. Locks the
// per-field dispatch — present / absent / null variants of every
// pointer field, escape-heavy strings, multi-turn arrays.
func TestUnmarshalMessageRequest_DirectShapes(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.95)
	topK := 64
	cases := []struct {
		name string
		in   string
		want MessageRequest
	}{
		{
			name: "minimal",
			in:   `{"model":"claude-3","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":256}`,
			want: MessageRequest{
				Model:     "claude-3",
				Messages:  []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
				MaxTokens: 256,
			},
		},
		{
			name: "all-optional-fields-set",
			in:   `{"model":"claude-3","system":"Be concise.","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":1024,"temperature":0.7,"top_p":0.95,"top_k":64,"stream":true,"stop_sequences":["</s>","<|eot|>"]}`,
			want: MessageRequest{
				Model:         "claude-3",
				System:        "Be concise.",
				Messages:      []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
				MaxTokens:     1024,
				Temperature:   &temp,
				TopP:          &topP,
				TopK:          &topK,
				Stream:        true,
				StopSequences: []string{"</s>", "<|eot|>"},
			},
		},
		{
			name: "pointer-fields-null-keeps-zero-value",
			in:   `{"model":"claude-3","messages":[],"max_tokens":256,"temperature":null,"top_p":null,"top_k":null,"stream":null}`,
			want: MessageRequest{
				Model:     "claude-3",
				MaxTokens: 256,
			},
		},
		{
			name: "stop-sequences-as-single-string",
			in:   `{"model":"claude-3","messages":[],"max_tokens":256,"stop_sequences":"</s>"}`,
			want: MessageRequest{
				Model:         "claude-3",
				MaxTokens:     256,
				StopSequences: []string{"</s>"},
			},
		},
		{
			name: "unknown-fields-ignored",
			in:   `{"model":"claude-3","messages":[],"max_tokens":256,"future_field":42,"another":"x"}`,
			want: MessageRequest{
				Model:     "claude-3",
				MaxTokens: 256,
			},
		},
		{
			name: "whitespace-friendly",
			in: `{
				"model": "claude-3",
				"messages": [
					{ "role": "user", "content": [ { "type": "text", "text": "hi" } ] }
				],
				"max_tokens": 256
			}`,
			want: MessageRequest{
				Model:     "claude-3",
				Messages:  []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
				MaxTokens: 256,
			},
		},
		{
			name: "escape-heavy-text",
			in:   `{"model":"claude-3","messages":[{"role":"user","content":[{"type":"text","text":"line1\nline2 \"quoted\" \\back"}]}],"max_tokens":256}`,
			want: MessageRequest{
				Model:     "claude-3",
				Messages:  []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "line1\nline2 \"quoted\" \\back"}}}},
				MaxTokens: 256,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got MessageRequest
			if err := json.Unmarshal([]byte(tc.in), &got); err != nil {
				t.Fatalf("Unmarshal error = %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("Unmarshal mismatch\ngot:  %+v\nwant: %+v", got, tc.want)
			}
		})
	}
}

// TestUnmarshalMessageRequest_InvalidShapes asserts the walker rejects
// malformed bodies cleanly — no panics, just errors.
func TestUnmarshalMessageRequest_InvalidShapes(t *testing.T) {
	cases := []string{
		``,
		`{`,
		`}`,
		`{"model":42}`,
		`{"messages":not-an-array}`,
		`{"temperature":"hot"}`,
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			var req MessageRequest
			if err := json.Unmarshal([]byte(in), &req); err == nil {
				t.Fatalf("Unmarshal(%q) returned nil error", in)
			}
		})
	}
}

// TestUnmarshalMessageResponse_DirectShapes pins the response decoder.
func TestUnmarshalMessageResponse_DirectShapes(t *testing.T) {
	in := `{"id":"msg_1","type":"message","role":"assistant","model":"claude-3","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`
	want := MessageResponse{
		ID:         "msg_1",
		Type:       "message",
		Role:       "assistant",
		Model:      "claude-3",
		Content:    []ContentBlock{{Type: "text", Text: "hello"}},
		StopReason: "end_turn",
		Usage:      Usage{InputTokens: 10, OutputTokens: 5},
	}
	var got MessageResponse
	if err := json.Unmarshal([]byte(in), &got); err != nil {
		t.Fatalf("Unmarshal error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got:  %+v\nwant: %+v", got, want)
	}
}
