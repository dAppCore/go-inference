// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestChatCompletionChunk_MarshalJSON_RoundTrip locks the hand-rolled
// chunk encoder shape to encoding/json's deserialiser. The encoder
// fires per streamed token; the wire output is consumed by both
// proxy clients and downstream services that re-decode the frame
// back into ChatCompletionChunk.
//
// Cases cover every branch the encoder walks:
//   - empty (no choices, no thought)
//   - priming frame (role-only delta, nil finish_reason -> null)
//   - mid-stream content delta (content-only delta, nil finish)
//   - thought-bearing frame (Thought pointer set)
//   - terminal frame (finish_reason set)
//   - escape-bearing content
func TestChatCompletionChunk_MarshalJSON_RoundTrip(t *testing.T) {
	finishStop := "stop"
	thought := "let me think"
	cases := []struct {
		name string
		in   ChatCompletionChunk
	}{
		{"empty", ChatCompletionChunk{ID: "id", Object: "chat.completion.chunk", Created: 1700000000, Model: "qwen3"}},
		{"priming", ChatCompletionChunk{
			ID: "id", Object: "chat.completion.chunk", Created: 1700000000, Model: "qwen3",
			Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{Role: "assistant"}}},
		}},
		{"delta", ChatCompletionChunk{
			ID: "id", Object: "chat.completion.chunk", Created: 1700000000, Model: "qwen3",
			Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{Content: "Answer"}}},
		}},
		{"thought-bearing", ChatCompletionChunk{
			ID: "id", Object: "chat.completion.chunk", Created: 1700000000, Model: "qwen3",
			Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{Content: "x"}}},
			Thought: &thought,
		}},
		{"terminal", ChatCompletionChunk{
			ID: "id", Object: "chat.completion.chunk", Created: 1700000000, Model: "qwen3",
			Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{}, FinishReason: &finishStop}},
		}},
		{"escapes", ChatCompletionChunk{
			ID: "id", Object: "chat.completion.chunk", Created: 1700000000, Model: "qwen3",
			Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{Content: "quote \" and tab\t"}}},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Round-trip via hand-rolled encoder.
			encoded := appendChatCompletionChunk(nil, tc.in)
			var back ChatCompletionChunk
			if err := json.Unmarshal(encoded, &back); err != nil {
				t.Fatalf("json.Unmarshal(%s) error = %v", encoded, err)
			}
			// Compare load-bearing fields.
			if back.ID != tc.in.ID || back.Object != tc.in.Object || back.Created != tc.in.Created || back.Model != tc.in.Model {
				t.Fatalf("identity: got %+v, want %+v", back, tc.in)
			}
			if len(back.Choices) != len(tc.in.Choices) {
				t.Fatalf("choices len = %d, want %d", len(back.Choices), len(tc.in.Choices))
			}
			for i := range tc.in.Choices {
				if back.Choices[i].Index != tc.in.Choices[i].Index {
					t.Fatalf("choices[%d].index = %d, want %d", i, back.Choices[i].Index, tc.in.Choices[i].Index)
				}
				if back.Choices[i].Delta.Role != tc.in.Choices[i].Delta.Role || back.Choices[i].Delta.Content != tc.in.Choices[i].Delta.Content {
					t.Fatalf("choices[%d].delta = %+v, want %+v", i, back.Choices[i].Delta, tc.in.Choices[i].Delta)
				}
				gotFinish := back.Choices[i].FinishReason
				wantFinish := tc.in.Choices[i].FinishReason
				if (gotFinish == nil) != (wantFinish == nil) {
					t.Fatalf("choices[%d].finish_reason nil mismatch: got=%v want=%v", i, gotFinish, wantFinish)
				}
				if gotFinish != nil && *gotFinish != *wantFinish {
					t.Fatalf("choices[%d].finish_reason = %q, want %q", i, *gotFinish, *wantFinish)
				}
			}
			if (back.Thought == nil) != (tc.in.Thought == nil) {
				t.Fatalf("thought nil mismatch: got=%v want=%v", back.Thought, tc.in.Thought)
			}
			if back.Thought != nil && *back.Thought != *tc.in.Thought {
				t.Fatalf("thought = %q, want %q", *back.Thought, *tc.in.Thought)
			}
		})
	}
}

// TestChatCompletionChunk_SSEFrame verifies the SSE framing helper —
// the streaming hot path embeds "data: " prefix + body + "\n\n" in
// one buffer. Output must match what proxy clients parse as one SSE
// event (LL-formatted: line "data: <json>" terminated by blank line).
func TestChatCompletionChunk_SSEFrame(t *testing.T) {
	finish := "stop"
	chunk := ChatCompletionChunk{
		ID: "chatcmpl-test", Object: "chat.completion.chunk", Created: 1700000000, Model: "qwen3",
		Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{}, FinishReason: &finish}},
	}
	frame := appendChatCompletionChunkSSE(nil, chunk)
	frameStr := string(frame)
	if !strings.HasPrefix(frameStr, "data: ") {
		t.Fatalf("frame missing data: prefix: %q", frameStr)
	}
	if !strings.HasSuffix(frameStr, "\n\n") {
		t.Fatalf("frame missing trailing newlines: %q", frameStr)
	}
	body := strings.TrimSuffix(strings.TrimPrefix(frameStr, "data: "), "\n\n")
	var back ChatCompletionChunk
	if err := json.Unmarshal([]byte(body), &back); err != nil {
		t.Fatalf("frame body json.Unmarshal error: %v body=%q", err, body)
	}
	if back.ID != chunk.ID || back.Choices[0].FinishReason == nil || *back.Choices[0].FinishReason != "stop" {
		t.Fatalf("frame body decoded mismatch: %+v", back)
	}
}
