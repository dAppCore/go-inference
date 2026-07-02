// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"dappco.re/go/inference"
)

// TestResponse_AppendRoundTrip locks the hand-rolled Responses-API
// non-streaming encoder to encoding/json's deserialiser.
func TestResponse_AppendRoundTrip(t *testing.T) {
	thought := "let me think"
	cases := []struct {
		name string
		in   Response
	}{
		{"minimal", NewTextResponse("resp_x", "qwen3", "Hi", inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 4})},
		{"with-thought", func() Response {
			r := NewTextResponse("resp_y", "qwen3", "Answer", inference.GenerateMetrics{PromptTokens: 10, GeneratedTokens: 20})
			r.Thought = &thought
			return r
		}()},
		{"with-id-on-msg", Response{
			ID: "resp_z", Object: "response", Created: 1700000000, Model: "qwen3",
			Output: []ResponseOutputMessage{{
				ID: "msg_1", Type: "message", Role: "assistant",
				Content: []ResponseOutputText{{Type: "output_text", Text: "text"}},
			}},
			Usage: ResponseUsage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3},
		}},
		{"escapes", Response{
			ID: "resp_e", Object: "response", Created: 1700000000, Model: "qwen3",
			Output: []ResponseOutputMessage{{
				Type: "message", Role: "assistant",
				Content: []ResponseOutputText{{Type: "output_text", Text: "quote \" tab\t"}},
			}},
			Usage: ResponseUsage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2},
		}},
		{"multiple-output-and-content", Response{
			// 2+ Output messages exercises appendResponse's i>0
			// comma-separator branch; 2+ Content items per message
			// exercises appendResponseOutputMessage's own i>0 branch —
			// every other case above carries exactly one of each.
			ID: "resp_m", Object: "response", Created: 1700000000, Model: "qwen3",
			Output: []ResponseOutputMessage{
				{
					Type: "message", Role: "assistant",
					Content: []ResponseOutputText{
						{Type: "output_text", Text: "part one"},
						{Type: "output_text", Text: "part two"},
					},
				},
				{
					ID: "msg_2", Type: "message", Role: "assistant",
					Content: []ResponseOutputText{{Type: "output_text", Text: "second message"}},
				},
			},
			Usage: ResponseUsage{InputTokens: 5, OutputTokens: 6, TotalTokens: 11},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := appendResponse(nil, tc.in)
			var back Response
			if err := json.Unmarshal(encoded, &back); err != nil {
				t.Fatalf("json.Unmarshal(%s) error = %v", encoded, err)
			}
			if back.ID != tc.in.ID || back.Object != tc.in.Object || back.Created != tc.in.Created || back.Model != tc.in.Model {
				t.Fatalf("identity: got %+v, want %+v", back, tc.in)
			}
			if back.Usage != tc.in.Usage {
				t.Fatalf("usage: got %+v, want %+v", back.Usage, tc.in.Usage)
			}
			if len(back.Output) != len(tc.in.Output) {
				t.Fatalf("output len = %d, want %d", len(back.Output), len(tc.in.Output))
			}
			for i := range tc.in.Output {
				if back.Output[i].ID != tc.in.Output[i].ID ||
					back.Output[i].Type != tc.in.Output[i].Type ||
					back.Output[i].Role != tc.in.Output[i].Role {
					t.Fatalf("output[%d] header: got %+v want %+v", i, back.Output[i], tc.in.Output[i])
				}
				if len(back.Output[i].Content) != len(tc.in.Output[i].Content) {
					t.Fatalf("output[%d].content len = %d, want %d", i, len(back.Output[i].Content), len(tc.in.Output[i].Content))
				}
				for j := range tc.in.Output[i].Content {
					if back.Output[i].Content[j] != tc.in.Output[i].Content[j] {
						t.Fatalf("output[%d].content[%d] = %+v, want %+v", i, j, back.Output[i].Content[j], tc.in.Output[i].Content[j])
					}
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

// TestResponseStreamEvent_AppendRoundTrip locks the hand-rolled
// stream-event encoder. Fires per delta on the streaming path; the
// "response.completed" event embeds a full Response payload.
func TestResponseStreamEvent_AppendRoundTrip(t *testing.T) {
	thought := "let me think"
	resp := NewTextResponse("resp_x", "qwen3", "Hi", inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 4})
	cases := []struct {
		name string
		in   ResponseStreamEvent
	}{
		{"delta-only", ResponseStreamEvent{Type: "response.output_text.delta", Delta: "Answer"}},
		{"thought-delta", ResponseStreamEvent{Type: "response.thought.delta", Delta: "thinking", Thought: &thought}},
		{"completed", ResponseStreamEvent{Type: "response.completed", Response: &resp}},
		{"type-only", ResponseStreamEvent{Type: "response.started"}},
		{"delta-with-escapes", ResponseStreamEvent{Type: "response.output_text.delta", Delta: "quote \" tab\t"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := appendResponseStreamEvent(nil, tc.in)
			var back ResponseStreamEvent
			if err := json.Unmarshal(encoded, &back); err != nil {
				t.Fatalf("json.Unmarshal(%s) error = %v", encoded, err)
			}
			if back.Type != tc.in.Type {
				t.Fatalf("type: got %q, want %q", back.Type, tc.in.Type)
			}
			if back.Delta != tc.in.Delta {
				t.Fatalf("delta: got %q, want %q", back.Delta, tc.in.Delta)
			}
			if (back.Response == nil) != (tc.in.Response == nil) {
				t.Fatalf("response nil mismatch")
			}
			if back.Response != nil && back.Response.ID != tc.in.Response.ID {
				t.Fatalf("response.id: got %q, want %q", back.Response.ID, tc.in.Response.ID)
			}
			if (back.Thought == nil) != (tc.in.Thought == nil) {
				t.Fatalf("thought nil mismatch")
			}
			if back.Thought != nil && *back.Thought != *tc.in.Thought {
				t.Fatalf("thought: got %q, want %q", *back.Thought, *tc.in.Thought)
			}
		})
	}
}

// TestResponseSize_Good pins responseSize's single-allocation
// contract for a fully-populated Response (multi-message output,
// multi-part content, a thought).
func TestResponseSize_Good(t *testing.T) {
	thought := "let me think"
	resp := Response{
		ID: "resp_x", Object: "response", Created: 1700000000, Model: "qwen3",
		Output: []ResponseOutputMessage{
			{ID: "msg_1", Type: "message", Role: "assistant", Content: []ResponseOutputText{
				{Type: "output_text", Text: "part one"},
				{Type: "output_text", Text: "part two"},
			}},
			{Type: "message", Role: "assistant", Content: []ResponseOutputText{{Type: "output_text", Text: "second"}}},
		},
		Usage:   ResponseUsage{InputTokens: 5, OutputTokens: 6, TotalTokens: 11},
		Thought: &thought,
	}
	buf := make([]byte, 0, responseSize(resp))
	got := appendResponse(buf, resp)
	if len(got) > cap(buf) {
		t.Fatalf("responseSize(%+v) = %d, actual encoded length %d exceeds capacity", resp, responseSize(resp), len(got))
	}
}

// TestResponseSize_Bad covers the empty-output floor (no messages, no
// thought).
func TestResponseSize_Bad(t *testing.T) {
	resp := Response{ID: "resp_x", Object: "response", Created: 1700000000, Model: "qwen3"}
	buf := make([]byte, 0, responseSize(resp))
	got := appendResponse(buf, resp)
	if len(got) > cap(buf) {
		t.Fatalf("responseSize(empty) = %d, actual encoded length %d exceeds capacity", responseSize(resp), len(got))
	}
}

// TestResponseSize_Ugly drives escape-heavy content through the
// estimator — the estimate counts raw len(), not escaped length, so
// this pins that append still produces correct, round-trippable
// output even when the estimate under-counts.
func TestResponseSize_Ugly(t *testing.T) {
	resp := NewTextResponse("resp_x", "qwen3", strings.Repeat("\"quoted\"\n\t", 64), inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 1})
	buf := appendResponse(make([]byte, 0, responseSize(resp)), resp)
	var back Response
	if err := json.Unmarshal(buf, &back); err != nil {
		t.Fatalf("Unmarshal(%s): %v", buf, err)
	}
	if back.Output[0].Content[0].Text != resp.Output[0].Content[0].Text {
		t.Fatalf("round-trip mismatch: got %q, want %q", back.Output[0].Content[0].Text, resp.Output[0].Content[0].Text)
	}
}

// TestResponseStreamEventSize_Good pins the estimator's single-
// allocation contract for the "response.completed" shape — the
// Response-embedding case the doc comment calls load-bearing.
func TestResponseStreamEventSize_Good(t *testing.T) {
	thought := "thinking"
	resp := NewTextResponse("resp_x", "qwen3", "answer", inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 4})
	event := ResponseStreamEvent{Type: "response.completed", Response: &resp, Delta: "answer", Thought: &thought}
	buf := make([]byte, 0, responseStreamEventSize(event))
	got := appendResponseStreamEvent(buf, event)
	if len(got) > cap(buf) {
		t.Fatalf("responseStreamEventSize(%+v) = %d, actual encoded length %d exceeds capacity", event, responseStreamEventSize(event), len(got))
	}
}

// TestResponseStreamEventSize_Bad covers the type-only floor (no
// response, delta, or thought).
func TestResponseStreamEventSize_Bad(t *testing.T) {
	event := ResponseStreamEvent{Type: "response.started"}
	buf := make([]byte, 0, responseStreamEventSize(event))
	got := appendResponseStreamEvent(buf, event)
	if len(got) > cap(buf) {
		t.Fatalf("responseStreamEventSize(type-only) = %d, actual encoded length %d exceeds capacity", responseStreamEventSize(event), len(got))
	}
}

// TestResponseStreamEventSize_Ugly drives an escape-heavy delta
// through the estimator, mirroring TestResponseSize_Ugly's
// round-trip-correctness-over-exact-size-match check.
func TestResponseStreamEventSize_Ugly(t *testing.T) {
	event := ResponseStreamEvent{Type: "response.output_text.delta", Delta: strings.Repeat("\"quoted\"\n\t", 64)}
	buf := appendResponseStreamEvent(make([]byte, 0, responseStreamEventSize(event)), event)
	var back ResponseStreamEvent
	if err := json.Unmarshal(buf, &back); err != nil {
		t.Fatalf("Unmarshal(%s): %v", buf, err)
	}
	if back.Delta != event.Delta {
		t.Fatalf("round-trip mismatch: got %q, want %q", back.Delta, event.Delta)
	}
}
