// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import (
	"encoding/json"
	"reflect"
	"testing"

	"dappco.re/go/inference"
)

// TestAppendMessageRequest_RoundTrip pins the hand-rolled MessageRequest
// encoder against encoding/json across every wire shape. Proxies and
// SDK clients that consume this body feed it back into the same Go
// type, so the round-trip must be exact.
func TestAppendMessageRequest_RoundTrip(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.95)
	topK := 64
	cases := []struct {
		name string
		req  MessageRequest
	}{
		{
			name: "Minimal_RequiredFieldsOnly",
			req: MessageRequest{
				Model:     "claude-3-5-sonnet",
				Messages:  []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
				MaxTokens: 256,
			},
		},
		{
			name: "AllOptionalFieldsSet",
			req: MessageRequest{
				Model:         "claude-3-5-sonnet",
				System:        "Be concise.",
				Messages:      []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
				MaxTokens:     1024,
				Temperature:   &temp,
				TopP:          &topP,
				TopK:          &topK,
				Stream:        true,
				StopSequences: []string{"</response>", "<|eot_id|>"},
			},
		},
		{
			name: "MultiTurn_MixedRoles",
			req: MessageRequest{
				Model: "claude-3-5-sonnet",
				Messages: []Message{
					{Role: "user", Content: []ContentBlock{{Type: "text", Text: "first"}}},
					{Role: "assistant", Content: []ContentBlock{{Type: "text", Text: "second"}}},
					{Role: "user", Content: []ContentBlock{{Type: "text", Text: "third"}}},
				},
				MaxTokens: 256,
			},
		},
		{
			name: "EscapeHeavy_System",
			req: MessageRequest{
				Model:     "claude-3-5-sonnet",
				System:    "Reply with \"quotes\" and\nnewlines\tand\x01control",
				Messages:  []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "back\\slash"}}}},
				MaxTokens: 256,
			},
		},
		{
			name: "EmptyStopSequences_OmittedNotEmptyArray",
			req: MessageRequest{
				Model:         "claude-3-5-sonnet",
				Messages:      []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
				MaxTokens:     256,
				StopSequences: []string{},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hand := AppendMessageRequest(make([]byte, 0, MessageRequestSize(tc.req)), tc.req)

			var got MessageRequest
			if err := json.Unmarshal(hand, &got); err != nil {
				t.Fatalf("json.Unmarshal hand-rolled output failed: %v\nbody: %s", err, hand)
			}
			ref, err := json.Marshal(tc.req)
			if err != nil {
				t.Fatalf("json.Marshal reference: %v", err)
			}
			var want MessageRequest
			if err := json.Unmarshal(ref, &want); err != nil {
				t.Fatalf("json.Unmarshal stdlib output failed: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("round-trip mismatch\ngot:  %+v\nwant: %+v\nhand: %s\nref:  %s", got, want, hand, ref)
			}
		})
	}
}

// TestAppendMessageRequest_SizeBoundsFits guards the request-side size
// estimator. Under-sizing forces append to grow the buffer, costing
// the alloc win we built the helper to claim.
func TestAppendMessageRequest_SizeBoundsFits(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.95)
	topK := 64
	cases := []struct {
		name string
		req  MessageRequest
	}{
		{"Minimal", MessageRequest{
			Model:     "claude-3-5-sonnet",
			Messages:  []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
			MaxTokens: 256,
		}},
		{"FullyPopulated", MessageRequest{
			Model:         "claude-3-5-sonnet",
			System:        "Be concise.",
			Messages:      []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "the question"}}}},
			MaxTokens:     1024,
			Temperature:   &temp,
			TopP:          &topP,
			TopK:          &topK,
			Stream:        true,
			StopSequences: []string{"</response>", "<|eot_id|>", "STOP"},
		}},
		{"FiveTurnMultiBlock", MessageRequest{
			Model: "claude-3-5-sonnet",
			Messages: []Message{
				{Role: "user", Content: []ContentBlock{{Type: "text", Text: "one"}, {Type: "text", Text: "two"}}},
				{Role: "assistant", Content: []ContentBlock{{Type: "text", Text: "three"}}},
				{Role: "user", Content: []ContentBlock{{Type: "text", Text: "four"}}},
				{Role: "assistant", Content: []ContentBlock{{Type: "text", Text: "five"}}},
				{Role: "user", Content: []ContentBlock{{Type: "text", Text: "six"}}},
			},
			MaxTokens: 256,
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			predicted := MessageRequestSize(tc.req)
			actual := len(AppendMessageRequest(nil, tc.req))
			if predicted < actual {
				t.Fatalf("MessageRequestSize=%d < actual encoded %d — under-sizing forces realloc", predicted, actual)
			}
		})
	}
}

// TestAppendMessageResponse_SizeBoundsFits checks the size estimator
// returns >= the actual encoded size across the round-trip cases.
// Pre-sizing is load-bearing — under-sizing forces append to grow
// the slice, which costs one more allocation per call.
func TestAppendMessageResponse_SizeBoundsFits(t *testing.T) {
	cases := []struct {
		name string
		resp MessageResponse
	}{
		{"Typical_NewTextResponse", NewTextResponse(
			"msg_bench",
			"claude-3-5-sonnet",
			"The summary is concise and faithful to the original text.",
			inference.GenerateMetrics{PromptTokens: 320, GeneratedTokens: 48},
		)},
		{"WithStopReasonAndSequence", MessageResponse{
			ID:           "msg_x",
			Type:         "message",
			Role:         "assistant",
			Model:        "claude-3-5-sonnet",
			Content:      []ContentBlock{{Type: "text", Text: "stopped early"}},
			StopReason:   "stop_sequence",
			StopSequence: "</response>",
			Usage:        Usage{InputTokens: 5, OutputTokens: 1},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			predicted := MessageResponseSize(tc.resp)
			actual := len(AppendMessageResponse(nil, tc.resp))
			if predicted < actual {
				t.Fatalf("MessageResponseSize=%d < actual encoded %d — under-sizing forces realloc", predicted, actual)
			}
		})
	}
}

// TestAppendMessageResponse_RoundTrip pins the hand-rolled
// MessageResponse encoder against encoding/json across every wire
// shape — the proxy / SDK clients that read this body feed it back
// into the same Go type, so the round-trip must be exact.
func TestAppendMessageResponse_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		resp MessageResponse
	}{
		{
			name: "Typical_SingleTextBlock",
			resp: MessageResponse{
				ID:      "msg_1",
				Type:    "message",
				Role:    "assistant",
				Model:   "claude-3-5-sonnet",
				Content: []ContentBlock{{Type: "text", Text: "hello"}},
				Usage:   Usage{InputTokens: 5, OutputTokens: 1},
			},
		},
		{
			name: "WithStopReason_AndStopSequence",
			resp: MessageResponse{
				ID:           "msg_2",
				Type:         "message",
				Role:         "assistant",
				Model:        "claude-3-5-sonnet",
				Content:      []ContentBlock{{Type: "text", Text: "stopped"}},
				StopReason:   "stop_sequence",
				StopSequence: "</response>",
				Usage:        Usage{InputTokens: 7, OutputTokens: 2},
			},
		},
		{
			name: "EmptyContent",
			resp: MessageResponse{
				ID:      "msg_3",
				Type:    "message",
				Role:    "assistant",
				Model:   "claude-3-5-sonnet",
				Content: []ContentBlock{},
				Usage:   Usage{InputTokens: 0, OutputTokens: 0},
			},
		},
		{
			name: "MultiBlock_MixedText",
			resp: MessageResponse{
				ID:    "msg_4",
				Type:  "message",
				Role:  "assistant",
				Model: "claude-3-5-sonnet",
				Content: []ContentBlock{
					{Type: "text", Text: "first"},
					{Type: "text", Text: "second"},
					{Type: "tool_use", Text: ""}, // text omitted when empty
				},
				Usage: Usage{InputTokens: 10, OutputTokens: 3},
			},
		},
		{
			name: "EscapeHeavy",
			resp: MessageResponse{
				ID:      `msg "5"`,
				Type:    "message",
				Role:    "assistant",
				Model:   "claude-3-5-sonnet",
				Content: []ContentBlock{{Type: "text", Text: "line1\nline2\twith\"quotes\\and\rcontrol\x01char"}},
				Usage:   Usage{InputTokens: 8, OutputTokens: 5},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hand := AppendMessageResponse(make([]byte, 0, MessageResponseSize(tc.resp)), tc.resp)

			var got MessageResponse
			if err := json.Unmarshal(hand, &got); err != nil {
				t.Fatalf("json.Unmarshal hand-rolled output failed: %v\nbody: %s", err, hand)
			}
			// Normalise: empty Content slice unmarshals into nil for
			// some shapes; compare via re-marshal-and-decode to a
			// reference produced by the stdlib encoder.
			ref, err := json.Marshal(tc.resp)
			if err != nil {
				t.Fatalf("json.Marshal reference: %v", err)
			}
			var want MessageResponse
			if err := json.Unmarshal(ref, &want); err != nil {
				t.Fatalf("json.Unmarshal stdlib output failed: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("round-trip mismatch\ngot:  %+v\nwant: %+v\nhand: %s\nref:  %s", got, want, hand, ref)
			}
		})
	}
}
