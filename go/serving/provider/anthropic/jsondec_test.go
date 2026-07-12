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
	minP := float32(0.05)
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
			in:   `{"model":"claude-3","system":"Be concise.","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":1024,"temperature":0.7,"top_p":0.95,"min_p":0.05,"top_k":64,"stream":true,"stop_sequences":["</s>","<|eot|>"]}`,
			want: MessageRequest{
				Model:         "claude-3",
				System:        "Be concise.",
				Messages:      []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
				MaxTokens:     1024,
				Temperature:   &temp,
				TopP:          &topP,
				MinP:          &minP,
				TopK:          &topK,
				Stream:        true,
				StopSequences: []string{"</s>", "<|eot|>"},
			},
		},
		{
			name: "pointer-fields-null-keeps-zero-value",
			in:   `{"model":"claude-3","messages":[],"max_tokens":256,"temperature":null,"top_p":null,"min_p":null,"top_k":null,"stream":null}`,
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

// TestJsondec_MessageRequest_UnmarshalJSON_Good pins a direct method call
// (rather than via encoding/json.Unmarshal) against a typical request.
func TestJsondec_MessageRequest_UnmarshalJSON_Good(t *testing.T) {
	var req MessageRequest
	data := []byte(`{"model":"gemma-4","system":"Be concise.","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":128,"stream":true}`)
	if err := req.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if req.Model != "gemma-4" || req.System != "Be concise." || req.MaxTokens != 128 || !req.Stream {
		t.Fatalf("req = %+v", req)
	}
	if len(req.Messages) != 1 || req.Messages[0].Content[0].Text != "hi" {
		t.Fatalf("req.Messages = %+v", req.Messages)
	}
}

// TestJsondec_MessageRequest_UnmarshalJSON_Bad pins a non-numeric top_k
// value rejected cleanly (distinct decode branch from the invalid-shapes
// table above).
func TestJsondec_MessageRequest_UnmarshalJSON_Bad(t *testing.T) {
	var req MessageRequest
	err := req.UnmarshalJSON([]byte(`{"model":"x","max_tokens":1,"messages":[],"top_k":"nope"}`))
	if err == nil {
		t.Fatalf("UnmarshalJSON with a non-numeric top_k returned nil error")
	}
}

// TestJsondec_MessageRequest_UnmarshalJSON_Ugly pins the "tools" field's
// reflect-fallback decode path — the one branch of MessageRequest.UnmarshalJSON
// that doesn't hand-roll its own walker.
func TestJsondec_MessageRequest_UnmarshalJSON_Ugly(t *testing.T) {
	var req MessageRequest
	data := []byte(`{"model":"x","max_tokens":5,"messages":[],"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}`)
	if err := req.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "get_weather" {
		t.Fatalf("req.Tools = %+v, want one get_weather tool (reflect-decoded nested schema)", req.Tools)
	}
	if req.Tools[0].InputSchema.Type != "object" || len(req.Tools[0].InputSchema.Required) != 1 {
		t.Fatalf("req.Tools[0].InputSchema = %+v", req.Tools[0].InputSchema)
	}
}

// TestJsondec_MessageRequest_UnmarshalJSON_ToolChoice pins tool_choice's
// reflect-fallback decode path (the same cold-path shape as "tools" above),
// and that the field omitted entirely leaves ToolChoice nil rather than a
// zero-value struct (InferenceMessages' nil check relies on that distinction).
func TestJsondec_MessageRequest_UnmarshalJSON_ToolChoice(t *testing.T) {
	var req MessageRequest
	data := []byte(`{"model":"x","max_tokens":5,"messages":[],"tool_choice":{"type":"tool","name":"get_weather"}}`)
	if err := req.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if req.ToolChoice == nil || req.ToolChoice.Type != "tool" || req.ToolChoice.Name != "get_weather" {
		t.Fatalf("req.ToolChoice = %+v, want type=tool name=get_weather", req.ToolChoice)
	}

	var reqOmitted MessageRequest
	if err := reqOmitted.UnmarshalJSON([]byte(`{"model":"x","max_tokens":5,"messages":[]}`)); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if reqOmitted.ToolChoice != nil {
		t.Fatalf("req.ToolChoice with the field omitted = %+v, want nil", reqOmitted.ToolChoice)
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

// TestJsondec_MessageResponse_UnmarshalJSON_Good pins a direct method call
// against a typical response.
func TestJsondec_MessageResponse_UnmarshalJSON_Good(t *testing.T) {
	var resp MessageResponse
	data := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"gemma-4","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}`)
	if err := resp.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if resp.ID != "msg_1" || resp.StopReason != "end_turn" || resp.Usage.InputTokens != 3 {
		t.Fatalf("resp = %+v", resp)
	}
}

// TestJsondec_MessageResponse_UnmarshalJSON_Bad pins a non-object usage
// value rejected cleanly.
func TestJsondec_MessageResponse_UnmarshalJSON_Bad(t *testing.T) {
	var resp MessageResponse
	err := resp.UnmarshalJSON([]byte(`{"id":"msg_1","usage":"not-an-object"}`))
	if err == nil {
		t.Fatalf("UnmarshalJSON with a non-object usage returned nil error")
	}
}

// TestJsondec_MessageResponse_UnmarshalJSON_Ugly pins a content array
// holding a tool_use block (nested input decode) alongside an explicit
// null stop_sequence.
func TestJsondec_MessageResponse_UnmarshalJSON_Ugly(t *testing.T) {
	var resp MessageResponse
	data := []byte(`{"id":"msg_2","type":"message","role":"assistant","model":"gemma-4","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}],"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`)
	if err := resp.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if resp.StopSequence != "" {
		t.Fatalf("resp.StopSequence = %q, want empty for explicit null", resp.StopSequence)
	}
	block := resp.Content[0]
	if block.Type != "tool_use" || block.Name != "get_weather" || block.Input["city"] != "Paris" {
		t.Fatalf("resp.Content[0] = %+v", block)
	}
}
