// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import "testing"

// TestRenderBlock_ToolResult pins that a tool_result block renders as a
// <|tool_response> span carrying the tool output — the channel Gemma 4 reads
// after a call before answering.
func TestRenderBlock_ToolResult(t *testing.T) {
	got := renderBlock(ContentBlock{Type: "tool_result", ToolUseID: "t1", Text: "15C sunny"})
	if got != "<|tool_response>15C sunny<tool_response|>" {
		t.Fatalf("tool_result render = %q, want the <|tool_response> span", got)
	}
}

// TestRenderBlock_ToolUse_Good pins that a prior-turn tool_use block re-renders
// as its gemma4 <|tool_call> span (string args wrapped in the arg-quote marker,
// numbers bare), so a stateless client replaying full history keeps the call
// context a following tool_result answers (#300).
func TestRenderBlock_ToolUse_Good(t *testing.T) {
	blk := ContentBlock{Type: "tool_use", ID: "toolu_1", Name: "get_weather", Input: map[string]any{"city": "Paris", "days": float64(5)}}
	want := "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:5}<tool_call|>"
	if got := renderBlock(blk); got != want {
		t.Fatalf("tool_use render = %q, want %q", got, want)
	}
}

// TestRenderBlock_ToolUse_Bad pins that a tool_use with no arguments renders an
// empty arg body rather than dropping the call.
func TestRenderBlock_ToolUse_Bad(t *testing.T) {
	if got := renderBlock(ContentBlock{Type: "tool_use", Name: "now"}); got != "<|tool_call>call:now{}<tool_call|>" {
		t.Fatalf("empty-arg tool_use render = %q, want the {} call span", got)
	}
}

// TestRenderBlock_Text pins the plain-text passthrough is unchanged.
func TestRenderBlock_Text(t *testing.T) {
	if got := renderBlock(ContentBlock{Type: "text", Text: "hi"}); got != "hi" {
		t.Fatalf("text render = %q, want hi", got)
	}
}

// TestMessageRequest_DecodesToolResult pins the hand-rolled decoder lifts a
// tool_result block's tool_use_id + string content.
func TestMessageRequest_DecodesToolResult(t *testing.T) {
	data := []byte(`{"model":"x","max_tokens":10,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"sunny"}]}]}`)
	var req MessageRequest
	if err := req.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if len(req.Messages) != 1 || len(req.Messages[0].Content) != 1 {
		t.Fatalf("decoded shape wrong: %+v", req.Messages)
	}
	block := req.Messages[0].Content[0]
	if block.Type != "tool_result" || block.ToolUseID != "t1" || block.Text != "sunny" {
		t.Fatalf("tool_result decode = %+v, want type/tool_use_id/content = tool_result/t1/sunny", block)
	}
}

// TestMessageRequest_DecodesToolResultBlockContent pins that a tool_result whose
// content is an array of text blocks (Anthropic's other accepted shape) is
// flattened into the content text.
func TestMessageRequest_DecodesToolResultBlockContent(t *testing.T) {
	data := []byte(`{"model":"x","max_tokens":10,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"cloudy"}]}]}]}`)
	var req MessageRequest
	if err := req.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if got := req.Messages[0].Content[0].Text; got != "cloudy" {
		t.Fatalf("array tool_result content = %q, want cloudy", got)
	}
}

// TestMessageRequest_DecodesToolUse pins the decoder now lifts a prior-turn
// tool_use block's name + input (previously skipped), so a replayed history can
// re-render the call for a stateless client (#300).
func TestMessageRequest_DecodesToolUse(t *testing.T) {
	data := []byte(`{"model":"x","max_tokens":10,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris","days":5}}]}]}`)
	var req MessageRequest
	if err := req.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	block := req.Messages[0].Content[0]
	if block.Type != "tool_use" || block.Name != "get_weather" {
		t.Fatalf("tool_use decode = type/name %q/%q, want tool_use/get_weather", block.Type, block.Name)
	}
	if block.Input["city"] != "Paris" || block.Input["days"] != float64(5) {
		t.Fatalf("tool_use input = %+v, want city=Paris days=5", block.Input)
	}
}

// TestInferenceMessages_ToolResultRendersResponseSpan pins the end-to-end render:
// a user tool_result turn becomes a <|tool_response> span in the inference
// message the engine prefills.
func TestInferenceMessages_ToolResultRendersResponseSpan(t *testing.T) {
	req := MessageRequest{Messages: []Message{
		{Role: "user", Content: []ContentBlock{{Type: "tool_result", ToolUseID: "t1", Text: "42"}}},
	}}
	msgs := InferenceMessages(req)
	if len(msgs) != 1 {
		t.Fatalf("messages = %d, want 1", len(msgs))
	}
	if msgs[0].Content != "<|tool_response>42<tool_response|>" {
		t.Fatalf("tool_result message content = %q, want the <|tool_response> span", msgs[0].Content)
	}
}
