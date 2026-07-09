// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"testing"

	core "dappco.re/go"
)

// TestRenderOpenAITools pins that OpenAI function declarations convert to the
// Gemma 4 tool syntax via the shared renderer.
func TestRenderOpenAITools(t *testing.T) {
	tools := []Tool{{
		Type: "function",
		Function: ToolFunction{
			Name:        "get_weather",
			Description: "Get the weather",
			Parameters: ToolParameters{
				Type:       "object",
				Properties: map[string]ToolProperty{"city": {Type: "string", Description: "the city"}},
				Required:   []string{"city"},
			},
		},
	}}
	got := renderOpenAITools(tools)
	if !core.Contains(got, "<|tool>declaration:get_weather") || !core.Contains(got, "city") {
		t.Fatalf("renderOpenAITools = %q, want the gemma4 declaration for get_weather", got)
	}
	if renderOpenAITools(nil) != "" {
		t.Fatal("no tools should render empty")
	}
}

// TestOpenAIMessageContent_ToolRole pins that a role:"tool" message renders as a
// <|tool_response> span, while other roles pass their content through.
func TestOpenAIMessageContent_ToolRole(t *testing.T) {
	if got := openAIMessageContent(ChatMessage{Role: "tool", Content: "42"}); got != "<|tool_response>42<tool_response|>" {
		t.Fatalf("tool-role content = %q, want the <|tool_response> span", got)
	}
	if got := openAIMessageContent(ChatMessage{Role: "user", Content: "hi"}); got != "hi" {
		t.Fatalf("user content = %q, want hi", got)
	}
}

// TestOpenAIMessageContent_ToolCalls pins that a prior assistant turn's tool_calls
// re-render as their <|tool_call> spans, so a stateless client replaying full
// history keeps the call context a following tool result answers (#300).
func TestOpenAIMessageContent_ToolCalls(t *testing.T) {
	msg := ChatMessage{Role: "assistant", ToolCalls: []ToolCall{{
		ID: "call_1", Type: "function",
		Function: ToolCallFunction{Name: "get_weather", Arguments: `{"city":"Paris","days":5}`},
	}}}
	want := "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:5}<tool_call|>"
	if got := openAIMessageContent(msg); got != want {
		t.Fatalf("assistant tool_calls render = %q, want %q", got, want)
	}
}

// TestChatCompletionRequest_DecodesTools pins the hand-rolled decoder lifts the
// nested tools array.
func TestChatCompletionRequest_DecodesTools(t *testing.T) {
	data := []byte(`{"model":"x","messages":[],"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}}}]}`)
	var req ChatCompletionRequest
	if err := req.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if len(req.Tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(req.Tools))
	}
	fn := req.Tools[0].Function
	if fn.Name != "f" || fn.Parameters.Properties["a"].Type != "integer" || len(fn.Parameters.Required) != 1 {
		t.Fatalf("tool decode = %+v, want name f + integer param a required", fn)
	}
}

// TestChatMessageDelta_ToolCallsMarshal pins that a streaming delta carrying
// tool_calls marshals through the reflect path (index + id + function), while an
// empty delta stays the zero-alloc {}.
func TestChatMessageDelta_ToolCallsMarshal(t *testing.T) {
	d := ChatMessageDelta{ToolCalls: []ToolCallDelta{{
		Index: 0, ID: "call_1", Type: "function", Function: &ToolCallFunctionDelta{Name: "get_weather"},
	}}}
	b, err := d.MarshalJSON()
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	s := string(b)
	if !core.Contains(s, `"tool_calls"`) || !core.Contains(s, `"get_weather"`) || !core.Contains(s, `"index":0`) {
		t.Fatalf("tool_calls delta = %s, want the indexed call", s)
	}
	if got, _ := (ChatMessageDelta{}).MarshalJSON(); string(got) != "{}" {
		t.Fatalf("empty delta = %s, want {}", got)
	}
}

// TestChatResponseHasToolCalls pins the writeJSON routing predicate.
func TestChatResponseHasToolCalls(t *testing.T) {
	with := ChatCompletionResponse{Choices: []ChatChoice{{Message: ChatMessage{ToolCalls: []ToolCall{{ID: "x"}}}}}}
	if !chatResponseHasToolCalls(with) {
		t.Fatal("response with tool_calls should route to reflect")
	}
	if chatResponseHasToolCalls(ChatCompletionResponse{Choices: []ChatChoice{{Message: ChatMessage{Content: "hi"}}}}) {
		t.Fatal("plain text response should use the fast path")
	}
}
