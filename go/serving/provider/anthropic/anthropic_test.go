// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestAnthropic_InferenceMessages_Good(t *testing.T) {
	req := MessageRequest{
		System: "system",
		Messages: []Message{{
			Role:    "user",
			Content: []ContentBlock{{Type: "text", Text: "hello"}},
		}},
	}

	messages := InferenceMessages(req)

	if len(messages) != 2 {
		t.Fatalf("len(messages) = %d, want 2", len(messages))
	}
	if messages[0].Role != "system" || messages[1].Content != "hello" {
		t.Fatalf("messages = %+v", messages)
	}
}

// TestAnthropic_InferenceMessages_Bad pins the edge case of a request with no
// system prompt, no tools, and no messages — the caller-facing empty case
// returns an empty slice rather than panicking or fabricating a turn.
func TestAnthropic_InferenceMessages_Bad(t *testing.T) {
	messages := InferenceMessages(MessageRequest{})
	if len(messages) != 0 {
		t.Fatalf("messages = %+v, want empty for a request with no system/messages/tools", messages)
	}
}

// TestAnthropic_InferenceMessages_Ugly pins that tools without an explicit
// system prompt still produce a system turn — built purely from the
// rendered tool declaration (RenderToolDeclarations).
func TestAnthropic_InferenceMessages_Ugly(t *testing.T) {
	req := MessageRequest{
		Tools:    []Tool{{Name: "get_weather", InputSchema: ToolInputSchema{Type: "object"}}},
		Messages: []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
	}
	messages := InferenceMessages(req)
	if len(messages) != 2 {
		t.Fatalf("messages = %+v, want system+user", messages)
	}
	if messages[0].Role != "system" || !core.Contains(messages[0].Content, "get_weather") {
		t.Fatalf("system turn = %+v, want the rendered tool declaration", messages[0])
	}
}

func TestAnthropic_GenerateOptions_Good(t *testing.T) {
	temp := float32(0.2)
	minP := float32(0.04)
	topK := 4
	opts := GenerateOptions(MessageRequest{MaxTokens: 9, Temperature: &temp, MinP: &minP, TopK: &topK})

	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.MaxTokens != 9 || cfg.Temperature != 0.2 || cfg.MinP != 0.04 || cfg.TopK != 4 {
		t.Fatalf("cfg = %+v", cfg)
	}
}

// TestAnthropic_GenerateOptions_Bad pins that a request with no sampling
// fields set at all (edge: MaxTokens<=0, every pointer nil) yields no
// options rather than a slice of zero-value entries.
func TestAnthropic_GenerateOptions_Bad(t *testing.T) {
	opts := GenerateOptions(MessageRequest{})
	if len(opts) != 0 {
		t.Fatalf("opts = %d entries, want 0 for a request with no sampling fields set", len(opts))
	}
}

// TestAnthropic_GenerateOptions_Ugly pins the edge combination of only one
// pointer field set (TopP) with MaxTokens left at its zero value — the
// unset fields must not leak a phantom option.
func TestAnthropic_GenerateOptions_Ugly(t *testing.T) {
	topP := float32(0.42)
	opts := GenerateOptions(MessageRequest{TopP: &topP})
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.TopP != 0.42 {
		t.Fatalf("cfg.TopP = %v, want 0.42", cfg.TopP)
	}
	if cfg.MaxTokens != 0 {
		t.Fatalf("cfg.MaxTokens = %d, want the default (0) since MaxTokens was never set", cfg.MaxTokens)
	}
}

func TestAnthropic_NewTextResponse_Good(t *testing.T) {
	resp := NewTextResponse("msg_1", "claude-ish", "ok", inference.GenerateMetrics{PromptTokens: 2, GeneratedTokens: 3})

	if resp.ID != "msg_1" || resp.Type != "message" || resp.Role != "assistant" {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.Content[0].Text != "ok" || resp.Usage.OutputTokens != 3 {
		t.Fatalf("resp = %+v", resp)
	}
}

// TestAnthropic_NewTextResponse_Bad pins the edge case of empty generated
// text — still a well-formed text block, not a dropped/nil content array.
func TestAnthropic_NewTextResponse_Bad(t *testing.T) {
	resp := NewTextResponse("msg_1", "gemma-4", "", inference.GenerateMetrics{})
	if resp.Content[0].Text != "" || resp.Content[0].Type != "text" {
		t.Fatalf("resp.Content = %+v, want a text block with empty text", resp.Content)
	}
	if resp.Usage.InputTokens != 0 || resp.Usage.OutputTokens != 0 {
		t.Fatalf("resp.Usage = %+v, want zero", resp.Usage)
	}
}

// TestAnthropic_NewTextResponse_Ugly pins that escape-heavy/unicode text
// passes through unmodified — NewTextResponse does no sanitisation, that's
// the encoder's job.
func TestAnthropic_NewTextResponse_Ugly(t *testing.T) {
	text := "line1\nline2 \"quoted\" \\back 日本語"
	resp := NewTextResponse("msg_2", "gemma-4", text, inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 1})
	if resp.Content[0].Text != text {
		t.Fatalf("resp.Content[0].Text = %q, want unmodified %q", resp.Content[0].Text, text)
	}
}

// TestAnthropic_ToolUseBlock_Good pins a normal tool_use block build with
// valid JSON arguments decoded into Input.
func TestAnthropic_ToolUseBlock_Good(t *testing.T) {
	block := ToolUseBlock("toolu_1", "get_weather", `{"city":"Paris","days":5}`)
	if block.Type != "tool_use" || block.ID != "toolu_1" || block.Name != "get_weather" {
		t.Fatalf("block = %+v", block)
	}
	if block.Input["city"] != "Paris" || block.Input["days"] != float64(5) {
		t.Fatalf("block.Input = %+v", block.Input)
	}
}

// TestAnthropic_ToolUseBlock_Bad pins that malformed JSON arguments fall
// back to an empty object rather than dropping the block.
func TestAnthropic_ToolUseBlock_Bad(t *testing.T) {
	block := ToolUseBlock("toolu_2", "now", `{not json`)
	if block.Type != "tool_use" || block.Name != "now" {
		t.Fatalf("block = %+v", block)
	}
	if block.Input == nil || len(block.Input) != 0 {
		t.Fatalf("malformed args should fall back to an empty (non-nil) object, got %+v", block.Input)
	}
}

// TestAnthropic_ToolUseBlock_Ugly pins the other empty-object path — an
// empty arguments string skips the unmarshal attempt entirely.
func TestAnthropic_ToolUseBlock_Ugly(t *testing.T) {
	block := ToolUseBlock("toolu_3", "now", "")
	if block.Input == nil || len(block.Input) != 0 {
		t.Fatalf("empty arguments string should yield an empty object, got %+v", block.Input)
	}
}

// TestAnthropic_NewToolUseResponse_Good pins the non-streaming tool-call
// response shape.
func TestAnthropic_NewToolUseResponse_Good(t *testing.T) {
	blocks := []ContentBlock{ToolUseBlock("toolu_1", "get_weather", `{"city":"Paris"}`)}
	resp := NewToolUseResponse("msg_1", "gemma-4", blocks, inference.GenerateMetrics{PromptTokens: 10, GeneratedTokens: 4})
	if resp.ID != "msg_1" || resp.Type != "message" || resp.Role != "assistant" || resp.Model != "gemma-4" {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.StopReason != "tool_use" || len(resp.Content) != 1 {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 4 {
		t.Fatalf("resp.Usage = %+v", resp.Usage)
	}
}

// TestAnthropic_NewToolUseResponse_Bad pins the edge case of no blocks at
// all — stop_reason stays "tool_use" even though there is nothing to call.
func TestAnthropic_NewToolUseResponse_Bad(t *testing.T) {
	resp := NewToolUseResponse("msg_2", "gemma-4", nil, inference.GenerateMetrics{})
	if resp.StopReason != "tool_use" {
		t.Fatalf("resp.StopReason = %q, want tool_use even with no blocks", resp.StopReason)
	}
	if len(resp.Content) != 0 {
		t.Fatalf("resp.Content = %+v, want empty", resp.Content)
	}
}

// TestAnthropic_NewToolUseResponse_Ugly pins the multi-call edge — a single
// turn ending in more than one tool_use block.
func TestAnthropic_NewToolUseResponse_Ugly(t *testing.T) {
	blocks := []ContentBlock{
		ToolUseBlock("toolu_1", "get_weather", `{"city":"Paris"}`),
		ToolUseBlock("toolu_2", "get_time", `{"tz":"UTC"}`),
	}
	resp := NewToolUseResponse("msg_3", "gemma-4", blocks, inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 1})
	if len(resp.Content) != 2 {
		t.Fatalf("resp.Content = %+v, want 2 tool_use blocks", resp.Content)
	}
	if resp.Content[0].Name != "get_weather" || resp.Content[1].Name != "get_time" {
		t.Fatalf("resp.Content = %+v", resp.Content)
	}
}

// TestAnthropic_AppendMessageResponse_Good pins the hand-rolled encoder
// against a literal single-text-block response.
func TestAnthropic_AppendMessageResponse_Good(t *testing.T) {
	resp := MessageResponse{
		ID: "msg_1", Type: "message", Role: "assistant", Model: "gemma-4",
		Content: []ContentBlock{{Type: "text", Text: "hi"}},
		Usage:   Usage{InputTokens: 3, OutputTokens: 1},
	}
	core.AssertEqual(t,
		`{"id":"msg_1","type":"message","role":"assistant","model":"gemma-4","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":3,"output_tokens":1}}`,
		string(AppendMessageResponse(nil, resp)))
}

// TestAnthropic_AppendMessageResponse_Bad pins the edge case of an empty
// Content array combined with both optional stop fields set.
func TestAnthropic_AppendMessageResponse_Bad(t *testing.T) {
	resp := MessageResponse{
		ID: "msg_2", Type: "message", Role: "assistant", Model: "gemma-4",
		StopReason:   "stop_sequence",
		StopSequence: "</s>",
		Usage:        Usage{InputTokens: 1, OutputTokens: 0},
	}
	core.AssertEqual(t,
		`{"id":"msg_2","type":"message","role":"assistant","model":"gemma-4","content":[],"stop_reason":"stop_sequence","stop_sequence":"</s>","usage":{"input_tokens":1,"output_tokens":0}}`,
		string(AppendMessageResponse(nil, resp)))
}

// TestAnthropic_AppendMessageResponse_Ugly pins two edges together: append
// onto a non-empty caller buffer, and an escape-heavy ID.
func TestAnthropic_AppendMessageResponse_Ugly(t *testing.T) {
	buf := []byte("PRE")
	resp := MessageResponse{ID: `msg "3"`, Type: "message", Role: "assistant", Model: "gemma-4"}
	buf = AppendMessageResponse(buf, resp)
	core.AssertEqual(t,
		`PRE{"id":"msg \"3\"","type":"message","role":"assistant","model":"gemma-4","content":[],"usage":{"input_tokens":0,"output_tokens":0}}`,
		string(buf))
}

// TestAnthropic_MessageResponseSize_Good pins the size estimator against a
// typical text response — predicted must be >= the actual encoded size.
func TestAnthropic_MessageResponseSize_Good(t *testing.T) {
	resp := NewTextResponse("msg_1", "gemma-4", "hello", inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 1})
	predicted := MessageResponseSize(resp)
	actual := len(AppendMessageResponse(nil, resp))
	if predicted < actual {
		t.Fatalf("MessageResponseSize=%d < actual %d", predicted, actual)
	}
}

// TestAnthropic_MessageResponseSize_Bad pins the zero-value edge — the
// estimator must not undercount an empty response.
func TestAnthropic_MessageResponseSize_Bad(t *testing.T) {
	var resp MessageResponse
	predicted := MessageResponseSize(resp)
	actual := len(AppendMessageResponse(nil, resp))
	if predicted < actual {
		t.Fatalf("MessageResponseSize=%d < actual %d for zero-value response", predicted, actual)
	}
}

// TestAnthropic_MessageResponseSize_Ugly pins the branch-heavy shape —
// multi-block content plus both optional stop fields.
func TestAnthropic_MessageResponseSize_Ugly(t *testing.T) {
	resp := MessageResponse{
		ID: "msg_x", Type: "message", Role: "assistant", Model: "gemma-4",
		Content: []ContentBlock{
			{Type: "text", Text: "one"},
			{Type: "text", Text: "two"},
			{Type: "tool_use"},
		},
		StopReason:   "stop_sequence",
		StopSequence: "</response>",
		Usage:        Usage{InputTokens: 100, OutputTokens: 200},
	}
	predicted := MessageResponseSize(resp)
	actual := len(AppendMessageResponse(nil, resp))
	if predicted < actual {
		t.Fatalf("MessageResponseSize=%d < actual %d", predicted, actual)
	}
}

// TestAnthropic_AppendMessageRequest_Good pins the hand-rolled encoder
// against a literal minimal request.
func TestAnthropic_AppendMessageRequest_Good(t *testing.T) {
	req := MessageRequest{
		Model:     "gemma-4",
		Messages:  []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
		MaxTokens: 256,
	}
	core.AssertEqual(t,
		`{"model":"gemma-4","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":256}`,
		string(AppendMessageRequest(nil, req)))
}

// TestAnthropic_AppendMessageRequest_Bad pins the edge case of no messages
// at all — messages still emits as an empty array, max_tokens still emits
// at its zero value (neither field carries omitempty).
func TestAnthropic_AppendMessageRequest_Bad(t *testing.T) {
	req := MessageRequest{Model: "gemma-4"}
	core.AssertEqual(t,
		`{"model":"gemma-4","messages":[],"max_tokens":0}`,
		string(AppendMessageRequest(nil, req)))
}

// TestAnthropic_AppendMessageRequest_Ugly pins three edges together: append
// onto a non-empty caller buffer, every scalar optional field set, and an
// escape-heavy stop sequence.
func TestAnthropic_AppendMessageRequest_Ugly(t *testing.T) {
	temp := float32(0.5)
	buf := []byte("PRE")
	req := MessageRequest{
		Model:         "gemma-4",
		System:        "Be concise.",
		Messages:      []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
		MaxTokens:     10,
		Temperature:   &temp,
		Stream:        true,
		StopSequences: []string{`</s>"quote"`},
	}
	buf = AppendMessageRequest(buf, req)
	core.AssertEqual(t,
		`PRE{"model":"gemma-4","system":"Be concise.","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":10,"temperature":0.5,"stream":true,"stop_sequences":["</s>\"quote\""]}`,
		string(buf))
}

// TestAnthropic_MessageRequestSize_Good pins the size estimator against a
// typical minimal request.
func TestAnthropic_MessageRequestSize_Good(t *testing.T) {
	req := MessageRequest{
		Model:     "gemma-4",
		Messages:  []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
		MaxTokens: 256,
	}
	predicted := MessageRequestSize(req)
	actual := len(AppendMessageRequest(nil, req))
	if predicted < actual {
		t.Fatalf("MessageRequestSize=%d < actual %d", predicted, actual)
	}
}

// TestAnthropic_MessageRequestSize_Bad pins the zero-value edge.
func TestAnthropic_MessageRequestSize_Bad(t *testing.T) {
	var req MessageRequest
	predicted := MessageRequestSize(req)
	actual := len(AppendMessageRequest(nil, req))
	if predicted < actual {
		t.Fatalf("MessageRequestSize=%d < actual %d for zero-value request", predicted, actual)
	}
}

// TestAnthropic_MessageRequestSize_Ugly pins the branch-heavy shape — every
// optional pointer field set, multi-turn multi-block messages, and
// multiple stop sequences.
func TestAnthropic_MessageRequestSize_Ugly(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.9)
	minP := float32(0.1)
	topK := 40
	req := MessageRequest{
		Model:  "gemma-4",
		System: "Be concise.",
		Messages: []Message{
			{Role: "user", Content: []ContentBlock{{Type: "text", Text: "one"}, {Type: "text", Text: "two"}}},
			{Role: "assistant", Content: []ContentBlock{{Type: "text", Text: "three"}}},
		},
		MaxTokens:     512,
		Temperature:   &temp,
		TopP:          &topP,
		MinP:          &minP,
		TopK:          &topK,
		Stream:        true,
		StopSequences: []string{"</s>", "<|eot|>"},
	}
	predicted := MessageRequestSize(req)
	actual := len(AppendMessageRequest(nil, req))
	if predicted < actual {
		t.Fatalf("MessageRequestSize=%d < actual %d", predicted, actual)
	}
}
