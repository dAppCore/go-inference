// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleToolUseBlock() {
	block := ToolUseBlock("toolu_1", "get_weather", `{"city":"Paris"}`)
	core.Println(block.Type, block.Name, block.Input["city"])
	// Output:
	// tool_use get_weather Paris
}

func ExampleNewToolUseResponse() {
	blocks := []ContentBlock{ToolUseBlock("toolu_1", "get_weather", `{"city":"Paris"}`)}
	resp := NewToolUseResponse("msg_1", "gemma-4", blocks, inference.GenerateMetrics{PromptTokens: 10, GeneratedTokens: 4})
	core.Println(resp.StopReason, resp.Usage.InputTokens, resp.Usage.OutputTokens)
	// Output:
	// tool_use 10 4
}

func ExampleAppendMessageResponse() {
	resp := NewTextResponse("msg_1", "gemma-4", "hi", inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 1})
	buf := AppendMessageResponse(make([]byte, 0, MessageResponseSize(resp)), resp)
	core.Println(string(buf))
	// Output:
	// {"id":"msg_1","type":"message","role":"assistant","model":"gemma-4","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}
}

func ExampleMessageResponseSize() {
	resp := NewTextResponse("msg_1", "gemma-4", "hi", inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 1})
	size := MessageResponseSize(resp)
	actual := len(AppendMessageResponse(nil, resp))
	core.Println(size >= actual)
	// Output:
	// true
}

func ExampleAppendMessageRequest() {
	req := MessageRequest{
		Model:     "gemma-4",
		Messages:  []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
		MaxTokens: 256,
	}
	buf := AppendMessageRequest(make([]byte, 0, MessageRequestSize(req)), req)
	core.Println(string(buf))
	// Output:
	// {"model":"gemma-4","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":256}
}

func ExampleMessageRequestSize() {
	req := MessageRequest{
		Model:     "gemma-4",
		Messages:  []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
		MaxTokens: 256,
	}
	size := MessageRequestSize(req)
	actual := len(AppendMessageRequest(nil, req))
	core.Println(size >= actual)
	// Output:
	// true
}

func ExampleInferenceMessages() {
	req := MessageRequest{
		System:   "Be concise.",
		Messages: []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
	}
	messages := InferenceMessages(req)
	core.Println(messages[0].Role, messages[1].Role, messages[1].Content)
	// Output:
	// system user hi
}

func ExampleGenerateOptions() {
	req := MessageRequest{MaxTokens: 128}
	opts := GenerateOptions(req)
	cfg := inference.ApplyGenerateOpts(opts)
	core.Println(cfg.MaxTokens)
	// Output:
	// 128
}

func ExampleNewTextResponse() {
	resp := NewTextResponse("msg_1", "gemma-4", "hi", inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 1})
	core.Println(resp.StopReason, resp.Content[0].Text)
	// Output:
	// end_turn hi
}
