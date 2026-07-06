// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleResponseMessages() {
	req := ResponseRequest{
		Instructions: "Be concise.",
		Input:        []ResponseInputMessage{{Role: "user", Content: "hello"}},
	}

	messages := ResponseMessages(req)

	core.Println(len(messages))
	core.Println(messages[0].Role, messages[1].Content)
	// Output:
	// 2
	// system hello
}

func ExampleResponseGenerateOptions() {
	req := ResponseRequest{
		Model: "qwen",
		Input: []ResponseInputMessage{{Role: "user", Content: "hi"}},
	}

	opts, err := ResponseGenerateOptions(req)
	if err != nil {
		core.Println(err)
		return
	}

	cfg := inference.ApplyGenerateOpts(opts)
	core.Println(cfg.MaxTokens)
	// Output:
	// 2048
}

func ExampleNewTextResponse() {
	resp := NewTextResponse("resp_1", "qwen", "ok", inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 2})

	core.Println(resp.Object)
	core.Println(resp.Output[0].Content[0].Text)
	core.Println(resp.Usage.TotalTokens)
	// Output:
	// response
	// ok
	// 5
}
