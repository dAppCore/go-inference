// SPDX-Licence-Identifier: EUPL-1.2

// A minimal Go app using the GENERATED lem SDK (task sdk → build/sdk/go) to
// chat with a local gemma4 serve — the OpenAPI standard doing the client work.
package main

import (
	"context"
	"fmt"
	"os"

	lemsdk "dappco.re/go/inference/lemsdk"
)

func main() {
	base := os.Getenv("LEM_BASE_URL")
	if base == "" {
		base = "http://localhost:36911"
	}
	cfg := lemsdk.NewConfiguration()
	cfg.Servers = lemsdk.ServerConfigurations{{URL: base}}
	client := lemsdk.NewAPIClient(cfg)
	ctx := context.Background()

	models, _, err := client.InferenceAPI.GetV1Models(ctx).Execute()
	if err != nil {
		fmt.Fprintln(os.Stderr, "models:", err)
		os.Exit(1)
	}
	for _, m := range models.GetData() {
		fmt.Println("serving:", m.GetId())
	}

	msg := lemsdk.NewPostV1ChatCompletionsRequestMessagesInner("user")
	msg.SetContent("In one sentence, why does local inference matter?")
	req := lemsdk.NewPostV1ChatCompletionsRequest(
		[]lemsdk.PostV1ChatCompletionsRequestMessagesInner{*msg}, "gemma4")
	req.SetMaxTokens(96)
	req.SetChatTemplateKwargs(map[string]interface{}{"enable_thinking": false})

	resp, _, err := client.InferenceAPI.PostV1ChatCompletions(ctx).
		PostV1ChatCompletionsRequest(*req).Execute()
	if err != nil {
		fmt.Fprintln(os.Stderr, "chat:", err)
		os.Exit(1)
	}
	choice := resp.GetChoices()[0]
	fmt.Println("gemma4:", choice.Message.GetContent())
	if usage, ok := resp.GetUsageOk(); ok {
		fmt.Printf("usage: %d prompt + %d completion tokens\n",
			usage.GetPromptTokens(), usage.GetCompletionTokens())
	}
}
