// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleNewProviderRouter() {
	routerResult := NewProviderRouter(ProviderRoute{
		Name:    "local",
		ModelID: "gemma-test",
		Model:   &routerFakeModel{modelType: "mlx", output: "hello from local"},
	})
	router := routerResult.Value.(*ProviderRouter)

	chatResult := router.Chat(context.Background(), ProviderChatRequest{Prompt: "hello"})
	response := chatResult.Value.(ProviderChatResponse)

	core.Println(response.Provider)
	core.Println(response.Text)
	// Output:
	// local
	// hello from local
}

func ExampleProviderContextAssemblerFunc_AssembleContext() {
	assembler := ProviderContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
		return core.Ok("retrieved context")
	})
	result := assembler.AssembleContext(context.Background(), nil)

	core.Println(result.Value.(string))
	// Output:
	// retrieved context
}

func ExampleNewProviderRouterWithOptions() {
	routerResult := NewProviderRouterWithOptions(ProviderRouterOptions{ContextRole: "system"}, ProviderRoute{
		Name:    "local",
		ModelID: "gemma-test",
		Model:   &routerFakeModel{modelType: "mlx", output: "hello"},
	})

	core.Println(routerResult.OK)
	// Output:
	// true
}

func ExampleProviderRouter_Providers() {
	router := core.MustCast[*ProviderRouter](NewProviderRouter(ProviderRoute{
		Name:    "local",
		ModelID: "gemma-test",
		Model:   &routerFakeModel{modelType: "mlx", output: "hello"},
	}))

	core.Println(router.Providers()[0].Name)
	// Output:
	// local
}

func ExampleProviderRouter_Chat() {
	router := core.MustCast[*ProviderRouter](NewProviderRouter(ProviderRoute{
		Name:    "local",
		ModelID: "gemma-test",
		Model:   &routerFakeModel{modelType: "mlx", output: "hello"},
	}))
	result := router.Chat(context.Background(), ProviderChatRequest{Prompt: "hi"})
	response := result.Value.(ProviderChatResponse)

	core.Println(response.Text)
	// Output:
	// hello
}
