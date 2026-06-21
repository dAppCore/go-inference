// SPDX-License-Identifier: EUPL-1.2

package openai

import (
	"context"
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
)

func ExampleNewBackend() {
	backend := NewBackend(Config{
		Name:         "openai",
		BaseURL:      "https://api.openai.com",
		DefaultModel: "gpt-4o-mini",
	})

	core.Println(backend.Name())
	core.Println(backend.Available())

	// Output:
	// openai
	// true
}

func ExampleContextAssemblerFunc() {
	assembler := ContextAssemblerFunc(func(ctx context.Context, messages []inference.Message) core.Result {
		return core.Ok("retrieved context")
	})
	contextResult := assembler.AssembleContext(context.Background(), nil)
	contextText := contextResult.Value.(string)

	core.Println(contextText)

	// Output:
	// retrieved context
}

func ExampleContextAssemblerFunc_AssembleContext() {
	assembler := ContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
		return core.Ok("context")
	})
	result := assembler.AssembleContext(context.Background(), nil)

	core.Println(result.Value.(string))
	// Output:
	// context
}

func ExampleRegister() {
	backend := Register(Config{Name: "example-openai-register", BaseURL: "https://api.example.test", DefaultModel: "gpt"})
	got, ok := inference.Get("example-openai-register")

	core.Println(ok)
	core.Println(got == backend)
	// Output:
	// true
	// true
}

func ExampleBackend_Name() {
	backend := NewBackend(Config{Name: "example"})

	core.Println(backend.Name())
	// Output:
	// example
}

func ExampleBackend_Available() {
	backend := NewBackend(Config{BaseURL: "https://api.example.test", DefaultModel: "gpt"})

	core.Println(backend.Available())
	// Output:
	// true
}

func ExampleBackend_LoadModel() {
	backend := NewBackend(Config{BaseURL: "https://api.example.test", DefaultModel: "gpt"})
	result := backend.LoadModel("")
	model := result.Value.(inference.TextModel)

	core.Println(result.OK)
	core.Println(model.ModelType())
	// Output:
	// true
	// openai-compatible
}

func ExampleBackend_Capabilities() {
	backend := NewBackend(Config{Name: "example", BaseURL: "https://api.example.test", DefaultModel: "gpt"})
	report := backend.Capabilities()

	core.Println(report.Runtime.Backend)
	core.Println(report.Supports(inference.CapabilityChat))
	// Output:
	// example
	// true
}

func ExampleModel_Generate() {
	model, cleanup := exampleOpenAIModel("hello")
	defer cleanup()

	var text string
	for token := range model.Generate(context.Background(), "hi") {
		text += token.Text
	}

	core.Println(text)
	// Output:
	// hello
}

func ExampleModel_Chat() {
	model, cleanup := exampleOpenAIModel("chat")
	defer cleanup()

	var text string
	for token := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		text += token.Text
	}

	core.Println(text)
	// Output:
	// chat
}

func ExampleModel_Classify() {
	model := &Model{}
	result := model.Classify(context.Background(), []string{"prompt"})

	core.Println(result.OK)
	core.Println(core.Contains(result.Error(), "not supported"))
	// Output:
	// false
	// true
}

func ExampleModel_BatchGenerate() {
	model, cleanup := exampleOpenAIModel("batch")
	defer cleanup()
	result := model.BatchGenerate(context.Background(), []string{"a", "b"})
	batches := result.Value.([]inference.BatchResult)

	core.Println(len(batches))
	core.Println(batches[0].Tokens[0].Text)
	// Output:
	// 2
	// batch
}

func ExampleModel_ModelType() {
	model := &Model{}

	core.Println(model.ModelType())
	// Output:
	// openai-compatible
}

func ExampleModel_Info() {
	model := &Model{}

	core.Println(model.Info().Architecture)
	// Output:
	// openai-compatible
}

func ExampleModel_Metrics() {
	model := &Model{metrics: inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 2}}
	metrics := model.Metrics()

	core.Println(metrics.PromptTokens)
	core.Println(metrics.GeneratedTokens)
	// Output:
	// 3
	// 2
}

func ExampleModel_Err() {
	model := &Model{lastErr: core.NewError("failed")}
	result := model.Err()

	core.Println(result.OK)
	core.Println(result.Error())
	// Output:
	// false
	// failed
}

func ExampleModel_Close() {
	model := &Model{}
	result := model.Close()

	core.Println(result.OK)
	// Output:
	// true
}

func ExampleModel_Capabilities() {
	backend := NewBackend(Config{Name: "example", BaseURL: "https://api.example.test", DefaultModel: "gpt"})
	model := &Model{backend: backend, modelID: "gpt"}
	report := model.Capabilities()

	core.Println(report.Model.ID)
	core.Println(report.Supports(inference.CapabilityGenerate))
	// Output:
	// gpt
	// true
}

func exampleOpenAIModel(content string) (*Model, func()) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(core.JSONMarshalString(openaicompat.ChatCompletionResponse{
			Model: "gpt",
			Choices: []openaicompat.ChatChoice{{
				Message: openaicompat.ChatMessage{Role: "assistant", Content: content},
			}},
			Usage: openaicompat.ChatUsage{PromptTokens: 1, CompletionTokens: 1},
		})))
	}))
	backend := NewBackend(Config{
		BaseURL:      server.URL,
		DefaultModel: "gpt",
		HTTPClient:   server.Client(),
	})
	return backend.LoadModel("").Value.(*Model), server.Close
}
