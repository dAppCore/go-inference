// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func Example_rocmModel_ApplyChatTemplate() {
	model := &rocmModel{native: &fakeNativeModel{chatTemplateResult: "user:hello"}}
	prompt, _ := model.ApplyChatTemplate([]inference.Message{{Role: "user", Content: "hello"}})
	core.Println(prompt)
	// Output: user:hello
}

func Example_rocmModel_LoadAdapter() {
	model := &rocmModel{native: &fakeNativeModel{}}
	identity, _ := model.LoadAdapter("domain.safetensors")
	core.Println(identity.Format)
	_ = model.UnloadAdapter()
	core.Println(model.ActiveAdapter().Path == "")
	// Output:
	// lora
	// true
}

func Example_rocmModel_Embed() {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "bert"},
		native:    &fakeNativeEmbeddingModel{fakeNativeModel: &fakeNativeModel{}},
	}
	result, _ := model.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"core"}})
	core.Println(result.Model.Architecture)
	core.Println(len(result.Vectors[0]))
	// Output:
	// bert
	// 2
}

func Example_rocmModel_Rerank() {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "bert"},
		native:    &fakeNativeEmbeddingModel{fakeNativeModel: &fakeNativeModel{}},
	}
	result, _ := model.Rerank(context.Background(), inference.RerankRequest{Query: "core", Documents: []string{"first", "second"}})
	core.Println(result.Model.Architecture)
	core.Println(result.Results[0].Text)
	// Output:
	// bert
	// second
}
