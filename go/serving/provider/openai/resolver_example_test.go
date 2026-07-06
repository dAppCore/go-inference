// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleResolverFunc_ResolveModel() {
	model := &stubModel{}
	fn := ResolverFunc(func(ctx context.Context, name string) (inference.TextModel, error) {
		return model, nil
	})

	got, err := fn.ResolveModel(context.Background(), "qwen")

	core.Println(err)
	core.Println(got == model)
	// Output:
	// <nil>
	// true
}

func ExampleNewStaticResolver() {
	resolver := NewStaticResolver(map[string]inference.TextModel{"Qwen3": &stubModel{}})

	_, err := resolver.ResolveModel(context.Background(), "qwen3")

	core.Println(err)
	// Output:
	// <nil>
}

func ExampleStaticResolver_ResolveModel() {
	model := &stubModel{}
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": model})

	got, err := resolver.ResolveModel(context.Background(), "qwen")

	core.Println(err)
	core.Println(got == model)
	// Output:
	// <nil>
	// true
}

func ExampleNewBackendResolver() {
	r := NewBackendResolver(" my-backend ", " /models/x ")

	core.Println(r.BackendName)
	core.Println(r.ModelPath)
	// Output:
	// my-backend
	// /models/x
}

func ExampleBackendResolver_ResolveModel() {
	backend := &fakeLoadBackend{name: "example-resolver-backend", available: true, model: &stubModel{}}
	inference.Register(backend)
	r := NewBackendResolver(backend.name, "/models/x")

	_, err := r.ResolveModel(context.Background(), "ignored")

	core.Println(err)
	// Output:
	// <nil>
}
