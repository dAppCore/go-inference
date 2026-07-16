// SPDX-Licence-Identifier: EUPL-1.2

package bert_test

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/arch/bert"
)

func ExampleNewServeModel() {
	sm := bert.NewServeModel(exampleModel())
	core.Println(sm.ModelType())
	// Output: bert
}

func ExampleServeModel_Generate() {
	sm := bert.NewServeModel(exampleModel())
	n := 0
	for range sm.Generate(context.Background(), "hello world") {
		n++
	}
	core.Println(n) // an encoder has no generative path
	// Output: 0
}

func ExampleServeModel_Chat() {
	sm := bert.NewServeModel(exampleModel())
	n := 0
	for range sm.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		n++
	}
	core.Println(n) // an encoder has no generative path
	// Output: 0
}

func ExampleServeModel_Classify() {
	sm := bert.NewServeModel(exampleModel())
	r := sm.Classify(context.Background(), []string{"hello world"})
	core.Println(r.OK)
	// Output: false
}

func ExampleServeModel_BatchGenerate() {
	sm := bert.NewServeModel(exampleModel())
	r := sm.BatchGenerate(context.Background(), []string{"hello world"})
	core.Println(r.OK)
	// Output: false
}

func ExampleServeModel_ModelType() {
	sm := bert.NewServeModel(exampleModel())
	core.Println(sm.ModelType())
	// Output: bert
}

func ExampleServeModel_Info() {
	sm := bert.NewServeModel(exampleModel())
	info := sm.Info()
	core.Println(info.Architecture, info.HiddenSize)
	// Output: bert 8
}

func ExampleServeModel_Metrics() {
	sm := bert.NewServeModel(exampleModel())
	m := sm.Metrics()
	core.Println(m == inference.GenerateMetrics{})
	// Output: true
}

func ExampleServeModel_Err() {
	sm := bert.NewServeModel(exampleModel())
	core.Println(sm.Err().OK)
	// Output: false
}

func ExampleServeModel_Close() {
	sm := bert.NewServeModel(exampleModel())
	core.Println(sm.Close().OK)
	// Output: true
}
