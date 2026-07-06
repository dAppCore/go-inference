// SPDX-Licence-Identifier: EUPL-1.2

package train_test

import (
	"context"
	"fmt"

	"dappco.re/go/inference"
	"dappco.re/go/inference/train"
	"dappco.re/go/inference/train/dataset"
)

// ExampleRunSSD shows the native SSD sampling loop: sample every prompt from
// the frozen base, capture the raw return, and stop at the trace. SSD never
// trains — the samples ARE the deliverable.
func ExampleRunSSD() {
	runner := train.SSDRunner{
		Generate: func(_ context.Context, prompt string, _ inference.GenerateConfig) (string, error) {
			return "self-said: " + prompt, nil
		},
	}
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "name a colour"}, {Prompt: "name a fruit"}})
	cfg := train.SSDConfig{SampleTemperature: 0.7, DisableCapture: true}

	result, err := train.RunSSD(context.Background(), runner, ds, cfg)
	if err != nil {
		panic(err)
	}
	for _, sample := range result.Samples {
		fmt.Println(sample.Response)
	}
	// Output:
	// self-said: name a colour
	// self-said: name a fruit
}

// ExampleDefaultSSDConfig shows the ml-ssd data-generation defaults a driver
// starts from before overriding any knob.
func ExampleDefaultSSDConfig() {
	cfg := train.DefaultSSDConfig()
	fmt.Println("sample max tokens:", cfg.SampleMaxTokens)
	fmt.Println("sample temperature:", cfg.SampleTemperature)
	fmt.Println("filter shortest percent:", cfg.FilterShortestPercent)
	// Output:
	// sample max tokens: 65536
	// sample temperature: 1.5
	// filter shortest percent: 10
}

// ExampleDefaultSSDCodeBenchmarkConfig shows the LiveCodeBench-v6 evaluation
// defaults for the SSD code-benchmark harness.
func ExampleDefaultSSDCodeBenchmarkConfig() {
	cfg := train.DefaultSSDCodeBenchmarkConfig()
	fmt.Println("benchmark:", cfg.Benchmark)
	fmt.Println("n repeat:", cfg.NRepeat)
	fmt.Println("generate max tokens:", cfg.Generate.MaxTokens)
	// Output:
	// benchmark: LiveCodeBench-v6
	// n repeat: 20
	// generate max tokens: 32768
}

// ExampleSSDRecipes shows the released ml-ssd model recipes with their
// native data-generation and evaluation defaults filled in.
func ExampleSSDRecipes() {
	for _, recipe := range train.SSDRecipes() {
		fmt.Println(recipe.Name, "->", recipe.Model)
	}
	// Output:
	// SimpleSD-4B-instruct -> apple/SimpleSD-4B-instruct
	// SimpleSD-4B-thinking -> apple/SimpleSD-4B-thinking
	// SimpleSD-30b-a3b-instruct -> apple/SimpleSD-30b-a3b-instruct
}

// ExampleLookupSSDRecipe resolves a named parity recipe by either its short
// name or its full model identifier.
func ExampleLookupSSDRecipe() {
	recipe, ok := train.LookupSSDRecipe(train.SSDRecipe4BInstruct)
	fmt.Println(ok, recipe.Model)
	// Output:
	// true apple/SimpleSD-4B-instruct
}

// ExampleSSDResult_SampleGenerateConfig recovers the frozen-model sampling
// configuration used to create the raw SSD training rows — the exact
// settings to reproduce a sample.
func ExampleSSDResult_SampleGenerateConfig() {
	result := &train.SSDResult{SampleMaxTokens: 256, SampleTemperature: 0.7, SampleTopK: 40}
	cfg := result.SampleGenerateConfig()
	fmt.Println("max tokens:", cfg.MaxTokens)
	fmt.Println("temperature:", cfg.Temperature)
	fmt.Println("top k:", cfg.TopK)
	// Output:
	// max tokens: 256
	// temperature: 0.7
	// top k: 40
}
