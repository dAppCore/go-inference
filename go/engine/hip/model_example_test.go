//go:build linux && amd64

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func exampleModel() *rocmModel {
	return &rocmModel{modelType: "llama", modelInfo: inference.ModelInfo{Architecture: "llama"}}
}

func Example_rocmModelGenerate() {
	count := 0
	for range exampleModel().Generate(context.Background(), "hello") {
		count++
	}
	core.Println(count)
	// Output: 0
}

func Example_rocmModelChat() {
	count := 0
	for range exampleModel().Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		count++
	}
	core.Println(count)
	// Output: 0
}

func Example_rocmModelClassify() {
	r := exampleModel().Classify(context.Background(), []string{"x"})
	core.Println(!r.OK)
	// Output: true
}

func Example_rocmModelBatchGenerate() {
	r := exampleModel().BatchGenerate(context.Background(), []string{"x"})
	core.Println(!r.OK)
	// Output: true
}

func Example_rocmModelModelType() { core.Println(exampleModel().ModelType()) /* Output: llama */ }
func Example_rocmModelInfo()      { core.Println(exampleModel().Info().Architecture) /* Output: llama */ }
func Example_rocmModelMetrics() {
	core.Println(exampleModel().Metrics().GeneratedTokens) /* Output: 0 */
}
func Example_rocmModelErr()   { core.Println(exampleModel().Err().OK) /* Output: true */ }
func Example_rocmModelClose() { core.Println(exampleModel().Close().OK) /* Output: true */ }
