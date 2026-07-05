// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"

	core "dappco.re/go"
)

// ExampleQuantizeModelPack converts a dense safetensors model pack into a
// quantised GGUF pack.
func ExampleQuantizeModelPack() {
	baseResult := core.MkdirTemp("", "gguf-quantize-example-*")
	if !baseResult.OK {
		core.Println("tempdir failed")
		return
	}
	base := baseResult.Value.(string)
	defer core.RemoveAll(base)

	sourceDir := core.Path(base, "source")
	core.MkdirAll(sourceDir, 0o755)
	weightPath := core.Path(sourceDir, "model.safetensors")
	// 256 values — block-aligned for every supported QuantizeFormat.
	values := make([]float32, 256)
	for i := range values {
		values[i] = float32(i) / 256
	}
	if err := writeMinimalExampleSafetensors(weightPath, "weight", values, []int{256}); err != nil {
		core.Println("write failed")
		return
	}

	result, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: Source{
			Root:         sourceDir,
			Architecture: "llama",
			WeightFiles:  []string{weightPath},
		},
		OutputPath: core.Path(base, "quantised"),
		Format:     QuantizeQ8_0,
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println(result.Format, result.QuantizedTensors)
	// Output: q8_0 1
}

// ExampleValidationSummary joins GGUF validation issue codes into a
// human-readable string.
func ExampleValidationSummary() {
	core.Println(ValidationSummary([]ValidationIssue{
		{Code: "invalid_tensor_shape", Tensor: "blk.0.weight"},
	}))
	// Output: invalid_tensor_shape:blk.0.weight
}
