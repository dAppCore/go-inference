// SPDX-Licence-Identifier: EUPL-1.2

package mlxaffine_test

import (
	"fmt"

	"dappco.re/go/inference/model/quant/mlxaffine"
)

// ExampleQuantizeTensor group-affine quantises a one-row weight and shows the byte
// layout the engine loads: a uint32-packed weight (32/bits codes per word) plus bf16
// scales and biases, one pair per group.
func ExampleQuantizeTensor() {
	const outDim, inDim, bits, groupSize = 1, 64, 4, 64
	w := make([]float32, inDim)
	for i := range w {
		w[i] = 0.01 * float32(i-32) // a spread of negative and positive weights
	}

	packed, scales, biases, err := mlxaffine.QuantizeTensor(w, outDim, inDim, bits, groupSize)
	if err != nil {
		panic(err)
	}

	// One group per row → one scale, one bias; 64 four-bit codes → 8 uint32 words.
	fmt.Printf("packed=%d bytes (%d words) scales=%d bytes biases=%d bytes\n",
		len(packed), mlxaffine.PackedWords(inDim, bits), len(scales), len(biases))

	got, _ := mlxaffine.DequantizeTensor(packed, scales, biases, outDim, inDim, bits, groupSize)
	fmt.Printf("w[0]=%.3f -> dequant=%.3f\n", w[0], got[0])
	// Output:
	// packed=32 bytes (8 words) scales=2 bytes biases=2 bytes
	// w[0]=-0.320 -> dequant=-0.320
}
