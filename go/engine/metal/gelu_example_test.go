// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleGelu composes the tanh-approximation GELU on the GPU. gelu(0) is exactly
// zero. The call needs MLX_METALLIB_PATH set, so the example guards on it (no
// Output: directive — the GPU dispatch is exercised under the test gate).
func ExampleGelu() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	out, err := Gelu([]float32{0})
	if err != nil {
		return
	}
	core.Println(out[0]) // gelu(0) == 0
}

// ExampleGeluGateMul shows gelu(gate)*up — gemma's MLP gate. With up all-zero the
// product is zero regardless of the gate, demonstrating the gate·up composition.
func ExampleGeluGateMul() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	out, err := GeluGateMul([]float32{8, -8}, []float32{0, 0})
	if err != nil {
		return
	}
	core.Println(out) // [0 0]
}
