// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleRoPE shows the single-token decode rotary embedding call shape: x is row-major
// (b,nHeads,1,headDim), offset is the absolute position. The call needs MLX_METALLIB_PATH set,
// so the example guards on it (no Output: directive — the GPU dispatch is exercised under the
// test gate).
func ExampleRoPE() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const b, nHeads, headDim = 1, 8, 64
	x := syntheticFloat32(b*nHeads*headDim, 3)

	out, err := RoPE(x, b, nHeads, headDim, 10000, 1, 17, false)
	if err != nil {
		return
	}
	core.Println(len(out)) // b*nHeads*headDim float32 elements
}

// ExampleRoPEInto is ExampleRoPE with caller-owned output storage.
func ExampleRoPEInto() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const b, nHeads, headDim = 1, 8, 64
	x := syntheticFloat32(b*nHeads*headDim, 3)
	out := make([]float32, b*nHeads*headDim)

	got, err := RoPEInto(out, x, b, nHeads, headDim, 10000, 1, 17, false)
	if err != nil {
		return
	}
	core.Println(len(got)) // b*nHeads*headDim float32 elements, reusing out's backing
}
