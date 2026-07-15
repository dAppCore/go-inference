// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleMatVec shows the matrix-vector projection call shape: a row-major
// (outDim x inDim) float32 matrix and an inDim vector in, the outDim product
// out. The call needs MLX_METALLIB_PATH set, so the example guards on it (no
// Output: directive — the GPU dispatch is exercised under the test gate).
func ExampleMatVec() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const outDim, inDim = 16, 64
	mat := syntheticFloat32(outDim*inDim, 3)
	vec := syntheticFloat32(inDim, 5)

	out, err := MatVec(mat, vec, outDim, inDim)
	if err != nil {
		return
	}
	core.Println(len(out)) // outDim values
}

// ExampleMatVecInto is ExampleMatVec with caller-owned output storage: pass a
// slice with capacity for outDim values and the kernel writes it directly (no
// per-call result allocation on the decode hot path).
func ExampleMatVecInto() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const outDim, inDim = 16, 64
	mat := syntheticFloat32(outDim*inDim, 3)
	vec := syntheticFloat32(inDim, 5)
	out := make([]float32, outDim)

	got, err := MatVecInto(out, mat, vec, outDim, inDim)
	if err != nil {
		return
	}
	core.Println(len(got)) // outDim values, backed by out
}
