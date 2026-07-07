// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleSoftmaxF32 shows the row-wise softmax call shape: a row-major
// [rows,axis] float32 slice in, each row softmaxed over the last axis, same
// shape out. The call needs MLX_METALLIB_PATH set, so the example guards on it
// (no Output: directive — the GPU dispatch is exercised under the test gate).
func ExampleSoftmaxF32() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const rows, axisSize = 2, 64
	in := syntheticFloat32(rows*axisSize, 5)

	out, err := SoftmaxF32(in, axisSize)
	if err != nil {
		return
	}
	core.Println(len(out)) // rows*axisSize values, each row summing to 1
}

// ExampleSoftmaxF32Into is ExampleSoftmaxF32 with caller-owned output storage:
// pass a slice with capacity for len(in) values and the kernel writes it
// directly (no per-call result allocation on the attention hot path).
func ExampleSoftmaxF32Into() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const rows, axisSize = 2, 64
	in := syntheticFloat32(rows*axisSize, 5)
	out := make([]float32, rows*axisSize)

	got, err := SoftmaxF32Into(out, in, axisSize)
	if err != nil {
		return
	}
	core.Println(len(got)) // rows*axisSize values, backed by out
}
