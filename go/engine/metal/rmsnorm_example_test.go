// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleRMSNorm shows the row-normalise call shape: a row-major [rows,axis]
// float32 slice in, the RMS-normalised weight-scaled rows out, same shape. The
// call needs MLX_METALLIB_PATH set, so the example guards on it (no Output:
// directive — the GPU dispatch is exercised under the test gate).
func ExampleRMSNorm() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const rows, axisSize = 2, 64
	x := syntheticFloat32(rows*axisSize, 3)
	weight := syntheticFloat32(axisSize, 5)

	out, err := RMSNorm(x, weight, rows, axisSize, 1e-5)
	if err != nil {
		return
	}
	core.Println(len(out)) // rows*axisSize values
}

// ExampleRMSNormInto is ExampleRMSNorm with caller-owned output storage: pass
// a slice with capacity for rows*axisSize values and the kernel writes it
// directly (no per-call result allocation on the decode hot path).
func ExampleRMSNormInto() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const rows, axisSize = 2, 64
	x := syntheticFloat32(rows*axisSize, 3)
	weight := syntheticFloat32(axisSize, 5)
	out := make([]float32, rows*axisSize)

	got, err := RMSNormInto(out, x, weight, rows, axisSize, 1e-5)
	if err != nil {
		return
	}
	core.Println(len(got)) // rows*axisSize values, backed by out
}
