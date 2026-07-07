// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleRMSNormResidualBF16 shows the fused gemma4 tail call shape: one bf16
// row each of x, weight and residual in, out = res + RMSNorm(x, weight) in a
// single dispatch. The call needs MLX_METALLIB_PATH set (plus the sibling
// lthn_kernels.metallib), so the example guards on it (no Output: directive —
// the GPU dispatch is exercised under the test gate).
func ExampleRMSNormResidualBF16() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const axisSize = 512
	x, w, res := rmsNormResidualFixture(axisSize)

	out, err := RMSNormResidualBF16(x, w, res, axisSize, 1e-6)
	if err != nil {
		return
	}
	core.Println(len(out)) // axisSize bf16 values, 2 bytes each
}

// ExampleRMSNormResidualBF16Into is ExampleRMSNormResidualBF16 with
// caller-owned output storage: pass a slice with capacity for axisSize bf16
// bytes and the kernel writes it directly (no per-call row allocation on the
// decode hot path).
func ExampleRMSNormResidualBF16Into() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const axisSize = 512
	x, w, res := rmsNormResidualFixture(axisSize)
	out := make([]byte, axisSize*bf16Size)

	got, err := RMSNormResidualBF16Into(out, x, w, res, axisSize, 1e-6)
	if err != nil {
		return
	}
	core.Println(len(got)) // axisSize bf16 values, backed by out
}
