// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/engine/enginetest"
)

// TestNativeAffineQuantParity is engine/metal's proving consumer of
// enginetest.QuantParity (rocm design #13): it fetches this backend's
// registered "native"/"affine" quant compute (model_quant.go, affineQMV.MatVec)
// and checks it against the pure-Go group-affine reference on a small
// deterministic fixture — real GPU dispatch through QMVBF16, not a mock.
// Skips cleanly without MLX_METALLIB_PATH, exactly like this package's other
// real-dispatch quant tests (requireNativeRuntime, test_helpers_test.go;
// e.g. TestQMVBF16AllocationBudget, qmv_test.go).
func TestNativeAffineQuantParity(t *testing.T) {
	requireNativeRuntime(t)
	enginetest.QuantParity(t, "native")
}
