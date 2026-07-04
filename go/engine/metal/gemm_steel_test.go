// SPDX-Licence-Identifier: EUPL-1.2

package native

import (
	"testing"
	"unsafe"
)

// TestEncGemvBF16BatchedAtSteelGEMMEngagesAndMatchesGemv pins the true GEMM fold (#252): at
// steelGEMMMinRows and above the batched projections route to MLX's steel_gemm_fused kernel
// (the weight read ONCE for all rows), and its per-element outputs agree with the grid-Z gemv
// within bf16 accumulation-order tolerance — the token-identity trade the large-row prefill
// makes, checked on both the tile-aligned and the bounds-checked (unaligned M/N/K) paths.
// Engagement is asserted via the steel dispatch counter: a GEMM and a gemv are one dispatch
// each, so plain dispatch counts cannot tell them apart.
func TestEncGemvBF16BatchedAtSteelGEMMEngagesAndMatchesGemv(t *testing.T) {
	requireNativeRuntime(t)
	shapes := []struct{ rows, outDim, inDim int }{
		{steelGEMMMinRows, 128, 64}, // fully tile-aligned (align_M/N/K fast path)
		{88, 96, 72},                // unaligned M, N and K — the bounds-checked path
	}
	for _, sh := range shapes {
		w := toBF16Bytes(syntheticFloat32(sh.outDim*sh.inDim, 31))
		x := toBF16Bytes(syntheticFloat32(sh.rows*sh.inDim, 47))

		run := func(disable bool) ([]float32, int64) {
			t.Helper()
			prev, prevTiming := steelGEMMDisabledForTest, pieceTimingOn
			steelGEMMDisabledForTest = disable
			pieceTimingOn = true
			steelGEMMDispatchesForTest = 0
			defer func() {
				steelGEMMDisabledForTest = prev
				pieceTimingOn = prevTiming
			}()
			outBytes := make([]byte, sh.rows*sh.outDim*bf16Size)
			var encErr error
			withAutoreleasePool(func() {
				wBuf := residentBytes(w)
				xBuf := residentBytes(x)
				oBuf := scratchBF16(sh.rows * sh.outDim)
				cb := commandBufferFast(queue)
				enc := computeCommandEncoderFast(cb)
				encErr = encGemvBF16BatchedAt(enc, wBuf, xBuf, oBuf, 0, 0, 0, sh.outDim, sh.inDim, sh.rows)
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
				copy(outBytes, unsafe.Slice((*byte)(oBuf.Contents()), len(outBytes)))
			})
			if encErr != nil {
				t.Fatalf("encGemvBF16BatchedAt (%+v, disableSteel=%v): %v", sh, disable, encErr)
			}
			out := make([]float32, sh.rows*sh.outDim)
			bf16ToF32Into(out, outBytes)
			return out, steelGEMMDispatchesForTest
		}

		steel, steelDispatches := run(false)
		gemv, gemvDispatches := run(true)
		if steelDispatches == 0 {
			t.Fatalf("steel GEMM did not engage for %+v (dispatch counter stayed 0)", sh)
		}
		if gemvDispatches != 0 {
			t.Fatalf("kill switch leaked: gemv run counted %d steel dispatches for %+v", gemvDispatches, sh)
		}
		// bf16 accumulation-order tolerance: a few ulps (bf16 ulp ≈ 0.4% relative). A layout or
		// transpose bug produces values wrong by orders of magnitude, far outside this band.
		for i := range steel {
			ref := gemv[i]
			diff := steel[i] - ref
			if diff < 0 {
				diff = -diff
			}
			limit := 0.03 * absf32(ref)
			if limit < 1e-2 {
				limit = 1e-2
			}
			if diff > limit {
				t.Fatalf("steel GEMM diverges from gemv at %+v element %d: steel=%g gemv=%g (|diff|=%g > %g)", sh, i, steel[i], ref, diff, limit)
			}
		}
	}
}

func absf32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
