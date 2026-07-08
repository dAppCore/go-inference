// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// TestQMMTAt31BDims is #348's fold-projection micro-repro: the batched prefill projects
// q/k/v/gate/up/down through MLX's affine qmm_t (encQMMTBF16At, one weight pass over all
// rows), while the per-token step uses the verified per-row qmv. The two must agree on the
// same affine weight at EVERY live geometry — 31B's inDim 5376 (%512 = 256) is the first
// family member off the 512 grid, the same boundary class as the f32-QMV wrapper bug.
func TestQMMTAt31BDims(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("MLX_METALLIB_PATH not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	const gs, bits = 64, 4
	cases := []struct {
		name                string
		rows, outDim, inDim int
	}{
		{"e2b-ctrl-2048", 27, 2048, 2048},
		{"31b-kv-5376", 27, 4096, 5376},
		{"31b-q-5376", 27, 8192, 5376},
	}
	rng := uint32(0x9e3779b9)
	next := func() uint32 { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng }

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			groups := tc.inDim / gs
			wq := make([]byte, tc.outDim*tc.inDim/2) // 4-bit packed, two weights per byte
			for i := range wq {
				wq[i] = byte(next())
			}
			// per-group VARYING scales/biases — uniform values are blind to any scale/bias
			// group-indexing defect (every group reads the same value regardless of index).
			scalesF := make([]float32, tc.outDim*groups)
			biasesF := make([]float32, tc.outDim*groups)
			for i := range scalesF {
				scalesF[i] = 0.01 + float32(i%97)*0.002
				biasesF[i] = -0.5 + float32(i%89)*0.011
			}
			scales := toBF16Bytes(scalesF)
			biases := toBF16Bytes(biasesF)
			xf := make([]float32, tc.rows*tc.inDim)
			for i := range xf {
				xf[i] = (float32(next()%2000) - 1000) / 1000
			}
			xb := toBF16Bytes(xf)

			// truth: the verified per-row qmv on each row
			want := make([][]byte, tc.rows)
			for r := range tc.rows {
				out, err := QMVBF16(xb[r*tc.inDim*2:(r+1)*tc.inDim*2], wq, scales, biases, tc.outDim, tc.inDim, gs, bits)
				if err != nil {
					t.Fatalf("QMVBF16 row %d: %v", r, err)
				}
				want[r] = out
			}

			// the fold's qmm_t: all rows in one dispatch
			outLen := tc.rows * tc.outDim * 2
			xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&xb[0]), uint(len(xb)), metal.MTLResourceStorageModeShared)
			outBuf := device.NewBufferWithLengthOptions(uint(outLen), metal.MTLResourceStorageModeShared)
			var encErr error
			withAutoreleasePool(func() {
				cb := commandBufferFast(queue)
				enc := computeCommandEncoderFast(cb)
				encErr = encQMMTBF16At(enc, residentBytes(wq), residentBytes(scales), residentBytes(biases), xBuf, outBuf, 0, 0, 0, 0, 0, tc.rows, tc.outDim, tc.inDim, gs, bits)
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
			})
			if encErr != nil {
				t.Fatalf("encQMMTBF16At: %v", encErr)
			}
			got := unsafe.Slice((*byte)(outBuf.Contents()), outLen)

			worst, worstRow := 2.0, -1
			var l2Got, l2Want float64
			for r := range tc.rows {
				var dot, ng, nw float64
				for i := range tc.outDim {
					o := (r*tc.outDim + i) * 2
					g := float64(bf16ToF32(got[o], got[o+1]))
					w := float64(bf16ToF32(want[r][i*2], want[r][i*2+1]))
					dot += g * w
					ng += g * g
					nw += w * w
				}
				cos := dot / (math.Sqrt(ng)*math.Sqrt(nw) + 1e-30)
				l2Got += ng
				l2Want += nw
				if cos < worst {
					worst, worstRow = cos, r
				}
			}
			t.Logf("%s: worst row cos=%.6f@row%d l2(qmm)=%.2f l2(qmv)=%.2f", tc.name, worst, worstRow, math.Sqrt(l2Got), math.Sqrt(l2Want))
			if worst < 0.999 {
				t.Errorf("qmm_t diverges from qmv at rows=%d outDim=%d inDim=%d: worst cos %.6f", tc.rows, tc.outDim, tc.inDim, worst)
			}
		})
	}
}
