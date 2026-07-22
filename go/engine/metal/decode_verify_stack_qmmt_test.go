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

// TestVerifyStackICBReplay_QMMTParity_Good is #71's decisive micro: the SAME
// affine qmm_t, dispatched three ways on synthetic weights — (a) the host
// per-row qmv truth, (b) the live encoder (encQMMTBF16At), (c) recorded into
// a verifyStackRecorder ICB and REPLAYED via executeInto — must agree at
// every live E-series geometry. The live E4B failure (all-NaN verify rows on
// the uniform-4-bit lean conversion, clean on lean E2B and on 8-bit-MLP qat)
// pins to the whole-stack replay, so the parity is swept across both models'
// MLP dims at bits 4 AND 8 at the MTP verify width.
func TestVerifyStackICBReplay_QMMTParity_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("MLX_METALLIB_PATH not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	const gs = 64
	const rows = 5
	cases := []struct {
		name          string
		outDim, inDim int
		bits          int
	}{
		{"e4b-gate-b4", 10240, 2560, 4},
		{"e4b-down-b4", 2560, 10240, 4},
		{"e4b-gate-b8", 10240, 2560, 8},
		{"e4b-down-b8", 2560, 10240, 8},
		{"e2b-gate-b4", 8192, 2048, 4},
		{"e2b-ctrl-b4", 2048, 2048, 4},
	}
	rng := uint32(0x2545f491)
	next := func() uint32 { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng }

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			groups := tc.inDim / gs
			wq := make([]byte, tc.outDim*tc.inDim*tc.bits/8)
			for i := range wq {
				wq[i] = byte(next())
			}
			scalesF := make([]float32, tc.outDim*groups)
			biasesF := make([]float32, tc.outDim*groups)
			for i := range scalesF {
				scalesF[i] = 0.01 + float32(i%97)*0.002
				biasesF[i] = -0.5 + float32(i%89)*0.011
			}
			scales := toBF16Bytes(scalesF)
			biases := toBF16Bytes(biasesF)
			xf := make([]float32, rows*tc.inDim)
			for i := range xf {
				xf[i] = (float32(next()%2000) - 1000) / 1000
			}
			xb := toBF16Bytes(xf)

			want := make([][]byte, rows)
			for r := range rows {
				out, err := QMVBF16(xb[r*tc.inDim*2:(r+1)*tc.inDim*2], wq, scales, biases, tc.outDim, tc.inDim, gs, tc.bits)
				if err != nil {
					t.Fatalf("QMVBF16 row %d: %v", r, err)
				}
				want[r] = out
			}

			outLen := rows * tc.outDim * 2
			wqBuf, scalesBuf, biasesBuf := residentBytes(wq), residentBytes(scales), residentBytes(biases)
			xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&xb[0]), uint(len(xb)), metal.MTLResourceStorageModeShared)
			outLive := device.NewBufferWithLengthOptions(uint(outLen), metal.MTLResourceStorageModeShared)
			outICB := device.NewBufferWithLengthOptions(uint(outLen), metal.MTLResourceStorageModeShared)

			// (b) live encoder.
			var encErr error
			withAutoreleasePool(func() {
				cb := commandBufferFast(queue)
				enc := computeCommandEncoderFast(cb)
				encErr = encQMMTBF16At(enc, wqBuf, scalesBuf, biasesBuf, xBuf, outLive, 0, 0, 0, 0, 0, rows, tc.outDim, tc.inDim, gs, tc.bits)
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
			})
			if encErr != nil {
				t.Fatalf("live encQMMTBF16At: %v", encErr)
			}

			// (c) record into a verifyStackRecorder ICB, then replay.
			rec := newVerifyStackRecorder(3, rows, verifyStackKey{}, nil)
			if rec == nil {
				t.Fatal("newVerifyStackRecorder returned nil")
			}
			rec.setLayer(1)
			rec.layerEntry()
			rec.recQMMT(wqBuf, scalesBuf, biasesBuf, xBuf, outICB, 0, 0, 0, 0, 0, rows, tc.outDim, tc.inDim, gs, tc.bits)
			if rec.failed {
				t.Fatal("recorder failed to record the qmm_t")
			}
			vs := rec.finish()
			if vs == nil {
				t.Fatal("finish returned nil — recording incomplete")
			}
			withAutoreleasePool(func() {
				cb := commandBufferFast(queue)
				enc := concurrentComputeEncoderFast(cb)
				vs.executeInto(enc, 0, nil)
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
			})

			live := unsafe.Slice((*byte)(outLive.Contents()), outLen)
			icb := unsafe.Slice((*byte)(outICB.Contents()), outLen)
			for lane, got := range map[string][]byte{"live": live, "icb-replay": icb} {
				worst, worstRow, nan := 2.0, -1, 0
				for r := range rows {
					var dot, ng, nw float64
					for i := range tc.outDim {
						o := (r*tc.outDim + i) * 2
						g := float64(bf16ToF32(got[o], got[o+1]))
						w := float64(bf16ToF32(want[r][i*2], want[r][i*2+1]))
						if math.IsNaN(g) {
							nan++
							continue
						}
						dot += g * w
						ng += g * g
						nw += w * w
					}
					cos := dot / (math.Sqrt(ng)*math.Sqrt(nw) + 1e-30)
					if cos < worst {
						worst, worstRow = cos, r
					}
				}
				t.Logf("%s %s: worst cos=%.6f@row%d nan=%d", tc.name, lane, worst, worstRow, nan)
				if worst < 0.999 || nan > 0 {
					t.Errorf("%s %s diverges from the per-row qmv truth: worst cos %.6f, %d NaN", tc.name, lane, worst, nan)
				}
			}
		})
	}
}
