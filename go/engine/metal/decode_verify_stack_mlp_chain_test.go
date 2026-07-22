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

// TestVerifyStackICBReplay_MLPChainParity_Good is #71's chain micro: the live
// bisect pinned the E4B first-replay NaN to the MLP block (fold commands
// 13..19 — mlp rms, gate, up, gelu, down, post-rms, residual add) at
// dFF 10240 / dModel 2560, entering the residual stream at the add; the same
// chain at E2B dims (8192/1536) replays clean. This records exactly that
// seven-op sequence (with the fold's up-after-gate overlap relaxation) on
// synthetic weights, replays it, and compares against the SAME PSOs encoded
// live — divergence or NaN in the replay lane reproduces the fault in-suite.
func TestVerifyStackICBReplay_MLPChainParity_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("MLX_METALLIB_PATH not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	const gs, bits, rows = 64, 4, 5
	const eps = 1e-6
	cases := []struct {
		name         string
		dModel, dFF  int
	}{
		{"e4b-2560x10240", 2560, 10240},
		{"e2b-1536x8192", 1536, 8192},
	}
	rng := uint32(0x9e3779b9)
	next := func() uint32 { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng }
	quantW := func(outDim, inDim int) (wq, scales, biases metal.MTLBuffer) {
		groups := inDim / gs
		wqb := make([]byte, outDim*inDim*bits/8)
		for i := range wqb {
			wqb[i] = byte(next())
		}
		sf := make([]float32, outDim*groups)
		bf := make([]float32, outDim*groups)
		for i := range sf {
			sf[i] = 0.001 + float32(i%97)*0.0002
			bf[i] = -0.05 + float32(i%89)*0.0011
		}
		return residentBytes(wqb), residentBytes(toBF16Bytes(sf)), residentBytes(toBF16Bytes(bf))
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gateWq, gateS, gateB := quantW(tc.dFF, tc.dModel)
			upWq, upS, upB := quantW(tc.dFF, tc.dModel)
			downWq, downS, downB := quantW(tc.dModel, tc.dFF)
			w1f := make([]float32, tc.dModel)
			w2f := make([]float32, tc.dModel)
			for i := range w1f {
				w1f[i] = -0.2 + float32(i%53)*0.01
				w2f[i] = -0.1 + float32(i%47)*0.008
			}
			w1 := residentBytes(toBF16Bytes(w1f))
			w2 := residentBytes(toBF16Bytes(w2f))
			xf := make([]float32, rows*tc.dModel)
			for i := range xf {
				xf[i] = (float32(next()%2000) - 1000) / 500
			}
			xb := toBF16Bytes(xf)
			xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&xb[0]), uint(len(xb)), metal.MTLResourceStorageModeShared)

			plan, ok := qmvRowsPlanFor(rows, tc.dFF, tc.dModel, gs, bits)
			if !ok {
				t.Fatal("no qmv rows plan at MLP dims")
			}
			planDown, okd := qmvRowsPlanFor(rows, tc.dModel, tc.dFF, gs, bits)
			if !okd {
				t.Fatal("no qmv rows plan at down dims")
			}
			t.Logf("plans: gate/up tiled=%v down tiled=%v", plan.tiled, planDown.tiled)

			newSlab := func(n int) metal.MTLBuffer {
				return device.NewBufferWithLengthOptions(uint(n*2), metal.MTLResourceStorageModeShared)
			}
			mkLane := func() (norm, gate, up, gated, down, scratch, out metal.MTLBuffer) {
				return newSlab(rows * tc.dModel), newSlab(rows * tc.dFF), newSlab(rows * tc.dFF),
					newSlab(rows * tc.dFF), newSlab(rows * tc.dModel), newSlab(rows * tc.dModel), newSlab(rows * tc.dModel)
			}

			rmsPSO, err := pipelineForICB(rmsKernelBF16(tc.dModel))
			if err != nil {
				t.Fatalf("rms pso: %v", err)
			}
			geluPSO, err := geluPipelineICB()
			if err != nil {
				t.Fatalf("gelu pso: %v", err)
			}
			addPSO, err := pipelineForICB("vv_Addbfloat16")
			if err != nil {
				t.Fatalf("add pso: %v", err)
			}
			gatherPSO := func(p qmvRowsPlan) metal.MTLComputePipelineState {
				if p.tiled {
					pso, ok := lthnQMVRowsPipelineICB(p.tiledKey)
					if !ok {
						t.Fatal("tiled pso missing")
					}
					return pso
				}
				pso, ok := lthnGatherQMVPipelineICB(p.gatherKey)
				if !ok {
					t.Fatal("gather pso missing")
				}
				return pso
			}
			gupPSO := gatherPSO(plan)
			downPSO := gatherPSO(planDown)
			lhs, rhs, okb := qmvRowsIndexBuffers()
			if !okb {
				t.Fatal("qmv rows index buffers unavailable")
			}

			// ---- live lane: serial encoder, same PSOs ----
			lNorm, lGate, lUp, lGated, lDown, lScratch, lOut := mkLane()
			withAutoreleasePool(func() {
				cb := commandBufferFast(queue)
				enc := computeCommandEncoderFast(cb)
				emitRMSNormRows(encSink{enc}, rmsPSO, xBuf, w1, lNorm, 0, 0, 0, tc.dModel, eps, rows, rmsThreadgroup(tc.dModel, rmsPSO))
				emitProj := func(p qmvRowsPlan, pso metal.MTLComputePipelineState, wq, s, b, in, out metal.MTLBuffer, outDim, inDim int) {
					if p.tiled {
						emitQMVRowsTiled(encSink{enc}, pso, wq, 0, s, 0, b, 0, in, 0, out, 0, inDim, outDim)
						return
					}
					emitLthnGatherQMVRoutes(encSink{enc}, pso, in, 0, wq, 0, s, 0, b, 0, lhs, rhs, 0, out, 0, outDim, inDim, gs, bits, 0, rows)
				}
				emitProj(plan, gupPSO, gateWq, gateS, gateB, lNorm, lGate, tc.dFF, tc.dModel)
				emitProj(plan, gupPSO, upWq, upS, upB, lNorm, lUp, tc.dFF, tc.dModel)
				emitBinary(encSink{enc}, geluPSO, lGate, 0, lUp, 0, lGated, 0, rows*tc.dFF)
				emitProj(planDown, downPSO, downWq, downS, downB, lGated, lDown, tc.dModel, tc.dFF)
				emitRMSNormRows(encSink{enc}, rmsPSO, lDown, w2, lScratch, 0, 0, 0, tc.dModel, eps, rows, rmsThreadgroup(tc.dModel, rmsPSO))
				emitBinary(encSink{enc}, addPSO, xBuf, 0, lScratch, 0, lOut, 0, rows*tc.dModel)
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
			})

			// ---- recorded lane: the recorder's own rec* mirrors, replayed ----
			rNorm, rGate, rUp, rGated, rDown, rScratch, rOut := mkLane()
			rec := newVerifyStackRecorder(3, rows, verifyStackKey{}, nil)
			if rec == nil {
				t.Fatal("newVerifyStackRecorder returned nil")
			}
			rec.setLayer(1)
			rec.layerEntry()
			rec.recRMSRows(xBuf, w1, rNorm, 0, 0, 0, rows, tc.dModel, eps)
			recProj := func(p qmvRowsPlan, pso metal.MTLComputePipelineState, wq, s, b, in, out metal.MTLBuffer, outDim, inDim int) {
				c, okc := rec.nextCmd()
				if !okc {
					t.Fatal("nextCmd failed")
				}
				if p.tiled {
					emitQMVRowsTiled(vsRecordSink{c, rec}, pso, wq, 0, s, 0, b, 0, in, 0, out, 0, inDim, outDim)
					return
				}
				emitLthnGatherQMVRoutes(vsRecordSink{c, rec}, pso, in, 0, wq, 0, s, 0, b, 0, lhs, rhs, 0, out, 0, outDim, inDim, gs, bits, 0, rows)
			}
			recProj(plan, gupPSO, gateWq, gateS, gateB, rNorm, rGate, tc.dFF, tc.dModel)
			rec.overlapNext() // the fold's proven up-after-gate sibling relaxation
			recProj(plan, gupPSO, upWq, upS, upB, rNorm, rUp, tc.dFF, tc.dModel)
			rec.recGeluGateMul(rGate, rUp, rGated, 0, 0, 0, rows*tc.dFF)
			recProj(planDown, downPSO, downWq, downS, downB, rGated, rDown, tc.dModel, tc.dFF)
			rec.recRMSRows(rDown, w2, rScratch, 0, 0, 0, rows, tc.dModel, eps)
			rec.recAdd(xBuf, 0, rScratch, 0, rOut, 0, rows*tc.dModel)
			if rec.failed {
				t.Fatal("recorder failed mid-chain")
			}
			vs := rec.finish()
			if vs == nil {
				t.Fatal("finish returned nil")
			}
			withAutoreleasePool(func() {
				cb := commandBufferFast(queue)
				enc := concurrentComputeEncoderFast(cb)
				vs.executeInto(enc, 0, nil)
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
			})

			// ---- verdicts: NaN census per stage, then live↔replay parity ----
			census := func(lane string, bufs map[string]metal.MTLBuffer) {
				for name, b := range bufs {
					n := int(b.Length()) / 2
					nan, inf, _, _, first := bf16BufStats(b, 0, n)
					if nan > 0 || inf > 0 {
						t.Errorf("%s %s: NaN=%d Inf=%d first=%d", lane, name, nan, inf, first)
					}
				}
			}
			census("live", map[string]metal.MTLBuffer{"norm": lNorm, "gate": lGate, "up": lUp, "gated": lGated, "down": lDown, "scratch": lScratch, "out": lOut})
			census("replay", map[string]metal.MTLBuffer{"norm": rNorm, "gate": rGate, "up": rUp, "gated": rGated, "down": rDown, "scratch": rScratch, "out": rOut})

			outLen := rows * tc.dModel * 2
			live := unsafe.Slice((*byte)(lOut.Contents()), outLen)
			icb := unsafe.Slice((*byte)(rOut.Contents()), outLen)
			var dot, ng, nw float64
			worst := 2.0
			for r := 0; r < rows; r++ {
				dot, ng, nw = 0, 0, 0
				for i := 0; i < tc.dModel; i++ {
					o := (r*tc.dModel + i) * 2
					g := float64(bf16ToF32(icb[o], icb[o+1]))
					w := float64(bf16ToF32(live[o], live[o+1]))
					dot += g * w
					ng += g * g
					nw += w * w
				}
				if cos := dot / (math.Sqrt(ng)*math.Sqrt(nw) + 1e-30); cos < worst {
					worst = cos
				}
			}
			t.Logf("%s live↔replay worst cos=%.6f", tc.name, worst)
			if worst < 0.9999 {
				t.Errorf("%s replay diverges from live: worst cos %.6f", tc.name, worst)
			}
		})
	}
}
