// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// TestPLEGateGeluRowsChainMatchesComposed gates the three-dispatch PLE
// epilogue (#372: gate+gelu fused, proj qmm_t, rms+add fused) against the
// composed five-dispatch chain it replaces, on synthetic quant weights over K
// rows. The rms+add half is byte-identical by construction (the proven
// lthn_rmsnorm_residual kernel, rows-widened grid); the gate half moves from
// qmm_t's MMA accumulation to the qgemv order — the fold's token-identity
// tier — so the gate is tolerance + cosine over every element.
func TestPLEGateGeluRowsChainMatchesComposed(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, pliDim, gs, bits, rows = 2048, 256, 64, 4, 5
	const eps = 1e-6
	rng := rand.New(rand.NewSource(372))

	gatePacked := make([]byte, pliDim*dModel*bits/8)
	projPacked := make([]byte, dModel*pliDim*bits/8)
	for i := range gatePacked {
		gatePacked[i] = byte(rng.Intn(256))
	}
	for i := range projPacked {
		projPacked[i] = byte(rng.Intn(256))
	}
	gateScales := mtpShapeRandBF16(rng, pliDim*dModel/gs)
	gateBiases := mtpShapeRandBF16(rng, pliDim*dModel/gs)
	projScales := mtpShapeRandBF16(rng, dModel*pliDim/gs)
	projBiases := mtpShapeRandBF16(rng, dModel*pliDim/gs)
	normW := mtpShapeRandBF16(rng, dModel)
	outRows := mtpShapeRandBF16(rng, rows*dModel)
	pleSlab := mtpShapeRandBF16(rng, rows*pliDim)

	run := func(fused bool) []byte {
		got := make([]byte, rows*dModel*bf16Size)
		var encErr error
		var handled bool
		withAutoreleasePool(func() {
			gpBuf, gsBuf, gbBuf := sharedBytes(gatePacked), sharedBytes(gateScales), sharedBytes(gateBiases)
			ppBuf, psBuf, pbBuf := sharedBytes(projPacked), sharedBytes(projScales), sharedBytes(projBiases)
			nwBuf, pleBuf := sharedBytes(normW), sharedBytes(pleSlab)
			outBuf := sharedBytes(outRows) // fresh copy each run — the chain writes in place
			gateSlab := scratchBF16(rows * pliDim)
			multSlab := scratchBF16(rows * pliDim)
			projSlab := scratchBF16(rows * dModel)
			normSlab := scratchBF16(rows * dModel)
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			fail := func(err error) {
				if err != nil && encErr == nil {
					encErr = err
				}
			}
			if fused {
				handled, encErr = encPLEGateGeluRows(enc,
					bufView{buf: gpBuf}, bufView{buf: gsBuf}, bufView{buf: gbBuf},
					outBuf, 0, pleBuf, 0, multSlab, 0, rows, dModel, pliDim, gs, bits)
				if encErr == nil && handled {
					fail(encQMMTBF16At(enc, ppBuf, psBuf, pbBuf, multSlab, projSlab, 0, 0, 0, 0, 0, rows, dModel, pliDim, gs, bits))
					fail(encRMSNormResidualRowsBF16At(enc, projSlab, nwBuf, outBuf, outBuf, 0, 0, 0, 0, rows, dModel, eps))
				}
			} else {
				handled = true
				fail(encQMMTBF16At(enc, gpBuf, gsBuf, gbBuf, outBuf, gateSlab, 0, 0, 0, 0, 0, rows, pliDim, dModel, gs, bits))
				fail(encGeluGateMulFusedTo(enc, gateSlab, pleBuf, multSlab, 0, 0, 0, rows*pliDim))
				fail(encQMMTBF16At(enc, ppBuf, psBuf, pbBuf, multSlab, projSlab, 0, 0, 0, 0, 0, rows, dModel, pliDim, gs, bits))
				fail(encRMSNormRowsBF16(enc, projSlab, nwBuf, normSlab, 0, 0, 0, rows, dModel, eps))
				fail(encAddBF16To(enc, outBuf, normSlab, outBuf, 0, 0, 0, rows*dModel))
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			copy(got, unsafe.Slice((*byte)(outBuf.Contents()), len(got)))
		})
		if encErr != nil {
			t.Fatalf("encode (fused=%v): %v", fused, encErr)
		}
		if !handled {
			t.Fatalf("encPLEGateGeluRows declined — kernel missing from metallib?")
		}
		return got
	}

	want := run(false)
	got := run(true)

	var maxDiff, dot, nw, ng float64
	for i := 0; i < rows*dModel; i++ {
		w := float64(bf16ToF32(want[2*i], want[2*i+1]))
		g := float64(bf16ToF32(got[2*i], got[2*i+1]))
		if d := math.Abs(g - w); d > maxDiff {
			maxDiff = d
		}
		dot += w * g
		nw += w * w
		ng += g * g
	}
	cos := dot / (math.Sqrt(nw)*math.Sqrt(ng) + 1e-30)
	t.Logf("three-dispatch PLE chain vs composed (rows=%d dModel=%d pliDim=%d gs=%d b=%d): maxDiff=%.5f cosine=%.7f", rows, dModel, pliDim, gs, bits, maxDiff, cos)
	if cos < 0.99999 || maxDiff > 0.25 {
		t.Fatalf("fused PLE chain diverges from the composed chain: maxDiff=%.5f cosine=%.7f", maxDiff, cos)
	}
}
