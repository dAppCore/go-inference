// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math/rand"
	"testing"
	"unsafe"
)

// TestPLEGateGeluQMVMatchesComposedBytes gates the fused PLE gate+gelu op's
// byte-identity contract (#373): the kernel is qmv_fast_impl verbatim with
// the gelu·pli product at the store, so its output must equal the composed
// pair (gate qmv → gelu·pli binary op) BYTE FOR BYTE — the property that lets
// the ICB record it without touching the re-encode twin.
func TestPLEGateGeluQMVMatchesComposedBytes(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, pliDim, gs, bits = 1536, 256, 64, 4
	rng := rand.New(rand.NewSource(373))

	packed := make([]byte, pliDim*dModel*bits/8)
	for i := range packed {
		packed[i] = byte(rng.Intn(256))
	}
	scales := mtpShapeRandBF16(rng, pliDim*dModel/gs)
	biases := mtpShapeRandBF16(rng, pliDim*dModel/gs)
	x := mtpShapeRandBF16(rng, dModel)
	pli := mtpShapeRandBF16(rng, pliDim)

	run := func(fused bool) []byte {
		got := make([]byte, pliDim*bf16Size)
		var encErr error
		withAutoreleasePool(func() {
			wqBuf, sBuf, bBuf := sharedBytes(packed), sharedBytes(scales), sharedBytes(biases)
			xBuf, pliBuf := sharedBytes(x), sharedBytes(pli)
			gateBuf := scratchBF16(pliDim)
			outBuf := scratchBF16(pliDim)
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			if fused {
				encErr = encPLEGateGeluQMV(enc, qmvWeight{
					wq:     bufView{buf: wqBuf},
					scales: bufView{buf: sBuf},
					biases: bufView{buf: bBuf},
					gs:     gs, bits: bits,
				}, xBuf, pliBuf, 0, outBuf, dModel, pliDim, gs, bits)
			} else {
				encErr = encQMVBF16(enc, wqBuf, sBuf, bBuf, xBuf, gateBuf, 0, 0, 0, 0, pliDim, dModel, gs, bits)
				if encErr == nil {
					encErr = encGeluGateMulFusedTo(enc, gateBuf, pliBuf, outBuf, 0, 0, 0, pliDim)
				}
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			copy(got, unsafe.Slice((*byte)(outBuf.Contents()), len(got)))
		})
		if encErr != nil {
			t.Fatalf("encode (fused=%v): %v", fused, encErr)
		}
		return got
	}

	want := run(false)
	got := run(true)
	if !bytes.Equal(got, want) {
		diff := 0
		for i := range want {
			if want[i] != got[i] {
				diff++
			}
		}
		t.Fatalf("fused PLE gate+gelu diverges from the composed pair: %d/%d bytes differ — the qmv_fast_impl port is not verbatim", diff, len(want))
	}
	t.Logf("fused PLE gate+gelu output == composed pair, byte-for-byte (%d out values, gs=%d b=%d)", pliDim, gs, bits)
}
