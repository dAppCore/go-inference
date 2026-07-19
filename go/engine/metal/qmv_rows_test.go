// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math/rand"
	"testing"
	"unsafe"
)

// withQMVRowsWide runs fn with the wide-tile lane forced on/off, restoring
// the process default after — the LTHN_QMV_ROWS_WIDE lever's test twin.
func withQMVRowsWide(t *testing.T, enabled bool, fn func()) {
	t.Helper()
	qmvRowsWideEnabled() // force the sync.Once so the override below sticks
	prev := qmvRowsWideOn
	qmvRowsWideOn = enabled
	defer func() { qmvRowsWideOn = prev }()
	fn()
}

// TestLthnQMVRowsWideByteBand gates the halved-tile wide kernel (M 5..8) at
// fast-twin dims: every output row through encQMVRowsBF16At must be
// BYTE-identical to the production per-row qmv — the same bar the flat M ≤ 4
// tile holds, on the shapes the MTP verify feeds (draft block + carry).
// Scales/biases vary per group (uniform values are blind to group/row
// indexing defects — the flat-tile test's lesson).
func TestLthnQMVRowsWideByteBand(t *testing.T) {
	requireNativeRuntime(t)
	withQMVRowsWide(t, true, func() { lthnQMVRowsWideByteBand(t) })
}

func lthnQMVRowsWideByteBand(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	const gs, bits = 64, 4
	for _, dims := range [][2]int{{512, 1024}, {3840, 7680}} { // fast-twin: outDim%8==0 && inDim%512==0
		outDim, inDim := dims[0], dims[1]
		packed := make([]byte, outDim*inDim/2)
		for i := range packed {
			packed[i] = byte(rng.Intn(256))
		}
		groups := inDim / gs
		scales := mtpShapeRandBF16(rng, outDim*groups)
		biases := mtpShapeRandBF16(rng, outDim*groups)
		for rows := lthnQMVRowsMaxM + 1; rows <= lthnQMVRowsWideMaxM; rows++ {
			plan, ok := qmvRowsPlanFor(rows, outDim, inDim, gs, bits)
			if !ok || !plan.tiled {
				t.Fatalf("out=%d in=%d rows=%d: wide plan not tiled (ok=%v tiled=%v)", outDim, inDim, rows, ok, plan.tiled)
			}
			x := mtpShapeRandBF16(rng, rows*inDim)
			want := make([]byte, rows*outDim*bf16Size)
			for m := range rows {
				row, err := QMVBF16(x[m*inDim*bf16Size:(m+1)*inDim*bf16Size], packed, scales, biases, outDim, inDim, gs, bits)
				if err != nil {
					t.Fatalf("QMVBF16 rows=%d m=%d: %v", rows, m, err)
				}
				copy(want[m*outDim*bf16Size:], row)
			}
			got := make([]byte, rows*outDim*bf16Size)
			var handled bool
			var encErr error
			withAutoreleasePool(func() {
				wqBuf, sBuf, bBuf := sharedBytes(packed), sharedBytes(scales), sharedBytes(biases)
				xBuf, outBuf := sharedBytes(x), scratchBF16(rows*outDim)
				cb := commandBufferFast(queue)
				enc := computeCommandEncoderFast(cb)
				handled, encErr = encQMVRowsBF16At(enc, wqBuf, sBuf, bBuf, xBuf, outBuf, 0, 0, 0, 0, 0, rows, outDim, inDim, gs, bits)
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
				copy(got, unsafe.Slice((*byte)(outBuf.Contents()), len(got)))
			})
			if encErr != nil {
				t.Fatalf("wide qmv out=%d rows=%d: %v", outDim, rows, encErr)
			}
			if !handled {
				t.Fatalf("wide qmv declined out=%d in=%d rows=%d", outDim, inDim, rows)
			}
			if nan, _ := bf16NaNScanBytes(got); nan > 0 {
				t.Fatalf("out=%d rows=%d produced %d NaN", outDim, rows, nan)
			}
			if !bytes.Equal(got, want) {
				bad := 0
				for i := 0; i < rows*outDim; i++ {
					if got[2*i] != want[2*i] || got[2*i+1] != want[2*i+1] {
						bad++
					}
				}
				t.Errorf("out=%d in=%d rows=%d: wide tile not byte-identical to per-row qmv (%d/%d elements differ)",
					outDim, inDim, rows, bad, rows*outDim)
			}
		}
	}
}

// TestQMVRowsTiledCap_Good pins the band rule both ways: wide lane on → cap 8
// and rows 5..8 plan tiled at fast-twin dims; wide lane off → cap 4 and rows
// 5..8 fall back to the gather plan (the pre-#53 route, the repro anchor).
func TestQMVRowsTiledCap_Good(t *testing.T) {
	requireNativeRuntime(t)
	const gs, bits, outDim, inDim = 64, 4, 512, 1024
	withQMVRowsWide(t, true, func() {
		if got := qmvRowsTiledCap(); got != lthnQMVRowsWideMaxM {
			t.Fatalf("wide-on cap = %d, want %d", got, lthnQMVRowsWideMaxM)
		}
		plan, ok := qmvRowsPlanFor(6, outDim, inDim, gs, bits)
		if !ok || !plan.tiled {
			t.Fatalf("wide-on rows=6 plan tiled = %v (ok=%v), want tiled", plan.tiled, ok)
		}
	})
	withQMVRowsWide(t, false, func() {
		if got := qmvRowsTiledCap(); got != lthnQMVRowsMaxM {
			t.Fatalf("wide-off cap = %d, want %d", got, lthnQMVRowsMaxM)
		}
		plan, ok := qmvRowsPlanFor(6, outDim, inDim, gs, bits)
		if !ok {
			t.Fatal("wide-off rows=6 plan missing (gather fallback expected)")
		}
		if plan.tiled {
			t.Fatal("wide-off rows=6 plan tiled, want the gather fallback")
		}
	})
}
