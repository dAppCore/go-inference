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

// TestQMVRowsTiledKeyFor_Good pins THE tiled-band rule on both envelopes: fast
// dims (outDim%8==0 && inDim%512==0) resolve the fast twin (general=false);
// non-fast dims resolve the general twin (qmv_impl's M-variant) at rows 2..4.
func TestQMVRowsTiledKeyFor_Good(t *testing.T) {
	for m := 2; m <= lthnQMVRowsMaxM; m++ {
		key, ok := qmvRowsTiledKeyFor(m, 512, 1024, 64, 4)
		if !ok || key.general || key.m != m || key.groupSize != 64 || key.bits != 4 {
			t.Fatalf("fast m=%d: key=%+v ok=%v, want the fast key", m, key, ok)
		}
		key, ok = qmvRowsTiledKeyFor(m, 704, 2816, 64, 4)
		if !ok || !key.general || key.m != m {
			t.Fatalf("general m=%d: key=%+v ok=%v, want the general key", m, key, ok)
		}
	}
}

// TestQMVRowsTiledKeyFor_Bad pins the band edges: m < 2 never keys (rows==1 is
// the plain per-row qmv), and the general tier never exceeds the flat cap.
func TestQMVRowsTiledKeyFor_Bad(t *testing.T) {
	if _, ok := qmvRowsTiledKeyFor(1, 704, 2816, 64, 4); ok {
		t.Fatal("m=1 keyed — the tiled band starts at 2")
	}
	withQMVRowsWide(t, false, func() {
		if _, ok := qmvRowsTiledKeyFor(lthnQMVRowsMaxM+1, 704, 2816, 64, 4); ok {
			t.Fatal("general m=5 keyed — the general tier is flat 2..4 only")
		}
		if _, ok := qmvRowsTiledKeyFor(lthnQMVRowsMaxM+1, 512, 1024, 64, 4); ok {
			t.Fatal("fast m=5 keyed with the wide lane off — cap is 4")
		}
	})
}

// TestQMVRowsTiledKeyFor_Ugly pins the wide-lane asymmetry: arming
// LTHN_QMV_ROWS_WIDE raises the FAST cap to 8 but never the general one —
// unaligned rows 5..8 have no single tiled dispatch and stay chunk-composed
// (qmvByteExactServable's fall-through), because there is no general wide
// kernel (docs/design-qmv-rows-unaligned.md records why).
func TestQMVRowsTiledKeyFor_Ugly(t *testing.T) {
	withQMVRowsWide(t, true, func() {
		key, ok := qmvRowsTiledKeyFor(6, 512, 1024, 64, 4)
		if !ok || key.general {
			t.Fatalf("wide-armed fast m=6: key=%+v ok=%v, want the fast wide key", key, ok)
		}
		if _, ok := qmvRowsTiledKeyFor(6, 704, 2816, 64, 4); ok {
			t.Fatal("wide-armed general m=6 keyed — no general wide variant exists")
		}
	})
}

// TestLthnQMVRowsGeneralParity gates the general tiled tier
// (lthn_qmv_rows_general — qmv_impl's M-variant) at NON-fast dims: every output
// row must be BYTE-identical to the production per-row qmv, whose oracle at
// these dims IS qmv_impl (qmvBF16KernelName routes every non-fast dim there).
// The sweep carries the real gemma4 26B-A4B MoE projections (2816/704/2112 —
// the geometry whose up-front decline motivated the unaligned tier), the
// 1408-class routed width, and the two ragged-out kernel branches (outDim 706 →
// the moved-back last tile, outDim 6 → the fully-guarded small-out branch),
// across every gs/bits pair the metallib instantiates (pairs with inDim%gs != 0
// are not quantisable and skip). Rows 2..4 ride the single tiled dispatch
// through encQMVRowsBF16At — the plan must report the GENERAL tiled key (a fast
// key here means the envelope rule broke); rows 5..8 byte-assert the chunked
// composition (encQMVRowsBF16ChunkedAt — the byte-tier fold/MoE entry, now
// serving unaligned dims). qmvByteExactServable must agree at every row count —
// it is the probe the MoE driver's eligibility walk consults. Scales/biases
// vary per group (uniform values are blind to group/row indexing defects — the
// flat-tile test's lesson).
func TestLthnQMVRowsGeneralParity(t *testing.T) {
	requireNativeRuntime(t)
	rng := rand.New(rand.NewSource(31))
	shapes := [][2]int{ // {outDim, inDim}, all NON-fast
		{704, 2816},  // real 26B expert gate/up (outDim=expertDFF, inDim=dModel)
		{2816, 704},  // real 26B expert down
		{2112, 2816}, // real 26B local gate/up
		{2816, 2112}, // real 26B local down
		{2560, 1408}, // 1408-class inDim (the Qwen1.5-MoE routed width)
		{706, 1408},  // outDim%8 != 0: the moved-back last out-tile branch
		{6, 1408},    // outDim < 8: the fully-guarded small-out branch
	}
	for _, gb := range [][2]int{{32, 4}, {32, 8}, {64, 4}, {64, 8}, {128, 4}, {128, 8}} {
		gs, bits := gb[0], gb[1]
		for _, dims := range shapes {
			outDim, inDim := dims[0], dims[1]
			if inDim%gs != 0 {
				continue // not quantisable at this group size (704/2112 at gs=128)
			}
			if outDim%8 == 0 && inDim%512 == 0 {
				t.Fatalf("sweep shape out=%d in=%d rides the FAST envelope — it belongs in TestLthnQMVRowsParity", outDim, inDim)
			}
			packed := make([]byte, outDim*inDim*bits/8)
			for i := range packed {
				packed[i] = byte(rng.Intn(256))
			}
			groups := inDim / gs
			scales := mtpShapeRandBF16(rng, outDim*groups)
			biases := mtpShapeRandBF16(rng, outDim*groups)
			for rows := 2; rows <= 2*lthnQMVRowsMaxM; rows++ {
				if !qmvByteExactServable(rows, outDim, inDim, gs, bits) {
					t.Fatalf("gs=%d b=%d out=%d in=%d rows=%d: qmvByteExactServable=false — the eligibility probe would decline this geometry", gs, bits, outDim, inDim, rows)
				}
				tiledBand := rows <= lthnQMVRowsMaxM
				if tiledBand {
					plan, ok := qmvRowsPlanFor(rows, outDim, inDim, gs, bits)
					if !ok || !plan.tiled || !plan.tiledKey.general {
						t.Fatalf("gs=%d b=%d out=%d in=%d rows=%d: plan not general-tiled (ok=%v tiled=%v general=%v)", gs, bits, outDim, inDim, rows, ok, plan.tiled, plan.tiledKey.general)
					}
				}
				x := mtpShapeRandBF16(rng, rows*inDim)
				want := make([]byte, rows*outDim*bf16Size)
				for m := range rows {
					row, err := QMVBF16(x[m*inDim*bf16Size:(m+1)*inDim*bf16Size], packed, scales, biases, outDim, inDim, gs, bits)
					if err != nil {
						t.Fatalf("QMVBF16 gs=%d b=%d out=%d rows=%d m=%d: %v", gs, bits, outDim, rows, m, err)
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
					if tiledBand {
						handled, encErr = encQMVRowsBF16At(enc, wqBuf, sBuf, bBuf, xBuf, outBuf, 0, 0, 0, 0, 0, rows, outDim, inDim, gs, bits)
					} else {
						handled, encErr = encQMVRowsBF16ChunkedAt(enc, wqBuf, sBuf, bBuf, xBuf, outBuf, 0, 0, 0, 0, 0, rows, outDim, inDim, gs, bits)
					}
					endEncodingFast(enc)
					commitCommandBufferFast(cb)
					waitUntilCompletedFast(cb)
					copy(got, unsafe.Slice((*byte)(outBuf.Contents()), len(got)))
				})
				if encErr != nil {
					t.Fatalf("general qmv gs=%d b=%d out=%d in=%d rows=%d: %v", gs, bits, outDim, inDim, rows, encErr)
				}
				if !handled {
					t.Fatalf("general qmv declined gs=%d b=%d out=%d in=%d rows=%d", gs, bits, outDim, inDim, rows)
				}
				if nan, _ := bf16NaNScanBytes(got); nan > 0 {
					t.Fatalf("gs=%d b=%d out=%d rows=%d produced %d NaN", gs, bits, outDim, rows, nan)
				}
				if !bytes.Equal(got, want) {
					bad := 0
					for i := 0; i < rows*outDim; i++ {
						if got[2*i] != want[2*i] || got[2*i+1] != want[2*i+1] {
							bad++
						}
					}
					t.Errorf("gs=%d b=%d out=%d in=%d rows=%d: general tier not byte-identical to per-row qmv (%d/%d elements differ)", gs, bits, outDim, inDim, rows, bad, rows*outDim)
				}
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
