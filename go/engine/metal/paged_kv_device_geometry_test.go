// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// TestDevicePagedKVGeometrySchedule pins the geometric page schedule's closed
// forms against a brute-force prefix walk: page capacities double from the
// base to the cap then stay flat, starts are their prefix sums, and every
// position maps back to the page whose [start, start+rows) range holds it.
func TestDevicePagedKVGeometrySchedule(t *testing.T) {
	for _, base := range []int{512, 2048, 16384, 32768} {
		c := &devicePagedKVCache{pageSize: base}
		start := 0
		for page := 0; page < 24; page++ {
			rows := c.pageRowsFor(page)
			wantRows := base << page
			if wantRows > maxPagedKVPageRows || wantRows < base { // < base guards shift overflow
				wantRows = maxPagedKVPageRows
			}
			if base >= maxPagedKVPageRows {
				wantRows = base
			}
			if rows != wantRows {
				t.Fatalf("base %d page %d rows = %d, want %d", base, page, rows, wantRows)
			}
			if got := c.pageStartFor(page); got != start {
				t.Fatalf("base %d page %d start = %d, want %d", base, page, got, start)
			}
			for _, pos := range []int{start, start + rows/2, start + rows - 1} {
				if got := c.pageForPos(pos); got != page {
					t.Fatalf("base %d pos %d page = %d, want %d", base, pos, got, page)
				}
			}
			start += rows
		}
	}
}

// TestDevicePagedKVGeometryRoundTrip writes rows across the doubling
// boundaries through slot(), snapshots to linear, reloads, truncates, and
// checks the visible lengths — the five call sites that consulted the fixed
// page math now share the schedule.
func TestDevicePagedKVGeometryRoundTrip(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const (
		nKVHeads = 2
		headDim  = 32
		rows     = 5000 // crosses the 2048 and 2048+4096=6144 boundaries? 5000 crosses page0->page1
	)
	c, err := newDevicePagedKVCache(nKVHeads, headDim, 0, 2048)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	rowBytes := nKVHeads * headDim * bf16Size
	for pos := range rows {
		kPage, vPage, off, serr := c.slot(pos)
		if serr != nil {
			t.Fatalf("slot(%d): %v", pos, serr)
		}
		kRow := unsafe.Slice((*byte)(unsafe.Add(kPage.Contents(), uintptr(off))), rowBytes)
		vRow := unsafe.Slice((*byte)(unsafe.Add(vPage.Contents(), uintptr(off))), rowBytes)
		for i := range rowBytes {
			kRow[i] = byte(pos + i)
			vRow[i] = byte(pos ^ i)
		}
	}
	if len(c.kPages) != 2 { // 2048 + 4096 covers 5000
		t.Fatalf("pages = %d, want 2 (geometric)", len(c.kPages))
	}
	if c.pageLens[0] != 2048 || c.pageLens[1] != 5000-2048 {
		t.Fatalf("pageLens = %v", c.pageLens)
	}
	_, _, kPtr, _, err := c.linearSnapshot(rows)
	if err != nil {
		t.Fatalf("linearSnapshot: %v", err)
	}
	lin := unsafe.Slice(kPtr, rows*rowBytes)
	for _, pos := range []int{0, 2047, 2048, 4095, 4999} {
		off := pos * rowBytes
		for i := range 4 {
			if lin[off+i] != byte(pos+i) {
				t.Fatalf("snapshot row %d byte %d = %d, want %d", pos, i, lin[off+i], byte(pos+i))
			}
		}
	}
	// reload the snapshot into a FRESH cache and spot-check via its own snapshot
	c2, err := newDevicePagedKVCache(nKVHeads, headDim, 0, 2048)
	if err != nil {
		t.Fatalf("new c2: %v", err)
	}
	vLin := unsafe.Slice(c.snapshotVPtr, rows*rowBytes)
	if err := c2.loadLinearSnapshot(lin, vLin, rows); err != nil {
		t.Fatalf("loadLinearSnapshot: %v", err)
	}
	_, _, k2Ptr, _, err := c2.linearSnapshot(rows)
	if err != nil {
		t.Fatalf("c2 snapshot: %v", err)
	}
	lin2 := unsafe.Slice(k2Ptr, rows*rowBytes)
	for _, pos := range []int{0, 2048, 4999} {
		off := pos * rowBytes
		if lin2[off] != byte(pos) {
			t.Fatalf("reloaded row %d byte 0 = %d, want %d", pos, lin2[off], byte(pos))
		}
	}
	// truncate back across the boundary and re-extend
	if err := c.truncate(2048); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	if c.pageLens[0] != 2048 || c.pageLens[1] != 0 || c.length != 2048 {
		t.Fatalf("post-truncate pageLens=%v len=%d", c.pageLens, c.length)
	}
	if _, _, _, err := c.slot(2048); err != nil {
		t.Fatalf("re-extend slot: %v", err)
	}
	if c.pageLens[1] != 1 {
		t.Fatalf("re-extend pageLens = %v", c.pageLens)
	}
}

func TestDevicePagedKVCache_Close_Good(t *testing.T) {
	c := &devicePagedKVCache{
		kPages: []metal.MTLBuffer{nil}, vPages: []metal.MTLBuffer{nil},
		kPagePtrs: []*byte{nil}, vPagePtrs: []*byte{nil}, pageLens: []int{1},
		keyScratch: []metal.MTLBuffer{nil}, valueScratch: []metal.MTLBuffer{nil}, lensScratch: []int{1},
		kHeadStrides: []int{1}, kSeqStrides: []int{2}, vHeadStrides: []int{3}, vSeqStrides: []int{4},
		kScalePages: []metal.MTLBuffer{nil}, vScalePages: []metal.MTLBuffer{nil},
		kScalePtrs: []*byte{nil}, vScalePtrs: []*byte{nil},
		kScaleScratch: []metal.MTLBuffer{nil}, vScaleScratch: []metal.MTLBuffer{nil},
		snapshotBytes: 9, sdpaScratch: []*sdpaPagedDecodeScratch{{}}, sdpaScratchCursor: 1,
		length: 3, offset: 4, linearSynced: 2,
	}
	c.Close()
	if c.kPages != nil || c.vPages != nil || c.pageLens != nil || c.kScalePages != nil || c.vScalePages != nil || c.sdpaScratch != nil {
		t.Fatalf("Close retained page ownership: %#v", c)
	}
	if c.snapshotBytes != 0 || c.sdpaScratchCursor != 0 || c.length != 0 || c.offset != 0 || c.linearSynced != 0 {
		t.Fatalf("Close retained scalar state: %#v", c)
	}
	var nilCache *devicePagedKVCache
	nilCache.Close()
}

func TestDevicePagedKVCache_Slot_Bad(t *testing.T) {
	var nilCache *devicePagedKVCache
	if _, _, _, err := nilCache.slot(0); err == nil {
		t.Fatal("slot accepted a nil cache")
	}
	c := &devicePagedKVCache{maxSize: 2, pageSize: 1}
	if _, _, _, err := c.slot(-1); err == nil {
		t.Fatal("slot accepted a negative position")
	}
	if _, _, _, err := c.slot(2); err == nil {
		t.Fatal("slot accepted a position at maxSize")
	}
}

func TestDevicePagedKVCache_LinearSnapshot_Bad(t *testing.T) {
	var nilCache *devicePagedKVCache
	if _, _, _, _, err := nilCache.linearSnapshot(1); err == nil {
		t.Fatal("linearSnapshot accepted a nil cache")
	}
	if _, _, _, _, err := (&devicePagedKVCache{length: 2}).linearSnapshot(1); err == nil {
		t.Fatal("linearSnapshot accepted fewer rows than the cache length")
	}
	if _, _, _, _, err := (&devicePagedKVCache{}).linearSnapshot(-1); err == nil {
		t.Fatal("linearSnapshot accepted a negative row count")
	}
}

func TestDevicePagedKVCache_LoadLinearSnapshot_Bad(t *testing.T) {
	var nilCache *devicePagedKVCache
	if err := nilCache.loadLinearSnapshot(nil, nil, 0); err == nil {
		t.Fatal("loadLinearSnapshot accepted a nil cache")
	}
	c := &devicePagedKVCache{kvDim: 2, maxSize: 2}
	if err := c.loadLinearSnapshot(nil, nil, -1); err == nil {
		t.Fatal("loadLinearSnapshot accepted negative tokens")
	}
	if err := c.loadLinearSnapshot(nil, nil, 3); err == nil {
		t.Fatal("loadLinearSnapshot accepted tokens beyond maxSize")
	}
	if err := c.loadLinearSnapshot([]byte{0}, []byte{0}, 1); err == nil {
		t.Fatal("loadLinearSnapshot accepted short snapshot rows")
	}
}

func TestDevicePagedKVCache_Truncate_Good(t *testing.T) {
	c := &devicePagedKVCache{
		pageSize: 2, pageLens: []int{2, 4, 6}, length: 12, offset: 12, linearSynced: 9,
	}
	if err := c.truncate(4); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	// The geometric schedule gives page starts 0, 2, 6. Token 4 fully keeps
	// page 0, partly keeps page 1, and clears every later page.
	if got, want := c.pageLens, []int{2, 2, 0}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("truncate pageLens = %v, want %v", got, want)
	}
	if c.length != 4 || c.offset != 4 || c.linearSynced != 4 {
		t.Fatalf("truncate state = len %d offset %d synced %d, want 4/4/4", c.length, c.offset, c.linearSynced)
	}
}

func TestDevicePagedKVCache_Truncate_Bad(t *testing.T) {
	var nilCache *devicePagedKVCache
	if err := nilCache.truncate(0); err == nil {
		t.Fatal("truncate accepted a nil cache")
	}
	c := &devicePagedKVCache{maxSize: 2, length: 1}
	if err := c.truncate(-1); err == nil {
		t.Fatal("truncate accepted negative tokens")
	}
	if err := c.truncate(3); err == nil {
		t.Fatal("truncate accepted tokens beyond maxSize")
	}
	if err := c.truncate(2); err == nil {
		t.Fatal("truncate extended a cache")
	}
}

func TestDevicePagedKVCache_State_Good(t *testing.T) {
	c := &devicePagedKVCache{
		kPages: []metal.MTLBuffer{nil}, vPages: []metal.MTLBuffer{nil}, pageLens: []int{3},
		headDim: 8, kvDim: 16,
	}
	keys, values, lens, kHead, kSeq, vHead, vSeq, err := c.state()
	if err != nil {
		t.Fatalf("state: %v", err)
	}
	if len(keys) != 1 || len(values) != 1 || lens[0] != 3 || kHead[0] != 8 || kSeq[0] != 16 || vHead[0] != 8 || vSeq[0] != 16 {
		t.Fatalf("state = keys=%d values=%d lens=%v k=(%v,%v) v=(%v,%v)", len(keys), len(values), lens, kHead, kSeq, vHead, vSeq)
	}
}

func TestDevicePagedKVCache_State_Bad(t *testing.T) {
	if _, _, _, _, _, _, _, err := (&devicePagedKVCache{}).state(); err == nil {
		t.Fatal("state accepted empty page slices")
	}
	if _, _, _, _, _, _, _, err := (&devicePagedKVCache{kPages: []metal.MTLBuffer{nil}, vPages: []metal.MTLBuffer{nil}}).state(); err == nil {
		t.Fatal("state accepted a missing page length")
	}
}

func TestDevicePagedKVCache_ScaleState_Good(t *testing.T) {
	c := &devicePagedKVCache{
		quantQ8: true, kPages: []metal.MTLBuffer{nil},
		kScalePages: []metal.MTLBuffer{nil}, vScalePages: []metal.MTLBuffer{nil},
	}
	k, v := c.scaleState()
	if len(k) != 1 || len(v) != 1 {
		t.Fatalf("scaleState lengths = %d/%d, want 1/1", len(k), len(v))
	}
	if k[0] != c.kScalePages[0] || v[0] != c.vScalePages[0] {
		t.Fatal("scaleState did not preserve scale-page membership")
	}
}

func TestDevicePagedKVCache_ScaleState_Bad(t *testing.T) {
	if k, v := (&devicePagedKVCache{}).scaleState(); k != nil || v != nil {
		t.Fatal("scaleState returned scales for a non-quantised cache")
	}
	if k, v := (&devicePagedKVCache{quantQ8: true, kPages: []metal.MTLBuffer{nil}}).scaleState(); k != nil || v != nil {
		t.Fatal("scaleState returned scales with missing scale pages")
	}
}

func TestEncAttnHalfKVPaged_Bad(t *testing.T) {
	var enc metal.MTLComputeCommandEncoderObject
	var cb metal.MTLCommandBufferObject
	got, concurrent, err := encAttnHalfKVPaged(enc, cb, nil, false, nil, nil, nil, nil, 0, bufView{}, bufView{}, bufView{}, bufView{}, nil, attnScratch{}, nil, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, nil)
	if err == nil {
		t.Fatal("encAttnHalfKVPaged accepted a sliding window without a cache")
	}
	if got != enc || concurrent {
		t.Fatal("encAttnHalfKVPaged changed the incoming encoder while declining")
	}
}

func TestEncAttnHalfSharedPaged_Bad(t *testing.T) {
	var enc metal.MTLComputeCommandEncoderObject
	if err := encAttnHalfSharedPaged(enc, nil, nil, nil, nil, 0, bufView{}, bufView{}, bufView{}, attnScratch{}, nil, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, nil); err == nil {
		t.Fatal("encAttnHalfSharedPaged accepted a nil cache")
	}
	c := &devicePagedKVCache{}
	if err := encAttnHalfSharedPaged(enc, nil, c, nil, nil, 0, bufView{}, bufView{}, bufView{}, attnScratch{}, nil, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, nil); err == nil {
		t.Fatal("encAttnHalfSharedPaged accepted a cache shorter than the requested position")
	}
}
