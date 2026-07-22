// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"unsafe"
)

// deltaTestFillRows lands rows [0, n) into cache and mirrors them into a
// position-ordered reference slab (pos*kvBytes), returning (kRef, vRef).
func deltaTestFillRows(t *testing.T, cache *devicePagedKVCache, n, kvBytes int) (kRef, vRef []byte) {
	t.Helper()
	kRef = make([]byte, n*kvBytes)
	vRef = make([]byte, n*kvBytes)
	for pos := range n {
		kRow := toBF16Bytes(syntheticFloat32(kvBytes/bf16Size, 1103+pos))
		vRow := toBF16Bytes(syntheticFloat32(kvBytes/bf16Size, 1201+pos))
		copy(kRef[pos*kvBytes:(pos+1)*kvBytes], kRow)
		copy(vRef[pos*kvBytes:(pos+1)*kvBytes], vRow)
		kPage, vPage, rowOff, err := cache.slot(pos)
		if err != nil {
			t.Fatalf("slot %d: %v", pos, err)
		}
		copy(unsafe.Slice((*byte)(unsafe.Add(kPage.Contents(), uintptr(rowOff))), kvBytes), kRow)
		copy(unsafe.Slice((*byte)(unsafe.Add(vPage.Contents(), uintptr(rowOff))), kvBytes), vRow)
	}
	return kRef, vRef
}

func TestDevicePagedKVCache_PagedRowRef_Good(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 6, 2
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	deltaTestFillRows(t, cache, 5, nKVHeads*headDim*bf16Size)
	for pos := range 5 {
		page, slotIdx, err := cache.pagedRowRef(pos)
		if err != nil {
			t.Fatalf("pagedRowRef(%d): %v", pos, err)
		}
		if want := cache.pageForPos(pos); page != want {
			t.Fatalf("pagedRowRef(%d) page = %d, want %d", pos, page, want)
		}
		if want := pos - cache.pageStartFor(page); slotIdx != want {
			t.Fatalf("pagedRowRef(%d) slot = %d, want %d", pos, slotIdx, want)
		}
	}
}

func TestDevicePagedKVCache_PagedRowRef_Bad(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 6, 2
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	deltaTestFillRows(t, cache, 2, nKVHeads*headDim*bf16Size)
	if _, _, err := cache.pagedRowRef(-1); err == nil {
		t.Fatal("pagedRowRef(-1) accepted a negative position")
	}
	if _, _, err := cache.pagedRowRef(2); err == nil {
		t.Fatal("pagedRowRef accepted a never-landed row")
	}
	var nilCache *devicePagedKVCache
	if _, _, err := nilCache.pagedRowRef(0); err == nil {
		t.Fatal("pagedRowRef on nil cache did not error")
	}
}

func TestDevicePagedKVCache_SyncRowsToLinear_Good(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 6, 2
	kvBytes := nKVHeads * headDim * bf16Size
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	kRef, vRef := deltaTestFillRows(t, cache, 5, kvBytes)

	kDst := make([]byte, maxLen*kvBytes)
	vDst := make([]byte, maxLen*kvBytes)
	// pre-fill rows [0,2) so the delta start is observable: they must survive.
	copy(kDst, kRef[:2*kvBytes])
	copy(vDst, vRef[:2*kvBytes])
	if err := cache.syncRowsToLinear(kDst, vDst, maxLen, 2, 5); err != nil {
		t.Fatalf("syncRowsToLinear: %v", err)
	}
	eqBytes(t, "delta-synced linear K rows", kDst[:5*kvBytes], kRef[:5*kvBytes])
	eqBytes(t, "delta-synced linear V rows", vDst[:5*kvBytes], vRef[:5*kvBytes])
}

func TestDevicePagedKVCache_SyncRowsToLinear_RingWrap_Good(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, window, pageSize = 1, 64, 4, 2
	kvBytes := nKVHeads * headDim * bf16Size
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, window, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	cache.ring = true
	// land rows 0..5 — positions 4,5 wrap onto slots 0,1.
	kRef, vRef := deltaTestFillRows(t, cache, 6, kvBytes)

	kDst := make([]byte, window*kvBytes)
	vDst := make([]byte, window*kvBytes)
	if err := cache.syncRowsToLinear(kDst, vDst, window, 2, 6); err != nil {
		t.Fatalf("syncRowsToLinear ring: %v", err)
	}
	// linear twin is slot-addressed pos%window: slot p holds the LAST landed
	// position congruent to p — rows 4,5 over 0,1; rows 2,3 in place.
	for pos := 2; pos < 6; pos++ {
		slot := pos % window
		eqBytes(t, "ring K slot", kDst[slot*kvBytes:(slot+1)*kvBytes], kRef[pos*kvBytes:(pos+1)*kvBytes])
		eqBytes(t, "ring V slot", vDst[slot*kvBytes:(slot+1)*kvBytes], vRef[pos*kvBytes:(pos+1)*kvBytes])
	}
}

func TestDevicePagedKVCache_SyncRowsToLinear_Bad(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 4, 2
	kvBytes := nKVHeads * headDim * bf16Size
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	deltaTestFillRows(t, cache, 2, kvBytes)
	dst := make([]byte, maxLen*kvBytes)
	if err := cache.syncRowsToLinear(dst, dst, maxLen, 3, 2); err == nil {
		t.Fatal("syncRowsToLinear accepted a reversed range")
	}
	if err := cache.syncRowsToLinear(dst[:kvBytes-1], dst, maxLen, 0, 2); err == nil {
		t.Fatal("syncRowsToLinear accepted a short destination")
	}
	if err := cache.syncRowsToLinear(dst, dst, maxLen, 0, 3); err == nil {
		t.Fatal("syncRowsToLinear read a never-landed row")
	}
}

func TestDevicePagedKVCache_LoadRowsFromLinear_Good(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 6, 2
	kvBytes := nKVHeads * headDim * bf16Size
	kRows := toBF16Bytes(syntheticFloat32(maxLen*nKVHeads*headDim, 1301))
	vRows := toBF16Bytes(syntheticFloat32(maxLen*nKVHeads*headDim, 1409))

	full, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache full: %v", err)
	}
	defer full.Close()
	if err := full.loadLinearSnapshot(kRows, vRows, 5); err != nil {
		t.Fatalf("loadLinearSnapshot: %v", err)
	}

	delta, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache delta: %v", err)
	}
	defer delta.Close()
	if err := delta.loadRowsFromLinear(kRows, vRows, maxLen, 0, 3); err != nil {
		t.Fatalf("loadRowsFromLinear [0,3): %v", err)
	}
	if err := delta.loadRowsFromLinear(kRows, vRows, maxLen, 3, 5); err != nil {
		t.Fatalf("loadRowsFromLinear [3,5): %v", err)
	}
	if delta.length != full.length || delta.offset != full.offset {
		t.Fatalf("delta length/offset = %d/%d, want %d/%d", delta.length, delta.offset, full.length, full.offset)
	}
	_, _, fk, fv, err := full.linearSnapshot(maxLen)
	if err != nil {
		t.Fatalf("full linearSnapshot: %v", err)
	}
	_, _, dk, dv, err := delta.linearSnapshot(maxLen)
	if err != nil {
		t.Fatalf("delta linearSnapshot: %v", err)
	}
	eqBytes(t, "delta-loaded K pages", unsafe.Slice(dk, maxLen*kvBytes), unsafe.Slice(fk, maxLen*kvBytes))
	eqBytes(t, "delta-loaded V pages", unsafe.Slice(dv, maxLen*kvBytes), unsafe.Slice(fv, maxLen*kvBytes))
}

func TestDevicePagedKVCache_LoadRowsFromLinear_Q8_Good(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 6, 2
	kvBytes := nKVHeads * headDim * bf16Size
	kRows := toBF16Bytes(syntheticFloat32(maxLen*nKVHeads*headDim, 1511))
	vRows := toBF16Bytes(syntheticFloat32(maxLen*nKVHeads*headDim, 1601))

	mk := func() *devicePagedKVCache {
		c, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
		if err != nil {
			t.Fatalf("newDevicePagedKVCache: %v", err)
		}
		c.quantQ8 = true
		return c
	}
	full := mk()
	defer full.Close()
	if err := full.loadLinearSnapshot(kRows, vRows, 5); err != nil {
		t.Fatalf("q8 loadLinearSnapshot: %v", err)
	}
	delta := mk()
	defer delta.Close()
	if err := delta.loadRowsFromLinear(kRows, vRows, maxLen, 0, 5); err != nil {
		t.Fatalf("q8 loadRowsFromLinear: %v", err)
	}
	// q8 pages code each row once from the same bf16 source — byte-identical.
	_, _, fk, fv, err := full.linearSnapshot(maxLen)
	if err != nil {
		t.Fatalf("full q8 linearSnapshot: %v", err)
	}
	_, _, dk, dv, err := delta.linearSnapshot(maxLen)
	if err != nil {
		t.Fatalf("delta q8 linearSnapshot: %v", err)
	}
	eqBytes(t, "q8 delta-loaded K dequant", unsafe.Slice(dk, maxLen*kvBytes), unsafe.Slice(fk, maxLen*kvBytes))
	eqBytes(t, "q8 delta-loaded V dequant", unsafe.Slice(dv, maxLen*kvBytes), unsafe.Slice(fv, maxLen*kvBytes))

	// and the delta SYNC of those q8 rows dequantises to the same bf16 bytes.
	kDst := make([]byte, maxLen*kvBytes)
	vDst := make([]byte, maxLen*kvBytes)
	if err := delta.syncRowsToLinear(kDst, vDst, maxLen, 0, 5); err != nil {
		t.Fatalf("q8 syncRowsToLinear: %v", err)
	}
	eqBytes(t, "q8 delta-synced K rows", kDst[:5*kvBytes], unsafe.Slice(fk, 5*kvBytes))
	eqBytes(t, "q8 delta-synced V rows", vDst[:5*kvBytes], unsafe.Slice(fv, 5*kvBytes))
}

func TestDevicePagedKVCache_LoadRowsFromLinear_Bad(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 4, 2
	kvBytes := nKVHeads * headDim * bf16Size
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	rows := make([]byte, maxLen*kvBytes)
	if err := cache.loadRowsFromLinear(rows, rows, maxLen, 3, 2); err == nil {
		t.Fatal("loadRowsFromLinear accepted a reversed range")
	}
	if err := cache.loadRowsFromLinear(rows[:kvBytes-1], rows, maxLen, 0, 1); err == nil {
		t.Fatal("loadRowsFromLinear accepted a short source")
	}
	if err := cache.loadRowsFromLinear(rows, rows, maxLen, maxLen, maxLen+1); err == nil {
		t.Fatal("loadRowsFromLinear wrote past maxSize on a non-ring cache")
	}
}

func TestDevicePagedKVCache_LinearSnapshot_Incremental_Good(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 6, 2
	kvBytes := nKVHeads * headDim * bf16Size
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	kRef, vRef := deltaTestFillRows(t, cache, 4, kvBytes)

	if _, _, kp, vp, err := cache.linearSnapshot(maxLen); err != nil {
		t.Fatalf("first linearSnapshot: %v", err)
	} else {
		eqBytes(t, "first snapshot K", unsafe.Slice(kp, 4*kvBytes), kRef)
		eqBytes(t, "first snapshot V", unsafe.Slice(vp, 4*kvBytes), vRef)
	}
	if cache.snapshotValidRows != 4 {
		t.Fatalf("snapshotValidRows after first materialise = %d, want 4", cache.snapshotValidRows)
	}

	// overwrite row 1 (below the watermark) and append row 4: the second
	// snapshot must carry BOTH — the overwrite via the slot() lowering, the
	// append via the [valid, length) sweep.
	newRow1K := toBF16Bytes(syntheticFloat32(nKVHeads*headDim, 2003))
	newRow1V := toBF16Bytes(syntheticFloat32(nKVHeads*headDim, 2011))
	kPage, vPage, rowOff, err := cache.slot(1)
	if err != nil {
		t.Fatalf("slot(1): %v", err)
	}
	copy(unsafe.Slice((*byte)(unsafe.Add(kPage.Contents(), uintptr(rowOff))), kvBytes), newRow1K)
	copy(unsafe.Slice((*byte)(unsafe.Add(vPage.Contents(), uintptr(rowOff))), kvBytes), newRow1V)
	row4K := toBF16Bytes(syntheticFloat32(nKVHeads*headDim, 2017))
	row4V := toBF16Bytes(syntheticFloat32(nKVHeads*headDim, 2027))
	kPage, vPage, rowOff, err = cache.slot(4)
	if err != nil {
		t.Fatalf("slot(4): %v", err)
	}
	copy(unsafe.Slice((*byte)(unsafe.Add(kPage.Contents(), uintptr(rowOff))), kvBytes), row4K)
	copy(unsafe.Slice((*byte)(unsafe.Add(vPage.Contents(), uintptr(rowOff))), kvBytes), row4V)

	_, _, kp, vp, err := cache.linearSnapshot(maxLen)
	if err != nil {
		t.Fatalf("second linearSnapshot: %v", err)
	}
	got := unsafe.Slice(kp, 5*kvBytes)
	eqBytes(t, "row 0 kept", got[:kvBytes], kRef[:kvBytes])
	eqBytes(t, "row 1 overwrite", got[kvBytes:2*kvBytes], newRow1K)
	eqBytes(t, "rows 2-3 kept", got[2*kvBytes:4*kvBytes], kRef[2*kvBytes:4*kvBytes])
	eqBytes(t, "row 4 append", got[4*kvBytes:5*kvBytes], row4K)
	gotV := unsafe.Slice(vp, 5*kvBytes)
	eqBytes(t, "V row 1 overwrite", gotV[kvBytes:2*kvBytes], newRow1V)
	eqBytes(t, "V row 4 append", gotV[4*kvBytes:5*kvBytes], row4V)
	if cache.snapshotValidRows != 5 {
		t.Fatalf("snapshotValidRows after second materialise = %d, want 5", cache.snapshotValidRows)
	}
}

func TestDevicePagedKVCache_LinearSnapshot_TruncateZeroesTail_Good(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, maxLen, pageSize = 1, 64, 6, 2
	kvBytes := nKVHeads * headDim * bf16Size
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, maxLen, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	deltaTestFillRows(t, cache, 5, kvBytes)
	if _, _, _, _, err := cache.linearSnapshot(maxLen); err != nil {
		t.Fatalf("first linearSnapshot: %v", err)
	}
	if err := cache.truncate(2); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	_, _, kp, vp, err := cache.linearSnapshot(maxLen)
	if err != nil {
		t.Fatalf("post-truncate linearSnapshot: %v", err)
	}
	zero := make([]byte, 3*kvBytes)
	eqBytes(t, "truncated K tail zeroed", unsafe.Slice(kp, 5*kvBytes)[2*kvBytes:], zero)
	eqBytes(t, "truncated V tail zeroed", unsafe.Slice(vp, 5*kvBytes)[2*kvBytes:], zero)
}

func TestDevicePagedKVCache_LinearSyncedAbs_Good(t *testing.T) {
	requireNativeRuntime(t)
	const nKVHeads, headDim, window, pageSize = 1, 64, 4, 2
	cache, err := newDevicePagedKVCache(nKVHeads, headDim, window, pageSize)
	if err != nil {
		t.Fatalf("newDevicePagedKVCache: %v", err)
	}
	defer cache.Close()
	cache.ring = true
	deltaTestFillRows(t, cache, 3, nKVHeads*headDim*bf16Size)

	cache.linearSyncedAbs = 3 // the seam declares mirrors equal through 3
	if _, _, _, err := cache.slot(5); err != nil {
		t.Fatalf("slot(5): %v", err)
	}
	if cache.linearSyncedAbs != 3 {
		t.Fatalf("append above watermark moved linearSyncedAbs to %d, want 3", cache.linearSyncedAbs)
	}
	if _, _, _, err := cache.slot(1); err != nil { // overwrite below the watermark
		t.Fatalf("slot(1): %v", err)
	}
	if cache.linearSyncedAbs != 1 {
		t.Fatalf("overwrite at 1 left linearSyncedAbs = %d, want 1", cache.linearSyncedAbs)
	}
	cache.linearSyncedAbs = 6
	if err := cache.truncate(4); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	if cache.linearSyncedAbs != 4 {
		t.Fatalf("truncate left linearSyncedAbs = %d, want 4", cache.linearSyncedAbs)
	}
}
