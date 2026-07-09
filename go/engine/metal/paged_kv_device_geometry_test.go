// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"unsafe"
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
