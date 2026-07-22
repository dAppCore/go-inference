// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import "unsafe"

// owned.go registers load-time SYNTHESISED tensors with their DirMapping. A normalising load pass
// may replace or add a tensor whose Data is a fresh heap buffer rather than a shard mmap view —
// the packed per-layer MoE expert tensors (packExperts), the b1→b2 exact repack (model.LoadLinear
// → mlxaffine.RepackB1ToB2). A zero-copy binder must treat those exactly like the F16→BF16-widened
// tensors (widen.go): bind resident instead of failing its "weight is not a view into any mapped
// shard" wrong-mapping guard. The registration is POSITIVE — an off-shard tensor that no load pass
// recorded still fails that guard, so a weight from a different mapping cannot bind silently —
// and per-mapping, so an engine can evict the owned tensors' device buffers with the session that
// bound them (see engine/metal shardBuffers.Close) and a dead registration can never false-match
// a recycled heap address (Close clears the set).

// AdoptOwnedTensors sweeps the mapping's tensors and records the heap range of every non-empty
// Tensor.Data that is neither a view into a mapped shard nor a widened (widen.go) buffer — i.e.
// every tensor some load pass synthesised on the heap. Call once, after the last tensor-producing
// load step (model.Load runs it after Assemble); calling again only adopts ranges added since
// (already-recorded ranges are kept, not duplicated). Returns the number of tensors adopted.
//
//	dm.AdoptOwnedTensors()
//	dm.IsOwned(t.Data) // → true for a packExperts pack or a repacked b1 weight
func (d *DirMapping) AdoptOwnedTensors() int {
	if d == nil {
		return 0
	}
	adopted := 0
	for _, t := range d.Tensors {
		if len(t.Data) == 0 || d.inShard(t.Data) || d.IsWidened(t.Data) || d.IsOwned(t.Data) {
			continue
		}
		start := uintptr(unsafe.Pointer(&t.Data[0]))
		d.owned = append(d.owned, widenedRange{start: start, end: start + uintptr(len(t.Data))})
		adopted++
	}
	return adopted
}

// IsOwned reports whether b is (a view into) a tensor AdoptOwnedTensors recorded — a load-time
// synthesised heap buffer this mapping vouches for. The zero-copy binder consults this beside
// IsWidened so a registered owned weight binds resident instead of failing the wrong-mapping guard.
func (d *DirMapping) IsOwned(b []byte) bool {
	if d == nil || len(b) == 0 {
		return false
	}
	p := uintptr(unsafe.Pointer(&b[0]))
	for _, r := range d.owned {
		if p >= r.start && p < r.end {
			return true
		}
	}
	return false
}

// OwnedRanges returns the adopted owned tensors' [start,end) heap spans as parallel base/end
// slices — the shape the engine-side resident-buffer eviction consumes when the owning session
// closes (engine/metal evictResidentBufsForRanges). Empty when nothing was adopted.
func (d *DirMapping) OwnedRanges() (bases, ends []uintptr) {
	if d == nil || len(d.owned) == 0 {
		return nil, nil
	}
	bases = make([]uintptr, len(d.owned))
	ends = make([]uintptr, len(d.owned))
	for i, r := range d.owned {
		bases[i], ends[i] = r.start, r.end
	}
	return bases, ends
}

// inShard reports whether b's first byte lies inside one of the mapping's shard mmaps — the
// same containment rule the zero-copy binder's shard scan applies (first byte decides; a tensor
// never straddles shards because each views exactly one shard's mmap).
func (d *DirMapping) inShard(b []byte) bool {
	if len(b) == 0 {
		return false
	}
	p := uintptr(unsafe.Pointer(&b[0]))
	for _, m := range d.Shards {
		if m == nil || len(m.Data) == 0 {
			continue
		}
		start := uintptr(unsafe.Pointer(&m.Data[0]))
		if p >= start && p < start+uintptr(len(m.Data)) {
			return true
		}
	}
	return false
}
