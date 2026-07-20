// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import "testing"

// ownedTestMapping builds a DirMapping over one in-memory "shard" with one shard-view tensor,
// one F16 tensor (widened by the caller when the case needs it), and one synthesised heap
// tensor — the three provenances AdoptOwnedTensors must tell apart.
func ownedTestMapping() (*DirMapping, Tensor, Tensor) {
	shard := make([]byte, 64)
	view := Tensor{Dtype: "BF16", Shape: []int{8}, Data: shard[16:32]}
	synth := Tensor{Dtype: "BF16", Shape: []int{8}, Data: make([]byte, 16)}
	dm := &DirMapping{
		Shards:  []*Mapping{{Data: shard}},
		Tensors: map[string]Tensor{"view": view, "synth": synth},
	}
	return dm, view, synth
}

// TestAdoptOwnedTensors_Good — the sweep records exactly the synthesised heap tensor: the shard
// view is skipped (it binds zero-copy), a second sweep adopts nothing new (idempotent), and a
// tensor added between sweeps is picked up by the next one.
func TestAdoptOwnedTensors_Good(t *testing.T) {
	dm, view, synth := ownedTestMapping()
	if got := dm.AdoptOwnedTensors(); got != 1 {
		t.Fatalf("AdoptOwnedTensors = %d, want 1 (only the synthesised tensor)", got)
	}
	if !dm.IsOwned(synth.Data) {
		t.Fatal("synthesised tensor not owned after adoption")
	}
	if dm.IsOwned(view.Data) {
		t.Fatal("shard-view tensor must NOT adopt as owned")
	}
	if got := dm.AdoptOwnedTensors(); got != 0 {
		t.Fatalf("second sweep adopted %d, want 0 (idempotent)", got)
	}
	late := Tensor{Dtype: "BF16", Shape: []int{4}, Data: make([]byte, 8)}
	dm.Tensors["late"] = late
	if got := dm.AdoptOwnedTensors(); got != 1 || !dm.IsOwned(late.Data) {
		t.Fatalf("late sweep = %d, IsOwned(late) = %v — want 1, true", got, dm.IsOwned(late.Data))
	}
}

// TestAdoptOwnedTensors_Bad — nil mapping and empty-Data tensors adopt nothing (no panic, no
// zero-length ranges), and a WIDENED tensor is NOT double-registered as owned (widen.go already
// vouches for it; the binder checks both sets).
func TestAdoptOwnedTensors_Bad(t *testing.T) {
	var nilDM *DirMapping
	if got := nilDM.AdoptOwnedTensors(); got != 0 {
		t.Fatalf("nil AdoptOwnedTensors = %d, want 0", got)
	}
	dm, _, _ := ownedTestMapping()
	dm.Tensors["empty"] = Tensor{Dtype: "BF16", Shape: []int{0}}
	f16 := Tensor{Dtype: "F16", Shape: []int{4}, Data: make([]byte, 8)}
	dm.Tensors["half"] = f16
	if n := dm.WidenF16TensorsToBF16(); n != 1 {
		t.Fatalf("WidenF16TensorsToBF16 = %d, want 1", n)
	}
	if got := dm.AdoptOwnedTensors(); got != 1 {
		t.Fatalf("AdoptOwnedTensors = %d, want 1 (synth only: empty skipped, widened already vouched)", got)
	}
	if dm.IsOwned(dm.Tensors["half"].Data) {
		t.Fatal("widened tensor must stay in the widened set, not adopt as owned")
	}
}

// TestIsOwned_Good — a SUB-SLICE of an adopted tensor is owned too (containment is by first
// byte inside the recorded [start,end) span, matching the binder's shard-scan rule).
func TestIsOwned_Good(t *testing.T) {
	dm, _, synth := ownedTestMapping()
	dm.AdoptOwnedTensors()
	if !dm.IsOwned(synth.Data[4:12]) {
		t.Fatal("interior view of an owned tensor should report owned")
	}
}

// TestIsOwned_Bad — nil mapping, empty slice, and an unregistered heap slice all report false:
// the wrong-mapping guard must keep failing for buffers no load pass vouched for.
func TestIsOwned_Bad(t *testing.T) {
	var nilDM *DirMapping
	if nilDM.IsOwned([]byte{1}) {
		t.Fatal("nil mapping reported owned")
	}
	dm, _, _ := ownedTestMapping()
	dm.AdoptOwnedTensors()
	if dm.IsOwned(nil) {
		t.Fatal("empty slice reported owned")
	}
	if stranger := make([]byte, 16); dm.IsOwned(stranger) {
		t.Fatal("unregistered heap slice reported owned — the wrong-mapping guard just died")
	}
}

// TestOwnedRanges_Good — the spans come back as parallel base/end slices covering exactly the
// adopted tensors (the engine-side eviction shape), and Close clears them so a dead
// registration can never match a recycled heap address.
func TestOwnedRanges_Good(t *testing.T) {
	dm, _, synth := ownedTestMapping()
	dm.AdoptOwnedTensors()
	bases, ends := dm.OwnedRanges()
	if len(bases) != 1 || len(ends) != 1 {
		t.Fatalf("OwnedRanges lengths = %d/%d, want 1/1", len(bases), len(ends))
	}
	if ends[0]-bases[0] != uintptr(len(synth.Data)) {
		t.Fatalf("owned span = %d bytes, want %d", ends[0]-bases[0], len(synth.Data))
	}
	// Close drops the registrations with the mapping. (Shards here are heap stand-ins, not real
	// mmaps; Close on them is the documented no-op path for an already-nil mapping region.)
	dm.Shards = nil
	_ = dm.Close()
	if b, e := dm.OwnedRanges(); b != nil || e != nil {
		t.Fatal("OwnedRanges after Close should be empty")
	}
}

// TestOwnedRanges_Bad — nil mapping and a mapping with no adoptions return nil/nil (the
// engine-side eviction treats that as "nothing to do").
func TestOwnedRanges_Bad(t *testing.T) {
	var nilDM *DirMapping
	if b, e := nilDM.OwnedRanges(); b != nil || e != nil {
		t.Fatal("nil mapping OwnedRanges should be nil/nil")
	}
	dm, _, _ := ownedTestMapping()
	if b, e := dm.OwnedRanges(); b != nil || e != nil {
		t.Fatal("unadopted mapping OwnedRanges should be nil/nil")
	}
}

// TestDirMapping_InShard_Good — the containment helper matches a view into a shard by its
// first byte and rejects heap slices, nil shards and empty input (the provenance test the
// adoption sweep is built on).
func TestDirMapping_InShard_Good(t *testing.T) {
	dm, view, synth := ownedTestMapping()
	if !dm.inShard(view.Data) {
		t.Fatal("shard view not recognised")
	}
	if dm.inShard(synth.Data) {
		t.Fatal("heap tensor misclassified as shard view")
	}
	if dm.inShard(nil) {
		t.Fatal("empty slice misclassified as shard view")
	}
	dm.Shards = append(dm.Shards, nil, &Mapping{})
	if !dm.inShard(view.Data) {
		t.Fatal("nil/empty shard entries broke containment")
	}
}
