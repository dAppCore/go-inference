// SPDX-Licence-Identifier: EUPL-1.2

// Forced-spill receipt: a tiny byte budget over several conversation chunks
// spills the coldest ones to a real filestore.Store .kv file in LRU order and
// revives them transparently on the next resolve — plus the unknown-chunk and
// cold-I/O-failure edges.
package ramspill

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv/kvtier"
	state "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/filestore"
)

// payload returns a 40-byte deterministic chunk body for conversation i —
// fixed-width so the forced-spill maths below (budget vs. resident bytes) is
// exact rather than approximate.
func payload(i int) string {
	return core.Sprintf("%040d", i)
}

func openColdStore(t *testing.T) *filestore.Store {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "spill.kv")
	cold, err := filestore.Create(context.Background(), path)
	if err != nil {
		t.Fatalf("filestore.Create() error = %v", err)
	}
	t.Cleanup(func() { _ = cold.Close() })
	return cold
}

func TestRamspill_New_Bad_BudgetWithoutCold(t *testing.T) {
	if _, err := New(Options{Budget: 100}); err == nil {
		t.Fatal("New() error = nil, want a budget-without-Cold error")
	}
}

func TestRamspill_New_Good_UnlimitedBudgetNeedsNoCold(t *testing.T) {
	store, err := New(Options{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx := context.Background()
	for i := range 20 {
		if _, err := store.Put(ctx, payload(i), state.PutOptions{}); err != nil {
			t.Fatalf("Put(%d) error = %v", i, err)
		}
	}
	if store.ChunkCount() != 20 {
		t.Fatalf("ChunkCount() = %d, want 20", store.ChunkCount())
	}
	for id := 1; id <= 20; id++ {
		if !store.Resident(id) {
			t.Fatalf("Resident(%d) = false, want true (Budget <= 0 never spills)", id)
		}
	}
}

// TestRamspill_Store_Good_SpillsColdestAndRevivesTransparently is the forced-
// spill receipt: a 100-byte budget over five 40-byte chunks holds at most two
// resident, spills the rest oldest-first, and a later ResolveURI on a spilled
// conversation transparently revives it — bumping it back to most-recently-used
// and pushing the (now coldest) previously-hot chunk out in its place.
func TestRamspill_Store_Good_SpillsColdestAndRevivesTransparently(t *testing.T) {
	cold := openColdStore(t)
	store, err := New(Options{Budget: 100, Cold: cold})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx := context.Background()

	uris := make([]string, 5)
	for i := range 5 {
		uris[i] = core.Sprintf("conv/%d", i)
		if _, err := store.Put(ctx, payload(i), state.PutOptions{URI: uris[i]}); err != nil {
			t.Fatalf("Put(%d) error = %v", i, err)
		}
	}

	idOf := func(i int) int {
		id, ok := store.ChunkIDForURI(uris[i])
		if !ok {
			t.Fatalf("ChunkIDForURI(%s): not found", uris[i])
		}
		return id
	}

	// LRU order after 5 sequential 40-byte puts under a 100-byte budget: the
	// two most recent (3, 4) fit (80 <= 100); 0, 1, 2 spilled oldest-first.
	wantResident := map[int]bool{0: false, 1: false, 2: false, 3: true, 4: true}
	for i, want := range wantResident {
		if got := store.Resident(idOf(i)); got != want {
			t.Fatalf("Resident(conv/%d) = %v, want %v (LRU spill order)", i, got, want)
		}
	}

	// Transparent revival: reading a spilled conversation's chunk returns its
	// original bytes without the caller doing anything special.
	chunk, err := store.ResolveURI(ctx, uris[0])
	if err != nil {
		t.Fatalf("ResolveURI(conv/0) error = %v", err)
	}
	if chunk.Text != payload(0) {
		t.Fatalf("ResolveURI(conv/0).Text = %q, want %q", chunk.Text, payload(0))
	}

	// The revive promotes 0 to most-recently-used, which re-triggers the
	// budget cascade: 3 (now the coldest of the three resident-or-just-
	// promoted chunks {3, 4, 0}) spills to make room.
	wantAfterRevive := map[int]bool{0: true, 1: false, 2: false, 3: false, 4: true}
	for i, want := range wantAfterRevive {
		if got := store.Resident(idOf(i)); got != want {
			t.Fatalf("Resident(conv/%d) after revive = %v, want %v", i, got, want)
		}
	}

	// The chunk that was already resident throughout never left RAM: its
	// content still round-trips without touching Cold.
	chunk4, err := store.ResolveURI(ctx, uris[4])
	if err != nil {
		t.Fatalf("ResolveURI(conv/4) error = %v", err)
	}
	if chunk4.Text != payload(4) {
		t.Fatalf("ResolveURI(conv/4).Text = %q, want %q", chunk4.Text, payload(4))
	}
}

// TestRamspill_Store_Good_LogsSpillAndRevivalReceipts mirrors serving's
// multimodel eviction notice ("serve: evicted model %s (%s), ~%d bytes
// freed") so a spill/revive is visible on the same boot-log stream an
// operator already watches.
func TestRamspill_Store_Good_LogsSpillAndRevivalReceipts(t *testing.T) {
	cold := openColdStore(t)
	log := core.NewBuilder()
	store, err := New(Options{Budget: 80, Cold: cold, Log: log})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx := context.Background()
	for i := range 3 {
		if _, err := store.Put(ctx, payload(i), state.PutOptions{URI: core.Sprintf("conv/%d", i)}); err != nil {
			t.Fatalf("Put(%d) error = %v", i, err)
		}
	}
	if !core.Contains(log.String(), "state: spilled conversation chunk") {
		t.Fatalf("log = %q, want a spill receipt line", log.String())
	}

	if _, err := store.ResolveURI(ctx, "conv/0"); err != nil {
		t.Fatalf("ResolveURI(conv/0) error = %v", err)
	}
	if !core.Contains(log.String(), "state: revived conversation chunk") {
		t.Fatalf("log = %q, want a revival receipt line", log.String())
	}
}

func TestRamspill_Store_Bad_UnknownChunk(t *testing.T) {
	store, err := New(Options{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx := context.Background()

	if _, err := store.Resolve(ctx, 999); err == nil {
		t.Fatal("Resolve(999) error = nil, want ChunkNotFoundError")
	} else {
		var notFound *state.ChunkNotFoundError
		if !core.As(err, &notFound) {
			t.Fatalf("Resolve(999) error = %v, want *state.ChunkNotFoundError", err)
		}
	}

	if _, err := store.ResolveURI(ctx, "conv/missing"); err == nil {
		t.Fatal("ResolveURI(missing) error = nil, want URIChunkNotFoundError")
	} else {
		var notFound *state.URIChunkNotFoundError
		if !core.As(err, &notFound) {
			t.Fatalf("ResolveURI(missing) error = %v, want *state.URIChunkNotFoundError", err)
		}
	}
}

// failingCold is a ColdStore whose PutBytes always fails — it exercises
// kvtier's rollback contract: a spill that can't durably land must leave the
// chunk exactly as resident (and readable) as before the attempt, never half
// evicted.
type failingCold struct{}

func (failingCold) Put(context.Context, string, state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, core.NewError("failingCold: Put always fails")
}

func (failingCold) PutBytes(context.Context, []byte, state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, core.NewError("failingCold: PutBytes always fails")
}

func (failingCold) ResolveURI(_ context.Context, uri string) (state.Chunk, error) {
	return state.Chunk{}, &state.URIChunkNotFoundError{URI: uri}
}

func (failingCold) ResolveBytes(_ context.Context, chunkID int) (state.Chunk, error) {
	return state.Chunk{}, &state.ChunkNotFoundError{ID: chunkID}
}

func TestRamspill_Store_Ugly_ColdStoreFailureKeepsChunkResident(t *testing.T) {
	store, err := New(Options{Budget: 100, Cold: failingCold{}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx := context.Background()

	var lastRef state.ChunkRef
	for i := range 5 {
		ref, err := store.Put(ctx, payload(i), state.PutOptions{})
		if err != nil {
			t.Fatalf("Put(%d) error = %v", i, err)
		}
		lastRef = ref
	}

	// Every chunk stays resident: eviction was ATTEMPTED (over budget by
	// chunk 2 onward) but Cold rejects every write, so kvtier's rollback
	// keeps the tier map exactly as it was — nothing is silently lost.
	for id := 1; id <= lastRef.ChunkID; id++ {
		if !store.Resident(id) {
			t.Fatalf("Resident(%d) = false, want true (a failed spill must never lose the chunk)", id)
		}
		chunk, err := store.Resolve(ctx, id)
		if err != nil {
			t.Fatalf("Resolve(%d) error = %v", id, err)
		}
		if chunk.Text != payload(id-1) {
			t.Fatalf("Resolve(%d).Text = %q, want %q", id, chunk.Text, payload(id-1))
		}
	}
}

// TestRamspill_Store_Ugly_MoveRejectsUnparseableBlockID documents the guard
// on the kvtier.Store seam: Move only ever receives block ids this Store
// itself minted (decimal chunk ids), but the seam is exported, so a foreign
// caller feeding it garbage gets a clean error instead of a panic.
func TestRamspill_Store_Ugly_MoveRejectsUnparseableBlockID(t *testing.T) {
	store, err := New(Options{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := store.Move(context.Background(), "not-an-id", kvtier.TierGPU, kvtier.TierDisk); err == nil {
		t.Fatal("Move(\"not-an-id\") error = nil, want a parse error")
	}
}
