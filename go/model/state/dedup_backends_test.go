// SPDX-Licence-Identifier: EUPL-1.2

// Per-backend DedupStore receipts: the store-savings, reclaim-safety, and
// ramspill spill/revive proofs against the two durable backends. External test
// package (state_test) so it can import filestore and ramspill — both of which
// import state — without an import cycle.
package state_test

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/filestore"
	"dappco.re/go/inference/model/state/ramspill"
)

// TestDedupStore_Filestore_Good_SharesPrefixAndSurvivesReclaim is the append-only
// backend receipt: two conversations sharing a prefix block store it once, and
// reclaiming the first (which cannot physically delete on an append-only log)
// leaves the shared block immortal so the second wakes byte-identical.
func TestDedupStore_Filestore_Good_SharesPrefixAndSurvivesReclaim(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "dedup.kv")
	fs, err := filestore.Create(ctx, path)
	if err != nil {
		t.Fatalf("filestore.Create() error = %v", err)
	}
	t.Cleanup(func() { _ = fs.Close() })
	store := state.NewDedupStore(fs)

	shared := []byte("shared-prefix-kv-block")
	privateA := []byte("A-divergent-tail-block")
	privateB := []byte("B-divergent-tail-block")

	// Conversation A sleeps two blocks; conversation B shares the prefix.
	refShared, _ := store.PutBytes(ctx, shared, state.PutOptions{URI: "a/block/0"})
	refPrivA, _ := store.PutBytes(ctx, privateA, state.PutOptions{URI: "a/block/1"})
	refSharedB, _ := store.PutBytes(ctx, shared, state.PutOptions{URI: "b/block/0"})
	refPrivB, _ := store.PutBytes(ctx, privateB, state.PutOptions{URI: "b/block/1"})

	if refSharedB.ChunkID != refShared.ChunkID {
		t.Fatalf("B prefix block = %d, want shared %d", refSharedB.ChunkID, refShared.ChunkID)
	}
	if got := fs.ChunkCount(); got != 3 {
		t.Fatalf("filestore ChunkCount = %d, want 3 (shared + 2 private)", got)
	}
	if stats := store.Stats(); stats.Dedups != 1 {
		t.Fatalf("stats.Dedups = %d, want 1", stats.Dedups)
	}

	// Reclaim A — append-only has no Deleter, so the shared chunk stays immortal.
	if err := store.Release(ctx, refShared, refPrivA); err != nil {
		t.Fatalf("Release(A) error = %v", err)
	}
	if stats := store.Stats(); stats.Reclaimed != 0 {
		t.Fatalf("stats.Reclaimed = %d, want 0 (append-only keeps chunks)", stats.Reclaimed)
	}
	if got := fs.ChunkCount(); got != 3 {
		t.Fatalf("filestore ChunkCount after reclaim = %d, want 3 (immortal)", got)
	}

	// B wakes byte-identical: its shared prefix block and its own tail resolve.
	assertRefBytes(t, store, refShared, shared)
	assertRefBytes(t, store, refPrivB, privateB)
}

// TestDedupStore_Ramspill_Good_SharedChunkSurvivesSpillRevive is the ramspill
// receipt: a deduped shared block is one chunk that spills to cold under budget
// pressure and revives on the next read byte-identical, then survives a reclaim
// of the conversation that first wrote it.
func TestDedupStore_Ramspill_Good_SharedChunkSurvivesSpillRevive(t *testing.T) {
	ctx := context.Background()
	coldPath := core.PathJoin(t.TempDir(), "spill.kv")
	cold, err := filestore.Create(ctx, coldPath)
	if err != nil {
		t.Fatalf("filestore.Create() error = %v", err)
	}
	t.Cleanup(func() { _ = cold.Close() })
	// 80-byte budget over 40-byte blocks holds two resident; the oldest spills.
	rs, err := ramspill.New(ramspill.Options{Budget: 80, Cold: cold})
	if err != nil {
		t.Fatalf("ramspill.New() error = %v", err)
	}
	store := state.NewDedupStore(rs)

	shared := []byte(core.Sprintf("%040d", 0))   // 40 bytes, written first (coldest)
	privateA := []byte(core.Sprintf("%040d", 1)) // 40 bytes
	privateB := []byte(core.Sprintf("%040d", 2)) // 40 bytes

	refShared, _ := store.PutBytes(ctx, shared, state.PutOptions{URI: "a/block/0"})
	refPrivA, _ := store.PutBytes(ctx, privateA, state.PutOptions{URI: "a/block/1"})
	refSharedB, _ := store.PutBytes(ctx, shared, state.PutOptions{URI: "b/block/0"}) // dedup: no ramspill write
	_, _ = store.PutBytes(ctx, privateB, state.PutOptions{URI: "b/block/1"})

	if refSharedB.ChunkID != refShared.ChunkID {
		t.Fatalf("B prefix block = %d, want shared %d", refSharedB.ChunkID, refShared.ChunkID)
	}
	// One physical chunk for the shared block despite two conversations using it.
	if rs.ChunkCount() != 3 {
		t.Fatalf("ramspill ChunkCount = %d, want 3 (shared stored once)", rs.ChunkCount())
	}
	// The shared block, written first and never re-touched by B's dedup, is the
	// coldest and has spilled to the cold store.
	if rs.Resident(refShared.ChunkID) {
		t.Fatal("shared chunk still resident, want spilled to cold under budget")
	}

	// Reading the shared block through the dedup store transparently revives it
	// byte-identical — the spill/revive cycle mid-test.
	assertRefBytes(t, store, refShared, shared)
	if !rs.Resident(refShared.ChunkID) {
		t.Fatal("shared chunk not resident after read, want revived")
	}

	// Reclaim A: ramspill has no Deleter, so the shared chunk (still referenced
	// by B) stays, and B keeps waking byte-identical across the reclaim.
	if err := store.Release(ctx, refShared, refPrivA); err != nil {
		t.Fatalf("Release(A) error = %v", err)
	}
	assertRefBytes(t, store, refShared, shared)
}

// assertRefBytes fails unless resolving ref through store yields want.
func assertRefBytes(t *testing.T, store *state.DedupStore, ref state.ChunkRef, want []byte) {
	t.Helper()
	chunk, err := store.ResolveRefBytes(context.Background(), ref)
	if err != nil {
		t.Fatalf("ResolveRefBytes(%d) error = %v", ref.ChunkID, err)
	}
	if string(chunk.Data) != string(want) {
		t.Fatalf("ResolveRefBytes(%d) = %q, want %q", ref.ChunkID, chunk.Data, want)
	}
}
