// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"testing"
)

// textOnlyStore is a Store+Writer with no BinaryWriter — the "inner cannot take
// a stable content write" fallback: a DedupStore over it must decline PutBytes
// plainly rather than dedup a write it cannot make.
type textOnlyStore struct{ inner *InMemoryStore }

func (s *textOnlyStore) Get(ctx context.Context, id int) (string, error) {
	return s.inner.Get(ctx, id)
}

func (s *textOnlyStore) Put(ctx context.Context, text string, opts PutOptions) (ChunkRef, error) {
	return s.inner.Put(ctx, text, opts)
}

// noDeleteStore is a Store+Writer+BinaryWriter with no Deleter — the append-only
// / immortal-backend fallback: a DedupStore over it keeps a zero-referenced
// chunk resident (safe) rather than reclaiming it. It wraps InMemoryStore
// without embedding it, so InMemoryStore.Delete is not promoted and the store is
// genuinely not a Deleter.
type noDeleteStore struct{ inner *InMemoryStore }

func (s *noDeleteStore) Get(ctx context.Context, id int) (string, error) {
	return s.inner.Get(ctx, id)
}

func (s *noDeleteStore) Resolve(ctx context.Context, id int) (Chunk, error) {
	return s.inner.Resolve(ctx, id)
}

func (s *noDeleteStore) ResolveBytes(ctx context.Context, id int) (Chunk, error) {
	return s.inner.ResolveBytes(ctx, id)
}

func (s *noDeleteStore) Put(ctx context.Context, text string, opts PutOptions) (ChunkRef, error) {
	return s.inner.Put(ctx, text, opts)
}

func (s *noDeleteStore) PutBytes(ctx context.Context, data []byte, opts PutOptions) (ChunkRef, error) {
	return s.inner.PutBytes(ctx, data, opts)
}

// TestDedupStore_PutBytes_Good_DedupsIdenticalContent is the store-savings
// keystone: a second write of byte-identical content returns the first chunk's
// ref and writes nothing, so the inner store holds one physical copy.
func TestDedupStore_PutBytes_Good_DedupsIdenticalContent(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemoryStore(nil)
	store := NewDedupStore(inner)

	block := []byte("shared-prefix-kv-block")
	first, err := store.PutBytes(ctx, block, PutOptions{URI: "a/block/0"})
	if err != nil {
		t.Fatalf("PutBytes(first) error = %v", err)
	}
	second, err := store.PutBytes(ctx, block, PutOptions{URI: "b/block/0"})
	if err != nil {
		t.Fatalf("PutBytes(second) error = %v", err)
	}
	if first.ChunkID != second.ChunkID {
		t.Fatalf("dedup ref = %d, want shared %d", second.ChunkID, first.ChunkID)
	}
	if got := inner.ChunkCount(); got != 1 {
		t.Fatalf("inner ChunkCount = %d, want 1 (deduped)", got)
	}
	stats := store.Stats()
	if stats.Writes != 1 || stats.Dedups != 1 {
		t.Fatalf("stats = %+v, want 1 write + 1 dedup", stats)
	}
	if stats.UniqueChunks != 1 {
		t.Fatalf("stats.UniqueChunks = %d, want 1", stats.UniqueChunks)
	}
}

// TestDedupStore_PutBytes_Good_DistinctContentWritesEach is the dedup-miss =
// full-write receipt: content that hashes differently is never merged.
func TestDedupStore_PutBytes_Good_DistinctContentWritesEach(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemoryStore(nil)
	store := NewDedupStore(inner)

	a, err := store.PutBytes(ctx, []byte("block-A"), PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes(A) error = %v", err)
	}
	b, err := store.PutBytes(ctx, []byte("block-B"), PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes(B) error = %v", err)
	}
	if a.ChunkID == b.ChunkID {
		t.Fatalf("distinct content shared ref %d — must not dedup", a.ChunkID)
	}
	if got := inner.ChunkCount(); got != 2 {
		t.Fatalf("inner ChunkCount = %d, want 2 (no dedup on a miss)", got)
	}
	if stats := store.Stats(); stats.Dedups != 0 || stats.Writes != 2 {
		t.Fatalf("stats = %+v, want 2 writes + 0 dedups", stats)
	}
}

// TestDedupStore_PutBytes_Bad_NonBinaryInner asserts a DedupStore over a store
// that cannot take a byte write declines PutBytes plainly — the "backend cannot
// support the mechanism" fallback, never a silent dangling ref.
func TestDedupStore_PutBytes_Bad_NonBinaryInner(t *testing.T) {
	ctx := context.Background()
	store := NewDedupStore(&textOnlyStore{inner: NewInMemoryStore(nil)})
	if _, err := store.PutBytes(ctx, []byte("x"), PutOptions{}); err == nil {
		t.Fatal("PutBytes(non-binary inner) error = nil, want a plain decline")
	}
	// The text path still works, so the wrapper stays usable as a transparent
	// proxy for URI-addressed manifests.
	if _, err := store.Put(ctx, "manifest", PutOptions{URI: "m"}); err != nil {
		t.Fatalf("Put(text) error = %v, want pass-through", err)
	}
}

// TestDedupStore_Put_Good_PassesThroughURIResolvable asserts text writes are not
// deduped, so two manifests keep independent URIs — the reason only ref-addressed
// binary blocks share.
func TestDedupStore_Put_Good_PassesThroughURIResolvable(t *testing.T) {
	ctx := context.Background()
	store := NewDedupStore(NewInMemoryStore(nil))

	if _, err := store.Put(ctx, "manifest-body", PutOptions{URI: "conv-a/bundle"}); err != nil {
		t.Fatalf("Put(A) error = %v", err)
	}
	if _, err := store.Put(ctx, "manifest-body", PutOptions{URI: "conv-b/bundle"}); err != nil {
		t.Fatalf("Put(B) error = %v", err)
	}
	// Even with identical bodies, both URIs must resolve — a deduped text write
	// would have left the second URI unregistered.
	for _, uri := range []string{"conv-a/bundle", "conv-b/bundle"} {
		chunk, err := store.ResolveURI(ctx, uri)
		if err != nil {
			t.Fatalf("ResolveURI(%q) error = %v", uri, err)
		}
		if chunk.Text != "manifest-body" {
			t.Fatalf("ResolveURI(%q) text = %q, want manifest-body", uri, chunk.Text)
		}
	}
}

// TestDedupStore_Reads_Good_PassThroughToInner asserts every read surface
// resolves a deduped chunk through the inner store byte-for-byte.
func TestDedupStore_Reads_Good_PassThroughToInner(t *testing.T) {
	ctx := context.Background()
	store := NewDedupStore(NewInMemoryStore(nil))
	payload := []byte("resolvable-block")
	ref, err := store.PutBytes(ctx, payload, PutOptions{URI: "blk"})
	if err != nil {
		t.Fatalf("PutBytes error = %v", err)
	}
	resolved, err := store.ResolveBytes(ctx, ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes error = %v", err)
	}
	if string(resolved.Data) != string(payload) {
		t.Fatalf("ResolveBytes data = %q, want %q", resolved.Data, payload)
	}
	borrowed, err := store.BorrowRefBytes(ctx, ref)
	if err != nil {
		t.Fatalf("BorrowRefBytes error = %v", err)
	}
	if string(borrowed.Data) != string(payload) {
		t.Fatalf("BorrowRefBytes data = %q, want %q", borrowed.Data, payload)
	}
}

// TestDedupStore_Release_Good_ReclaimsPrivateKeepsShared is the reclaim-safety
// receipt on InMemoryStore: two conversations share a prefix block; reclaiming
// the first physically frees only its private block, and the shared block plus
// the second conversation's own blocks survive byte-identical.
func TestDedupStore_Release_Good_ReclaimsPrivateKeepsShared(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemoryStore(nil)
	store := NewDedupStore(inner)

	shared := []byte("shared-prefix-block")
	privateA := []byte("A-divergent-tail")
	privateB := []byte("B-divergent-tail")

	// Conversation A sleeps its two blocks.
	refShared, _ := store.PutBytes(ctx, shared, PutOptions{URI: "a/block/0"})
	refPrivA, _ := store.PutBytes(ctx, privateA, PutOptions{URI: "a/block/1"})
	// Conversation B sleeps: prefix block dedups against A, tail is fresh.
	refSharedB, _ := store.PutBytes(ctx, shared, PutOptions{URI: "b/block/0"})
	refPrivB, _ := store.PutBytes(ctx, privateB, PutOptions{URI: "b/block/1"})

	if refSharedB.ChunkID != refShared.ChunkID {
		t.Fatalf("B prefix block = %d, want shared %d", refSharedB.ChunkID, refShared.ChunkID)
	}
	if got := inner.ChunkCount(); got != 3 {
		t.Fatalf("inner ChunkCount = %d, want 3 (shared + 2 private)", got)
	}

	// Reclaim conversation A: release the block refs its bundle recorded.
	if err := store.Release(ctx, refShared, refPrivA); err != nil {
		t.Fatalf("Release(A) error = %v", err)
	}

	// The shared block survives (B still references it) and B wakes byte-identical.
	if chunk, err := store.ResolveBytes(ctx, refShared.ChunkID); err != nil || string(chunk.Data) != string(shared) {
		t.Fatalf("shared block after reclaim = (%q, %v), want %q surviving", chunkData(chunk), err, shared)
	}
	if chunk, err := store.ResolveBytes(ctx, refPrivB.ChunkID); err != nil || string(chunk.Data) != string(privateB) {
		t.Fatalf("B private block after reclaim = (%q, %v), want %q surviving", chunkData(chunk), err, privateB)
	}
	// A's private block was the only reference — it is physically reclaimed.
	if _, err := store.ResolveBytes(ctx, refPrivA.ChunkID); err == nil {
		t.Fatal("A private block still resolves after reclaim — refcount 0 should have deleted it")
	}
	if got := inner.ChunkCount(); got != 2 {
		t.Fatalf("inner ChunkCount = %d after reclaim, want 2", got)
	}
	if stats := store.Stats(); stats.Reclaimed != 1 {
		t.Fatalf("stats.Reclaimed = %d, want 1", stats.Reclaimed)
	}
}

// TestDedupStore_Release_Good_NoDeleterKeepsChunk asserts the append-only
// fallback: with no Deleter, a chunk whose last reference is released stays
// resident and resolvable — safe, never a dangling ref.
func TestDedupStore_Release_Good_NoDeleterKeepsChunk(t *testing.T) {
	ctx := context.Background()
	store := NewDedupStore(&noDeleteStore{inner: NewInMemoryStore(nil)})

	block := []byte("immortal-block")
	ref, _ := store.PutBytes(ctx, block, PutOptions{})
	if err := store.Release(ctx, ref); err != nil {
		t.Fatalf("Release error = %v", err)
	}
	if chunk, err := store.ResolveBytes(ctx, ref.ChunkID); err != nil || string(chunk.Data) != string(block) {
		t.Fatalf("block after release without Deleter = (%q, %v), want %q resident", chunkData(chunk), err, block)
	}
	if stats := store.Stats(); stats.Reclaimed != 0 {
		t.Fatalf("stats.Reclaimed = %d, want 0 (no Deleter)", stats.Reclaimed)
	}
}

// TestDedupStore_Release_Ugly_UnknownRefIsNoop asserts releasing a ref the store
// never deduped (a manifest, a foreign id) is a harmless no-op.
func TestDedupStore_Release_Ugly_UnknownRefIsNoop(t *testing.T) {
	ctx := context.Background()
	store := NewDedupStore(NewInMemoryStore(nil))
	if err := store.Release(ctx, ChunkRef{ChunkID: 999}); err != nil {
		t.Fatalf("Release(unknown) error = %v, want no-op", err)
	}
}

// TestDedupStore_Good_BaselineWithoutDedupWritesTwice pins the counterfactual: a
// bare store (no dedup wrapper) writes both identical payloads, so the savings
// are the wrapper's and not the fixture's.
func TestDedupStore_Good_BaselineWithoutDedupWritesTwice(t *testing.T) {
	ctx := context.Background()
	inner := NewInMemoryStore(nil)
	block := []byte("shared-prefix-kv-block")
	if _, err := inner.PutBytes(ctx, block, PutOptions{}); err != nil {
		t.Fatalf("PutBytes(first) error = %v", err)
	}
	if _, err := inner.PutBytes(ctx, block, PutOptions{}); err != nil {
		t.Fatalf("PutBytes(second) error = %v", err)
	}
	if got := inner.ChunkCount(); got != 2 {
		t.Fatalf("bare-store ChunkCount = %d, want 2 (no dedup mechanism)", got)
	}
}

// chunkData is a nil-safe view of a chunk's bytes for failure messages.
func chunkData(c Chunk) string { return string(c.Data) }
