// SPDX-Licence-Identifier: EUPL-1.2

// Tests for InMemoryStore method-level guard branches and cross-field
// backfill behaviour that state_test.go's native-store happy path never
// reaches — nil ctx/receiver guards, cancelled-context short-circuits,
// the stored-ref ChunkID backfill, and the Text<->Data cross-population
// that only fires when a chunk was seeded through the "other" write path.

package state

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// --- NewInMemoryStoreWithManifest -------------------------------------------

func TestMemory_NewInMemoryStoreWithManifest_Good(t *testing.T) {
	store := NewInMemoryStoreWithManifest(
		map[int]string{1: "seeded", 5: "seeded-five"},
		map[int]ChunkRef{
			5:  {Codec: "custom/codec", Segment: "epoch-1"},
			10: {Codec: "manifest/only"},
		},
	)

	chunk, err := store.Resolve(context.Background(), 5)
	if err != nil {
		t.Fatalf("Resolve(manifest ref) error = %v", err)
	}
	// The manifest ref overwrites the auto-derived default, and its
	// ChunkID is force-set to the map key regardless of the input value.
	if chunk.Ref.ChunkID != 5 || chunk.Ref.Codec != "custom/codec" || chunk.Ref.Segment != "epoch-1" {
		t.Fatalf("Resolve(manifest ref) = %+v, want ChunkID 5 with manifest codec/segment", chunk.Ref)
	}
	if chunk.Text != "seeded-five" {
		t.Fatalf("Resolve(manifest ref) Text = %q, want seeded-five", chunk.Text)
	}

	// nextID advances past the highest manifest-only ref key (10), not
	// just the highest chunks key (5), so a later Put doesn't collide.
	ref, err := store.Put(context.Background(), "next", PutOptions{})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if ref.ChunkID != 11 {
		t.Fatalf("Put() ChunkID = %d, want 11 (past the manifest high-water mark)", ref.ChunkID)
	}
}

// --- Resolve (method) --------------------------------------------------------

func TestMemory_Resolve_Good(t *testing.T) {
	// A chunk stored without a matching refs entry (bypassing Put)
	// backfills ChunkID from the lookup key rather than the zero-value ref.
	store := &InMemoryStore{chunks: map[int]string{3: "no ref entry"}}
	chunk, err := store.Resolve(context.Background(), 3)
	if err != nil {
		t.Fatalf("Resolve(no ref) error = %v", err)
	}
	if chunk.Ref.ChunkID != 3 {
		t.Fatalf("Resolve(no ref) ChunkID = %d, want backfilled 3", chunk.Ref.ChunkID)
	}

	// A chunk written via PutBytes has no chunks[id] text entry — Resolve
	// (not ResolveBytes) still backfills Text from the binary payload.
	bin := NewInMemoryStore(nil)
	ref, err := bin.PutBytes(context.Background(), []byte("binary payload"), PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	chunk, err = bin.Resolve(context.Background(), ref.ChunkID)
	if err != nil {
		t.Fatalf("Resolve(after PutBytes) error = %v", err)
	}
	if chunk.Text != "binary payload" {
		t.Fatalf("Resolve(after PutBytes) Text = %q, want backfilled from Data", chunk.Text)
	}
}

func TestMemory_Resolve_Bad(t *testing.T) {
	store := NewInMemoryStore(map[int]string{1: "x"})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.Resolve(ctx, 1); !core.Is(err, context.Canceled) {
		t.Fatalf("Resolve(cancelled ctx) error = %v, want context.Canceled", err)
	}

	var nilStore *InMemoryStore
	if _, err := nilStore.Resolve(context.Background(), 1); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Resolve(nil receiver) error = %v, want ErrChunkNotFound", err)
	}

	// nil ctx normalises rather than panicking.
	if _, err := store.Resolve(nil, 1); err != nil {
		t.Fatalf("Resolve(nil ctx) error = %v", err)
	}
}

// --- ResolveBytes (method) ---------------------------------------------------

func TestMemory_ResolveBytes_Good(t *testing.T) {
	// A chunk written via Put (text-only) has no data[id] entry —
	// ResolveBytes derives Data from Text.
	store := NewInMemoryStore(nil)
	ref, err := store.Put(context.Background(), "text payload", PutOptions{})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	chunk, err := store.ResolveBytes(context.Background(), ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(after Put) error = %v", err)
	}
	if string(chunk.Data) != "text payload" {
		t.Fatalf("ResolveBytes(after Put) Data = %q, want derived from Text", chunk.Data)
	}

	// ChunkID backfill mirrors Resolve's.
	raw := &InMemoryStore{data: map[int][]byte{4: []byte("raw")}}
	chunk, err = raw.ResolveBytes(context.Background(), 4)
	if err != nil {
		t.Fatalf("ResolveBytes(no ref) error = %v", err)
	}
	if chunk.Ref.ChunkID != 4 {
		t.Fatalf("ResolveBytes(no ref) ChunkID = %d, want backfilled 4", chunk.Ref.ChunkID)
	}
}

func TestMemory_ResolveBytes_Bad(t *testing.T) {
	store := NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.ResolveBytes(ctx, 1); !core.Is(err, context.Canceled) {
		t.Fatalf("ResolveBytes(cancelled ctx) error = %v, want context.Canceled", err)
	}

	var nilStore *InMemoryStore
	if _, err := nilStore.ResolveBytes(context.Background(), 1); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveBytes(nil receiver) error = %v, want ErrChunkNotFound", err)
	}

	if _, err := store.ResolveBytes(context.Background(), 999); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveBytes(missing id) error = %v, want ErrChunkNotFound", err)
	}

	if _, err := store.ResolveBytes(nil, 999); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveBytes(nil ctx, missing id) error = %v, want ErrChunkNotFound", err)
	}
}

// --- BorrowBytes (method) -----------------------------------------------------

func TestMemory_BorrowBytes_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	ref, err := store.Put(context.Background(), "borrow text", PutOptions{})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	// nil ctx normalises rather than panicking.
	if _, err := store.BorrowBytes(nil, ref.ChunkID); err != nil {
		t.Fatalf("BorrowBytes(nil ctx) error = %v", err)
	}
	borrowed, err := store.BorrowBytes(context.Background(), ref.ChunkID)
	if err != nil {
		t.Fatalf("BorrowBytes(after Put) error = %v", err)
	}
	if string(borrowed.Data) != "borrow text" {
		t.Fatalf("BorrowBytes(after Put) Data = %q, want derived from Text", borrowed.Data)
	}

	raw := &InMemoryStore{data: map[int][]byte{6: []byte("raw")}}
	borrowed, err = raw.BorrowBytes(context.Background(), 6)
	if err != nil {
		t.Fatalf("BorrowBytes(no ref) error = %v", err)
	}
	if borrowed.Ref.ChunkID != 6 {
		t.Fatalf("BorrowBytes(no ref) ChunkID = %d, want backfilled 6", borrowed.Ref.ChunkID)
	}
}

func TestMemory_BorrowBytes_Bad(t *testing.T) {
	store := NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.BorrowBytes(ctx, 1); !core.Is(err, context.Canceled) {
		t.Fatalf("BorrowBytes(cancelled ctx) error = %v, want context.Canceled", err)
	}

	var nilStore *InMemoryStore
	if _, err := nilStore.BorrowBytes(context.Background(), 1); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowBytes(nil receiver) error = %v, want ErrChunkNotFound", err)
	}

	if _, err := store.BorrowBytes(context.Background(), 999); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowBytes(missing id) error = %v, want ErrChunkNotFound", err)
	}
}

// --- BorrowRefBytes (method) --------------------------------------------------

func TestMemory_BorrowRefBytes_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte("seg"), PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	ref.Segment = "epoch-7"
	borrowed, err := store.BorrowRefBytes(context.Background(), ref)
	if err != nil {
		t.Fatalf("BorrowRefBytes(segment overlay) error = %v", err)
	}
	if borrowed.Ref.Segment != "epoch-7" {
		t.Fatalf("BorrowRefBytes(segment overlay) Segment = %q, want epoch-7", borrowed.Ref.Segment)
	}
}

func TestMemory_BorrowRefBytes_Bad(t *testing.T) {
	store := NewInMemoryStore(nil)
	if _, err := store.BorrowRefBytes(context.Background(), ChunkRef{ChunkID: 0}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowRefBytes(zero id) error = %v, want ErrChunkNotFound", err)
	}
	// A missing id propagates the BorrowBytes error through BorrowRefBytes.
	if _, err := store.BorrowRefBytes(context.Background(), ChunkRef{ChunkID: 999}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowRefBytes(missing id) error = %v, want ErrChunkNotFound", err)
	}
}

// --- ResolveURI (method) ------------------------------------------------------

func TestMemory_ResolveURI_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	if _, err := store.Put(context.Background(), "uri text", PutOptions{URI: "state://a/1"}); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if _, err := store.ResolveURI(nil, "state://a/1"); err != nil {
		t.Fatalf("ResolveURI(nil ctx) error = %v", err)
	}
}

func TestMemory_ResolveURI_Bad(t *testing.T) {
	store := NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.ResolveURI(ctx, "state://x"); !core.Is(err, context.Canceled) {
		t.Fatalf("ResolveURI(cancelled ctx) error = %v, want context.Canceled", err)
	}

	var nilStore *InMemoryStore
	if _, err := nilStore.ResolveURI(context.Background(), "state://x"); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveURI(nil receiver) error = %v, want ErrChunkNotFound", err)
	}

	if _, err := store.ResolveURI(context.Background(), "state://missing"); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveURI(missing uri) error = %v, want ErrChunkNotFound", err)
	}
}

// --- Put (method) -------------------------------------------------------------

func TestMemory_Put_Bad(t *testing.T) {
	store := NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.Put(ctx, "x", PutOptions{}); !core.Is(err, context.Canceled) {
		t.Fatalf("Put(cancelled ctx) error = %v, want context.Canceled", err)
	}

	var nilStore *InMemoryStore
	if _, err := nilStore.Put(context.Background(), "x", PutOptions{}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Put(nil receiver) error = %v, want ErrChunkNotFound", err)
	}
}

func TestMemory_Put_Ugly(t *testing.T) {
	// A zero-value store (bypassing NewInMemoryStore) self-initialises
	// every backing map AND recovers a non-positive nextID to 1 — proves
	// Put is safe on a bare &InMemoryStore{} rather than requiring the
	// constructor.
	zero := &InMemoryStore{}
	ref, err := zero.Put(context.Background(), "first", PutOptions{})
	if err != nil {
		t.Fatalf("Put(zero-value store) error = %v", err)
	}
	if ref.ChunkID != 1 {
		t.Fatalf("Put(zero-value store) ChunkID = %d, want 1", ref.ChunkID)
	}

	if _, err := zero.Put(nil, "second", PutOptions{}); err != nil {
		t.Fatalf("Put(nil ctx) error = %v", err)
	}
}

// --- PutBytes (method) ---------------------------------------------------------

func TestMemory_PutBytes_Bad(t *testing.T) {
	store := NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.PutBytes(ctx, []byte("x"), PutOptions{}); !core.Is(err, context.Canceled) {
		t.Fatalf("PutBytes(cancelled ctx) error = %v, want context.Canceled", err)
	}

	var nilStore *InMemoryStore
	if _, err := nilStore.PutBytes(context.Background(), []byte("x"), PutOptions{}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("PutBytes(nil receiver) error = %v, want ErrChunkNotFound", err)
	}
}

func TestMemory_PutBytes_Ugly(t *testing.T) {
	zero := &InMemoryStore{}
	ref, err := zero.PutBytes(context.Background(), []byte("first"), PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes(zero-value store) error = %v", err)
	}
	if ref.ChunkID != 1 {
		t.Fatalf("PutBytes(zero-value store) ChunkID = %d, want 1", ref.ChunkID)
	}

	if _, err := zero.PutBytes(nil, []byte("second"), PutOptions{}); err != nil {
		t.Fatalf("PutBytes(nil ctx) error = %v", err)
	}
}
