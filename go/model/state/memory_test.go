// SPDX-Licence-Identifier: EUPL-1.2

// Tests for InMemoryStore construction and method-level guard branches —
// nil ctx/receiver guards, cancelled-context short-circuits, the
// stored-ref ChunkID backfill, the Text<->Data cross-population that only
// fires when a chunk was seeded through the "other" write path, and the
// copy-on-write/copy-on-read vs live-view aliasing contract that
// distinguishes Resolve*/Put* from Borrow*.

package state

import (
	"context"
	"sync"
	"testing"

	core "dappco.re/go"
)

// --- NewInMemoryStore --------------------------------------------------------

func TestMemory_NewInMemoryStore_Good(t *testing.T) {
	store := NewInMemoryStore(map[int]string{2: "two", 5: "five"})

	chunk, err := store.Resolve(context.Background(), 5)
	if err != nil {
		t.Fatalf("Resolve(seeded) error = %v", err)
	}
	if chunk.Text != "five" || chunk.Ref.ChunkID != 5 || !chunk.Ref.HasFrameOffset || chunk.Ref.FrameOffset != 5 || chunk.Ref.Codec != CodecMemory {
		t.Fatalf("Resolve(seeded) = %+v, want derived default ref for chunk 5", chunk.Ref)
	}

	// nextID advances past the highest seeded key (5), not from 1.
	ref, err := store.Put(context.Background(), "next", PutOptions{})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if ref.ChunkID != 6 {
		t.Fatalf("Put() ChunkID = %d, want 6 (past the seeded high-water mark)", ref.ChunkID)
	}
}

// TestMemory_NewInMemoryStore_Bad proves a nil chunks map still yields a
// usable, empty store rather than a store that panics on first use.
func TestMemory_NewInMemoryStore_Bad(t *testing.T) {
	store := NewInMemoryStore(nil)

	if _, err := store.Get(context.Background(), 42); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Get(empty store) error = %v, want ErrChunkNotFound", err)
	}

	ref, err := store.Put(context.Background(), "first", PutOptions{})
	if err != nil {
		t.Fatalf("Put(empty store) error = %v", err)
	}
	if ref.ChunkID != 1 {
		t.Fatalf("Put(empty store) ChunkID = %d, want 1", ref.ChunkID)
	}
}

// TestMemory_NewInMemoryStore_Ugly seeds chunk ID 0 — a legal but unusual
// key that never advances nextID (0 < the constructor's starting
// nextID of 1) — proving id-0 content is retrievable without colliding
// with the auto-assigned Put sequence.
func TestMemory_NewInMemoryStore_Ugly(t *testing.T) {
	store := NewInMemoryStore(map[int]string{0: "zero id"})

	chunk, err := store.Resolve(context.Background(), 0)
	if err != nil {
		t.Fatalf("Resolve(id 0) error = %v", err)
	}
	if chunk.Text != "zero id" {
		t.Fatalf("Resolve(id 0) Text = %q, want zero id", chunk.Text)
	}

	ref, err := store.Put(context.Background(), "auto", PutOptions{})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if ref.ChunkID != 1 {
		t.Fatalf("Put() ChunkID = %d, want 1 (id-0 seed does not perturb the auto sequence)", ref.ChunkID)
	}
}

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

// TestMemory_NewInMemoryStoreWithManifest_Bad proves a manifest-only ref
// (present in refs, absent from chunks/data) does NOT become resolvable
// content on its own — a caller who assumes a manifest entry implies a
// retrievable chunk gets ErrChunkNotFound.
func TestMemory_NewInMemoryStoreWithManifest_Bad(t *testing.T) {
	store := NewInMemoryStoreWithManifest(nil, map[int]ChunkRef{10: {Codec: "manifest/only"}})

	if _, err := store.Resolve(context.Background(), 10); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Resolve(manifest-only, no content) error = %v, want ErrChunkNotFound", err)
	}
}

// TestMemory_NewInMemoryStoreWithManifest_Ugly proves the manifest ref's
// own ChunkID field is discarded in favour of the map key, and that
// calling with both maps nil/empty still constructs a safe, usable store.
func TestMemory_NewInMemoryStoreWithManifest_Ugly(t *testing.T) {
	store := NewInMemoryStoreWithManifest(
		map[int]string{7: "seven"},
		map[int]ChunkRef{7: {ChunkID: 999, Codec: "mismatched"}},
	)

	chunk, err := store.Resolve(context.Background(), 7)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if chunk.Ref.ChunkID != 7 {
		t.Fatalf("Resolve() ChunkID = %d, want map key 7 to win over the ref's own field (999)", chunk.Ref.ChunkID)
	}

	empty := NewInMemoryStoreWithManifest(nil, nil)
	ref, err := empty.Put(context.Background(), "first", PutOptions{})
	if err != nil {
		t.Fatalf("Put(fully empty manifest store) error = %v", err)
	}
	if ref.ChunkID != 1 {
		t.Fatalf("Put(fully empty manifest store) ChunkID = %d, want 1", ref.ChunkID)
	}
}

// --- Get ---------------------------------------------------------------------

func TestMemory_InMemoryStore_Get_Good(t *testing.T) {
	store := NewInMemoryStore(map[int]string{7: "chunk seven"})

	text, err := store.Get(context.Background(), 7)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if text != "chunk seven" {
		t.Fatalf("Get() = %q, want chunk seven", text)
	}
}

func TestMemory_InMemoryStore_Get_Bad(t *testing.T) {
	store := NewInMemoryStore(map[int]string{1: "x"})
	if _, err := store.Get(context.Background(), 999); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Get(missing id) error = %v, want ErrChunkNotFound", err)
	}

	var nilStore *InMemoryStore
	if _, err := nilStore.Get(context.Background(), 1); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Get(nil receiver) error = %v, want ErrChunkNotFound", err)
	}
}

// TestMemory_InMemoryStore_Get_Ugly proves Get has no guards of its own —
// it delegates straight to Resolve, so a cancelled context and a nil
// context are handled exactly as Resolve handles them.
func TestMemory_InMemoryStore_Get_Ugly(t *testing.T) {
	store := NewInMemoryStore(map[int]string{1: "x"})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.Get(ctx, 1); !core.Is(err, context.Canceled) {
		t.Fatalf("Get(cancelled ctx) error = %v, want context.Canceled", err)
	}

	if text, err := store.Get(nil, 1); err != nil || text != "x" {
		t.Fatalf("Get(nil ctx) = %q, %v, want x, nil", text, err)
	}
}

// --- Resolve (method) --------------------------------------------------------

func TestMemory_InMemoryStore_Resolve_Good(t *testing.T) {
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

func TestMemory_InMemoryStore_Resolve_Bad(t *testing.T) {
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

// TestMemory_InMemoryStore_Resolve_Ugly seeds BOTH the text and binary
// maps for the same ID directly (bypassing Put/PutBytes, which always
// clear the other map) — proving the Text<->Data cross-population guard
// (`if chunk.Text == ""`) preserves an already-non-empty Text rather than
// overwriting it from Data.
func TestMemory_InMemoryStore_Resolve_Ugly(t *testing.T) {
	store := &InMemoryStore{
		chunks: map[int]string{4: "original text"},
		data:   map[int][]byte{4: []byte("binary payload")},
	}

	chunk, err := store.Resolve(context.Background(), 4)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if chunk.Text != "original text" {
		t.Fatalf("Resolve() Text = %q, want original text preserved (not overwritten from Data)", chunk.Text)
	}
	if string(chunk.Data) != "binary payload" {
		t.Fatalf("Resolve() Data = %q, want binary payload populated alongside Text", chunk.Data)
	}
}

// --- ResolveBytes (method) ---------------------------------------------------

func TestMemory_InMemoryStore_ResolveBytes_Good(t *testing.T) {
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

func TestMemory_InMemoryStore_ResolveBytes_Bad(t *testing.T) {
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

// TestMemory_InMemoryStore_ResolveBytes_Ugly proves ResolveBytes always
// hands back a defensive copy: mutating one call's returned Data has no
// effect on a second, independent ResolveBytes call for the same chunk —
// the opposite of BorrowBytes' live-view contract.
func TestMemory_InMemoryStore_ResolveBytes_Ugly(t *testing.T) {
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte{1, 2, 3}, PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	first, err := store.ResolveBytes(context.Background(), ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(first) error = %v", err)
	}
	first.Data[1] = 99

	second, err := store.ResolveBytes(context.Background(), ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(second) error = %v", err)
	}
	if second.Data[1] != 2 {
		t.Fatalf("ResolveBytes(second) = %v, want unaffected by mutation of the first call's copy", second.Data)
	}
}

// --- BorrowBytes (method) -----------------------------------------------------

func TestMemory_InMemoryStore_BorrowBytes_Good(t *testing.T) {
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

func TestMemory_InMemoryStore_BorrowBytes_Bad(t *testing.T) {
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

// TestMemory_InMemoryStore_BorrowBytes_Ugly proves BorrowBytes returns a
// LIVE view onto the store's own backing slice (per its doc comment) —
// mutating the returned Data is visible on a second BorrowBytes call for
// the same chunk, the opposite of ResolveBytes' defensive-copy contract.
func TestMemory_InMemoryStore_BorrowBytes_Ugly(t *testing.T) {
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte{1, 2, 3}, PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	first, err := store.BorrowBytes(context.Background(), ref.ChunkID)
	if err != nil {
		t.Fatalf("BorrowBytes(first) error = %v", err)
	}
	first.Data[1] = 99

	second, err := store.BorrowBytes(context.Background(), ref.ChunkID)
	if err != nil {
		t.Fatalf("BorrowBytes(second) error = %v", err)
	}
	if second.Data[1] != 99 {
		t.Fatalf("BorrowBytes(second) = %v, want the live mutation visible (no defensive copy)", second.Data)
	}
}

// --- BorrowRefBytes (method) --------------------------------------------------

func TestMemory_InMemoryStore_BorrowRefBytes_Good(t *testing.T) {
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

func TestMemory_InMemoryStore_BorrowRefBytes_Bad(t *testing.T) {
	store := NewInMemoryStore(nil)
	if _, err := store.BorrowRefBytes(context.Background(), ChunkRef{ChunkID: 0}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowRefBytes(zero id) error = %v, want ErrChunkNotFound", err)
	}
	// A missing id propagates the BorrowBytes error through BorrowRefBytes.
	if _, err := store.BorrowRefBytes(context.Background(), ChunkRef{ChunkID: 999}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowRefBytes(missing id) error = %v, want ErrChunkNotFound", err)
	}
}

// TestMemory_InMemoryStore_BorrowRefBytes_Ugly chains Borrow -> mutate ->
// Borrow -> Resolve -> mutate -> Borrow to prove the live-view/defensive-
// copy asymmetry holds even when the overlay logic (FrameOffset/Codec/
// Segment) is in play: a borrowed mutation is visible on the next borrow,
// but mutating a *resolved* (copied) chunk never corrupts the store.
func TestMemory_InMemoryStore_BorrowRefBytes_Ugly(t *testing.T) {
	store := NewInMemoryStore(nil)
	payload := []byte{4, 3, 2, 1}
	ref, err := store.PutBytes(context.Background(), payload, PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	borrowed, err := store.BorrowRefBytes(context.Background(), ref)
	if err != nil {
		t.Fatalf("BorrowRefBytes(first) error = %v", err)
	}
	borrowed.Data[2] = 99

	again, err := store.BorrowRefBytes(context.Background(), ref)
	if err != nil {
		t.Fatalf("BorrowRefBytes(second) error = %v", err)
	}
	if again.Data[2] != 99 {
		t.Fatalf("BorrowRefBytes(second) = %v, want the live mutation visible", again.Data)
	}

	resolved, err := store.ResolveBytes(context.Background(), ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes() error = %v", err)
	}
	resolved.Data[2] = 7

	view, err := store.BorrowRefBytes(context.Background(), ref)
	if err != nil {
		t.Fatalf("BorrowRefBytes(third) error = %v", err)
	}
	if view.Data[2] != 99 {
		t.Fatalf("BorrowRefBytes(third) = %v, want unaffected by mutating ResolveBytes' independent copy", view.Data)
	}
}

// --- ResolveURI (method) ------------------------------------------------------

func TestMemory_InMemoryStore_ResolveURI_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	if _, err := store.Put(context.Background(), "uri text", PutOptions{URI: "state://a/1"}); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if _, err := store.ResolveURI(nil, "state://a/1"); err != nil {
		t.Fatalf("ResolveURI(nil ctx) error = %v", err)
	}
}

func TestMemory_InMemoryStore_ResolveURI_Bad(t *testing.T) {
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

// TestMemory_InMemoryStore_ResolveURI_Ugly resolves a URI registered
// against a BINARY chunk (PutBytes, not Put) — proving ResolveURI's
// delegation to Resolve carries the Text<->Data backfill through to the
// URI lookup path too.
func TestMemory_InMemoryStore_ResolveURI_Ugly(t *testing.T) {
	store := NewInMemoryStore(nil)
	payload := []byte{0, 1, 2, 255}
	if _, err := store.PutBytes(context.Background(), payload, PutOptions{URI: "state://binary/1"}); err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	chunk, err := store.ResolveURI(context.Background(), "state://binary/1")
	if err != nil {
		t.Fatalf("ResolveURI(binary) error = %v", err)
	}
	if len(chunk.Data) != 4 || chunk.Data[0] != 0 || chunk.Data[3] != 255 {
		t.Fatalf("ResolveURI(binary) = %+v, want the binary payload resolved by URI", chunk)
	}
}

// --- Put (method) -------------------------------------------------------------

func TestMemory_InMemoryStore_Put_Good(t *testing.T) {
	store := NewInMemoryStore(nil)

	ref, err := store.Put(context.Background(), "hello", PutOptions{URI: "state://put/1"})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if ref.ChunkID != 1 || !ref.HasFrameOffset || ref.FrameOffset != 1 || ref.Codec != CodecMemory {
		t.Fatalf("Put() ref = %+v, want derived defaults for chunk 1", ref)
	}

	text, err := store.Get(context.Background(), ref.ChunkID)
	if err != nil || text != "hello" {
		t.Fatalf("Get(after Put) = %q, %v, want hello, nil", text, err)
	}

	byURI, err := store.ResolveURI(context.Background(), "state://put/1")
	if err != nil || byURI.Text != "hello" {
		t.Fatalf("ResolveURI(after Put) = %+v, %v, want hello", byURI, err)
	}
}

func TestMemory_InMemoryStore_Put_Bad(t *testing.T) {
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

func TestMemory_InMemoryStore_Put_Ugly(t *testing.T) {
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

func TestMemory_InMemoryStore_PutBytes_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	payload := []byte{0, 1, 2, 255}

	ref, err := store.PutBytes(context.Background(), payload, PutOptions{URI: "state://binary/1"})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if ref.ChunkID != 1 || !ref.HasFrameOffset || ref.FrameOffset != 1 || ref.Codec != CodecMemory {
		t.Fatalf("PutBytes() ref = %+v, want derived defaults for chunk 1", ref)
	}

	// PutBytes defensively copies its input — mutating the caller's
	// slice afterwards must not corrupt the stored payload.
	payload[1] = 99

	chunk, err := store.ResolveBytes(context.Background(), ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes() error = %v", err)
	}
	if chunk.Ref.ChunkID != ref.ChunkID || len(chunk.Data) != 4 || chunk.Data[1] != 1 || chunk.Data[3] != 255 {
		t.Fatalf("ResolveBytes() chunk = %+v, want copied binary payload unaffected by the caller's mutation", chunk)
	}

	byURI, err := store.ResolveURI(context.Background(), "state://binary/1")
	if err != nil {
		t.Fatalf("ResolveURI(binary) error = %v", err)
	}
	if len(byURI.Data) != 4 || byURI.Data[0] != 0 {
		t.Fatalf("ResolveURI(binary) chunk = %+v, want binary data", byURI)
	}
}

func TestMemory_InMemoryStore_PutBytes_Bad(t *testing.T) {
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

func TestMemory_InMemoryStore_PutBytes_Ugly(t *testing.T) {
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

// TestMemory_InMemoryStore_Concurrent_Good proves the store is safe to share
// across goroutines — the contract the concurrent serve path assumes when the
// RAM store is the default conversation tier. Without the RWMutex this races
// the backing maps (fatal "concurrent map read and map write"); run under
// -race to see the guard hold. Mixed writers (Put/PutBytes), readers
// (Resolve/ResolveBytes/BorrowBytes), and URI lookups run at once.
func TestMemory_InMemoryStore_Concurrent_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()

	const workers = 8
	const opsPerWorker = 200
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func(w int) {
			defer wg.Done()
			for i := 0; i < opsPerWorker; i++ {
				uri := "mem://c/" + core.Itoa(w) + "/" + core.Itoa(i)
				if i%2 == 0 {
					store.Put(ctx, "text", PutOptions{URI: uri})
				} else {
					store.PutBytes(ctx, []byte("bytes"), PutOptions{URI: uri})
				}
				// Read back through every read seam concurrently with peers'
				// writes; correctness is "no race, no panic", so ignore the
				// not-found errors a just-issued id may still be racing.
				_, _ = store.Resolve(ctx, 1)
				_, _ = store.ResolveBytes(ctx, 1)
				_, _ = store.BorrowBytes(ctx, 1)
				_, _ = store.ResolveURI(ctx, uri)
			}
		}(w)
	}
	wg.Wait()
}
