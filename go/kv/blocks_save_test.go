// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"bytes"
	"context"
	"testing"

	state "dappco.re/go/inference/model/state"
)

// TestBlocksSave_Snapshot_SaveStateBlocks_Good saves the four-token fixture as a
// two-block State bundle and asserts the manifest carries both blocks, the State
// kind, and a snapshot hash.
func TestBlocksSave_Snapshot_SaveStateBlocks_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/save-good",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	if len(bundle.Blocks) != 2 || bundle.TokenCount != 4 {
		t.Fatalf("SaveStateBlocks() bundle = %+v, want two blocks covering four tokens", bundle)
	}
	if bundle.Kind != StateBlockBundleKind || bundle.SnapshotHash == "" {
		t.Fatalf("SaveStateBlocks() bundle metadata = kind %q hash %q, want State kind + hash", bundle.Kind, bundle.SnapshotHash)
	}
}

// TestBlocksSave_Snapshot_SaveStateBlocks_Bad drives SaveStateBlocks' guard arms:
// a nil snapshot, a nil store, and an unsupported KV encoding all fail.
func TestBlocksSave_Snapshot_SaveStateBlocks_Bad(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	var nilSnapshot *Snapshot
	if _, err := nilSnapshot.SaveStateBlocks(ctx, store, StateBlockOptions{}); err == nil {
		t.Fatal("SaveStateBlocks(nil snapshot) error = nil")
	}
	if _, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, nil, StateBlockOptions{}); err == nil {
		t.Fatal("SaveStateBlocks(nil store) error = nil")
	}
	if _, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{KVEncoding: "q2"}); err == nil {
		t.Fatal("SaveStateBlocks(bad encoding) error = nil")
	}
}

// TestBlocksSave_Snapshot_SaveStateBlocks_Ugly drives the reuse-prefix path: a
// child save that adopts the parent's first prefix block by reference, so the
// bundle records one reused block sharing the parent's chunk ref.
func TestBlocksSave_Snapshot_SaveStateBlocks_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent := kvSnapshotBlocksTestSnapshot()
	parentBundle, err := parent.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://reuse-parent",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}

	child := kvSnapshotBlocksTestSnapshot()
	child.Tokens[2] = 9
	child.Tokens[3] = 10
	childBundle, err := child.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:         2,
		KVEncoding:        EncodingNative,
		URI:               "mlx://reuse-child",
		ReusePrefix:       parentBundle,
		ReusePrefixTokens: 2,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(child reuse) error = %v", err)
	}
	if childBundle.ReusedBlocks != 1 {
		t.Fatalf("child reused blocks = %d, want 1", childBundle.ReusedBlocks)
	}
	if childBundle.Blocks[0].State.ChunkID != parentBundle.Blocks[0].State.ChunkID {
		t.Fatalf("child first block = %+v, want shared parent ref", childBundle.Blocks[0])
	}
}

// TestBlocksSave_Snapshot_SaveMemvidBlocks_Good asserts the deprecated
// SaveMemvidBlocks alias forwards to SaveStateBlocks and stamps the memvid kind.
func TestBlocksSave_Snapshot_SaveMemvidBlocks_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	bundle, err := kvSnapshotBlocksTestSnapshot().SaveMemvidBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8})
	if err != nil {
		t.Fatalf("SaveMemvidBlocks() error = %v", err)
	}
	if len(bundle.Blocks) != 2 || bundle.Kind != MemvidBlockBundleKind {
		t.Fatalf("SaveMemvidBlocks() bundle = %+v, want two blocks with memvid kind", bundle)
	}
}

// TestBlocksSave_Snapshot_SaveMemvidBlocks_Bad asserts the SaveMemvidBlocks alias
// surfaces the same nil-snapshot and nil-store guards as SaveStateBlocks.
func TestBlocksSave_Snapshot_SaveMemvidBlocks_Bad(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	var nilSnapshot *Snapshot
	if _, err := nilSnapshot.SaveMemvidBlocks(ctx, store, StateBlockOptions{}); err == nil {
		t.Fatal("SaveMemvidBlocks(nil snapshot) error = nil")
	}
	if _, err := kvSnapshotBlocksTestSnapshot().SaveMemvidBlocks(ctx, nil, StateBlockOptions{}); err == nil {
		t.Fatal("SaveMemvidBlocks(nil store) error = nil")
	}
}

// TestBlocksSave_Snapshot_SaveMemvidBlocks_Ugly asserts the SaveMemvidBlocks
// alias rejects an unsupported KV encoding, forwarding the encode-time guard.
func TestBlocksSave_Snapshot_SaveMemvidBlocks_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := kvSnapshotBlocksTestSnapshot().SaveMemvidBlocks(ctx, store, StateBlockOptions{KVEncoding: "q2"}); err == nil {
		t.Fatal("SaveMemvidBlocks(bad encoding) error = nil, want unsupported-encoding error")
	}
}

// TestBlocksSave_SaveStateBlocksFromStream_Good streams a single whole-snapshot
// block through SaveStateBlocksFromStream and asserts a bundle is produced.
func TestBlocksSave_SaveStateBlocksFromStream_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	bundle, err := SaveStateBlocksFromStream(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()})
		return err
	})
	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream() error = %v", err)
	}
	if len(bundle.Blocks) == 0 || bundle.SnapshotHash == "" {
		t.Fatalf("SaveStateBlocksFromStream() bundle = %+v, want at least one block + hash", bundle)
	}
}

// TestBlocksSave_Q8NativeStreamRoundTrip pins the DISK block round trip for the
// raw q8 KV payload (#1846 block lane): a q8-native block streamed through
// SaveStateBlocksFromStream (the native sleep lane) and loaded back per block
// carries the opaque int8-codes+f32-scales payload byte-exact, and the bundle
// hash is produced. The engine restores native blocks one at a time (never
// AssembleBlocks), so this locks the blocks_save / blocks_load codec that lane
// depends on — the encoded-size, stream-write, and decode arms of KVNativeDTypeQ8.
func TestBlocksSave_Q8NativeStreamRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	kBlock := make([]byte, 30)
	vBlock := make([]byte, 30)
	for i := range kBlock {
		kBlock[i] = byte(i*7 + 1)
		vBlock[i] = byte(i*11 + 3)
	}
	block := Block{Index: 0, TokenStart: 0, TokenCount: 3, Snapshot: &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        3,
		HeadDim:       4,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  "fixed",
			KeyDType:   KVNativeDTypeQ8,
			KeyBytes:   kBlock,
			KeyShape:   []int32{1, 2, 3, 4},
			ValueDType: KVNativeDTypeQ8,
			ValueBytes: vBlock,
			ValueShape: []int32{1, 2, 3, 4},
		}},
	}}

	bundle, err := SaveStateBlocksFromStream(ctx, store, StateBlockOptions{BlockSize: 4, KVEncoding: EncodingNative}, func(yield func(Block) (bool, error)) error {
		_, err := yield(block)
		return err
	})
	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream(q8-native) error = %v", err)
	}
	if len(bundle.Blocks) != 1 || bundle.SnapshotHash == "" {
		t.Fatalf("bundle = %+v, want one block + hash", bundle)
	}

	loaded, err := LoadStateBlockWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadStateBlockWithOptions(q8-native) error = %v", err)
	}
	if loaded.Snapshot == nil || len(loaded.Snapshot.Layers) != 1 {
		t.Fatalf("loaded block = %+v, want one layer", loaded)
	}
	layer := loaded.Snapshot.Layers[0]
	if layer.KeyDType != KVNativeDTypeQ8 || layer.ValueDType != KVNativeDTypeQ8 {
		t.Fatalf("loaded dtype key=%q value=%q, want %q", layer.KeyDType, layer.ValueDType, KVNativeDTypeQ8)
	}
	if !bytes.Equal(layer.KeyBytes, kBlock) || !bytes.Equal(layer.ValueBytes, vBlock) {
		t.Fatal("q8-native block payload diverged across the disk round trip")
	}
}

// TestBlocksSave_SaveStateBlocksFromStream_Bad asserts SaveStateBlocksFromStream
// rejects a nil store before consuming the stream.
func TestBlocksSave_SaveStateBlocksFromStream_Bad(t *testing.T) {
	ctx := context.Background()

	_, err := SaveStateBlocksFromStream(ctx, nil, StateBlockOptions{BlockSize: 2}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()})
		return err
	})
	if err == nil {
		t.Fatal("SaveStateBlocksFromStream(nil store) error = nil, want store error")
	}
}

// TestBlocksSave_SaveStateBlocksFromStream_Ugly asserts a stream callback that
// returns an error aborts the save and the error propagates.
func TestBlocksSave_SaveStateBlocksFromStream_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	boom := context.Canceled

	_, err := SaveStateBlocksFromStream(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8}, func(yield func(Block) (bool, error)) error {
		return boom
	})
	if err == nil {
		t.Fatal("SaveStateBlocksFromStream(failing stream) error = nil, want stream error")
	}
}

// TestBlocksSave_SaveMemvidBlocksFromStream_Good streams one block through the
// deprecated SaveMemvidBlocksFromStream alias and asserts a bundle is produced.
func TestBlocksSave_SaveMemvidBlocksFromStream_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	bundle, err := SaveMemvidBlocksFromStream(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()})
		return err
	})
	if err != nil || len(bundle.Blocks) == 0 {
		t.Fatalf("SaveMemvidBlocksFromStream() = %+v, err = %v, want a bundle", bundle, err)
	}
}

// TestBlocksSave_SaveMemvidBlocksFromStream_Bad asserts the deprecated
// SaveMemvidBlocksFromStream alias rejects a nil store.
func TestBlocksSave_SaveMemvidBlocksFromStream_Bad(t *testing.T) {
	ctx := context.Background()

	_, err := SaveMemvidBlocksFromStream(ctx, nil, StateBlockOptions{BlockSize: 2}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()})
		return err
	})
	if err == nil {
		t.Fatal("SaveMemvidBlocksFromStream(nil store) error = nil, want store error")
	}
}

// TestBlocksSave_SaveMemvidBlocksFromStream_Ugly asserts a failing stream
// callback aborts the deprecated SaveMemvidBlocksFromStream alias.
func TestBlocksSave_SaveMemvidBlocksFromStream_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	_, err := SaveMemvidBlocksFromStream(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8}, func(yield func(Block) (bool, error)) error {
		return context.Canceled
	})
	if err == nil {
		t.Fatal("SaveMemvidBlocksFromStream(failing stream) error = nil, want stream error")
	}
}

// TestBlocksSave_SaveStateBlockBundle_Good saves a manifest chunk for a valid
// bundle and asserts a non-zero chunk ref is returned.
func TestBlocksSave_SaveStateBlockBundle_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	ref, err := SaveStateBlockBundle(ctx, store, bundle, "mlx://session/manifest")
	if err != nil {
		t.Fatalf("SaveStateBlockBundle() error = %v", err)
	}
	if ref.ChunkID == 0 {
		t.Fatalf("SaveStateBlockBundle() ref = %+v, want written manifest chunk", ref)
	}
}

// TestBlocksSave_SaveStateBlockBundle_Bad covers the bundle-save guard branches:
// nil store, blank URI, and an invalid (empty) bundle.
func TestBlocksSave_SaveStateBlockBundle_Bad(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := SaveStateBlockBundle(ctx, nil, bundle, "mlx://x"); err == nil {
		t.Fatal("SaveStateBlockBundle(nil store) error = nil")
	}
	if _, err := SaveStateBlockBundle(ctx, store, bundle, "   "); err == nil {
		t.Fatal("SaveStateBlockBundle(blank URI) error = nil")
	}
	if _, err := SaveStateBlockBundle(ctx, store, &StateBlockBundle{}, "mlx://x"); err == nil {
		t.Fatal("SaveStateBlockBundle(invalid bundle) error = nil")
	}
}

// TestBlocksSave_SaveStateBlockBundle_Ugly asserts SaveStateBlockBundle rejects
// a nil bundle pointer rather than dereferencing it.
func TestBlocksSave_SaveStateBlockBundle_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := SaveStateBlockBundle(ctx, store, nil, "mlx://x"); err == nil {
		t.Fatal("SaveStateBlockBundle(nil bundle) error = nil, want validation error")
	}
}

// TestBlocksSave_SaveMemvidBlockBundle_Good saves a manifest via the deprecated
// SaveMemvidBlockBundle alias and asserts a non-zero chunk ref is returned.
func TestBlocksSave_SaveMemvidBlockBundle_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	ref, err := SaveMemvidBlockBundle(ctx, store, bundle, "mlx://session/memvid-manifest")
	if err != nil {
		t.Fatalf("SaveMemvidBlockBundle() error = %v", err)
	}
	if ref.ChunkID == 0 {
		t.Fatalf("SaveMemvidBlockBundle() ref = %+v, want written manifest chunk", ref)
	}
}

// TestBlocksSave_SaveMemvidBlockBundle_Bad asserts the deprecated
// SaveMemvidBlockBundle alias surfaces the nil-store and blank-URI guards.
func TestBlocksSave_SaveMemvidBlockBundle_Bad(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := SaveMemvidBlockBundle(ctx, nil, bundle, "mlx://x"); err == nil {
		t.Fatal("SaveMemvidBlockBundle(nil store) error = nil")
	}
	if _, err := SaveMemvidBlockBundle(ctx, store, bundle, "  "); err == nil {
		t.Fatal("SaveMemvidBlockBundle(blank URI) error = nil")
	}
}

// TestBlocksSave_SaveMemvidBlockBundle_Ugly asserts the deprecated
// SaveMemvidBlockBundle alias rejects an invalid (empty) bundle.
func TestBlocksSave_SaveMemvidBlockBundle_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := SaveMemvidBlockBundle(ctx, store, &MemvidBlockBundle{}, "mlx://x"); err == nil {
		t.Fatal("SaveMemvidBlockBundle(invalid bundle) error = nil, want validation error")
	}
}

// TestBlocksSave_TrustedReuseBoundary_Good asserts TrustedReuseBoundary returns
// the full reuse boundary when a trusted parent bundle's prefix blocks align
// with the block size.
func TestBlocksSave_TrustedReuseBoundary_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}

	boundary := TrustedReuseBoundary(StateBlockOptions{
		ReusePrefix:        parent,
		ReusePrefixTrusted: true,
		ReusePrefixTokens:  2,
	}, 2)
	if boundary != 2 {
		t.Fatalf("TrustedReuseBoundary() = %d, want 2 (one aligned prefix block)", boundary)
	}
}

// TestBlocksSave_TrustedReuseBoundary_Bad asserts TrustedReuseBoundary returns 0
// when reuse is not trusted, even with a valid parent bundle present.
func TestBlocksSave_TrustedReuseBoundary_Bad(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}

	if boundary := TrustedReuseBoundary(StateBlockOptions{ReusePrefix: parent, ReusePrefixTrusted: false}, 2); boundary != 0 {
		t.Fatalf("TrustedReuseBoundary(untrusted) = %d, want 0", boundary)
	}
}

// TestBlocksSave_TrustedReuseBoundary_Ugly asserts TrustedReuseBoundary returns
// 0 for a nil parent and for a parent whose block size disagrees with the
// requested block size.
func TestBlocksSave_TrustedReuseBoundary_Ugly(t *testing.T) {
	if boundary := TrustedReuseBoundary(StateBlockOptions{ReusePrefixTrusted: true, ReusePrefix: nil}, 2); boundary != 0 {
		t.Fatalf("TrustedReuseBoundary(nil parent) = %d, want 0", boundary)
	}

	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}
	// Parent BlockSize is 2; requesting boundary for block size 4 mismatches.
	if boundary := TrustedReuseBoundary(StateBlockOptions{ReusePrefix: parent, ReusePrefixTrusted: true}, 4); boundary != 0 {
		t.Fatalf("TrustedReuseBoundary(block-size mismatch) = %d, want 0", boundary)
	}
}

// TestBlocksSave_DedupStore_Good_SharesPrefixWritesOnlyDivergentTail is the
// cross-conversation store-savings receipt: two DIFFERENT conversations (no
// parent-prefix reuse wired between them) that share a leading token span sleep
// into a content-addressed DedupStore, and the second physically writes ONLY its
// divergent-tail blocks — the shared prefix blocks dedup to the first
// conversation's chunks. The second still wakes byte-identical.
func TestBlocksSave_DedupStore_Good_SharesPrefixWritesOnlyDivergentTail(t *testing.T) {
	ctx := context.Background()
	inner := state.NewInMemoryStore(nil)
	store := state.NewDedupStore(inner)

	// Same first four tokens (blocks 0,1), divergent last four (blocks 2,3).
	convA := stateTokenOnlyTestSnapshot([]int32{1, 2, 3, 4, 5, 6, 7, 8}, 8, 2)
	convB := stateTokenOnlyTestSnapshot([]int32{1, 2, 3, 4, 90, 91, 92, 93}, 8, 2)
	opts := func(uri string) StateBlockOptions {
		return StateBlockOptions{BlockSize: 2, KVEncoding: EncodingNative, URI: uri}
	}

	bundleA, err := convA.SaveStateBlocks(ctx, store, opts("mlx://conv-a"))
	if err != nil {
		t.Fatalf("SaveStateBlocks(A) error = %v", err)
	}
	afterA := inner.ChunkCount()
	bundleB, err := convB.SaveStateBlocks(ctx, store, opts("mlx://conv-b"))
	if err != nil {
		t.Fatalf("SaveStateBlocks(B) error = %v", err)
	}
	afterB := inner.ChunkCount()

	if len(bundleA.Blocks) != 4 || len(bundleB.Blocks) != 4 {
		t.Fatalf("block counts = A %d / B %d, want 4 each", len(bundleA.Blocks), len(bundleB.Blocks))
	}
	// The savings: B's sleep added only its two divergent-tail chunks.
	if added := afterB - afterA; added != 2 {
		t.Fatalf("B added %d physical chunks, want 2 (only the divergent tail)", added)
	}
	if stats := store.Stats(); stats.Dedups != 2 {
		t.Fatalf("stats.Dedups = %d, want 2 (both shared prefix blocks)", stats.Dedups)
	}
	// The two shared prefix blocks point at the SAME chunks; the tail diverges.
	if bundleB.Blocks[0].State.ChunkID != bundleA.Blocks[0].State.ChunkID ||
		bundleB.Blocks[1].State.ChunkID != bundleA.Blocks[1].State.ChunkID {
		t.Fatalf("shared prefix blocks not shared: A %d,%d B %d,%d",
			bundleA.Blocks[0].State.ChunkID, bundleA.Blocks[1].State.ChunkID,
			bundleB.Blocks[0].State.ChunkID, bundleB.Blocks[1].State.ChunkID)
	}
	if bundleB.Blocks[2].State.ChunkID == bundleA.Blocks[2].State.ChunkID {
		t.Fatal("divergent-tail block shared a chunk, want distinct")
	}
	// B wakes byte-identical (the loader re-verifies each block hash on read).
	assertLoadedTokens(t, ctx, store, bundleB, convB.Tokens)
}

// TestBlocksSave_DedupStore_Good_ReclaimKeepsSharedBreaksNothing is the
// reclaim-safety receipt: after B dedups against A, reclaiming A physically frees
// only A's private-tail chunks; the shared prefix (still referenced by B) and B's
// own tail survive, so B wakes byte-identical — while A, whose tail is gone, no
// longer loads.
func TestBlocksSave_DedupStore_Good_ReclaimKeepsSharedBreaksNothing(t *testing.T) {
	ctx := context.Background()
	store := state.NewDedupStore(state.NewInMemoryStore(nil))

	convA := stateTokenOnlyTestSnapshot([]int32{1, 2, 3, 4, 5, 6, 7, 8}, 8, 2)
	convB := stateTokenOnlyTestSnapshot([]int32{1, 2, 3, 4, 90, 91, 92, 93}, 8, 2)
	opts := func(uri string) StateBlockOptions {
		return StateBlockOptions{BlockSize: 2, KVEncoding: EncodingNative, URI: uri}
	}
	bundleA, err := convA.SaveStateBlocks(ctx, store, opts("mlx://conv-a"))
	if err != nil {
		t.Fatalf("SaveStateBlocks(A) error = %v", err)
	}
	bundleB, err := convB.SaveStateBlocks(ctx, store, opts("mlx://conv-b"))
	if err != nil {
		t.Fatalf("SaveStateBlocks(B) error = %v", err)
	}
	// B wakes byte-identical before any reclaim.
	assertLoadedTokens(t, ctx, store, bundleB, convB.Tokens)

	// Reclaim A: release the block refs A's bundle recorded.
	refs := make([]state.ChunkRef, 0, len(bundleA.Blocks))
	for _, blk := range bundleA.Blocks {
		refs = append(refs, StateBlockChunkRef(blk))
	}
	if err := store.Release(ctx, refs...); err != nil {
		t.Fatalf("Release(A) error = %v", err)
	}
	// Only A's two private-tail blocks reached refcount 0; the shared prefix
	// (referenced by B) stayed.
	if stats := store.Stats(); stats.Reclaimed != 2 {
		t.Fatalf("stats.Reclaimed = %d, want 2 (A's private tail only)", stats.Reclaimed)
	}
	// B still wakes byte-identical across A's reclaim.
	assertLoadedTokens(t, ctx, store, bundleB, convB.Tokens)
	// A itself can no longer load — its tail chunks were physically reclaimed,
	// proving the reclaim actually happened rather than being a no-op.
	if _, err := LoadFromStateBlocks(ctx, store, bundleA); err == nil {
		t.Fatal("LoadFromStateBlocks(A) error = nil after reclaim, want its tail gone")
	}
}

// assertLoadedTokens loads bundle through store and fails unless the restored
// token sequence equals want.
func assertLoadedTokens(t *testing.T, ctx context.Context, store state.Store, bundle *StateBlockBundle, want []int32) {
	t.Helper()
	loaded, err := LoadFromStateBlocks(ctx, store, bundle)
	if err != nil {
		t.Fatalf("LoadFromStateBlocks error = %v", err)
	}
	if len(loaded.Tokens) != len(want) {
		t.Fatalf("loaded tokens = %v, want %v", loaded.Tokens, want)
	}
	for i := range want {
		if loaded.Tokens[i] != want[i] {
			t.Fatalf("loaded tokens = %v, want %v", loaded.Tokens, want)
		}
	}
}
