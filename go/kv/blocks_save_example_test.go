// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// ExampleSnapshot_SaveStateBlocks_native saves a native-dtype snapshot to a
// State store with EncodingNative, then reloads it raw-only. The durable
// save→load→assemble path reconstructs the layer-level slabs from the stored
// blocks without re-expanding them into per-head vectors.
func ExampleSnapshot_SaveStateBlocks_native() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	source := exampleNativeLayerSnapshot()

	bundle, err := source.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://session/native",
	})
	if err != nil {
		core.Println("save error:", err)
		return
	}

	loaded, err := LoadFromStateBlocksWithOptions(ctx, store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		core.Println("load error:", err)
		return
	}
	layer := loaded.Layers[0]
	core.Println("blocks:", len(bundle.Blocks))
	core.Println("value bytes recovered:", equalBytes(layer.ValueBytes, source.Layers[0].ValueBytes))
	core.Println("per-head bytes empty:", len(layer.Heads[0].KeyBytes) == 0)
	// Output:
	// blocks: 2
	// value bytes recovered: true
	// per-head bytes empty: true
}

// ExampleSnapshot_SaveStateBlocks splits a snapshot into fixed-size KV blocks
// and writes each one to a State store, returning a manifest.
func ExampleSnapshot_SaveStateBlocks() {
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/blocks",
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("blocks:", len(bundle.Blocks))
	// Output: blocks: 2
}

// ExampleSnapshot_SaveMemvidBlocks saves a snapshot via the deprecated
// memvid-named alias, which forwards to SaveStateBlocks.
func ExampleSnapshot_SaveMemvidBlocks() {
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveMemvidBlocks(context.Background(), store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("blocks:", len(bundle.Blocks))
	// Output: blocks: 2
}

// ExampleSaveStateBlocksFromStream saves blocks yielded one at a time by a
// generator, avoiding holding the whole snapshot's blocks in memory.
func ExampleSaveStateBlocksFromStream() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := SaveStateBlocksFromStream(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()})
		return err
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("blocks:", len(bundle.Blocks) > 0)
	// Output: blocks: true
}

// ExampleSaveMemvidBlocksFromStream streams blocks via the deprecated
// memvid-named alias.
func ExampleSaveMemvidBlocksFromStream() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := SaveMemvidBlocksFromStream(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()})
		return err
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("blocks:", len(bundle.Blocks) > 0)
	// Output: blocks: true
}

// ExampleTrustedReuseBoundary computes how many leading tokens of a trusted
// parent bundle a child save can reuse without re-capturing them.
func ExampleTrustedReuseBoundary() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8})
	if err != nil {
		core.Println("error:", err)
		return
	}
	boundary := TrustedReuseBoundary(StateBlockOptions{ReusePrefix: parent, ReusePrefixTrusted: true, ReusePrefixTokens: 2}, 2)
	core.Println("reuse boundary:", boundary)
	// Output: reuse boundary: 2
}

// ExampleSaveStateBlockBundle writes a bundle manifest chunk to the State store
// so the block layout can be resolved later by URI.
func ExampleSaveStateBlockBundle() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8})
	if err != nil {
		core.Println("error:", err)
		return
	}
	ref, err := SaveStateBlockBundle(ctx, store, bundle, "mlx://session/manifest")
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("written:", ref.ChunkID > 0)
	// Output: written: true
}

// ExampleSaveMemvidBlockBundle writes a bundle manifest via the deprecated
// memvid-named alias.
func ExampleSaveMemvidBlockBundle() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8})
	if err != nil {
		core.Println("error:", err)
		return
	}
	ref, err := SaveMemvidBlockBundle(ctx, store, bundle, "mlx://session/memvid-manifest")
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("written:", ref.ChunkID > 0)
	// Output: written: true
}
