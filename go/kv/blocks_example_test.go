// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
)

// exampleNativeLayerSnapshot builds a 4-token, 2-head snapshot whose K/V live
// as layer-level native float16 slabs ([B,H,L,D] = [1,2,4,1]) rather than
// per-head float32 vectors — the raw-tensor capture shape an MLX layer-cache
// export produces.
func exampleNativeLayerSnapshot() *Snapshot {
	keyBytes := []byte{
		1, 0, 2, 0, 3, 0, 4, 0,
		5, 0, 6, 0, 7, 0, 8, 0,
	}
	valueBytes := []byte{
		11, 0, 12, 0, 13, 0, 14, 0,
		15, 0, 16, 0, 17, 0, 18, 0,
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        4,
		HeadDim:       1,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{1, 2, 4, 1},
			ValueDType: "float16",
			ValueBytes: valueBytes,
			ValueShape: []int32{1, 2, 4, 1},
			Heads:      make([]HeadSnapshot, 2),
		}},
	}
}

// ExampleEffectiveSeqLen reports the effective sequence length, preferring the
// recorded SeqLen and falling back to the token count.
func ExampleEffectiveSeqLen() {
	core.Println(EffectiveSeqLen(&Snapshot{SeqLen: 7}))
	// Output: 7
}

// Example_loadStateBlock shows the canonical State block load path: save a
// snapshot as blocks, then read one block back into a Snapshot.
func Example_loadStateBlock() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	snapshot := &Snapshot{
		Version: SnapshotVersion, Architecture: "gemma4_text",
		Tokens: []int32{1, 2}, TokenOffset: 2,
		NumLayers: 1, NumHeads: 1, SeqLen: 2, HeadDim: 2, NumQueryHeads: 1,
		Layers: []LayerSnapshot{{Heads: []HeadSnapshot{{
			Key: []float32{1, 0, 0, 1}, Value: []float32{0, 1, 1, 0},
		}}}},
	}
	bundle, err := snapshot.SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, URI: "mlx://ex"})
	if err != nil {
		core.Println("error:", err)
		return
	}
	block, err := LoadStateBlockWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", block.TokenCount)
	// Output: tokens: 2
}

// ExampleSnapshot_SplitBlocks splits a four-token snapshot into two-token blocks
// for incremental durable storage.
func ExampleSnapshot_SplitBlocks() {
	blocks, err := kvSnapshotBlocksTestSnapshot().SplitBlocks(2)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("blocks:", len(blocks))
	// Output: blocks: 2
}

// ExampleSnapshot_RangeBlocks iterates a snapshot's blocks, stopping early when
// the callback returns false.
func ExampleSnapshot_RangeBlocks() {
	count := 0
	err := kvSnapshotBlocksTestSnapshot().RangeBlocks(1, func(Block) bool {
		count++
		return count < 2
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("visited:", count)
	// Output: visited: 2
}

// ExampleSnapshot_SliceBlock extracts a token window from a snapshot as a new
// standalone snapshot.
func ExampleSnapshot_SliceBlock() {
	slice, err := kvSnapshotBlocksTestSnapshot().SliceBlock(0, 2, 0, false)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", len(slice.Tokens))
	// Output: tokens: 2
}

// ExampleValidateStateBlockBundle checks a bundle manifest for structural
// validity before saving or loading it.
func ExampleValidateStateBlockBundle() {
	err := ValidateStateBlockBundle(&StateBlockBundle{})
	core.Println("empty bundle valid:", err == nil)
	// Output: empty bundle valid: false
}

// ExampleValidateMemvidBlockBundle validates a manifest via the deprecated
// memvid-named alias.
func ExampleValidateMemvidBlockBundle() {
	err := ValidateMemvidBlockBundle(&MemvidBlockBundle{})
	core.Println("empty bundle valid:", err == nil)
	// Output: empty bundle valid: false
}

// ExampleClearTerminalState strips the generated tokens and logits from a
// snapshot so a resumed session starts from a clean prompt boundary.
func ExampleClearTerminalState() {
	snapshot := kvSnapshotBlocksTestSnapshot()
	ClearTerminalState(snapshot)
	core.Println("generated cleared:", snapshot.Generated == nil)
	core.Println("logits cleared:", snapshot.Logits == nil)
	// Output:
	// generated cleared: true
	// logits cleared: true
}

// ExampleLoadStateBlockWithOptions loads a single durable block back into a
// Block value with explicit decode options.
func ExampleLoadStateBlockWithOptions() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, URI: "mlx://ex-lsbwo"})
	if err != nil {
		core.Println("error:", err)
		return
	}
	block, err := LoadStateBlockWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("token count:", block.TokenCount)
	// Output: token count: 2
}

// ExampleLoadMemvidBlockWithOptions loads a single block via the deprecated
// memvid-named alias.
func ExampleLoadMemvidBlockWithOptions() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, URI: "mlx://ex-lmbwo"})
	if err != nil {
		core.Println("error:", err)
		return
	}
	block, err := LoadMemvidBlockWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("token count:", block.TokenCount)
	// Output: token count: 2
}

// ExampleLoadStateBlockTokens reads only the token IDs of a durable block,
// skipping K/V tensor assembly.
func ExampleLoadStateBlockTokens() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, URI: "mlx://ex-lsbt"})
	if err != nil {
		core.Println("error:", err)
		return
	}
	block, err := LoadStateBlockTokens(ctx, store, bundle.Blocks[0])
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", len(block.Tokens))
	// Output: tokens: 2
}

// ExampleLoadStateBlockTokensWithOptions reads a block's token IDs with explicit
// decode options.
func ExampleLoadStateBlockTokensWithOptions() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, URI: "mlx://ex-lsbtwo"})
	if err != nil {
		core.Println("error:", err)
		return
	}
	block, err := LoadStateBlockTokensWithOptions(ctx, store, bundle.Blocks[1], LoadOptions{})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("token start:", block.TokenStart)
	// Output: token start: 2
}

// ExampleStateBlockChunkRef resolves a block ref to its underlying State chunk
// ref, preferring the State ref over the deprecated memvid ref.
func ExampleStateBlockChunkRef() {
	ref := StateBlockRef{State: state.ChunkRef{ChunkID: 42}}
	core.Println("chunk:", StateBlockChunkRef(ref).ChunkID)
	// Output: chunk: 42
}
