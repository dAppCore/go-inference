// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"testing"
)

// TestBlocksAssemble_AssembleBlocks_Good splits a snapshot into blocks then
// reassembles them, asserting AssembleBlocks recovers the full token stream.
func TestBlocksAssemble_AssembleBlocks_Good(t *testing.T) {
	source := kvSnapshotBlocksTestSnapshot()
	blocks, err := source.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}

	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		t.Fatalf("AssembleBlocks() error = %v", err)
	}
	if len(assembled.Tokens) != 4 || assembled.Tokens[3] != 4 {
		t.Fatalf("assembled = %+v, want four tokens", assembled)
	}
}

// TestBlocksAssemble_AssembleBlocks_WindowedMultiHeadByteIdentical splits then
// reassembles a grouped-query snapshot that mixes a full-attention layer (data
// in every block) with a sliding-window layer whose cache covers only the last
// tokens — the global+sliding shape engine/metal captures for a 31B model. The
// windowed layer's leading blocks carry NO data, so block 0 is the wrong shape
// donor for it; without adopting a later block's shape the assembled layer got
// no placement buffer and appendKVSnapshotLayerRawBlock fell to the O(N^2)
// merged-rebuild path. The round-trip must stay byte-identical either way, and
// with multiple windowed data blocks the placement path is exercised directly.
func TestBlocksAssemble_AssembleBlocks_WindowedMultiHeadByteIdentical(t *testing.T) {
	const seqLen, blockSize, heads, window = 6, 2, 2, 4
	seqBytes := func(n int, base byte) []byte {
		out := make([]byte, n)
		for i := range out {
			out[i] = base + byte(i)
		}
		return out
	}
	source := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4, 5, 6},
		TokenOffset:   seqLen,
		NumLayers:     2,
		NumHeads:      heads,
		SeqLen:        seqLen,
		HeadDim:       1,
		NumQueryHeads: heads,
		Layers: []LayerSnapshot{
			{
				Layer: 0, CacheIndex: 0,
				KeyDType: "float16", KeyBytes: seqBytes(heads*seqLen*2, 1), KeyShape: []int32{1, heads, seqLen, 1},
				ValueDType: "float16", ValueBytes: seqBytes(heads*seqLen*2, 101), ValueShape: []int32{1, heads, seqLen, 1},
				Heads: make([]HeadSnapshot, heads),
			},
			{
				Layer: 1, CacheIndex: 1, MaxSize: window,
				KeyDType: "float16", KeyBytes: seqBytes(heads*window*2, 41), KeyShape: []int32{1, heads, window, 1},
				ValueDType: "float16", ValueBytes: seqBytes(heads*window*2, 151), ValueShape: []int32{1, heads, window, 1},
				Heads: make([]HeadSnapshot, heads),
			},
		},
	}
	blocks, err := source.SplitBlocks(blockSize)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}
	if len(blocks) != 3 {
		t.Fatalf("blocks = %d, want 3", len(blocks))
	}
	// The shape the fix targets: the windowed layer's first block is empty, and
	// the window spans more than one block, so >1 data block folds in.
	if got := len(blocks[0].Snapshot.Layers[1].KeyBytes); got != 0 {
		t.Fatalf("block[0] windowed layer bytes = %d, want 0 (empty leading block)", got)
	}
	if got := len(blocks[1].Snapshot.Layers[1].KeyBytes); got == 0 {
		t.Fatal("block[1] windowed layer bytes = 0, want the window's first data block")
	}
	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		t.Fatalf("AssembleBlocks() error = %v", err)
	}
	for i, want := range source.Layers {
		got := assembled.Layers[i]
		if !equalBytes(got.KeyBytes, want.KeyBytes) {
			t.Fatalf("layer %d KeyBytes = %v, want %v", i, got.KeyBytes, want.KeyBytes)
		}
		if !equalBytes(got.ValueBytes, want.ValueBytes) {
			t.Fatalf("layer %d ValueBytes = %v, want %v", i, got.ValueBytes, want.ValueBytes)
		}
		if !equalInt32s(got.KeyShape, want.KeyShape) || !equalInt32s(got.ValueShape, want.ValueShape) {
			t.Fatalf("layer %d shapes = %v/%v, want %v/%v", i, got.KeyShape, got.ValueShape, want.KeyShape, want.ValueShape)
		}
	}
}

// TestBlocksAssemble_AssembleBlocks_Bad asserts AssembleBlocks rejects an empty
// block slice and a block carrying a nil snapshot.
func TestBlocksAssemble_AssembleBlocks_Bad(t *testing.T) {
	if _, err := AssembleBlocks(nil); err == nil {
		t.Fatal("AssembleBlocks(nil) error = nil")
	}
	if _, err := AssembleBlocks([]Block{{Index: 0, TokenStart: 0, TokenCount: 1, Snapshot: nil}}); err == nil {
		t.Fatal("AssembleBlocks(nil snapshot block) error = nil")
	}
}

// TestBlocksAssemble_AssembleBlocks_Ugly asserts AssembleBlocks rejects blocks
// presented out of contiguous order (the order-validation guard).
func TestBlocksAssemble_AssembleBlocks_Ugly(t *testing.T) {
	source := kvSnapshotBlocksTestSnapshot()
	blocks, err := source.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}

	disordered := []Block{blocks[1], blocks[0]}
	if _, err := AssembleBlocks(disordered); err == nil {
		t.Fatal("AssembleBlocks(non-contiguous) error = nil")
	}
}
