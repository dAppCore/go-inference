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
