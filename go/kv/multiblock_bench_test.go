// SPDX-Licence-Identifier: EUPL-1.2

// Multi-block path benches. Existing blocks_benchmark_test.go covers
// the 3-block load case; this file widens coverage along block count
// (3 / 5 / 10), the SliceBlock primitive at varying boundaries, and
// the walkBlocks traversal cost via RangeBlocks.
//
// Coverage map (W7-F deepening pass):
//
//   - SaveStateBlocks + LoadFromStateBlocks @ 3 / 5 / 10 blocks — block
//     count scaling on the persisted path (W7-A inlined LoadFromStateBlocks
//     stream-assembly, so this bench should resolve linear in blocks).
//   - SliceBlock at left edge (0..256), middle (1024..1536), and right
//     edge (1792..2048) — slice arithmetic + per-head cloneSlices cost
//     vs. layer-window overlap.
//   - SplitBlocks at 512 / 256 / 128 block sizes — exercises the
//     blockBoundaries + walkBlocks(includeHash=true) clone path.
//   - RangeBlocks streaming — zero-retention iteration cost, the path
//     SaveStateBlocksFromStream uses for streamed checkpoints.
//   - LoadPrefixFromStateBlocks at half / 3/4 / full prefix — measures
//     the partial-restore branch's trim-via-SliceBlock cost.
//
// Run: go test -bench='BenchmarkMultiblock' -benchmem -run='^$' ./go/kv

package kv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/model/state"
)

// --- SaveStateBlocks + LoadFromStateBlocks block-count scaling ---

func BenchmarkMultiblock_SaveAndLoad_3Blocks(b *testing.B) {
	benchSaveLoadStateBlocks(b, 1536, 512)
}

func BenchmarkMultiblock_SaveAndLoad_5Blocks(b *testing.B) {
	benchSaveLoadStateBlocks(b, 2560, 512)
}

func BenchmarkMultiblock_SaveAndLoad_10Blocks(b *testing.B) {
	benchSaveLoadStateBlocks(b, 5120, 512)
}

func benchSaveLoadStateBlocks(b *testing.B, tokens, blockSize int) {
	b.Helper()
	snap := benchSnapshot(tokens)
	opts := StateBlockOptions{BlockSize: blockSize, KVEncoding: EncodingNative}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		bundle, err := snap.SaveStateBlocks(ctx, store, opts)
		if err != nil {
			b.Fatal(err)
		}
		restored, err := LoadFromStateBlocks(ctx, store, bundle)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = restored
	}
}

// --- SliceBlock at varying boundaries ---

func BenchmarkMultiblock_SliceBlock_LeftEdge(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := snap.SliceBlock(0, 256, 0, false)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

func BenchmarkMultiblock_SliceBlock_Middle(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := snap.SliceBlock(1024, 1536, 0, false)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

func BenchmarkMultiblock_SliceBlock_RightEdge(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := snap.SliceBlock(1792, 2048, 0, true)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

// --- SplitBlocks @ varying block sizes (cloneSlices=true) ---

func BenchmarkMultiblock_SplitBlocks_512(b *testing.B) {
	benchSplitBlocks(b, 2048, 512)
}

func BenchmarkMultiblock_SplitBlocks_256(b *testing.B) {
	benchSplitBlocks(b, 2048, 256)
}

func BenchmarkMultiblock_SplitBlocks_128(b *testing.B) {
	benchSplitBlocks(b, 2048, 128)
}

func benchSplitBlocks(b *testing.B, tokens, blockSize int) {
	b.Helper()
	snap := benchSnapshot(tokens)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blocks, err := snap.SplitBlocks(blockSize)
		if err != nil {
			b.Fatal(err)
		}
		if len(blocks) == 0 {
			b.Fatal("expected blocks > 0")
		}
		benchSinkSnapshot = blocks[0].Snapshot
	}
}

// --- RangeBlocks (streaming, zero-retention) ---

func BenchmarkMultiblock_RangeBlocks_2048Tokens_Bsz256(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var count int
		err := snap.RangeBlocks(256, func(block Block) bool {
			count++
			benchSinkSnapshot = block.Snapshot
			return true
		})
		if err != nil {
			b.Fatal(err)
		}
		if count == 0 {
			b.Fatal("expected count > 0")
		}
	}
}

// --- LoadPrefixFromStateBlocks at varying prefix sizes ---

func BenchmarkMultiblock_LoadPrefix_HalfBlocks(b *testing.B) {
	benchLoadPrefixStateBlocks(b, 2560, 512, 1280) // 5 blocks, take ~2.5
}

func BenchmarkMultiblock_LoadPrefix_ThreeQuarterBlocks(b *testing.B) {
	benchLoadPrefixStateBlocks(b, 2560, 512, 1920) // 5 blocks, take 3.75
}

func benchLoadPrefixStateBlocks(b *testing.B, tokens, blockSize, prefix int) {
	b.Helper()
	snap := benchSnapshot(tokens)
	opts := StateBlockOptions{BlockSize: blockSize, KVEncoding: EncodingNative}
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := snap.SaveStateBlocks(ctx, store, opts)
	if err != nil {
		b.Fatalf("SaveStateBlocks: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefix, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}
