// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"
)

func BenchmarkLoadPrefixFromStateBlocks_MixedWindowThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}

func BenchmarkLoadPrefixTokensFromStateBlocks_MixedWindowThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		tokens, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkTokens = tokens
	}
}

func BenchmarkLoadPrefixFromStateBlocks_NativeLayerSingleHeadSlabThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkNativeLayerSlabStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}

// BenchmarkLoadFromStateBlocks_NativeLayerMultiHeadSlabThreeBlocks wakes a
// grouped-query (multi-KV-head) native bundle — the engine/metal capture shape.
// It drives the interleaved multi-head layer-raw assembly, which folds each
// block at a fill cursor rather than rebuilding the whole tensor per block.
func BenchmarkLoadFromStateBlocks_NativeLayerMultiHeadSlabThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkNativeLayerMultiHeadSlabStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadFromStateBlocks(ctx, store, bundle)
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}

func BenchmarkLoadPrefixFromStateBlocks_NativeLayerMultiHeadSlabThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkNativeLayerMultiHeadSlabStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}

func BenchmarkLoadPrefixFromStateBlocks_NativeLayerSingleHeadSlabPartialPrefix(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkNativeLayerSlabStateBlocksFixture(b)
	prefixTokens := 1024
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		if len(snapshot.Tokens) != prefixTokens {
			b.Fatalf("tokens = %d, want %d", len(snapshot.Tokens), prefixTokens)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}
