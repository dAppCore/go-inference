// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/model/state"
)

// These benches cover the Load/restore surface the bench-coverage audit
// flagged as 0%-bench-covered: the bundle-by-URI resolver, the deprecated
// memvid-named forwarders (driven BY NAME so their entry points register as
// covered — a forwarder is not marked covered by benching its State target),
// the single-block loaders, the token-only block loader, and Snapshot.Clone.
//
// All fixtures are in-memory (state.NewInMemoryStore) over the existing
// benchmarkStateBlocksFixture / benchmarkNativeLayerSlabStateBlocksFixture
// helpers — no model, no disk.

var (
	stateBlocksBenchmarkBlock      Block
	stateBlocksBenchmarkTokenBlock StateTokenBlock
	stateBlocksBenchmarkBundle     *StateBlockBundle
)

// benchmarkStateBlockBundleURIFixture saves a manifest to a URI in the same
// in-memory store as its blocks, so LoadStateBlockBundle can resolve it.
func benchmarkStateBlockBundleURIFixture(tb testing.TB) (state.Store, string) {
	tb.Helper()
	store, bundle := benchmarkStateBlocksFixture(tb)
	const uri = "mlx://bench/manifest"
	writer, ok := store.(state.Writer)
	if !ok {
		tb.Fatalf("benchmark store %T does not implement state.Writer", store)
	}
	if _, err := SaveStateBlockBundle(context.Background(), writer, bundle, uri); err != nil {
		tb.Fatalf("SaveStateBlockBundle() error = %v", err)
	}
	return store, uri
}

// --- bundle-by-URI resolve (LoadStateBlockBundle: resolve + JSON parse + validate) ---

func BenchmarkLoadStateBlockBundle_ThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, uri := benchmarkStateBlockBundleURIFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		bundle, err := LoadStateBlockBundle(ctx, store, uri)
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkBundle = bundle
	}
}

// --- full-snapshot load entry points (named, including the deprecated forwarders) ---

func BenchmarkLoadFromStateBlocks_ThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadFromStateBlocks(ctx, store, bundle)
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}

func BenchmarkLoadFromMemvidBlocks_ThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadFromMemvidBlocks(ctx, store, bundle) //nolint:staticcheck // deprecated forwarder, benched by name for coverage
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}

func BenchmarkLoadFromMemvidBlocksWithOptions_ThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadFromMemvidBlocksWithOptions(ctx, store, bundle, LoadOptions{RawKVOnly: true}) //nolint:staticcheck // deprecated forwarder, benched by name for coverage
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}

// --- prefix load entry points (the bare State/Memvid entries, not ...WithOptions) ---

func BenchmarkLoadPrefixFromStateBlocks_FullThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadPrefixFromStateBlocks(ctx, store, bundle, bundle.TokenCount)
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}

func BenchmarkLoadPrefixFromMemvidBlocks_FullThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snapshot, err := LoadPrefixFromMemvidBlocks(ctx, store, bundle, bundle.TokenCount) //nolint:staticcheck // deprecated forwarder, benched by name for coverage
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkSnapshot = snapshot
	}
}

// --- single-block loaders (raw native fast-path is the production shape) ---

func BenchmarkLoadStateBlockWithOptions_SingleBlock(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkNativeLayerSlabStateBlocksFixture(b)
	ref := bundle.Blocks[0]
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		block, err := LoadStateBlockWithOptions(ctx, store, ref, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkBlock = block
	}
}

func BenchmarkLoadMemvidBlockWithOptions_SingleBlock(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkNativeLayerSlabStateBlocksFixture(b)
	ref := bundle.Blocks[0]
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		block, err := LoadMemvidBlockWithOptions(ctx, store, ref, LoadOptions{RawKVOnly: true}) //nolint:staticcheck // deprecated forwarder, benched by name for coverage
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkBlock = block
	}
}

// --- token-only single-block loader (LoadStateBlockTokens: raw fast-path) ---

func BenchmarkLoadStateBlockTokens_SingleBlock(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	ref := bundle.Blocks[0]
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		block, err := LoadStateBlockTokens(ctx, store, ref)
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkTokenBlock = block
	}
}

// --- Snapshot.Clone deep copy over an assembled multi-block snapshot ---

func BenchmarkSnapshotClone_AssembledThreeBlocks(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkStateBlocksFixture(b)
	snapshot, err := LoadFromStateBlocks(ctx, store, bundle)
	if err != nil {
		b.Fatalf("LoadFromStateBlocks() error = %v", err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateBlocksBenchmarkSnapshot = snapshot.Clone()
	}
}

func BenchmarkSnapshotClone_NativeLayerSlab(b *testing.B) {
	ctx := context.Background()
	store, bundle := benchmarkNativeLayerSlabStateBlocksFixture(b)
	snapshot, err := LoadFromStateBlocks(ctx, store, bundle)
	if err != nil {
		b.Fatalf("LoadFromStateBlocks() error = %v", err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateBlocksBenchmarkSnapshot = snapshot.Clone()
	}
}
