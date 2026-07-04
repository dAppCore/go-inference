// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// cloneBundleShallow copies a bundle and its block slice so a test can mutate
// refs without corrupting the shared fixture for parallel sub-tests.
func cloneBundleShallow(b *StateBlockBundle) *StateBlockBundle {
	out := *b
	out.Blocks = append([]StateBlockRef(nil), b.Blocks...)
	return &out
}

// TestBlocksLoadCover_NilContextDefaults drives the ctx == nil → Background()
// fallback in each public loader entry point. Each call succeeds with a real
// bundle, so the only branch under test is the nil-ctx default.
func TestBlocksLoadCover_NilContextDefaults(t *testing.T) {
	store, bundle := kvSnapshotBlocksTestBundle(t)
	if _, err := SaveStateBlockBundle(context.Background(), store, bundle, "mlx://session/manifest"); err != nil {
		t.Fatalf("SaveStateBlockBundle() error = %v", err)
	}

	if _, err := LoadStateBlockBundle(nil, store, "mlx://session/manifest"); err != nil { //nolint:staticcheck
		t.Fatalf("LoadStateBlockBundle(nil ctx) error = %v", err)
	}
	if _, err := LoadFromStateBlocksWithOptions(nil, store, bundle, LoadOptions{}); err != nil { //nolint:staticcheck
		t.Fatalf("LoadFromStateBlocksWithOptions(nil ctx) error = %v", err)
	}
	if _, err := LoadPrefixFromStateBlocksWithOptions(nil, store, bundle, 2, LoadOptions{}); err != nil { //nolint:staticcheck
		t.Fatalf("LoadPrefixFromStateBlocksWithOptions(nil ctx) error = %v", err)
	}
	if _, err := LoadPrefixTokensFromStateBlocksWithOptions(nil, store, bundle, 2, LoadOptions{}); err != nil { //nolint:staticcheck
		t.Fatalf("LoadPrefixTokensFromStateBlocksWithOptions(nil ctx) error = %v", err)
	}
}

// TestBlocksLoadCover_LoadStateBlockBundle_Errors drives the bundle-resolve
// error arms: a missing URI, a blank URI, and an unresolvable URI.
func TestBlocksLoadCover_LoadStateBlockBundle_Errors(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := LoadStateBlockBundle(ctx, nil, "mlx://x"); err == nil {
		t.Fatal("LoadStateBlockBundle(nil store) error = nil, want store error")
	}
	if _, err := LoadStateBlockBundle(ctx, store, "   "); err == nil {
		t.Fatal("LoadStateBlockBundle(blank uri) error = nil, want uri error")
	}
	if _, err := LoadStateBlockBundle(ctx, store, "mlx://session/missing"); err == nil {
		t.Fatal("LoadStateBlockBundle(unresolvable uri) error = nil, want resolve error")
	}
}

// TestBlocksLoadCover_FromStateBlocks_BundleGuards drives the bundle-shape
// guards of LoadFromStateBlocksWithOptions: bad version, bad kind, no blocks.
func TestBlocksLoadCover_FromStateBlocks_BundleGuards(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	badVersion := cloneBundleShallow(bundle)
	badVersion.Version = StateBlockVersion + 1
	if _, err := LoadFromStateBlocksWithOptions(ctx, store, badVersion, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateBlocksWithOptions(bad version) error = nil")
	}

	badKind := cloneBundleShallow(bundle)
	badKind.Kind = "not-a-bundle"
	if _, err := LoadFromStateBlocksWithOptions(ctx, store, badKind, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateBlocksWithOptions(bad kind) error = nil")
	}

	noBlocks := cloneBundleShallow(bundle)
	noBlocks.Blocks = nil
	if _, err := LoadFromStateBlocksWithOptions(ctx, store, noBlocks, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateBlocksWithOptions(no blocks) error = nil")
	}
}

// TestBlocksLoadCover_FromStateBlocks_OrderingGuards drives the up-front block
// ordering validation in loadAndAssembleStateBlocks: an out-of-order index and
// a non-contiguous token range.
func TestBlocksLoadCover_FromStateBlocks_OrderingGuards(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	outOfOrder := cloneBundleShallow(bundle)
	outOfOrder.Blocks[0].Index = 9 // ref.Index != index → errBlocksOutOfOrder
	if _, err := LoadFromStateBlocksWithOptions(ctx, store, outOfOrder, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateBlocksWithOptions(out of order) error = nil")
	}

	notContiguous := cloneBundleShallow(bundle)
	notContiguous.Blocks[0].TokenStart = 5 // != nextStart(0) → errBlocksNotContiguous
	if _, err := LoadFromStateBlocksWithOptions(ctx, store, notContiguous, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateBlocksWithOptions(not contiguous) error = nil")
	}
}

// TestBlocksLoadCover_FromStateBlocks_TokenOffsetMismatch drives the bundle
// TokenOffset cross-check that fires after assembly when the bundle declares a
// TokenOffset that disagrees with the assembled snapshot.
func TestBlocksLoadCover_FromStateBlocks_TokenOffsetMismatch(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	mismatch := cloneBundleShallow(bundle)
	mismatch.TokenOffset = 999 // > 0 and != assembled.TokenOffset → mismatch
	if _, err := LoadFromStateBlocksWithOptions(ctx, store, mismatch, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateBlocksWithOptions(token offset mismatch) error = nil")
	}
}

// TestBlocksLoadCover_PrefixFromStateBlocks drives the prefix loader's branch
// matrix: store guard, validation, the full-bundle fast path, exceeds-bundle,
// and a genuine mid-bundle prefix that requires trim+assemble.
func TestBlocksLoadCover_PrefixFromStateBlocks(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, nil, bundle, 2, LoadOptions{}); err == nil {
		t.Fatal("prefix(nil store) error = nil")
	}
	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, &StateBlockBundle{}, 2, LoadOptions{}); err == nil {
		t.Fatal("prefix(invalid bundle) error = nil")
	}
	// prefixTokens == bundle.TokenCount → full-bundle fast path.
	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount, LoadOptions{}); err != nil {
		t.Fatalf("prefix(full) error = %v", err)
	}
	// prefixTokens > bundle.TokenCount → exceeds error.
	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount+1, LoadOptions{}); err == nil {
		t.Fatal("prefix(exceeds) error = nil")
	}
	// A 3-token prefix straddles the second 2-token block → trim path.
	prefix, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, 3, LoadOptions{})
	if err != nil {
		t.Fatalf("prefix(3) error = %v", err)
	}
	if len(prefix.Tokens) != 3 {
		t.Fatalf("prefix(3) tokens = %d, want 3", len(prefix.Tokens))
	}
}

// TestBlocksLoadCover_PrefixCoverageGuards drives stateBlockPrefixCoverage's
// error arms through the prefix loader: a nil/empty bundle, out-of-order and
// non-contiguous blocks, and a prefix the blocks cannot cover.
func TestBlocksLoadCover_PrefixCoverageGuards(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	outOfOrder := cloneBundleShallow(bundle)
	outOfOrder.Blocks[1].Index = 9 // second block index wrong
	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, outOfOrder, 3, LoadOptions{}); err == nil {
		t.Fatal("prefix coverage(out of order) error = nil")
	}

	notContiguous := cloneBundleShallow(bundle)
	notContiguous.Blocks[1].TokenStart = 99 // gap before the second block
	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, notContiguous, 3, LoadOptions{}); err == nil {
		t.Fatal("prefix coverage(not contiguous) error = nil")
	}
}

// TestBlocksLoadCover_PrefixTokensFromStateBlocks drives the token-only prefix
// loader: store guard, validation, the prefixTokens<=0 → full default, an
// exceeds-bundle request, and a real mid-bundle token prefix.
func TestBlocksLoadCover_PrefixTokensFromStateBlocks(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, nil, bundle, 2, LoadOptions{}); err == nil {
		t.Fatal("token prefix(nil store) error = nil")
	}
	if _, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, &StateBlockBundle{}, 2, LoadOptions{}); err == nil {
		t.Fatal("token prefix(invalid bundle) error = nil")
	}
	// prefixTokens <= 0 → defaults to bundle.TokenCount.
	all, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, 0, LoadOptions{})
	if err != nil || len(all) != bundle.TokenCount {
		t.Fatalf("token prefix(0) = %v, err = %v, want %d tokens", all, err, bundle.TokenCount)
	}
	// prefixTokens > bundle.TokenCount → exceeds error.
	if _, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount+1, LoadOptions{}); err == nil {
		t.Fatal("token prefix(exceeds) error = nil")
	}
	// A 2-token prefix covers exactly the first block.
	two, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, 2, LoadOptions{})
	if err != nil || len(two) != 2 {
		t.Fatalf("token prefix(2) = %v, err = %v, want 2 tokens", two, err)
	}
}

// TestBlocksLoadCover_PrefixTokens_ContiguityGuard drives the token-only
// loader's per-block contiguity guard (a non-contiguous second block).
func TestBlocksLoadCover_PrefixTokens_ContiguityGuard(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	notContiguous := cloneBundleShallow(bundle)
	notContiguous.Blocks[1].TokenStart = 99
	if _, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, notContiguous, 4, LoadOptions{}); err == nil {
		t.Fatal("token prefix(not contiguous) error = nil")
	}
}

// TestBlocksLoadCover_LoadBundle_ParseAndValidate drives the bundle parse and
// validation error arms of LoadStateBlockBundle: a chunk whose text is not a
// JSON bundle, and a chunk whose JSON is a bundle that fails validation.
func TestBlocksLoadCover_LoadBundle_ParseAndValidate(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	// Non-JSON bundle text → parse error.
	if _, err := store.Put(ctx, "definitely not json", state.PutOptions{URI: "mlx://garbage-bundle"}); err != nil {
		t.Fatalf("Put(garbage) error = %v", err)
	}
	if _, err := LoadStateBlockBundle(ctx, store, "mlx://garbage-bundle"); err == nil {
		t.Fatal("LoadStateBlockBundle(garbage) error = nil, want parse error")
	}

	// Valid JSON but an empty bundle → validation error.
	if _, err := store.Put(ctx, core.JSONMarshalString(StateBlockBundle{}), state.PutOptions{URI: "mlx://empty-bundle"}); err != nil {
		t.Fatalf("Put(empty bundle) error = %v", err)
	}
	if _, err := LoadStateBlockBundle(ctx, store, "mlx://empty-bundle"); err == nil {
		t.Fatal("LoadStateBlockBundle(empty bundle) error = nil, want validation error")
	}
}

// TestBlocksLoadCover_AssembleTokenOffsetFallback drives the assembled
// TokenOffset == 0 → len(Tokens) fallback in both the full and prefix
// assemblers by saving a bundle whose snapshot carries TokenOffset 0.
func TestBlocksLoadCover_AssembleTokenOffsetFallback(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	src := kvSnapshotBlocksTestSnapshot()
	src.TokenOffset = 0 // EffectiveTokenOffset falls back to SeqLen on the source,
	src.Generated = nil // but per-block snapshots carry TokenOffset 0 → fallback.
	bundle, err := src.SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8, URI: "mlx://offset0"})
	if err != nil {
		t.Fatalf("SaveStateBlocks(offset 0) error = %v", err)
	}
	// Clear the bundle's TokenOffset so the post-assembly cross-check passes and
	// the assembled TokenOffset == 0 fallback path is reached.
	bundle.TokenOffset = 0

	loaded, err := LoadFromStateBlocks(ctx, store, bundle)
	if err != nil {
		t.Fatalf("LoadFromStateBlocks(offset 0) error = %v", err)
	}
	if loaded.TokenOffset != len(loaded.Tokens) {
		t.Fatalf("assembled TokenOffset = %d, want len(Tokens)=%d", loaded.TokenOffset, len(loaded.Tokens))
	}

	// Prefix assembler over the same bundle (2-token prefix straddling block 0).
	prefix, err := LoadPrefixFromStateBlocks(ctx, store, bundle, 2)
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocks(offset 0) error = %v", err)
	}
	if prefix.TokenOffset != len(prefix.Tokens) {
		t.Fatalf("prefix TokenOffset = %d, want len(Tokens)=%d", prefix.TokenOffset, len(prefix.Tokens))
	}
}

// TestBlocksLoadCover_PrefixTrimStraddle drives the trim path of
// loadAndAssembleStateBlockPrefix where the covering prefix ends partway
// through a block, exercising the baseOffset and SliceBlock trim arms over a
// multi-block bundle.
func TestBlocksLoadCover_PrefixTrimStraddle(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	// Six tokens, block size 2 → three blocks at [0,2) [2,4) [4,6).
	src := stateTokenOnlyTestSnapshot([]int32{1, 2, 3, 4, 5, 6}, 6, 2)
	bundle, err := src.SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8, URI: "mlx://trim-straddle"})
	if err != nil {
		t.Fatalf("SaveStateBlocks(6 tokens) error = %v", err)
	}
	if len(bundle.Blocks) != 3 {
		t.Fatalf("bundle blocks = %d, want 3", len(bundle.Blocks))
	}

	// A 5-token prefix straddles the third block [4,6) → trimEnd = 1.
	prefix, err := LoadPrefixFromStateBlocks(ctx, store, bundle, 5)
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocks(5) error = %v", err)
	}
	if len(prefix.Tokens) != 5 {
		t.Fatalf("prefix(5) tokens = %d, want 5", len(prefix.Tokens))
	}

	// A 3-token prefix straddles the second block [2,4) → trimEnd = 1.
	prefix3, err := LoadPrefixFromStateBlocks(ctx, store, bundle, 3)
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocks(3) error = %v", err)
	}
	if len(prefix3.Tokens) != 3 {
		t.Fatalf("prefix(3) tokens = %d, want 3", len(prefix3.Tokens))
	}
}

// TestBlocksLoadCover_PrefixNoCoveringBlocks drives the blockCount == 0 arm of
// stateBlockPrefixCoverage and the !covered arm of the token-only loader: the
// first block starts past the requested prefix so the coverage loop breaks
// before counting any block.
func TestBlocksLoadCover_PrefixNoCoveringBlocks(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	// First block's TokenStart sits beyond a small prefix request → the
	// coverage / token loops break immediately and report "no covering blocks".
	noCover := cloneBundleShallow(bundle)
	noCover.Blocks[0].TokenStart = 5 // >= prefixTokens(3)

	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, noCover, 3, LoadOptions{}); err == nil {
		t.Fatal("prefix(no covering blocks) error = nil")
	}
	if _, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, noCover, 3, LoadOptions{}); err == nil {
		t.Fatal("token prefix(no covering blocks) error = nil")
	}
}
