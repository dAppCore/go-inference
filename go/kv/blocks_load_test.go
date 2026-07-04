// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/state"
)

// TestBlocksLoad_LoadFromStateBlocks_Good loads a full snapshot from the
// two-block fixture bundle and asserts the token stream is recovered.
func TestBlocksLoad_LoadFromStateBlocks_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	loaded, err := LoadFromStateBlocks(ctx, store, bundle)
	if err != nil {
		t.Fatalf("LoadFromStateBlocks() error = %v", err)
	}
	if len(loaded.Tokens) != 4 || loaded.Tokens[3] != 4 {
		t.Fatalf("LoadFromStateBlocks() tokens = %v, want four tokens", loaded.Tokens)
	}
}

// TestBlocksLoad_LoadFromStateBlocks_Bad asserts LoadFromStateBlocks rejects a
// nil store before resolving any block.
func TestBlocksLoad_LoadFromStateBlocks_Bad(t *testing.T) {
	ctx := context.Background()
	_, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadFromStateBlocks(ctx, nil, bundle); err == nil {
		t.Fatal("LoadFromStateBlocks(nil store) error = nil, want store error")
	}
}

// TestBlocksLoad_LoadFromStateBlocks_Ugly asserts LoadFromStateBlocks rejects a
// nil bundle rather than dereferencing it.
func TestBlocksLoad_LoadFromStateBlocks_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := LoadFromStateBlocks(ctx, store, nil); err == nil {
		t.Fatal("LoadFromStateBlocks(nil bundle) error = nil, want bundle error")
	}
}

// TestBlocksLoad_LoadFromMemvidBlocks_Good asserts the deprecated LoadFromMemvidBlocks
// alias loads a bundle written by SaveStateBlocks.
func TestBlocksLoad_LoadFromMemvidBlocks_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	loaded, err := LoadFromMemvidBlocks(ctx, store, bundle)
	if err != nil {
		t.Fatalf("LoadFromMemvidBlocks() error = %v", err)
	}
	if len(loaded.Tokens) != 4 {
		t.Fatalf("LoadFromMemvidBlocks() tokens = %d, want 4", len(loaded.Tokens))
	}
}

// TestBlocksLoad_LoadFromMemvidBlocks_Bad asserts the deprecated LoadFromMemvidBlocks
// alias surfaces the nil-store guard.
func TestBlocksLoad_LoadFromMemvidBlocks_Bad(t *testing.T) {
	ctx := context.Background()
	_, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadFromMemvidBlocks(ctx, nil, bundle); err == nil {
		t.Fatal("LoadFromMemvidBlocks(nil store) error = nil, want store error")
	}
}

// TestBlocksLoad_LoadFromMemvidBlocks_Ugly asserts the deprecated LoadFromMemvidBlocks
// alias rejects a nil bundle.
func TestBlocksLoad_LoadFromMemvidBlocks_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := LoadFromMemvidBlocks(ctx, store, nil); err == nil {
		t.Fatal("LoadFromMemvidBlocks(nil bundle) error = nil, want bundle error")
	}
}

// TestBlocksLoad_LoadStateBlockBundle_Good saves a manifest then resolves it by
// URI, asserting the reloaded bundle matches the saved snapshot hash and block
// count.
func TestBlocksLoad_LoadStateBlockBundle_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)
	if _, err := SaveStateBlockBundle(ctx, store, bundle, "mlx://session/manifest"); err != nil {
		t.Fatalf("SaveStateBlockBundle() error = %v", err)
	}

	reloaded, err := LoadStateBlockBundle(ctx, store, "mlx://session/manifest")
	if err != nil {
		t.Fatalf("LoadStateBlockBundle() error = %v", err)
	}
	if reloaded.SnapshotHash != bundle.SnapshotHash || len(reloaded.Blocks) != len(bundle.Blocks) {
		t.Fatalf("LoadStateBlockBundle() = %+v, want bundle round trip", reloaded)
	}
}

// TestBlocksLoad_LoadStateBlockBundle_Bad covers the bundle-load guard branches:
// nil store and a blank URI.
func TestBlocksLoad_LoadStateBlockBundle_Bad(t *testing.T) {
	ctx := context.Background()
	store, _ := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadStateBlockBundle(ctx, nil, "mlx://x"); err == nil {
		t.Fatal("LoadStateBlockBundle(nil store) error = nil")
	}
	if _, err := LoadStateBlockBundle(ctx, store, ""); err == nil {
		t.Fatal("LoadStateBlockBundle(blank URI) error = nil")
	}
}

// TestBlocksLoad_LoadStateBlockBundle_Ugly asks LoadStateBlockBundle to resolve
// a URI that was never written; the resolve step must fail.
func TestBlocksLoad_LoadStateBlockBundle_Ugly(t *testing.T) {
	ctx := context.Background()
	store, _ := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadStateBlockBundle(ctx, store, "mlx://does-not-exist"); err == nil {
		t.Fatal("LoadStateBlockBundle(missing URI) error = nil, want resolve error")
	}
}

// TestBlocksLoad_LoadMemvidBlockBundle_Good saves a manifest then reloads it via
// the deprecated LoadMemvidBlockBundle alias.
func TestBlocksLoad_LoadMemvidBlockBundle_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)
	if _, err := SaveStateBlockBundle(ctx, store, bundle, "mlx://session/memvid-manifest"); err != nil {
		t.Fatalf("SaveStateBlockBundle() error = %v", err)
	}

	reloaded, err := LoadMemvidBlockBundle(ctx, store, "mlx://session/memvid-manifest")
	if err != nil {
		t.Fatalf("LoadMemvidBlockBundle() error = %v", err)
	}
	if reloaded.SnapshotHash != bundle.SnapshotHash {
		t.Fatalf("LoadMemvidBlockBundle() = %+v, want bundle round trip", reloaded)
	}
}

// TestBlocksLoad_LoadMemvidBlockBundle_Bad asserts the deprecated
// LoadMemvidBlockBundle alias surfaces the nil-store and blank-URI guards.
func TestBlocksLoad_LoadMemvidBlockBundle_Bad(t *testing.T) {
	ctx := context.Background()
	store, _ := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadMemvidBlockBundle(ctx, nil, "mlx://x"); err == nil {
		t.Fatal("LoadMemvidBlockBundle(nil store) error = nil")
	}
	if _, err := LoadMemvidBlockBundle(ctx, store, ""); err == nil {
		t.Fatal("LoadMemvidBlockBundle(blank URI) error = nil")
	}
}

// TestBlocksLoad_LoadMemvidBlockBundle_Ugly asks the deprecated
// LoadMemvidBlockBundle alias to resolve a missing URI; the resolve must fail.
func TestBlocksLoad_LoadMemvidBlockBundle_Ugly(t *testing.T) {
	ctx := context.Background()
	store, _ := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadMemvidBlockBundle(ctx, store, "mlx://nope"); err == nil {
		t.Fatal("LoadMemvidBlockBundle(missing URI) error = nil, want resolve error")
	}
}

// TestBlocksLoad_LoadFromStateBlocksWithOptions_Good loads the fixture bundle
// with explicit options and asserts the full snapshot is recovered.
func TestBlocksLoad_LoadFromStateBlocksWithOptions_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	loaded, err := LoadFromStateBlocksWithOptions(ctx, store, bundle, LoadOptions{})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions() error = %v", err)
	}
	if len(loaded.Tokens) != 4 || loaded.NumLayers != 1 {
		t.Fatalf("LoadFromStateBlocksWithOptions() = %+v, want four tokens, one layer", loaded)
	}
}

// TestBlocksLoad_LoadFromStateBlocksWithOptions_Bad asserts the guard arms: a
// nil store and a nil bundle both fail.
func TestBlocksLoad_LoadFromStateBlocksWithOptions_Bad(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadFromStateBlocksWithOptions(ctx, nil, bundle, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateBlocksWithOptions(nil store) error = nil")
	}
	if _, err := LoadFromStateBlocksWithOptions(ctx, store, nil, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateBlocksWithOptions(nil bundle) error = nil")
	}
}

// TestBlocksLoad_LoadFromStateBlocksWithOptions_Ugly tampers the bundle kind so
// the kind guard rejects it, proving the validation runs before any block load.
func TestBlocksLoad_LoadFromStateBlocksWithOptions_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	broken := *bundle
	broken.Kind = "not-a-state-bundle"
	if _, err := LoadFromStateBlocksWithOptions(ctx, store, &broken, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateBlocksWithOptions(wrong kind) error = nil, want kind error")
	}
}

// TestBlocksLoad_LoadMemvidBlocksWithOptions_Good asserts the deprecated
// LoadMemvidBlocksWithOptions alias loads the fixture bundle with options.
func TestBlocksLoad_LoadFromMemvidBlocksWithOptions_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	loaded, err := LoadFromMemvidBlocksWithOptions(ctx, store, bundle, LoadOptions{})
	if err != nil {
		t.Fatalf("LoadFromMemvidBlocksWithOptions() error = %v", err)
	}
	if len(loaded.Tokens) != 4 {
		t.Fatalf("LoadFromMemvidBlocksWithOptions() tokens = %d, want 4", len(loaded.Tokens))
	}
}

// TestBlocksLoad_LoadMemvidBlocksWithOptions_Bad asserts the deprecated
// LoadMemvidBlocksWithOptions alias surfaces the nil-store guard.
func TestBlocksLoad_LoadFromMemvidBlocksWithOptions_Bad(t *testing.T) {
	ctx := context.Background()
	_, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadFromMemvidBlocksWithOptions(ctx, nil, bundle, LoadOptions{}); err == nil {
		t.Fatal("LoadFromMemvidBlocksWithOptions(nil store) error = nil, want store error")
	}
}

// TestBlocksLoad_LoadMemvidBlocksWithOptions_Ugly asserts the deprecated
// LoadMemvidBlocksWithOptions alias rejects a nil bundle.
func TestBlocksLoad_LoadFromMemvidBlocksWithOptions_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	if _, err := LoadFromMemvidBlocksWithOptions(ctx, store, nil, LoadOptions{}); err == nil {
		t.Fatal("LoadFromMemvidBlocksWithOptions(nil bundle) error = nil, want bundle error")
	}
}

// TestBlocksLoad_LoadPrefixFromStateBlocks_Good loads a two-token prefix from the
// four-token fixture bundle and asserts exactly the requested tokens come back.
func TestBlocksLoad_LoadPrefixFromStateBlocks_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	prefix, err := LoadPrefixFromStateBlocks(ctx, store, bundle, 2)
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocks() error = %v", err)
	}
	if len(prefix.Tokens) != 2 || prefix.Tokens[1] != 2 {
		t.Fatalf("LoadPrefixFromStateBlocks() tokens = %v, want first two", prefix.Tokens)
	}
}

// TestBlocksLoad_LoadPrefixFromStateBlocks_Bad asserts LoadPrefixFromStateBlocks
// rejects a nil store.
func TestBlocksLoad_LoadPrefixFromStateBlocks_Bad(t *testing.T) {
	ctx := context.Background()
	_, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixFromStateBlocks(ctx, nil, bundle, 2); err == nil {
		t.Fatal("LoadPrefixFromStateBlocks(nil store) error = nil, want store error")
	}
}

// TestBlocksLoad_LoadPrefixFromStateBlocks_Ugly asks for a prefix larger than the
// bundle's token count, tripping the oversized-prefix guard.
func TestBlocksLoad_LoadPrefixFromStateBlocks_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixFromStateBlocks(ctx, store, bundle, bundle.TokenCount+1); err == nil {
		t.Fatal("LoadPrefixFromStateBlocks(oversized) error = nil, want oversized-prefix error")
	}
}

// TestBlocksLoad_LoadPrefixFromMemvidBlocks_Good asserts the deprecated
// LoadPrefixFromMemvidBlocks alias returns the requested prefix.
func TestBlocksLoad_LoadPrefixFromMemvidBlocks_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	prefix, err := LoadPrefixFromMemvidBlocks(ctx, store, bundle, 2)
	if err != nil || len(prefix.Tokens) != 2 {
		t.Fatalf("LoadPrefixFromMemvidBlocks() = %+v, err = %v, want 2 tokens", prefix, err)
	}
}

// TestBlocksLoad_LoadPrefixFromMemvidBlocks_Bad asserts the deprecated
// LoadPrefixFromMemvidBlocks alias surfaces the nil-store guard.
func TestBlocksLoad_LoadPrefixFromMemvidBlocks_Bad(t *testing.T) {
	ctx := context.Background()
	_, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixFromMemvidBlocks(ctx, nil, bundle, 2); err == nil {
		t.Fatal("LoadPrefixFromMemvidBlocks(nil store) error = nil, want store error")
	}
}

// TestBlocksLoad_LoadPrefixFromMemvidBlocks_Ugly asks the deprecated alias for an
// oversized prefix, tripping the guard.
func TestBlocksLoad_LoadPrefixFromMemvidBlocks_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixFromMemvidBlocks(ctx, store, bundle, bundle.TokenCount+10); err == nil {
		t.Fatal("LoadPrefixFromMemvidBlocks(oversized) error = nil, want oversized-prefix error")
	}
}

// TestBlocksLoad_LoadPrefixFromStateBlocksWithOptions_Good loads a partial prefix
// with options and asserts the requested token count is returned.
func TestBlocksLoad_LoadPrefixFromStateBlocksWithOptions_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	prefix, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, 2, LoadOptions{})
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocksWithOptions() error = %v", err)
	}
	if len(prefix.Tokens) != 2 {
		t.Fatalf("LoadPrefixFromStateBlocksWithOptions() tokens = %d, want 2", len(prefix.Tokens))
	}
}

// TestBlocksLoad_LoadPrefixFromStateBlocksWithOptions_Bad exercises the guard and
// edge branches: a nil store and an oversized prefix both fail.
func TestBlocksLoad_LoadPrefixFromStateBlocksWithOptions_Bad(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, nil, bundle, 1, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromStateBlocksWithOptions(nil store) error = nil")
	}
	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount+1, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromStateBlocksWithOptions(oversized prefix) error = nil")
	}
}

// TestBlocksLoad_LoadPrefixFromStateBlocksWithOptions_Ugly covers the boundary
// edges: a zero prefix and an exact full prefix both fall back to the full
// bundle and return every token.
func TestBlocksLoad_LoadPrefixFromStateBlocksWithOptions_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	full, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount, LoadOptions{})
	if err != nil || len(full.Tokens) != bundle.TokenCount {
		t.Fatalf("LoadPrefixFromStateBlocksWithOptions(full) = %+v, err = %v", full, err)
	}
	zero, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, 0, LoadOptions{})
	if err != nil || len(zero.Tokens) != bundle.TokenCount {
		t.Fatalf("LoadPrefixFromStateBlocksWithOptions(zero) = %+v, err = %v", zero, err)
	}
}

// TestBlocksLoad_LoadPrefixFromMemvidBlocksWithOptions_Good asserts the
// deprecated alias returns the requested prefix with options.
func TestBlocksLoad_LoadPrefixFromMemvidBlocksWithOptions_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	prefix, err := LoadPrefixFromMemvidBlocksWithOptions(ctx, store, bundle, 2, LoadOptions{})
	if err != nil || len(prefix.Tokens) != 2 {
		t.Fatalf("LoadPrefixFromMemvidBlocksWithOptions() = %+v, err = %v, want 2 tokens", prefix, err)
	}
}

// TestBlocksLoad_LoadPrefixFromMemvidBlocksWithOptions_Bad asserts the deprecated
// alias surfaces the nil-store guard.
func TestBlocksLoad_LoadPrefixFromMemvidBlocksWithOptions_Bad(t *testing.T) {
	ctx := context.Background()
	_, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixFromMemvidBlocksWithOptions(ctx, nil, bundle, 1, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromMemvidBlocksWithOptions(nil store) error = nil, want store error")
	}
}

// TestBlocksLoad_LoadPrefixFromMemvidBlocksWithOptions_Ugly asks the deprecated
// alias for an oversized prefix, tripping the guard.
func TestBlocksLoad_LoadPrefixFromMemvidBlocksWithOptions_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixFromMemvidBlocksWithOptions(ctx, store, bundle, bundle.TokenCount+5, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromMemvidBlocksWithOptions(oversized) error = nil, want oversized-prefix error")
	}
}

// TestBlocksLoad_LoadPrefixTokensFromStateBlocks_Good loads only the prefix
// tokens (no K/V assembly) and asserts exactly the requested tokens return.
func TestBlocksLoad_LoadPrefixTokensFromStateBlocks_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	tokens, err := LoadPrefixTokensFromStateBlocks(ctx, store, bundle, 3)
	if err != nil {
		t.Fatalf("LoadPrefixTokensFromStateBlocks() error = %v", err)
	}
	if len(tokens) != 3 || tokens[0] != 1 || tokens[2] != 3 {
		t.Fatalf("LoadPrefixTokensFromStateBlocks() tokens = %v, want first three", tokens)
	}
}

// TestBlocksLoad_LoadPrefixTokensFromStateBlocks_Bad asserts the guard arms: a
// nil store and an oversized prefix both fail.
func TestBlocksLoad_LoadPrefixTokensFromStateBlocks_Bad(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixTokensFromStateBlocks(ctx, nil, bundle, 1); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(nil store) error = nil")
	}
	if _, err := LoadPrefixTokensFromStateBlocks(ctx, store, bundle, bundle.TokenCount+1); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(oversized) error = nil")
	}
}

// TestBlocksLoad_LoadPrefixTokensFromStateBlocks_Ugly tampers the manifest so
// block indices are non-contiguous, tripping the contiguity check.
func TestBlocksLoad_LoadPrefixTokensFromStateBlocks_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	broken := *bundle
	broken.Blocks = append([]StateBlockRef(nil), bundle.Blocks...)
	broken.Blocks[0].Index = 5
	if _, err := LoadPrefixTokensFromStateBlocks(ctx, store, &broken, 4); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(non-contiguous) error = nil, want contiguity error")
	}
}

// TestBlocksLoad_LoadPrefixTokensFromStateBlocksWithOptions_Good loads the prefix
// tokens with options and asserts the requested count returns without assembly.
func TestBlocksLoad_LoadPrefixTokensFromStateBlocksWithOptions_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	tokens, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, 2, LoadOptions{})
	if err != nil {
		t.Fatalf("LoadPrefixTokensFromStateBlocksWithOptions() error = %v", err)
	}
	if len(tokens) != 2 || tokens[0] != 1 {
		t.Fatalf("LoadPrefixTokensFromStateBlocksWithOptions() tokens = %v, want first two", tokens)
	}
}

// TestBlocksLoad_LoadPrefixTokensFromStateBlocksWithOptions_Bad asserts the
// guard arms: a nil store and an oversized prefix both fail.
func TestBlocksLoad_LoadPrefixTokensFromStateBlocksWithOptions_Bad(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	if _, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, nil, bundle, 1, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocksWithOptions(nil store) error = nil")
	}
	if _, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount+1, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocksWithOptions(oversized) error = nil")
	}
}

// TestBlocksLoad_LoadPrefixTokensFromStateBlocksWithOptions_Ugly tampers the
// manifest indices non-contiguously, tripping the contiguity check.
func TestBlocksLoad_LoadPrefixTokensFromStateBlocksWithOptions_Ugly(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	broken := *bundle
	broken.Blocks = append([]StateBlockRef(nil), bundle.Blocks...)
	broken.Blocks[0].Index = 9
	if _, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, &broken, 4, LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocksWithOptions(non-contiguous) error = nil, want contiguity error")
	}
}
