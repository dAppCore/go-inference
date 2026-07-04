// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// ExampleLoadFromStateBlocks restores a full snapshot from a block manifest.
func ExampleLoadFromStateBlocks() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2})
	if err != nil {
		core.Println("error:", err)
		return
	}

	loaded, err := LoadFromStateBlocks(ctx, store, bundle)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", len(loaded.Tokens))
	// Output: tokens: 4
}

// ExampleLoadPrefixFromStateBlocks restores only the blocks needed to cover a
// token prefix — the prompt-cache warmup path.
func ExampleLoadPrefixFromStateBlocks() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2})
	if err != nil {
		core.Println("error:", err)
		return
	}

	prefix, err := LoadPrefixFromStateBlocks(ctx, store, bundle, 2)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("prefix tokens:", len(prefix.Tokens))
	// Output: prefix tokens: 2
}

// exampleBlocksBundle saves the four-token fixture as a two-block bundle and
// returns both the store and manifest for the load examples below.
func exampleBlocksBundle(uri string) (*state.InMemoryStore, *StateBlockBundle, error) {
	store := state.NewInMemoryStore(nil)
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        uri,
	})
	return store, bundle, err
}

// ExampleLoadFromMemvidBlocks restores a full snapshot via the deprecated
// memvid-named alias.
func ExampleLoadFromMemvidBlocks() {
	store, bundle, err := exampleBlocksBundle("mlx://ex/from-memvid")
	if err != nil {
		core.Println("error:", err)
		return
	}
	loaded, err := LoadFromMemvidBlocks(context.Background(), store, bundle)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", len(loaded.Tokens))
	// Output: tokens: 4
}

// ExampleLoadStateBlockBundle resolves a saved bundle manifest by URI.
func ExampleLoadStateBlockBundle() {
	ctx := context.Background()
	store, bundle, err := exampleBlocksBundle("mlx://ex/bundle")
	if err != nil {
		core.Println("error:", err)
		return
	}
	if _, err := SaveStateBlockBundle(ctx, store, bundle, "mlx://ex/manifest"); err != nil {
		core.Println("error:", err)
		return
	}
	reloaded, err := LoadStateBlockBundle(ctx, store, "mlx://ex/manifest")
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("blocks:", len(reloaded.Blocks))
	// Output: blocks: 2
}

// ExampleLoadMemvidBlockBundle resolves a manifest via the deprecated alias.
func ExampleLoadMemvidBlockBundle() {
	ctx := context.Background()
	store, bundle, err := exampleBlocksBundle("mlx://ex/memvid-bundle")
	if err != nil {
		core.Println("error:", err)
		return
	}
	if _, err := SaveStateBlockBundle(ctx, store, bundle, "mlx://ex/memvid-manifest"); err != nil {
		core.Println("error:", err)
		return
	}
	reloaded, err := LoadMemvidBlockBundle(ctx, store, "mlx://ex/memvid-manifest")
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("blocks:", len(reloaded.Blocks))
	// Output: blocks: 2
}

// ExampleLoadFromStateBlocksWithOptions restores a full snapshot with explicit
// decode options.
func ExampleLoadFromStateBlocksWithOptions() {
	store, bundle, err := exampleBlocksBundle("mlx://ex/with-options")
	if err != nil {
		core.Println("error:", err)
		return
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", len(loaded.Tokens))
	// Output: tokens: 4
}

// ExampleLoadFromMemvidBlocksWithOptions restores a snapshot with options via
// the deprecated alias.
func ExampleLoadFromMemvidBlocksWithOptions() {
	store, bundle, err := exampleBlocksBundle("mlx://ex/memvid-with-options")
	if err != nil {
		core.Println("error:", err)
		return
	}
	loaded, err := LoadFromMemvidBlocksWithOptions(context.Background(), store, bundle, LoadOptions{})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", len(loaded.Tokens))
	// Output: tokens: 4
}

// ExampleLoadPrefixFromMemvidBlocks restores a token prefix via the deprecated
// alias.
func ExampleLoadPrefixFromMemvidBlocks() {
	store, bundle, err := exampleBlocksBundle("mlx://ex/prefix-memvid")
	if err != nil {
		core.Println("error:", err)
		return
	}
	prefix, err := LoadPrefixFromMemvidBlocks(context.Background(), store, bundle, 2)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("prefix tokens:", len(prefix.Tokens))
	// Output: prefix tokens: 2
}

// ExampleLoadPrefixFromStateBlocksWithOptions restores a token prefix with
// explicit decode options.
func ExampleLoadPrefixFromStateBlocksWithOptions() {
	store, bundle, err := exampleBlocksBundle("mlx://ex/prefix-options")
	if err != nil {
		core.Println("error:", err)
		return
	}
	prefix, err := LoadPrefixFromStateBlocksWithOptions(context.Background(), store, bundle, 2, LoadOptions{})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("prefix tokens:", len(prefix.Tokens))
	// Output: prefix tokens: 2
}

// ExampleLoadPrefixFromMemvidBlocksWithOptions restores a token prefix with
// options via the deprecated alias.
func ExampleLoadPrefixFromMemvidBlocksWithOptions() {
	store, bundle, err := exampleBlocksBundle("mlx://ex/prefix-memvid-options")
	if err != nil {
		core.Println("error:", err)
		return
	}
	prefix, err := LoadPrefixFromMemvidBlocksWithOptions(context.Background(), store, bundle, 2, LoadOptions{})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("prefix tokens:", len(prefix.Tokens))
	// Output: prefix tokens: 2
}

// ExampleLoadPrefixTokensFromStateBlocks restores only the token IDs covering a
// prefix, skipping K/V assembly entirely.
func ExampleLoadPrefixTokensFromStateBlocks() {
	store, bundle, err := exampleBlocksBundle("mlx://ex/prefix-tokens")
	if err != nil {
		core.Println("error:", err)
		return
	}
	tokens, err := LoadPrefixTokensFromStateBlocks(context.Background(), store, bundle, 3)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", len(tokens))
	// Output: tokens: 3
}

// ExampleLoadPrefixTokensFromStateBlocksWithOptions restores prefix token IDs
// with explicit decode options.
func ExampleLoadPrefixTokensFromStateBlocksWithOptions() {
	store, bundle, err := exampleBlocksBundle("mlx://ex/prefix-tokens-options")
	if err != nil {
		core.Println("error:", err)
		return
	}
	tokens, err := LoadPrefixTokensFromStateBlocksWithOptions(context.Background(), store, bundle, 2, LoadOptions{})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("tokens:", len(tokens))
	// Output: tokens: 2
}
