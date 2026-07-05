// SPDX-Licence-Identifier: EUPL-1.2

// blockcache_example_test.go is the usage-in-situ companion to
// blockcache_test.go: runnable Example functions that double as
// documentation and as coverage for the package's public surface. Each one
// drives the Service the way a real caller (the Metal cache adapter) does —
// New → WarmCache → CacheStats/CacheEntries/ClearCache — and pins a
// deterministic result via // Output.
//
// No model is loaded (AX-11): every Example feeds synthetic int32 token
// slices and explicit identity hashes through the in-memory metadata layer.
// The cache is memory-only (no DiskPath, no StateStore) so the examples are
// portable and need no temp directory or filesystem state.
//
// Block IDs are SHA-256 over a fixed identity header plus the cumulative
// token prefix, so they are fully deterministic and safe to assert in
// // Output — the stable, portable identity is the package's entire reason
// to exist, and printing it here guards that the hash composition does not
// drift. Map-valued fields (ref.Labels, stats.Labels) are NEVER printed
// whole — Go randomises map iteration order — only specific keys, counts,
// token ranges, and hit rate, all of which are order-independent.

package blockcache

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
)

// ExampleNew shows the zero-config constructor: an unset BlockSize falls
// back to DefaultBlockSize, and the service starts empty.
func ExampleNew() {
	service := New(Config{})

	stats, err := service.CacheStats(context.Background())
	if err != nil {
		core.Println(err)
		return
	}
	core.Println("blocks", stats.Blocks, "block_size", stats.Labels["block_size"], "cache_mode", stats.CacheMode)
	// Output: blocks 0 block_size 512 cache_mode block-prefix
}

// ExampleService_WarmCache warms a six-token prompt at BlockSize 4. The
// service chunks the prefix into two blocks (4 + 2 tokens) and returns a
// stable SHA-256 ref for each. The IDs are deterministic: the same identity
// hashes and tokens always produce these exact digests, which is what makes
// the block cache portable across processes and machines.
func ExampleService_WarmCache() {
	service := New(Config{
		BlockSize:     4,
		ModelHash:     "sha256:demo-model",
		AdapterHash:   "sha256:demo-adapter",
		TokenizerHash: "sha256:demo-tokenizer",
	})

	result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3, 4, 5, 6},
	})
	if err != nil {
		core.Println(err)
		return
	}
	for _, ref := range result.Blocks {
		core.Println(ref.ID, "tokens", ref.TokenStart, "..", ref.TokenStart+ref.TokenCount, "bytes", ref.SizeBytes, "prefix_tokens", ref.Labels["prefix_tokens"])
	}
	core.Println("misses", result.Stats.Misses, "hits", result.Stats.Hits)
	// Output:
	// c947c050fb9ce9268e6472bb002ce6ac54b0acbdb58048dcf361231dcb84099a tokens 0 .. 4 bytes 16 prefix_tokens 4
	// 540c24db6153f86a434f651d9c8b8072bf6fe7298498fb55805a1f2d52fcb921 tokens 4 .. 6 bytes 8 prefix_tokens 6
	// misses 2 hits 0
}

// ExampleService_WarmCache_repeatHits shows the hit path: warming the same
// prefix a second time matches every existing block by ID, so the second
// warm is all hits and the cumulative hit rate settles at 0.5 (two misses
// on the first warm, two hits on the second).
func ExampleService_WarmCache_repeatHits() {
	service := New(Config{
		BlockSize:     4,
		ModelHash:     "sha256:demo-model",
		TokenizerHash: "sha256:demo-tokenizer",
	})

	tokens := []int32{1, 2, 3, 4, 5, 6}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens}); err != nil {
		core.Println(err)
		return
	}
	second, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
	if err != nil {
		core.Println(err)
		return
	}
	core.Println("blocks", second.Stats.Blocks, "hits", second.Stats.Hits, "misses", second.Stats.Misses, "hit_rate", second.Stats.HitRate)
	// Output: blocks 2 hits 2 misses 2 hit_rate 0.5
}

// ExampleService_WarmCache_tokenize shows the prompt path: when a request
// carries a Prompt instead of pre-tokenised input, the configured Tokenize
// hook turns it into tokens and the optional WarmPrompt hook is invoked to
// warm the underlying native cache. No real tokenizer or model is involved —
// the hooks are plain synthetic functions.
func ExampleService_WarmCache_tokenize() {
	var warmed string
	service := New(Config{
		BlockSize:     2,
		ModelHash:     "sha256:demo-model",
		TokenizerHash: "sha256:demo-tokenizer",
		Tokenize: func(prompt string) ([]int32, error) {
			return []int32{10, 11, 12}, nil
		},
		WarmPrompt: func(_ context.Context, prompt string) error {
			warmed = prompt
			return nil
		},
	})

	result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"})
	if err != nil {
		core.Println(err)
		return
	}
	core.Println("warmed_prompt", warmed, "blocks", len(result.Blocks), "first_count", result.Blocks[0].TokenCount, "last_count", result.Blocks[1].TokenCount)
	// Output: warmed_prompt hello blocks 2 first_count 2 last_count 1
}

// ExampleService_CacheStats reports the in-memory block metadata and the
// cumulative warm hit/miss counters. Warming a six-token prefix at BlockSize 4
// records two blocks (two misses); warming the identical prefix again matches
// both by ID (two hits), so the cumulative hit rate settles at 0.5.
func ExampleService_CacheStats() {
	service := New(Config{
		BlockSize:     4,
		ModelHash:     "sha256:demo-model",
		TokenizerHash: "sha256:demo-tokenizer",
	})

	tokens := []int32{1, 2, 3, 4, 5, 6}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens}); err != nil {
		core.Println(err)
		return
	}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens}); err != nil {
		core.Println(err)
		return
	}

	stats, err := service.CacheStats(context.Background())
	if err != nil {
		core.Println(err)
		return
	}
	core.Println("blocks", stats.Blocks, "hits", stats.Hits, "misses", stats.Misses, "hit_rate", stats.HitRate, "cache_mode", stats.CacheMode)
	// Output: blocks 2 hits 2 misses 2 hit_rate 0.5 cache_mode block-prefix
}

// ExampleService_CacheEntries lists the stable refs the service holds,
// filtered by label. Entries arrive sorted by token start, and each is a
// clone — mutating a returned ref never disturbs the service's own copy.
func ExampleService_CacheEntries() {
	service := New(Config{BlockSize: 2, ModelHash: "sha256:demo-model"})

	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "alpha"},
		Tokens: []int32{1, 2, 3},
	}); err != nil {
		core.Println(err)
		return
	}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "beta"},
		Tokens: []int32{4, 5},
	}); err != nil {
		core.Println(err)
		return
	}

	entries, err := service.CacheEntries(context.Background(), map[string]string{"tenant": "alpha"})
	if err != nil {
		core.Println(err)
		return
	}
	for _, ref := range entries {
		core.Println("entry tokens", ref.TokenStart, "..", ref.TokenStart+ref.TokenCount, "tenant", ref.Labels["tenant"])
	}
	// Output:
	// entry tokens 0 .. 2 tenant alpha
	// entry tokens 2 .. 3 tenant alpha
}

// ExampleService_ClearCache shows label-scoped clearing: only blocks whose
// metadata matches the filter are dropped, the rest stay warm. Passing nil
// labels (not shown) would clear everything.
func ExampleService_ClearCache() {
	service := New(Config{BlockSize: 2, ModelHash: "sha256:demo-model"})

	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "alpha"},
		Tokens: []int32{1, 2, 3},
	}); err != nil {
		core.Println(err)
		return
	}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "beta"},
		Tokens: []int32{4, 5},
	}); err != nil {
		core.Println(err)
		return
	}

	stats, err := service.ClearCache(context.Background(), map[string]string{"tenant": "alpha"})
	if err != nil {
		core.Println(err)
		return
	}
	core.Println("remaining_blocks", stats.Blocks, "cleared", stats.Labels["cleared"])
	// Output: remaining_blocks 1 cleared 2
}

// ExampleService_WarmCache_stateColdStore shows the cold-store path: with a
// DiskPath and a state.Writer configured together, each block's KV payload is
// written to the store and the returned ref is tagged with cold-store labels.
// The in-memory store keeps the payload off any real backend; only a scratch
// directory for the block metadata records touches the filesystem. The label
// values are deterministic and independent of the temp path.
func ExampleService_WarmCache_stateColdStore() {
	diskPath := core.MkdirTemp("", "blockcache-example-*").Value.(string)
	defer core.RemoveAll(diskPath)

	store := state.NewInMemoryStore(nil)
	service := New(Config{
		BlockSize:     2,
		ModelHash:     "sha256:demo-model",
		TokenizerHash: "sha256:demo-tokenizer",
		DiskPath:      diskPath,
		StateStore:    store,
	})

	result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	if err != nil {
		core.Println(err)
		return
	}
	ref := result.Blocks[0]
	core.Println("blocks", len(result.Blocks), "cold_store", ref.Labels["cold_store"], "codec", ref.Labels["state_codec"])
	// Output: blocks 2 cold_store state codec memory/plaintext
}

// ExampleHashModelParts shows the standalone identity helper callers use to
// derive a stable model or tokenizer hash from arbitrary parts (architecture,
// vocab size, ...). The digest is deterministic for a given argument list.
func ExampleHashModelParts() {
	core.Println(HashModelParts("qwen3", 151936))
	// Output: aa5dab1cd4dbf496368ad47e056a6595e3cb3fc46864a094b0d72b15e7cf92eb
}
