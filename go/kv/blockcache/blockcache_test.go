// SPDX-Licence-Identifier: EUPL-1.2

package blockcache

import (
	"context"
	"sync"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/model/state"
)

// failingStateWriter is a test stub that always errors on Put. Used to
// exercise the State-write failure path inside blockcache.WarmCache.
type failingStateWriter struct{}

func (failingStateWriter) Put(_ context.Context, _ string, _ state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, context.Canceled
}

// ---------------------------------------------------------------------------
// New
// ---------------------------------------------------------------------------

func TestBlockcache_New_Good(t *testing.T) {
	// A configured BlockSize is honoured: the constructor records the chosen
	// size and reports it back through the stats block_size label, and the
	// fresh service starts with zero blocks.
	service := New(Config{
		BlockSize:     3,
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
	})
	if service == nil {
		t.Fatal("New() returned nil service")
	}
	stats, err := service.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Blocks != 0 {
		t.Fatalf("New() stats = %+v, want empty service", stats)
	}
	if stats.Labels["block_size"] != "3" {
		t.Fatalf("New() block_size label = %q, want 3", stats.Labels["block_size"])
	}
	if stats.CacheMode != "block-prefix" {
		t.Fatalf("New() cache_mode = %q, want block-prefix", stats.CacheMode)
	}
}

func TestBlockcache_New_Bad(t *testing.T) {
	// A non-positive BlockSize is invalid input; New clamps it to
	// DefaultBlockSize rather than producing a degenerate zero-size service.
	service := New(Config{BlockSize: -5, ModelHash: "sha256:model"})
	if service == nil {
		t.Fatal("New(negative block size) returned nil service")
	}
	stats, err := service.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Labels["block_size"] != core.Itoa(DefaultBlockSize) {
		t.Fatalf("New(-5) block_size label = %q, want clamp to %d", stats.Labels["block_size"], DefaultBlockSize)
	}
	// The clamp is observable in behaviour: a five-token warm chunks into a
	// single DefaultBlockSize block, not five.
	result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5}})
	if err != nil {
		t.Fatalf("WarmCache() error = %v", err)
	}
	if len(result.Blocks) != 1 || result.Blocks[0].TokenCount != 5 {
		t.Fatalf("New(-5) warm blocks = %+v, want one default-sized block", result.Blocks)
	}
}

func TestBlockcache_New_Ugly(t *testing.T) {
	// The zero-value Config is the documented zero-config path: every field
	// unset. BlockSize defaults, DiskPath stays empty (in-memory only), and
	// the service is immediately usable.
	service := New(Config{})
	if service == nil {
		t.Fatal("New(zero config) returned nil service")
	}
	stats, err := service.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Blocks != 0 {
		t.Fatalf("New(zero) stats = %+v, want empty service", stats)
	}
	if stats.Labels["block_size"] != core.Itoa(DefaultBlockSize) {
		t.Fatalf("New(zero) block_size label = %q, want default %d", stats.Labels["block_size"], DefaultBlockSize)
	}
	if _, ok := stats.Labels["disk_path"]; ok {
		t.Fatalf("New(zero) leaked disk_path label = %+v, want in-memory only", stats.Labels)
	}
}

// ---------------------------------------------------------------------------
// (*Service) CacheStats
// ---------------------------------------------------------------------------

func TestBlockcache_Service_CacheStats_Good(t *testing.T) {
	// Stats report in-memory block metadata and cumulative warm hit/miss
	// counters. Warming a seven-token prefix at BlockSize 3 yields three
	// blocks (3+3+1); warming the identical prefix again is all hits, so the
	// cumulative stats settle at three blocks, three hits, three misses, and
	// a 0.5 hit rate.
	service := New(Config{
		BlockSize:     3,
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
	})
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5, 6, 7}}); err != nil {
		t.Fatalf("WarmCache(first) error = %v", err)
	}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5, 6, 7}}); err != nil {
		t.Fatalf("WarmCache(second) error = %v", err)
	}
	stats, err := service.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Blocks != 3 || stats.Hits != 3 || stats.Misses != 3 || stats.HitRate != 0.5 {
		t.Fatalf("stats = %+v, want 3 blocks, 3 hits, 3 misses, 0.5 hit rate", stats)
	}

	// Disk-backed stats: a corrupt on-disk record is dropped on the first
	// load (CacheStats triggers ensureDiskLoaded), counted as one eviction,
	// and surfaced via the disk_corrupt label.
	corruptPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(corruptPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll() error = %s", result.Error())
	}
	if result := core.WriteFile(core.PathJoin(corruptPath, "broken.json"), []byte("{broken"), 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	diskService := New(Config{BlockSize: 2, DiskPath: corruptPath})
	diskStats, err := diskService.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats(disk) error = %v", err)
	}
	if diskStats.Blocks != 0 || diskStats.Evictions != 1 || diskStats.Labels["disk_corrupt"] != "1" {
		t.Fatalf("disk stats = %+v, want corrupt record ignored and counted", diskStats)
	}
}

func TestBlockcache_Service_CacheStats_Bad(t *testing.T) {
	// A nil *Service is a programming error: CacheStats reports it rather
	// than panicking on the nil receiver.
	if _, err := (*Service)(nil).CacheStats(context.Background()); err == nil {
		t.Fatal("CacheStats(nil service) error = nil")
	}
	// A cancelled context short-circuits before any work: CacheStats returns
	// the context error.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	service := New(Config{})
	if _, err := service.CacheStats(cancelled); err == nil {
		t.Fatal("CacheStats(cancelled) error = nil")
	}
}

func TestBlockcache_Service_CacheStats_Ugly(t *testing.T) {
	// A nil context is the documented fast path: cacheContextError returns
	// nil and CacheStats proceeds normally rather than treating nil as an
	// error.
	service := New(Config{BlockSize: 2, ModelHash: "sha256:model"})
	//nolint:staticcheck // SA1012: passing a nil Context is the path under test.
	stats, err := service.CacheStats(nil)
	if err != nil {
		t.Fatalf("CacheStats(nil ctx) error = %v, want nil", err)
	}
	if stats.Blocks != 0 || stats.CacheMode != "block-prefix" {
		t.Fatalf("CacheStats(nil ctx) = %+v, want empty block-prefix stats", stats)
	}
}

// ---------------------------------------------------------------------------
// (*Service) CacheEntries
// ---------------------------------------------------------------------------

func TestBlockcache_Service_CacheEntries_Good(t *testing.T) {
	// CacheEntries returns stable refs filtered by label, ordered by token
	// start, and each entry is a clone — mutating a returned ref never
	// disturbs the service's own copy.
	service := New(Config{BlockSize: 2, ModelHash: "sha256:model"})
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "alpha"},
		Tokens: []int32{1, 2, 3},
	}); err != nil {
		t.Fatalf("WarmCache(alpha) error = %v", err)
	}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "beta"},
		Tokens: []int32{4, 5},
	}); err != nil {
		t.Fatalf("WarmCache(beta) error = %v", err)
	}

	entries, err := service.CacheEntries(context.Background(), map[string]string{"tenant": "alpha"})
	if err != nil {
		t.Fatalf("CacheEntries(alpha) error = %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("entries = %+v, want two alpha prefix blocks", entries)
	}
	if entries[0].TokenStart != 0 || entries[1].TokenStart != 2 {
		t.Fatalf("entries = %+v, want deterministic token order", entries)
	}
	for _, ref := range entries {
		if ref.Labels["tenant"] != "alpha" {
			t.Fatalf("entry labels = %+v, want alpha tenant", ref.Labels)
		}
	}

	entries[0].Labels["tenant"] = "mutated"
	again, err := service.CacheEntries(context.Background(), map[string]string{"tenant": "alpha"})
	if err != nil {
		t.Fatalf("CacheEntries(alpha again) error = %v", err)
	}
	if again[0].Labels["tenant"] != "alpha" {
		t.Fatalf("entry labels were not cloned: %+v", again[0].Labels)
	}
}

func TestBlockcache_Service_CacheEntries_Bad(t *testing.T) {
	// A nil *Service is reported, not dereferenced.
	if _, err := (*Service)(nil).CacheEntries(context.Background(), nil); err == nil {
		t.Fatal("CacheEntries(nil service) error = nil")
	}
	// A cancelled context short-circuits CacheEntries with the context error.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	service := New(Config{})
	if _, err := service.CacheEntries(cancelled, nil); err == nil {
		t.Fatal("CacheEntries(cancelled) error = nil")
	}
}

func TestBlockcache_Service_CacheEntries_Ugly(t *testing.T) {
	// Edge: a nil label filter returns every entry (no filtering), and an
	// empty service returns an empty, non-nil slice. Both boundaries are
	// exercised on the same service.
	service := New(Config{BlockSize: 2, ModelHash: "sha256:model"})
	empty, err := service.CacheEntries(context.Background(), nil)
	if err != nil {
		t.Fatalf("CacheEntries(empty, nil) error = %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("CacheEntries(empty) = %+v, want no entries", empty)
	}
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}}); err != nil {
		t.Fatalf("WarmCache() error = %v", err)
	}
	all, err := service.CacheEntries(context.Background(), nil)
	if err != nil {
		t.Fatalf("CacheEntries(nil filter) error = %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("CacheEntries(nil filter) = %+v, want all blocks unfiltered", all)
	}
}

// ---------------------------------------------------------------------------
// (*Service) WarmCache
// ---------------------------------------------------------------------------

func TestBlockcache_Service_WarmCache_Good(t *testing.T) {
	// WarmCache creates stable, distinct, repeatable block refs for a token
	// request. A seven-token prefix at BlockSize 3 chunks into 3+3+1 blocks
	// with deterministic IDs and token ranges; warming the same prefix again
	// reproduces the identical IDs.
	t.Run("StablePrefixBlocks", func(t *testing.T) {
		service := New(Config{
			BlockSize:     3,
			ModelHash:     "sha256:model",
			AdapterHash:   "sha256:adapter",
			TokenizerHash: "sha256:tokenizer",
		})
		first, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5, 6, 7}})
		if err != nil {
			t.Fatalf("WarmCache(first) error = %v", err)
		}
		if len(first.Blocks) != 3 {
			t.Fatalf("blocks = %+v, want 3 prefix blocks", first.Blocks)
		}
		if first.Blocks[0].ID == "" || first.Blocks[0].ID == first.Blocks[1].ID {
			t.Fatalf("block IDs = %+v, want stable distinct IDs", first.Blocks)
		}
		if first.Blocks[0].TokenStart != 0 || first.Blocks[0].TokenCount != 3 || first.Blocks[2].TokenStart != 6 || first.Blocks[2].TokenCount != 1 {
			t.Fatalf("blocks = %+v, want chunked token ranges", first.Blocks)
		}
		second, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5, 6, 7}})
		if err != nil {
			t.Fatalf("WarmCache(second) error = %v", err)
		}
		for i := range first.Blocks {
			if first.Blocks[i].ID != second.Blocks[i].ID {
				t.Fatalf("block %d ID changed: %q != %q", i, first.Blocks[i].ID, second.Blocks[i].ID)
			}
		}
	})

	// The prompt path: a request carrying a Prompt instead of tokens runs the
	// configured Tokenize hook, then the WarmPrompt hook is invoked to warm
	// the underlying native cache.
	t.Run("WarmPromptUsesTokenizerAndWarmer", func(t *testing.T) {
		var warmedPrompt string
		service := New(Config{
			BlockSize:     2,
			ModelHash:     "sha256:model",
			TokenizerHash: "sha256:tokenizer",
			Tokenize: func(prompt string) ([]int32, error) {
				if prompt != "hello" {
					t.Fatalf("tokenized prompt = %q, want hello", prompt)
				}
				return []int32{10, 11, 12}, nil
			},
			WarmPrompt: func(_ context.Context, prompt string) error {
				warmedPrompt = prompt
				return nil
			},
		})
		result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"})
		if err != nil {
			t.Fatalf("WarmCache(prompt) error = %v", err)
		}
		if warmedPrompt != "hello" {
			t.Fatalf("warmed prompt = %q, want hello", warmedPrompt)
		}
		if len(result.Blocks) != 2 || result.Blocks[0].TokenCount != 2 || result.Blocks[1].TokenCount != 1 {
			t.Fatalf("blocks = %+v, want tokenized prompt blocks", result.Blocks)
		}
	})

	// Compatibility labels: when request identities differ from the service's
	// configured identities, the result and per-block labels carry the
	// mismatch flags.
	t.Run("CompatibilityLabels", func(t *testing.T) {
		service := New(Config{
			BlockSize:     2,
			ModelHash:     "sha256:model-a",
			AdapterHash:   "sha256:adapter-a",
			TokenizerHash: "sha256:tokenizer-a",
		})
		result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
			Model:   inference.ModelIdentity{Hash: "sha256:model-b"},
			Adapter: inference.AdapterIdentity{Hash: "sha256:adapter-b"},
			Labels:  map[string]string{"tokenizer_hash": "sha256:tokenizer-b"},
			Tokens:  []int32{1, 2},
		})
		if err != nil {
			t.Fatalf("WarmCache() error = %v", err)
		}
		if result.Labels["model_match"] != "false" || result.Labels["adapter_match"] != "false" || result.Labels["tokenizer_match"] != "false" {
			t.Fatalf("labels = %+v, want mismatch labels", result.Labels)
		}
		if result.Blocks[0].Labels["adapter_match"] != "false" {
			t.Fatalf("block labels = %+v, want adapter mismatch", result.Blocks[0].Labels)
		}
	})

	// Disk-backed warm: with a DiskPath set, every warmed block is persisted,
	// tagged with disk metadata, and contributes to DiskBytes. A fresh
	// service over the same path loads the persisted blocks and treats a
	// repeat warm as all hits.
	t.Run("DiskBackedBlocksSurviveRestart", func(t *testing.T) {
		diskPath := core.PathJoin(t.TempDir(), "blocks")
		cfg := Config{
			BlockSize:     2,
			ModelHash:     "sha256:model",
			AdapterHash:   "sha256:adapter",
			TokenizerHash: "sha256:tokenizer",
			DiskPath:      diskPath,
		}
		first := New(cfg)
		result, err := first.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5}})
		if err != nil {
			t.Fatalf("WarmCache(first) error = %v", err)
		}
		if len(result.Blocks) != 3 {
			t.Fatalf("blocks = %+v, want 3 persisted prefix blocks", result.Blocks)
		}
		for _, ref := range result.Blocks {
			if ref.Labels["disk"] != "true" || ref.Labels["disk_path"] == "" {
				t.Fatalf("block labels = %+v, want disk metadata", ref.Labels)
			}
			if stat := core.Stat(ref.Labels["disk_path"]); !stat.OK {
				t.Fatalf("persisted block %q was not written: %s", ref.Labels["disk_path"], stat.Error())
			}
		}
		if result.Stats.DiskBytes == 0 {
			t.Fatalf("warm stats = %+v, want disk bytes", result.Stats)
		}
		second := New(cfg)
		stats, err := second.CacheStats(context.Background())
		if err != nil {
			t.Fatalf("CacheStats(second) error = %v", err)
		}
		if stats.Blocks != 3 || stats.DiskBytes == 0 {
			t.Fatalf("second stats = %+v, want persisted blocks and disk bytes", stats)
		}
		hit, err := second.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5}})
		if err != nil {
			t.Fatalf("WarmCache(second) error = %v", err)
		}
		if hit.Stats.Hits != 3 || hit.Stats.Misses != 0 || hit.Stats.HitRate != 1 {
			t.Fatalf("second warm stats = %+v, want persisted block hits", hit.Stats)
		}
	})

	// State cold-store: with a DiskPath and a state.Writer configured, each
	// block's KV payload is written to the store, the returned ref carries
	// cold-store labels, and a fresh service reloads the state-backed blocks.
	t.Run("StateColdStoreRecordsPayload", func(t *testing.T) {
		diskPath := core.PathJoin(t.TempDir(), "blocks")
		store := state.NewInMemoryStore(nil)
		service := New(Config{
			BlockSize:     2,
			ModelHash:     "sha256:model",
			TokenizerHash: "sha256:tokenizer",
			DiskPath:      diskPath,
			StateStore:    store,
		})
		result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
		if err != nil {
			t.Fatalf("WarmCache() error = %v", err)
		}
		if len(result.Blocks) != 2 {
			t.Fatalf("blocks = %+v, want two state-backed blocks", result.Blocks)
		}
		ref := result.Blocks[0]
		if ref.Labels["cold_store"] != "state" || ref.Labels["state_chunk_id"] == "" || ref.Labels["state_codec"] != state.CodecMemory {
			t.Fatalf("block labels = %+v, want State cold-store labels", ref.Labels)
		}
		chunkIDResult := core.Atoi(ref.Labels["state_chunk_id"])
		if !chunkIDResult.OK {
			t.Fatalf("State chunk id %q did not parse: %s", ref.Labels["state_chunk_id"], chunkIDResult.Error())
		}
		chunk, err := state.Resolve(context.Background(), store, chunkIDResult.Value.(int))
		if err != nil {
			t.Fatalf("Resolve(State chunk) error = %v", err)
		}
		if !core.Contains(chunk.Text, `"block_id":"`+ref.ID+`"`) || !core.Contains(chunk.Text, `"tokens":[1,2]`) {
			t.Fatalf("State chunk = %s, want block payload", chunk.Text)
		}
		second := New(Config{
			BlockSize:     2,
			ModelHash:     "sha256:model",
			TokenizerHash: "sha256:tokenizer",
			DiskPath:      diskPath,
			StateStore:    store,
		})
		stats, err := second.CacheStats(context.Background())
		if err != nil {
			t.Fatalf("CacheStats(second) error = %v", err)
		}
		if stats.Blocks != 2 || stats.Labels["cold_store"] != "state" {
			t.Fatalf("second stats = %+v, want state-backed persisted blocks", stats)
		}
	})

	// Concurrency: blockRefs runs lock-free (before WarmCache takes
	// service.mu), so concurrent warms run blockRefs — and its package-level
	// sha256/encode-buffer pool — concurrently. blockCacheID shares that pool.
	// Many goroutines each warm a distinct token set in a loop and assert
	// every goroutine's block IDs equal a serially-computed baseline. Under
	// -race this demonstrates the pooled scratch carries no shared mutable
	// state across goroutines and the recycling is byte-identical to the
	// unpooled per-call form.
	t.Run("ConcurrentWarmIsRaceFreeAndStable", func(t *testing.T) {
		const (
			goroutines = 16
			iterations = 40
		)
		cfg := Config{
			BlockSize:     4,
			ModelHash:     "sha256:model",
			AdapterHash:   "sha256:adapter",
			TokenizerHash: "sha256:tokenizer",
		}
		tokenSets := make([][]int32, goroutines)
		wantIDs := make([][]string, goroutines)
		for g := range tokenSets {
			tokens := make([]int32, 10+g) // 3 blocks at size 4, last partial
			for i := range tokens {
				tokens[i] = int32(g*1000 + i + 1)
			}
			tokenSets[g] = tokens
			baseline, err := New(cfg).WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
			if err != nil {
				t.Fatalf("baseline WarmCache(g=%d) error = %v", g, err)
			}
			ids := make([]string, len(baseline.Blocks))
			for i, ref := range baseline.Blocks {
				ids[i] = ref.ID
			}
			wantIDs[g] = ids
		}

		var wg sync.WaitGroup
		errs := make(chan error, goroutines*iterations)
		for g := 0; g < goroutines; g++ {
			wg.Add(1)
			go func(g int) {
				defer wg.Done()
				service := New(cfg)
				tokens := tokenSets[g]
				for it := 0; it < iterations; it++ {
					result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
					if err != nil {
						errs <- core.NewError("WarmCache error in goroutine")
						return
					}
					if len(result.Blocks) != len(wantIDs[g]) {
						errs <- core.NewError("block count mismatch under concurrency")
						return
					}
					for i, ref := range result.Blocks {
						if ref.ID != wantIDs[g][i] {
							errs <- core.NewError("block ID mismatch under concurrency: pool leaked state")
							return
						}
					}
					if id := blockCacheID(cfg.ModelHash, cfg.AdapterHash, cfg.TokenizerHash, "", tokens); id != wantIDs[g][len(wantIDs[g])-1] {
						errs <- core.NewError("blockCacheID(full prefix) != final block ID under concurrency")
						return
					}
				}
			}(g)
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Fatal(err)
		}
	})
}

func TestBlockcache_Service_WarmCache_Bad(t *testing.T) {
	// A nil *Service is reported, not dereferenced.
	if _, err := (*Service)(nil).WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1}}); err == nil {
		t.Fatal("WarmCache(nil service) error = nil")
	}
	// A cancelled context short-circuits WarmCache with the context error.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	service := New(Config{})
	if _, err := service.WarmCache(cancelled, inference.CacheWarmRequest{Tokens: []int32{1}}); err == nil {
		t.Fatal("WarmCache(cancelled) error = nil")
	}
	// An empty request (no prompt, no tokens) has nothing to warm.
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{}); err == nil {
		t.Fatal("WarmCache(empty request) error = nil")
	}
	// A prompt without a configured tokenizer cannot be tokenised.
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"}); err == nil {
		t.Fatal("WarmCache(prompt without tokenizer) error = nil")
	}
	// A tokenizer that errors propagates the error.
	tokenizerErr := New(Config{
		Tokenize: func(string) ([]int32, error) {
			return nil, core.NewError("tokenize failed")
		},
	})
	if _, err := tokenizerErr.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"}); err == nil {
		t.Fatal("WarmCache(tokenizer error) error = nil")
	}
	// A warmer hook that errors propagates the error.
	warmerErr := New(Config{
		Tokenize: func(string) ([]int32, error) { return []int32{1}, nil },
		WarmPrompt: func(context.Context, string) error {
			return core.NewError("warm failed")
		},
	})
	if _, err := warmerErr.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello"}); err == nil {
		t.Fatal("WarmCache(warmer error) error = nil")
	}
	// A failing cold-store Put surfaces as a WarmCache error.
	stateErr := New(Config{
		DiskPath:   core.PathJoin(t.TempDir(), "blocks"),
		StateStore: failingStateWriter{},
	})
	if _, err := stateErr.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1}}); err == nil {
		t.Fatal("WarmCache(State write error) error = nil")
	}
}

func TestBlockcache_Service_WarmCache_Ugly(t *testing.T) {
	// The awkward-but-real corners the happy path skips, all reachable with
	// synthetic inputs — no model, no disk.

	// Hasher buffer-grow: a header longer than the pooled 256-byte default
	// (16 length-prefix bytes + the four identity strings) forces
	// acquireBlockCacheHasher to grow scratch.buf. The resulting IDs must
	// still be stable across repeated warms — exercising the grow path and
	// confirming the grown buffer is reused cleanly.
	longHash := "sha256:" + core.Repeat("ab", 200) // ~407-byte header
	longService := New(Config{
		BlockSize:     4,
		ModelHash:     longHash,
		AdapterHash:   longHash,
		TokenizerHash: longHash,
	})
	first, err := longService.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5}})
	if err != nil {
		t.Fatalf("WarmCache(long header, first) error = %v", err)
	}
	second, err := longService.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4, 5}})
	if err != nil {
		t.Fatalf("WarmCache(long header, second) error = %v", err)
	}
	if len(first.Blocks) == 0 || first.Blocks[0].ID == "" {
		t.Fatalf("WarmCache(long header) blocks = %+v, want stable IDs", first.Blocks)
	}
	for i := range first.Blocks {
		if first.Blocks[i].ID != second.Blocks[i].ID {
			t.Fatalf("long-header block %d ID changed across warms: %q != %q", i, first.Blocks[i].ID, second.Blocks[i].ID)
		}
	}

	// prefixTokenLabel beyond the pre-rendered cap: with BlockSize 1 the 33rd
	// aligned end (33) sits past prefixTokenLabelCacheSize (32), so the label
	// is produced by the Itoa fallback rather than the cached slice.
	capService := New(Config{BlockSize: 1, ModelHash: "sha256:model"})
	tokens := make([]int32, prefixTokenLabelCacheSize+2) // 34 single-token blocks
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	capResult, err := capService.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
	if err != nil {
		t.Fatalf("WarmCache(beyond cap) error = %v", err)
	}
	if got := len(capResult.Blocks); got != prefixTokenLabelCacheSize+2 {
		t.Fatalf("beyond-cap blocks = %d, want %d", got, prefixTokenLabelCacheSize+2)
	}
	lastLabel := capResult.Blocks[len(capResult.Blocks)-1].Labels["prefix_tokens"]
	if lastLabel != core.Itoa(prefixTokenLabelCacheSize+2) {
		t.Fatalf("beyond-cap prefix_tokens = %q, want %q", lastLabel, core.Itoa(prefixTokenLabelCacheSize+2))
	}
	if got := capResult.Blocks[0].Labels["prefix_tokens"]; got != "1" {
		t.Fatalf("in-cap prefix_tokens = %q, want 1", got)
	}

	// Nil context is the documented fast path: cacheContextError returns nil
	// and WarmCache substitutes context.Background internally.
	nilCtxService := New(Config{BlockSize: 2, ModelHash: "sha256:model"})
	//nolint:staticcheck // SA1012: passing a nil Context is the path under test.
	if _, err := nilCtxService.WarmCache(nil, inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}}); err != nil {
		t.Fatalf("WarmCache(nil ctx) error = %v, want nil", err)
	}
}

// ---------------------------------------------------------------------------
// (*Service) ClearCache
// ---------------------------------------------------------------------------

func TestBlockcache_Service_ClearCache_Good(t *testing.T) {
	// Clearing with nil labels drops every block and zeroes the in-memory
	// counters.
	t.Run("ClearAll", func(t *testing.T) {
		service := New(Config{BlockSize: 2, ModelHash: "sha256:model"})
		if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4}}); err != nil {
			t.Fatalf("WarmCache() error = %v", err)
		}
		stats, err := service.ClearCache(context.Background(), nil)
		if err != nil {
			t.Fatalf("ClearCache() error = %v", err)
		}
		if stats.Blocks != 0 {
			t.Fatalf("ClearCache stats = %+v, want zero blocks", stats)
		}
	})

	// Disk-backed clear-all also removes the persisted block files and resets
	// DiskBytes to zero.
	t.Run("ClearCacheRemovesDiskBlocks", func(t *testing.T) {
		diskPath := core.PathJoin(t.TempDir(), "blocks")
		service := New(Config{BlockSize: 2, ModelHash: "sha256:model", DiskPath: diskPath})
		result, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4}})
		if err != nil {
			t.Fatalf("WarmCache() error = %v", err)
		}
		var diskFiles []string
		for _, ref := range result.Blocks {
			diskFiles = append(diskFiles, ref.Labels["disk_path"])
		}
		stats, err := service.ClearCache(context.Background(), nil)
		if err != nil {
			t.Fatalf("ClearCache() error = %v", err)
		}
		if stats.Blocks != 0 || stats.DiskBytes != 0 {
			t.Fatalf("ClearCache stats = %+v, want no persisted blocks", stats)
		}
		for _, path := range diskFiles {
			if stat := core.Stat(path); stat.OK {
				t.Fatalf("persisted block still exists at %s", path)
			}
		}
	})

	// Label-scoped clear drops only matching blocks (and their disk files),
	// leaving the rest warm and on disk, and bumps the cleared counter per
	// removed block.
	t.Run("ClearCacheWithLabelsRemovesOnlyMatchingBlocks", func(t *testing.T) {
		diskPath := core.PathJoin(t.TempDir(), "blocks")
		service := New(Config{BlockSize: 2, ModelHash: "sha256:model", DiskPath: diskPath})
		alpha, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
			Labels: map[string]string{"tenant": "alpha"},
			Tokens: []int32{1, 2, 3},
		})
		if err != nil {
			t.Fatalf("WarmCache(alpha) error = %v", err)
		}
		beta, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
			Labels: map[string]string{"tenant": "beta"},
			Tokens: []int32{4, 5},
		})
		if err != nil {
			t.Fatalf("WarmCache(beta) error = %v", err)
		}
		stats, err := service.ClearCache(context.Background(), map[string]string{"tenant": "alpha"})
		if err != nil {
			t.Fatalf("ClearCache(alpha) error = %v", err)
		}
		if stats.Blocks != 1 || stats.Labels["cleared"] != "2" {
			t.Fatalf("ClearCache(alpha) stats = %+v, want one beta block remaining and two clears", stats)
		}
		for _, ref := range alpha.Blocks {
			if stat := core.Stat(ref.Labels["disk_path"]); stat.OK {
				t.Fatalf("alpha disk block still exists at %s", ref.Labels["disk_path"])
			}
		}
		if stat := core.Stat(beta.Blocks[0].Labels["disk_path"]); !stat.OK {
			t.Fatalf("beta disk block was removed: %s", beta.Blocks[0].Labels["disk_path"])
		}
		entries, err := service.CacheEntries(context.Background(), nil)
		if err != nil {
			t.Fatalf("CacheEntries() error = %v", err)
		}
		if len(entries) != 1 || entries[0].Labels["tenant"] != "beta" {
			t.Fatalf("remaining entries = %+v, want only beta", entries)
		}
	})
}

func TestBlockcache_Service_ClearCache_Bad(t *testing.T) {
	// A nil *Service is reported, not dereferenced.
	if _, err := (*Service)(nil).ClearCache(context.Background(), nil); err == nil {
		t.Fatal("ClearCache(nil service) error = nil")
	}
	// A cancelled context short-circuits ClearCache with the context error.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	service := New(Config{})
	if _, err := service.ClearCache(cancelled, nil); err == nil {
		t.Fatal("ClearCache(cancelled) error = nil")
	}
}

func TestBlockcache_Service_ClearCache_Ugly(t *testing.T) {
	// Edge: clearing an already-empty service is a no-op that still succeeds
	// and reports zero blocks, and the cleared counter increments on the
	// clear-all path even with nothing to remove.
	service := New(Config{BlockSize: 2, ModelHash: "sha256:model"})
	stats, err := service.ClearCache(context.Background(), nil)
	if err != nil {
		t.Fatalf("ClearCache(empty) error = %v", err)
	}
	if stats.Blocks != 0 {
		t.Fatalf("ClearCache(empty) stats = %+v, want zero blocks", stats)
	}
	if stats.Labels["cleared"] != "1" {
		t.Fatalf("ClearCache(empty) cleared = %q, want 1 (clear-all bumps the counter)", stats.Labels["cleared"])
	}
	// A label-scoped clear that matches nothing removes nothing and leaves the
	// cleared counter untouched.
	again, err := service.ClearCache(context.Background(), map[string]string{"tenant": "nope"})
	if err != nil {
		t.Fatalf("ClearCache(no match) error = %v", err)
	}
	if again.Blocks != 0 || again.Labels["cleared"] != "1" {
		t.Fatalf("ClearCache(no match) stats = %+v, want nothing cleared", again)
	}
}

// ---------------------------------------------------------------------------
// HashModelParts
// ---------------------------------------------------------------------------

func TestBlockcache_HashModelParts_Good(t *testing.T) {
	// HashModelParts returns a stable SHA-256 hex digest of the supplied
	// identity parts. The same arguments always produce the same 64-char hex
	// hash — the property callers rely on for portable cache identity.
	const want = "aa5dab1cd4dbf496368ad47e056a6595e3cb3fc46864a094b0d72b15e7cf92eb"
	got := HashModelParts("qwen3", 151936)
	if got != want {
		t.Fatalf("HashModelParts(qwen3, 151936) = %q, want %q", got, want)
	}
	if again := HashModelParts("qwen3", 151936); again != got {
		t.Fatalf("HashModelParts is not deterministic: %q != %q", again, got)
	}
}

func TestBlockcache_HashModelParts_Bad(t *testing.T) {
	// The no-argument call is the degenerate input: it still yields a valid,
	// stable 64-char hex digest (the hash of an empty parts list), and that
	// digest differs from any non-empty argument list — an empty identity is
	// not confusable with a real one.
	got := HashModelParts()
	if len(got) != 64 {
		t.Fatalf("HashModelParts() = %q, want a 64-char hex digest", got)
	}
	if got == HashModelParts("qwen3", 151936) {
		t.Fatal("HashModelParts() (no args) collided with a non-empty identity")
	}
}

func TestBlockcache_HashModelParts_Ugly(t *testing.T) {
	// Order sensitivity is the edge that matters for identity: the parts are
	// hashed as an ordered list, so swapping two parts changes the digest —
	// ("a","b") and ("b","a") must not collide.
	ab := HashModelParts("a", "b")
	ba := HashModelParts("b", "a")
	if ab == ba {
		t.Fatalf("HashModelParts is order-insensitive: (a,b)=%q == (b,a)=%q", ab, ba)
	}
	// A large, mixed argument list must not panic and must stay 64-char hex.
	big := make([]any, 0, 1024)
	for i := 0; i < 512; i++ {
		big = append(big, i, "part")
	}
	if got := HashModelParts(big...); len(got) != 64 {
		t.Fatalf("HashModelParts(large) = %q, want a 64-char hex digest", got)
	}
}

// ---------------------------------------------------------------------------
// Disk record compatibility (incompatible-record path) — exercised through the
// public CacheStats surface, kept here so the scenario is not lost in the fold.
// ---------------------------------------------------------------------------

func TestBlockcache_Service_CacheStats_IncompatibleDiskRecordIgnored(t *testing.T) {
	// A persisted record whose model hash does not match the service's
	// configured identity is loaded, found incompatible, and skipped — not
	// counted as corrupt (it is well-formed, just for a different model).
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll() error = %s", result.Error())
	}
	record := diskRecord{
		Version: diskVersion,
		Ref: inference.CacheBlockRef{
			ID:            "incompatible",
			ModelHash:     "sha256:other-model",
			AdapterHash:   "sha256:adapter",
			TokenizerHash: "sha256:tokenizer",
		},
	}
	if data := core.JSONMarshal(record); !data.OK {
		t.Fatalf("JSONMarshal(record) error = %s", data.Error())
	} else if result := core.WriteFile(core.PathJoin(diskPath, "incompatible.json"), data.Value.([]byte), 0o600); !result.OK {
		t.Fatalf("WriteFile(record) error = %s", result.Error())
	}
	service := New(Config{
		DiskPath:      diskPath,
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
	})
	stats, err := service.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Blocks != 0 || stats.Evictions != 0 || stats.Labels["disk_corrupt"] != "0" {
		t.Fatalf("stats = %+v, want incompatible record ignored without corruption", stats)
	}
}

// ---------------------------------------------------------------------------
// Unexported helper coverage — these symbols have no public canonical slot, so
// they live here alongside the public triplets. The name carries the required
// `_` separator and "Helpers" is not a real symbol, so the AX-7 triplet and
// non-canonical-triplet checks correctly ignore it.
// ---------------------------------------------------------------------------

func TestBlockCacheHelpers_Good(t *testing.T) {
	if !blockRefMatchesLabels(inference.CacheBlockRef{ModelHash: "m", AdapterHash: "a", TokenizerHash: "t", Labels: map[string]string{"tenant": "alpha"}}, map[string]string{
		"model_hash":     "m",
		"adapter_hash":   "a",
		"tokenizer_hash": "t",
		"tenant":         "alpha",
	}) {
		t.Fatal("blockRefMatchesLabels() returned false for matching labels")
	}
	if blockRefMatchesLabels(inference.CacheBlockRef{ModelHash: "m"}, map[string]string{"model_hash": "other"}) {
		t.Fatal("blockRefMatchesLabels() returned true for model mismatch")
	}
	if cacheIdentityMatches("actual", "requested") {
		t.Fatal("cacheIdentityMatches() returned true for mismatch")
	}
	if boolLabel(true) != "true" || boolLabel(false) != "false" {
		t.Fatal("boolLabel() returned unexpected text")
	}
	if got := firstNonEmptyString("", "  ", "value"); got != "value" {
		t.Fatalf("firstNonEmptyString() = %q, want value", got)
	}
	labels := map[string]string{"a": "b"}
	cloned := cloneBlockCacheLabels(labels)
	cloned["a"] = "changed"
	if labels["a"] != "b" {
		t.Fatalf("cloneBlockCacheLabels mutated source = %+v", labels)
	}
	refs := []inference.CacheBlockRef{
		{ID: "b", TokenStart: 2},
		{ID: "a", TokenStart: 0},
	}
	sortCacheBlockRefs(refs)
	if refs[0].ID != "a" || !cacheBlockRefLess(refs[0], refs[1]) {
		t.Fatalf("sorted refs = %+v, want token order", refs)
	}
	// cacheBlockRefLess tie-break: equal TokenStart falls through to the ID
	// comparison.
	if !cacheBlockRefLess(
		inference.CacheBlockRef{TokenStart: 4, ID: "aaa"},
		inference.CacheBlockRef{TokenStart: 4, ID: "bbb"},
	) {
		t.Fatal("cacheBlockRefLess(equal start, aaa<bbb) = false, want true")
	}
	if cacheBlockRefLess(
		inference.CacheBlockRef{TokenStart: 4, ID: "bbb"},
		inference.CacheBlockRef{TokenStart: 4, ID: "aaa"},
	) {
		t.Fatal("cacheBlockRefLess(equal start, bbb<aaa) = true, want false")
	}
	if err := resultError(core.Result{OK: true}); err != nil {
		t.Fatalf("resultError(OK) = %v", err)
	}
	if err := resultError(core.Result{Value: core.NewError("explicit")}); err == nil || err.Error() != "explicit" {
		t.Fatalf("resultError(error) = %v", err)
	}
	if err := resultError(core.Result{}); err == nil {
		t.Fatal("resultError(empty) = nil")
	}
}
