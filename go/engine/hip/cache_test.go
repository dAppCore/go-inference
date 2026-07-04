// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/json"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/state"
)

func TestCacheService_Good_WarmStatsClear(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{ModelHash: "model", AdapterHash: "adapter", TokenizerHash: "tok", CacheMode: "q8"})

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Model:   inference.ModelIdentity{Hash: "model"},
		Adapter: inference.AdapterIdentity{Hash: "adapter"},
		Tokens:  []int32{1, 2, 3},
		Labels:  map[string]string{"tokenizer_hash": "tok", "tenant": "a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(warmed.Blocks))
	core.AssertEqual(t, "q8", warmed.Stats.CacheMode)
	stats, err := service.CacheStats(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, stats.Blocks)
	core.AssertGreater(t, stats.MemoryBytes, uint64(0))
	core.AssertEqual(t, "3", stats.Labels["cached_tokens"])
	core.AssertEqual(t, core.Sprintf("%d", defaultROCmKVBlockSize), stats.Labels["kv_cache_block_size"])
	core.AssertEqual(t, "1", stats.Labels["kv_key_width"])
	core.AssertEqual(t, "1", stats.Labels["kv_value_width"])

	stats, err = service.ClearCache(context.Background(), nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, stats.Blocks)
}

func TestCacheService_Good_StatsReportExplicitWarmMode(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{})

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Mode:   rocmKVCacheModeKQ8VQ4,
	})
	core.RequireNoError(t, err)
	stats, err := service.CacheStats(context.Background())
	core.RequireNoError(t, err)

	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, warmed.Stats.CacheMode)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, stats.CacheMode)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, warmed.Blocks[0].Encoding)
}

func TestCacheService_Good_RecordsHitsForOverlappingPrefix(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{ModelHash: "m", TokenizerHash: "tok"})
	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Model: inference.ModelIdentity{Hash: "m"}, Tokens: []int32{1, 2}, Labels: map[string]string{"tokenizer_hash": "tok"}})
	core.RequireNoError(t, err)

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Model: inference.ModelIdentity{Hash: "m"}, Tokens: []int32{1, 2, 3}, Labels: map[string]string{"tokenizer_hash": "tok"}})

	core.RequireNoError(t, err)
	core.RequireTrue(t, len(warmed.Blocks) == 1)
	core.AssertEqual(t, "prompt", warmed.Blocks[0].Kind)
	core.AssertEqual(t, 3, warmed.Blocks[0].TokenCount)
	core.AssertEqual(t, "true", warmed.Labels["prefix_hit"])
	core.AssertEqual(t, uint64(1), warmed.Stats.Hits)
	core.AssertEqual(t, uint64(1), warmed.Stats.Misses)
	if warmed.Stats.RestoreMillis <= 0 {
		t.Fatalf("stats = %+v, want prefix restore time accounted", warmed.Stats)
	}
}

func TestCacheService_Good_CreatesFullBlockAfterPrefixHit(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{})
	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2}})
	core.RequireNoError(t, err)
	_, err = service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.RequireNoError(t, err)

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3, 4}})

	core.RequireNoError(t, err)
	core.RequireTrue(t, len(warmed.Blocks) == 1)
	core.AssertEqual(t, 4, warmed.Blocks[0].TokenCount)
	core.AssertEqual(t, 3, warmed.Stats.Blocks)
}

func TestCacheService_Good_BlockIdentityIncludesRequestCompatibility(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{})
	first, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Model: inference.ModelIdentity{Hash: "model-a"}, Adapter: inference.AdapterIdentity{Hash: "adapter-a"}, Tokens: []int32{1, 2}, Labels: map[string]string{"tokenizer_hash": "tok-a"}})
	core.RequireNoError(t, err)

	second, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Model: inference.ModelIdentity{Hash: "model-b"}, Adapter: inference.AdapterIdentity{Hash: "adapter-b"}, Tokens: []int32{1, 2}, Labels: map[string]string{"tokenizer_hash": "tok-b"}})

	core.RequireNoError(t, err)
	if first.Blocks[0].ID == second.Blocks[0].ID {
		t.Fatalf("cache IDs both %q, want compatibility hashes in block identity", first.Blocks[0].ID)
	}
	core.AssertEqual(t, uint64(2), second.Stats.Misses)
	core.AssertEqual(t, uint64(0), second.Stats.Hits)
	core.AssertEqual(t, 2, second.Stats.Blocks)
	core.AssertEqual(t, "model-b", second.Blocks[0].ModelHash)
	core.AssertEqual(t, "adapter-b", second.Blocks[0].AdapterHash)
	core.AssertEqual(t, "tok-b", second.Blocks[0].TokenizerHash)
}

func TestCacheService_Good_WarmCacheReturnsClonedBlockLabels(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{})
	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2},
		Labels: map[string]string{"tenant": "a"},
	})
	core.RequireNoError(t, err)

	warmed.Blocks[0].Labels["tenant"] = "mutated"
	stats, err := service.ClearCache(context.Background(), map[string]string{"tenant": "a"})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, stats.Blocks)
}

func TestCacheService_Good_ReturnsClonedResultLabelsAndStats(t *testing.T) {
	configLabels := map[string]string{"service": "cache"}
	service := NewBlockCacheService(BlockCacheConfig{Labels: configLabels})
	configLabels["service"] = "mutated"
	warmLabels := map[string]string{"tenant": "a"}
	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2},
		Labels: warmLabels,
	})
	core.RequireNoError(t, err)
	warmLabels["tenant"] = "mutated"

	warmed.Labels["tenant"] = "mutated"
	warmed.Stats.Labels["service"] = "mutated"
	warmed.Stats.Labels["kv_backing"] = "mutated"
	warmed.Blocks[0].Labels["tenant"] = "mutated"

	stats, err := service.CacheStats(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, "cache", stats.Labels["service"])
	core.AssertEqual(t, "metadata", stats.Labels["kv_backing"])

	stats.Labels["service"] = "mutated"
	stats.Labels["kv_backing"] = "mutated"
	stats, err = service.CacheStats(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, "cache", stats.Labels["service"])
	core.AssertEqual(t, "metadata", stats.Labels["kv_backing"])

	stats, err = service.ClearCache(context.Background(), map[string]string{"tenant": "a"})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, stats.Blocks)
}

func TestCacheService_Good_MetadataWarmScrubsRuntimeOnlyLabels(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{Labels: map[string]string{
		"service":           "cache",
		"kv_device_restore": "hit",
		"kv_device_tokens":  "99",
	}})

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2},
		Labels: map[string]string{
			"tenant":                 "a",
			"kv_cache_constructible": "true",
			"kv_cache_snapshot":      "portable",
			"kv_device_backing":      "mirrored",
			"kv_device_bytes":        "4096",
			"kv_device_error":        "spoofed",
			"kv_device_pages":        "7",
		},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "metadata", warmed.Labels["kv_backing"])
	core.AssertEqual(t, "metadata", warmed.Blocks[0].Labels["kv_backing"])
	core.AssertEqual(t, "metadata", warmed.Stats.Labels["kv_backing"])
	for _, item := range []struct {
		name   string
		labels map[string]string
	}{
		{name: "result", labels: warmed.Labels},
		{name: "block", labels: warmed.Blocks[0].Labels},
		{name: "stats", labels: warmed.Stats.Labels},
	} {
		for _, key := range []string{"kv_cache_constructible", "kv_cache_snapshot", "kv_device_backing", "kv_device_bytes", "kv_device_error", "kv_device_pages", "kv_device_restore", "kv_device_tokens"} {
			if item.labels[key] != "" {
				t.Fatalf("%s labels[%q] = %q, want scrubbed from metadata warm", item.name, key, item.labels[key])
			}
		}
	}

	stats, err := service.CacheStats(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, "metadata", stats.Labels["kv_backing"])
	for _, key := range []string{"kv_cache_constructible", "kv_cache_snapshot", "kv_device_backing", "kv_device_bytes", "kv_device_error", "kv_device_pages", "kv_device_restore", "kv_device_tokens"} {
		if stats.Labels[key] != "" {
			t.Fatalf("cache stats labels[%q] = %q, want scrubbed from metadata warm", key, stats.Labels[key])
		}
	}
}

func TestCacheService_Good_MetadataWarmScrubsKVShapeLabels(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{Labels: map[string]string{
		"service":             "cache",
		"kv_cache_block_size": "2",
	}})

	first, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2},
		Labels: map[string]string{
			"tenant":         "a",
			"kv_key_width":   "2",
			"kv_value_width": "3",
		},
	})
	core.RequireNoError(t, err)
	second, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2},
		Labels: map[string]string{"tenant": "a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, first.Blocks[0].ID, second.Blocks[0].ID)
	core.AssertEqual(t, uint64(1), second.Stats.Hits)
	core.AssertEqual(t, "metadata", first.Labels["kv_backing"])
	core.AssertEqual(t, "metadata", first.Blocks[0].Labels["kv_backing"])
	core.AssertEqual(t, "metadata", first.Stats.Labels["kv_backing"])
	for _, item := range []struct {
		name   string
		labels map[string]string
	}{
		{name: "result", labels: first.Labels},
		{name: "block", labels: first.Blocks[0].Labels},
		{name: "stats", labels: first.Stats.Labels},
	} {
		for _, key := range []string{"kv_cache_block_size", "kv_key_width", "kv_value_width"} {
			if item.labels[key] != "" {
				t.Fatalf("%s labels[%q] = %q, want scrubbed from metadata warm", item.name, key, item.labels[key])
			}
		}
	}

	stats, err := service.CacheStats(context.Background())
	core.RequireNoError(t, err)
	for _, key := range []string{"kv_cache_block_size", "kv_key_width", "kv_value_width"} {
		if stats.Labels[key] != "" {
			t.Fatalf("cache stats labels[%q] = %q, want scrubbed from metadata warm", key, stats.Labels[key])
		}
	}
}

func TestCacheService_Good_WarmScrubsDiskRuntimeLabels(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{Labels: map[string]string{
		"service":            "cache",
		"disk_cache_restore": "hit",
		"disk_chunk_id":      "99",
		"disk_uri":           "state://cache/spoofed-service",
	}})

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2},
		Labels: map[string]string{
			"tenant":             "a",
			"disk_cache_restore": "hit",
			"disk_codec":         "spoofed",
			"disk_chunk_id":      "7",
			"disk_encoding":      rocmKVSnapshotEncoding,
			"disk_kind":          "rocm-cache-kv-state",
			"disk_uri":           "state://cache/spoofed-request",
		},
	})

	core.RequireNoError(t, err)
	for _, item := range []struct {
		name   string
		labels map[string]string
	}{
		{name: "result", labels: warmed.Labels},
		{name: "block", labels: warmed.Blocks[0].Labels},
		{name: "stats", labels: warmed.Stats.Labels},
	} {
		for _, key := range []string{"disk_cache_restore", "disk_codec", "disk_chunk_id", "disk_encoding", "disk_kind", "disk_uri"} {
			if item.labels[key] != "" {
				t.Fatalf("%s labels[%q] = %q, want scrubbed from live warm", item.name, key, item.labels[key])
			}
		}
	}

	stats, err := service.CacheStats(context.Background())
	core.RequireNoError(t, err)
	for _, key := range []string{"disk_cache_restore", "disk_codec", "disk_chunk_id", "disk_encoding", "disk_kind", "disk_uri"} {
		if stats.Labels[key] != "" {
			t.Fatalf("cache stats labels[%q] = %q, want scrubbed from live warm", key, stats.Labels[key])
		}
	}
}

func TestCacheService_Good_WarmAllowsDiskURIWithStore(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: "block-prefix", DiskStore: store})

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2},
		Labels: map[string]string{
			"disk_uri": "state://cache/request-specific",
		},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "state://cache/request-specific", warmed.Labels["disk_uri"])
	core.AssertEqual(t, "state://cache/request-specific", warmed.Blocks[0].Labels["disk_uri"])
	core.AssertEqual(t, "state://cache/request-specific", warmed.Stats.Labels["disk_uri"])
	_, err = store.ResolveURI(context.Background(), "state://cache/request-specific")
	core.RequireNoError(t, err)
}

func TestCacheService_Good_PrefixHitsRequireMatchingCompatibility(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{})
	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Model: inference.ModelIdentity{Hash: "model-a"}, Tokens: []int32{1, 2}})
	core.RequireNoError(t, err)

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Model: inference.ModelIdentity{Hash: "model-b"}, Tokens: []int32{1, 2, 3}})

	core.RequireNoError(t, err)
	if warmed.Labels["prefix_hit"] == "true" {
		t.Fatalf("labels = %+v, want no prefix hit across model hash", warmed.Labels)
	}
	core.AssertEqual(t, uint64(2), warmed.Stats.Misses)
	core.AssertEqual(t, uint64(0), warmed.Stats.Hits)
}

func TestCacheService_Good_UsesKVCacheModeByteAccounting(t *testing.T) {
	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	fp16 := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeFP16})
	q8 := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8})

	fp16Warm, err := fp16.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
	core.RequireNoError(t, err)
	q8Warm, err := q8.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
	core.RequireNoError(t, err)

	core.AssertEqual(t, "package_local", fp16Warm.Blocks[0].Labels["kv_backing"])
	core.AssertEqual(t, "planned", fp16Warm.Blocks[0].Labels["kv_device_backing"])
	core.AssertEqual(t, "true", q8Warm.Blocks[0].Labels["kv_cache_constructible"])
	if q8Warm.Blocks[0].SizeBytes >= fp16Warm.Blocks[0].SizeBytes {
		t.Fatalf("q8 block size = %d, fp16 block size = %d, want q8 KV accounting lower than fp16", q8Warm.Blocks[0].SizeBytes, fp16Warm.Blocks[0].SizeBytes)
	}
}

func TestCacheService_Good_UsesKVVectorWidthByteAccounting(t *testing.T) {
	tokens := []int32{1, 2, 3, 4}
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeKQ8VQ4})
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	keys, values := cacheWarmKVTensors(tokens, 2, 4)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 4, keys, values))

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: tokens,
		Labels: map[string]string{
			"kv_cache_block_size": "2",
			"kv_key_width":        "2",
			"kv_value_width":      "4",
		},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "2", warmed.Blocks[0].Labels["kv_cache_block_size"])
	core.AssertEqual(t, "2", warmed.Blocks[0].Labels["kv_key_width"])
	core.AssertEqual(t, "4", warmed.Blocks[0].Labels["kv_value_width"])
	core.AssertEqual(t, cache.MemoryBytes(), warmed.Blocks[0].SizeBytes)
}

func TestCacheService_Good_BlockIdentityIncludesKVVectorShape(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8})
	first, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2},
		Labels: map[string]string{"kv_cache_block_size": "16", "kv_key_width": "1", "kv_value_width": "1"},
	})
	core.RequireNoError(t, err)

	second, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2},
		Labels: map[string]string{"kv_cache_block_size": "16", "kv_key_width": "2", "kv_value_width": "2"},
	})

	core.RequireNoError(t, err)
	if first.Blocks[0].ID == second.Blocks[0].ID {
		t.Fatalf("cache IDs both %q, want KV vector shape in block identity", first.Blocks[0].ID)
	}
	core.AssertEqual(t, uint64(2), second.Stats.Misses)
	core.AssertEqual(t, uint64(0), second.Stats.Hits)
	core.AssertEqual(t, 2, second.Stats.Blocks)
	core.AssertEqual(t, "2", second.Blocks[0].Labels["kv_key_width"])
}

func TestCacheService_Good_BlockIdentityIncludesKVBlockSize(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8})
	first, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Labels: map[string]string{"kv_cache_block_size": "1", "kv_key_width": "2", "kv_value_width": "2"},
	})
	core.RequireNoError(t, err)

	second, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Labels: map[string]string{"kv_cache_block_size": "3", "kv_key_width": "2", "kv_value_width": "2"},
	})

	core.RequireNoError(t, err)
	if first.Blocks[0].ID == second.Blocks[0].ID {
		t.Fatalf("cache IDs both %q, want KV block size in block identity", first.Blocks[0].ID)
	}
	core.AssertEqual(t, uint64(2), second.Stats.Misses)
	core.AssertEqual(t, uint64(0), second.Stats.Hits)
	core.AssertEqual(t, "3", second.Blocks[0].Labels["kv_cache_block_size"])
}

func TestCacheService_Good_WritesPortableKVSnapshotDiskRefsWithOpaqueStateStore(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, DiskStore: store})

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})

	core.RequireNoError(t, err)
	core.RequireTrue(t, len(warmed.Blocks) == 1)
	block := warmed.Blocks[0]
	core.AssertContains(t, block.Labels["disk_uri"], "rocm://cache/")
	core.AssertEqual(t, state.CodecMemory, block.Labels["disk_codec"])
	core.AssertEqual(t, rocmKVSnapshotEncoding, block.Labels["disk_encoding"])
	core.AssertEqual(t, "rocm-cache-kv-state", block.Labels["disk_kind"])
	core.AssertEqual(t, "portable", block.Labels["kv_cache_snapshot"])
	core.AssertGreater(t, warmed.Stats.DiskBytes, uint64(0))
	core.AssertEqual(t, block.Labels["disk_uri"], warmed.Stats.Labels["disk_uri"])
	core.AssertEqual(t, state.CodecMemory, warmed.Stats.Labels["disk_codec"])
	core.AssertEqual(t, rocmKVSnapshotEncoding, warmed.Stats.Labels["disk_encoding"])
	core.AssertNotEmpty(t, warmed.Stats.Labels["disk_chunk_id"])
	core.AssertEqual(t, "3", warmed.Stats.Labels["cached_tokens"])
	core.AssertEqual(t, core.Sprintf("%d", defaultROCmKVBlockSize), warmed.Stats.Labels["kv_cache_block_size"])
	chunk, err := store.ResolveURI(context.Background(), block.Labels["disk_uri"])
	core.RequireNoError(t, err)
	var snapshot rocmKVCacheSnapshot
	core.RequireNoError(t, json.Unmarshal(chunk.Data, &snapshot))
	core.AssertEqual(t, block.ID, snapshot.CacheBlockID)
	core.AssertEqual(t, rocmKVCacheModeQ8, snapshot.Mode)
	restored, err := newROCmKVCacheFromSnapshot(chunk.Data)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVCacheModeQ8, restored.Stats().CacheMode)
	core.AssertEqual(t, 3, restored.TokenCount())

	hit, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint64(1), hit.Stats.Hits)
	core.AssertEqual(t, block.Labels["disk_uri"], hit.Labels["disk_uri"])
	core.AssertEqual(t, block.Labels["disk_chunk_id"], hit.Labels["disk_chunk_id"])
	core.AssertEqual(t, rocmKVSnapshotEncoding, hit.Labels["disk_encoding"])
	core.AssertEqual(t, block.Labels["kv_cache_block_size"], hit.Labels["kv_cache_block_size"])
}

func TestCacheService_Good_RestoresPortableKVSnapshotDiskRefOnColdWarm(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	warming := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, DiskStore: store})
	first, err := warming.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Labels: map[string]string{
			"kv_key_width":   "2",
			"kv_value_width": "2",
		},
	})
	core.RequireNoError(t, err)
	cold := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, DiskStore: store})

	restored, err := cold.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Labels: map[string]string{
			"kv_key_width":   "2",
			"kv_value_width": "2",
		},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, first.Blocks[0].ID, restored.Blocks[0].ID)
	core.AssertEqual(t, uint64(1), restored.Stats.Hits)
	core.AssertEqual(t, uint64(0), restored.Stats.Misses)
	core.AssertEqual(t, "hit", restored.Labels["disk_cache_restore"])
	core.AssertEqual(t, "hit", restored.Stats.Labels["disk_cache_restore"])
	core.AssertEqual(t, rocmKVSnapshotEncoding, restored.Labels["disk_encoding"])
	core.AssertEqual(t, "portable", restored.Labels["kv_cache_snapshot"])
	core.AssertEqual(t, "2", restored.Labels["kv_key_width"])
	core.AssertEqual(t, "2", restored.Labels["kv_value_width"])
	core.AssertEqual(t, rocmKVCacheModeQ8, restored.Stats.CacheMode)
	core.AssertGreater(t, restored.Stats.DiskBytes, uint64(0))
}

func BenchmarkCacheValidateKVSnapshotTokens_KQ8VQ4Page(b *testing.B) {
	tokens := make([]int32, 512)
	for index := range tokens {
		tokens[index] = int32(index + 1)
	}
	keys, values := cacheWarmKVTensors(tokens, 128, 128)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 512)
	if err != nil {
		b.Fatalf("create KV cache: %v", err)
	}
	if err := cache.AppendVectors(0, 128, 128, keys, values); err != nil {
		b.Fatalf("append KV cache vectors: %v", err)
	}
	b.SetBytes(int64((len(keys) + len(values)) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := validateCacheKVSnapshotTokens(cache, tokens); err != nil {
			b.Fatalf("validate KV snapshot tokens: %v", err)
		}
	}
}

func BenchmarkCacheValidateKVSnapshotTokens_KQ8VQ4FourPages(b *testing.B) {
	tokens := make([]int32, 2048)
	for index := range tokens {
		tokens[index] = int32(index + 1)
	}
	keys, values := cacheWarmKVTensors(tokens, 128, 128)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 512)
	if err != nil {
		b.Fatalf("create KV cache: %v", err)
	}
	if err := cache.AppendVectors(0, 128, 128, keys, values); err != nil {
		b.Fatalf("append KV cache vectors: %v", err)
	}
	b.SetBytes(int64((len(keys) + len(values)) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := validateCacheKVSnapshotTokens(cache, tokens); err != nil {
			b.Fatalf("validate KV snapshot tokens: %v", err)
		}
	}
}

func TestCacheService_Good_MirrorsWarmKVSnapshotToHIPDevice(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver})

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Labels: map[string]string{
			"kv_cache_block_size": "2",
		},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "package_local", warmed.Blocks[0].Labels["kv_backing"])
	core.AssertEqual(t, "mirrored", warmed.Blocks[0].Labels["kv_device_backing"])
	core.AssertEqual(t, "2", warmed.Blocks[0].Labels["kv_device_pages"])
	core.AssertEqual(t, "3", warmed.Blocks[0].Labels["kv_device_tokens"])
	core.AssertNotEmpty(t, warmed.Blocks[0].Labels["kv_device_bytes"])
	core.AssertEqual(t, "mirrored", warmed.Stats.Labels["kv_device_backing"])
	core.AssertEqual(t, "2", warmed.Stats.Labels["kv_device_pages"])
	if len(driver.allocations) != 4 || len(driver.copies) != 4 {
		t.Fatalf("driver allocations=%+v copies=%+v, want mirrored key/value pages", driver.allocations, driver.copies)
	}

	stats, err := service.ClearCache(context.Background(), nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, stats.Blocks)
	core.AssertEqual(t, 4, len(driver.frees))
}

func TestCacheService_Good_CloseClosesMirroredKVPages(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver})
	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Labels: map[string]string{
			"kv_cache_block_size": "2",
		},
	})
	core.RequireNoError(t, err)

	core.RequireNoError(t, service.Close())
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
	core.RequireNoError(t, service.Close())
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
	stats, err := service.CacheStats(context.Background())
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, stats.Blocks)
}

func TestCacheService_Bad_ClosePropagatesDeviceFreeFailure(t *testing.T) {
	driver := &failingHIPDriver{available: true, freeErr: core.NewError("free failed")}
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver})
	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Labels: map[string]string{
			"kv_cache_block_size": "2",
		},
	})
	core.RequireNoError(t, err)

	err = service.Close()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "free KV")
	core.AssertContains(t, err.Error(), "free failed")
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
	stats, statsErr := service.CacheStats(context.Background())
	core.RequireNoError(t, statsErr)
	core.AssertEqual(t, 0, stats.Blocks)
}

func TestCacheService_Bad_ClearPropagatesDeviceFreeFailure(t *testing.T) {
	driver := &failingHIPDriver{available: true, freeErr: core.NewError("free failed")}
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver})
	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: []int32{1, 2, 3},
		Labels: map[string]string{
			"kv_cache_block_size": "2",
			"tenant":              "a",
		},
	})
	core.RequireNoError(t, err)

	stats, err := service.ClearCache(context.Background(), map[string]string{"tenant": "a"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "free KV")
	core.AssertContains(t, err.Error(), "free failed")
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
	core.AssertEqual(t, 0, stats.Blocks)
}

func TestCacheService_Good_RestoresDiskKVSnapshotToHIPDeviceOnColdWarm(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	warming := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, DiskStore: store})
	first, err := warming.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.RequireNoError(t, err)
	driver := &fakeHIPDriver{available: true}
	cold := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, DiskStore: store, deviceDriver: driver})

	restored, err := cold.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})

	core.RequireNoError(t, err)
	core.AssertEqual(t, first.Blocks[0].ID, restored.Blocks[0].ID)
	core.AssertEqual(t, "hit", restored.Labels["disk_cache_restore"])
	core.AssertEqual(t, "mirrored", restored.Labels["kv_device_backing"])
	core.AssertEqual(t, "mirrored", restored.Labels["kv_device_restore"])
	core.AssertEqual(t, "mirrored", restored.Stats.Labels["kv_device_backing"])
	core.AssertEqual(t, "mirrored", restored.Stats.Labels["kv_device_restore"])
	if len(driver.allocations) == 0 || len(driver.copies) == 0 {
		t.Fatalf("driver allocations=%+v copies=%+v, want cold disk restore mirrored to HIP device", driver.allocations, driver.copies)
	}
}

func TestCacheService_Bad_DeviceMirrorFailureKeepsPortableKVBlock(t *testing.T) {
	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed"), copyErrAt: 2}
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver})

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "package_local", warmed.Labels["kv_backing"])
	core.AssertEqual(t, "failed", warmed.Labels["kv_device_backing"])
	core.AssertContains(t, warmed.Labels["kv_device_error"], "copy KV value page")
	core.AssertEqual(t, 1, warmed.Stats.Blocks)
	core.AssertEqual(t, "failed", warmed.Stats.Labels["kv_device_backing"])
	core.AssertContains(t, warmed.Stats.Labels["kv_device_error"], "copy KV value page")
	core.AssertEqual(t, 2, len(driver.frees))
}

func TestCacheService_Bad_RejectsMismatchedPortableKVSnapshotDiskRef(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	warming := NewBlockCacheService(BlockCacheConfig{
		CacheMode: rocmKVCacheModeQ8,
		DiskStore: store,
		DiskURI:   "state://cache/shared",
	})
	_, err := warming.WarmCache(context.Background(), inference.CacheWarmRequest{
		Model:  inference.ModelIdentity{Hash: "model-a"},
		Tokens: []int32{1, 2, 3},
	})
	core.RequireNoError(t, err)
	cold := NewBlockCacheService(BlockCacheConfig{
		CacheMode: rocmKVCacheModeQ8,
		DiskStore: store,
		DiskURI:   "state://cache/shared",
	})

	_, err = cold.WarmCache(context.Background(), inference.CacheWarmRequest{
		Model:  inference.ModelIdentity{Hash: "model-b"},
		Tokens: []int32{7, 8, 9},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "disk KV cache ref does not match warm request")
}

func TestCacheService_Bad_RejectsMismatchedPortableKVSnapshotLabels(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(map[string]string)
	}{
		{
			name: "model label",
			mutate: func(labels map[string]string) {
				labels["model_hash"] = "model-b"
			},
		},
		{
			name: "adapter label",
			mutate: func(labels map[string]string) {
				labels["adapter_hash"] = "adapter-b"
			},
		},
		{
			name: "tokenizer label",
			mutate: func(labels map[string]string) {
				labels["tokenizer_hash"] = "tok-b"
			},
		},
		{
			name: "backing label",
			mutate: func(labels map[string]string) {
				labels["kv_backing"] = "metadata"
			},
		},
		{
			name: "block size label",
			mutate: func(labels map[string]string) {
				labels["kv_cache_block_size"] = "99"
			},
		},
		{
			name: "key width label",
			mutate: func(labels map[string]string) {
				labels["kv_key_width"] = "2"
			},
		},
		{
			name: "value width label",
			mutate: func(labels map[string]string) {
				labels["kv_value_width"] = "2"
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := state.NewInMemoryStore(nil)
			uri := "state://cache/kv-" + tt.name
			req := inference.CacheWarmRequest{
				Model:   inference.ModelIdentity{Hash: "model-a"},
				Adapter: inference.AdapterIdentity{Hash: "adapter-a"},
				Tokens:  []int32{1, 2, 3},
				Labels:  map[string]string{"tokenizer_hash": "tok-a"},
			}
			warming := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, DiskStore: store, DiskURI: uri})
			first, err := warming.WarmCache(context.Background(), req)
			core.RequireNoError(t, err)
			chunk, err := store.ResolveURI(context.Background(), first.Blocks[0].Labels["disk_uri"])
			core.RequireNoError(t, err)
			var snapshot rocmKVCacheSnapshot
			core.RequireNoError(t, json.Unmarshal(chunk.Data, &snapshot))
			if snapshot.Labels == nil {
				snapshot.Labels = map[string]string{}
			}
			tt.mutate(snapshot.Labels)
			corrupt, err := json.Marshal(snapshot)
			core.RequireNoError(t, err)
			_, err = store.PutBytes(context.Background(), corrupt, state.PutOptions{URI: uri, Kind: "rocm-cache-kv-state", Track: rocmKVSnapshotEncoding})
			core.RequireNoError(t, err)
			cold := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, DiskStore: store, DiskURI: uri})

			_, err = cold.WarmCache(context.Background(), req)

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), "disk KV cache ref does not match warm request")
		})
	}
}

func TestCacheService_Good_RestoresLegacyRawPortableKVSnapshotDiskRef(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	keys, values := cacheWarmKVTensors([]int32{1, 2, 3}, 1, 1)
	core.RequireNoError(t, cache.AppendVectors(0, 1, 1, keys, values))
	payload, err := cache.Snapshot()
	core.RequireNoError(t, err)
	_, err = store.PutBytes(context.Background(), payload, state.PutOptions{URI: "state://cache/raw", Kind: "rocm-cache-kv-state", Track: rocmKVSnapshotEncoding})
	core.RequireNoError(t, err)
	service := NewBlockCacheService(BlockCacheConfig{
		CacheMode: rocmKVCacheModeQ8,
		DiskStore: store,
		DiskURI:   "state://cache/raw",
	})

	restored, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})

	core.RequireNoError(t, err)
	core.AssertEqual(t, uint64(1), restored.Stats.Hits)
	core.AssertEqual(t, "hit", restored.Labels["disk_cache_restore"])
	core.AssertEqual(t, rocmKVSnapshotEncoding, restored.Labels["disk_encoding"])
}

func TestCacheService_Bad_RejectsMismatchedLegacyRawPortableKVSnapshotDiskRef(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	keys, values := cacheWarmKVTensors([]int32{1, 2, 3}, 1, 1)
	core.RequireNoError(t, cache.AppendVectors(0, 1, 1, keys, values))
	payload, err := cache.Snapshot()
	core.RequireNoError(t, err)
	_, err = store.PutBytes(context.Background(), payload, state.PutOptions{URI: "state://cache/raw", Kind: "rocm-cache-kv-state", Track: rocmKVSnapshotEncoding})
	core.RequireNoError(t, err)
	service := NewBlockCacheService(BlockCacheConfig{
		CacheMode: rocmKVCacheModeQ8,
		DiskStore: store,
		DiskURI:   "state://cache/raw",
	})

	_, err = service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{7, 8, 9}})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "disk KV cache ref does not match warm request")
}

func TestCacheService_Good_WritesMetadataDiskRefsForBlockPrefixCache(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: "block-prefix", DiskStore: store})

	warmed, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})

	core.RequireNoError(t, err)
	block := warmed.Blocks[0]
	core.AssertEqual(t, "rocm/cache-block+json", block.Labels["disk_encoding"])
	core.AssertEqual(t, "rocm-cache-block", block.Labels["disk_kind"])
	chunk, err := store.ResolveURI(context.Background(), block.Labels["disk_uri"])
	core.RequireNoError(t, err)
	core.AssertContains(t, string(chunk.Data), block.ID)
	var payload cacheBlockDiskPayload
	core.RequireNoError(t, json.Unmarshal(chunk.Data, &payload))
	core.AssertEqual(t, block.ID, payload.ID)
	core.AssertEqual(t, block.Labels["disk_uri"], payload.Labels["disk_uri"])
}

func TestCacheService_Good_RestoresMetadataDiskRefOnColdWarm(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	warming := NewBlockCacheService(BlockCacheConfig{CacheMode: "block-prefix", DiskStore: store})
	first, err := warming.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.RequireNoError(t, err)
	cold := NewBlockCacheService(BlockCacheConfig{CacheMode: "block-prefix", DiskStore: store})

	restored, err := cold.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})

	core.RequireNoError(t, err)
	core.AssertEqual(t, first.Blocks[0].ID, restored.Blocks[0].ID)
	core.AssertEqual(t, uint64(1), restored.Stats.Hits)
	core.AssertEqual(t, uint64(0), restored.Stats.Misses)
	core.AssertEqual(t, "hit", restored.Labels["disk_cache_restore"])
	core.AssertEqual(t, "hit", restored.Stats.Labels["disk_cache_restore"])
	core.AssertEqual(t, "rocm/cache-block+json", restored.Labels["disk_encoding"])
	core.AssertEqual(t, "rocm-cache-block", restored.Labels["disk_kind"])
	core.AssertEqual(t, "metadata", restored.Labels["kv_backing"])
	core.AssertGreater(t, restored.Stats.DiskBytes, uint64(0))
}

func TestCacheService_Bad_RejectsMismatchedMetadataDiskRef(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*cacheBlockDiskPayload)
	}{
		{
			name: "kind",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.Kind = "foreign"
			},
		},
		{
			name: "model hash",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.ModelHash = "model-b"
			},
		},
		{
			name: "adapter hash",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.AdapterHash = "adapter-b"
			},
		},
		{
			name: "tokenizer hash",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.TokenizerHash = "tok-b"
			},
		},
		{
			name: "token start",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.TokenStart = 1
			},
		},
		{
			name: "size bytes",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.SizeBytes++
			},
		},
		{
			name: "model label",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.Labels["model_hash"] = "model-b"
			},
		},
		{
			name: "adapter label",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.Labels["adapter_hash"] = "adapter-b"
			},
		},
		{
			name: "tokenizer label",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.Labels["tokenizer_hash"] = "tok-b"
			},
		},
		{
			name: "cache snapshot label",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.Labels["kv_cache_snapshot"] = "portable"
			},
		},
		{
			name: "device backing label",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.Labels["kv_device_backing"] = "mirrored"
			},
		},
		{
			name: "shape label",
			mutate: func(payload *cacheBlockDiskPayload) {
				payload.Labels["kv_backing"] = "foreign"
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := state.NewInMemoryStore(nil)
			uri := "state://cache/" + tt.name
			warming := NewBlockCacheService(BlockCacheConfig{CacheMode: "block-prefix", DiskStore: store, DiskURI: uri})
			req := inference.CacheWarmRequest{
				Model:   inference.ModelIdentity{Hash: "model-a"},
				Adapter: inference.AdapterIdentity{Hash: "adapter-a"},
				Tokens:  []int32{1, 2, 3},
				Labels:  map[string]string{"tokenizer_hash": "tok-a"},
			}
			first, err := warming.WarmCache(context.Background(), req)
			core.RequireNoError(t, err)
			chunk, err := store.ResolveURI(context.Background(), first.Blocks[0].Labels["disk_uri"])
			core.RequireNoError(t, err)
			var payload cacheBlockDiskPayload
			core.RequireNoError(t, json.Unmarshal(chunk.Data, &payload))
			tt.mutate(&payload)
			corrupt, err := json.Marshal(payload)
			core.RequireNoError(t, err)
			_, err = store.PutBytes(context.Background(), corrupt, state.PutOptions{URI: uri, Kind: "rocm-cache-block", Track: "rocm/cache-block+json"})
			core.RequireNoError(t, err)
			cold := NewBlockCacheService(BlockCacheConfig{CacheMode: "block-prefix", DiskStore: store, DiskURI: uri})

			_, err = cold.WarmCache(context.Background(), req)

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), "disk cache ref does not match warm request")
		})
	}
}

func TestCacheService_Bad_DiskWriteFailureHasContext(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, DiskStore: failingCacheDiskWriter{}})

	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rocm.CacheWarm")
	core.AssertContains(t, err.Error(), "write disk cache ref")
}

func TestCacheService_Bad_RejectsUnsupportedKVMode(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: "not-a-mode"})

	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2}})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported cache mode")
}

func TestCacheService_Bad_RejectsInvalidKVVectorWidth(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8})

	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2}, Labels: map[string]string{"kv_key_width": "0"}})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "kv_key_width")
}

func TestCacheService_Bad_RejectsTokenizerMismatch(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{TokenizerHash: "tok-a"})

	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1}, Labels: map[string]string{"tokenizer_hash": "tok-b"}})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tokenizer hash mismatch")
}

func TestCacheService_Bad_RejectsAdapterMismatch(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{AdapterHash: "adapter-a"})

	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Adapter: inference.AdapterIdentity{Hash: "adapter-b"}, Tokens: []int32{1}})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter hash mismatch")
}

func TestCacheService_Ugly_ClearByLabelsOnlyClearsMatchingBlocks(t *testing.T) {
	service := NewBlockCacheService(BlockCacheConfig{})
	_, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1}, Labels: map[string]string{"tenant": "a"}})
	core.RequireNoError(t, err)
	_, err = service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{2}, Labels: map[string]string{"tenant": "b"}})
	core.RequireNoError(t, err)

	stats, err := service.ClearCache(context.Background(), map[string]string{"tenant": "a"})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, stats.Blocks)
}

func TestCacheService_Good_RocmModelImplementsCacheService(t *testing.T) {
	var _ inference.CacheService = (*rocmModel)(nil)
	var _ ROCmCacheProfileReporter = (*rocmModel)(nil)

	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	warmed, err := model.WarmCache(context.Background(), inference.CacheWarmRequest{Prompt: "hello world"})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(warmed.Blocks))
}

func TestCacheService_Good_RocmModelReportsCacheProfile(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "gemma4_text"}}

	_, err := model.WarmCache(context.Background(), inference.CacheWarmRequest{
		Mode:   rocmKVCacheModeKQ8VQ4,
		Tokens: []int32{1, 2, 3, 4},
	})
	core.RequireNoError(t, err)
	profile, err := model.CacheProfile(context.Background())
	core.RequireNoError(t, err)

	if !profile.Matched() ||
		profile.Architecture != "gemma4_text" ||
		profile.TotalCaches != 1 ||
		profile.QuantizedCaches != 1 ||
		profile.MaxCacheTokens != 4 {
		t.Fatalf("rocmModel.CacheProfile = %+v, want model-scoped cache profile", profile)
	}
}

func TestCacheService_Bad_RocmModelWarmCacheRecordsErr(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}

	_, err := model.WarmCache(context.Background(), inference.CacheWarmRequest{})

	if err == nil {
		t.Fatal("WarmCache missing input error = nil")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "prompt or tokens are required")

	_, err = model.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})

	core.RequireNoError(t, err)
	if resultError(model.Err()) != nil {
		t.Fatalf("WarmCache success Err() = %v, want nil", resultError(model.Err()))
	}
}

func TestCacheService_Good_RocmModelWarmCacheUsesHIPDeviceDriver(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	model := &rocmModel{
		native:    &hipLoadedModel{driver: driver},
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
	}

	warmed, err := model.WarmCache(context.Background(), inference.CacheWarmRequest{
		Mode:   rocmKVCacheModeQ8,
		Tokens: []int32{1, 2, 3},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "mirrored", warmed.Labels["kv_device_backing"])
	core.AssertEqual(t, "mirrored", warmed.Stats.Labels["kv_device_backing"])
	if len(driver.allocations) == 0 || len(driver.copies) == 0 {
		t.Fatalf("driver allocations=%+v copies=%+v, want rocmModel cache warm to mirror KV pages", driver.allocations, driver.copies)
	}
}

func TestCacheService_Good_RocmModelAdapterChangeResetsCache(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}, modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	_, err := model.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2}})
	core.RequireNoError(t, err)

	_, err = model.LoadAdapter("domain.safetensors")
	core.RequireNoError(t, err)
	if model.cache != nil {
		t.Fatalf("cache service should reset after adapter load")
	}
	warmed, err := model.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2}})
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint64(0), warmed.Stats.Hits)
	core.AssertEqual(t, uint64(1), warmed.Stats.Misses)

	core.RequireNoError(t, model.UnloadAdapter())
	if model.cache != nil {
		t.Fatalf("cache service should reset after adapter unload")
	}
}

func TestCacheService_Good_RocmModelAdapterChangeClosesMirroredCache(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cache := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver})
	_, err := cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.RequireNoError(t, err)
	model := &rocmModel{
		native:    &fakeNativeModel{},
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		cache:     cache,
	}

	_, err = model.LoadAdapter("domain.safetensors")

	core.RequireNoError(t, err)
	if model.cache != nil {
		t.Fatalf("cache service should reset after adapter load")
	}
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
}

func TestCacheService_Good_RocmModelCloseClosesMirroredCache(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cache := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver})
	_, err := cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.RequireNoError(t, err)
	native := &fakeNativeModel{}
	model := &rocmModel{
		native:    native,
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		cache:     cache,
	}

	err = resultError(model.Close())

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, native.closeCalls)
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
	if model.cache != nil || model.native != nil {
		t.Fatalf("model did not clear cache/native on close")
	}
}

func TestCacheService_Bad_RocmModelLoadAdapterStopsOnCacheCloseFailure(t *testing.T) {
	driver := &failingHIPDriver{available: true, freeErr: core.NewError("free failed")}
	cache := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver})
	_, err := cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.RequireNoError(t, err)
	native := &fakeNativeModel{}
	model := &rocmModel{
		native:    native,
		adapter:   inference.AdapterIdentity{Path: "previous.safetensors", Format: "lora"},
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		cache:     cache,
	}

	identity, err := model.LoadAdapter("next.safetensors")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "close cache runtime")
	core.AssertContains(t, err.Error(), "free failed")
	if !adapterIdentityIsZero(identity) {
		t.Fatalf("identity = %+v, want zero", identity)
	}
	core.AssertEqual(t, 0, len(native.adapterLoads))
	if got := model.ActiveAdapter(); got.Path != "previous.safetensors" || got.Format != "lora" {
		t.Fatalf("active adapter = %+v, want previous adapter", got)
	}
	if model.cache != cache {
		t.Fatal("cache service was cleared after load-adapter cache close failure")
	}
}

func TestCacheService_Bad_RocmModelUnloadAdapterStopsOnCacheCloseFailure(t *testing.T) {
	driver := &failingHIPDriver{available: true, freeErr: core.NewError("free failed")}
	cache := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver})
	_, err := cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.RequireNoError(t, err)
	native := &fakeNativeModel{adapter: inference.AdapterIdentity{Path: "previous.safetensors", Format: "lora"}}
	model := &rocmModel{
		native:    native,
		adapter:   inference.AdapterIdentity{Path: "previous.safetensors", Format: "lora"},
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		cache:     cache,
	}

	err = model.UnloadAdapter()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "close cache runtime")
	core.AssertContains(t, err.Error(), "free failed")
	core.AssertEqual(t, 0, native.unloadCalls)
	if got := model.ActiveAdapter(); got.Path != "previous.safetensors" || got.Format != "lora" {
		t.Fatalf("active adapter = %+v, want previous adapter", got)
	}
	if model.cache != cache {
		t.Fatal("cache service was cleared after unload-adapter cache close failure")
	}
}

func TestCacheService_Bad_RocmModelCloseStopsOnCacheCloseFailure(t *testing.T) {
	driver := &failingHIPDriver{available: true, freeErr: core.NewError("free failed")}
	cache := NewBlockCacheService(BlockCacheConfig{CacheMode: rocmKVCacheModeQ8, deviceDriver: driver})
	_, err := cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.RequireNoError(t, err)
	native := &fakeNativeModel{}
	model := &rocmModel{
		native:    native,
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		cache:     cache,
	}

	err = resultError(model.Close())

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "free failed")
	core.AssertEqual(t, 0, native.closeCalls)
	if model.cache != cache || model.native != native {
		t.Fatalf("model cleared cache/native after cache close failure: cache=%p native=%p", model.cache, model.native)
	}
}

type failingCacheDiskWriter struct{}

func (failingCacheDiskWriter) PutBytes(context.Context, []byte, state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, core.NewError("disk write failed")
}
