// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/state"
)

func ExampleBlockCacheService_WarmCache() {
	cache := NewBlockCacheService(BlockCacheConfig{CacheMode: "q8"})
	result, _ := cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.Println(result.Stats.Blocks)
	// Output: 1
}

func ExampleBlockCacheService_WarmCache_diskRefs() {
	store := state.NewInMemoryStore(nil)
	cache := NewBlockCacheService(BlockCacheConfig{CacheMode: "q8", DiskStore: store})
	result, _ := cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	core.Println(result.Blocks[0].Labels["disk_codec"])
	core.Println(result.Stats.DiskBytes > 0)
	// Output:
	// memory/plaintext
	// true
}

func ExampleBlockCacheService_CacheStats() {
	cache := NewBlockCacheService(BlockCacheConfig{CacheMode: "q8"})
	_, _ = cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	_, _ = cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	stats, _ := cache.CacheStats(context.Background())
	core.Println(stats.Blocks)
	core.Println(stats.Hits)
	core.Println(stats.Misses)
	// Output:
	// 1
	// 1
	// 1
}

func ExampleBlockCacheService_ClearCache() {
	cache := NewBlockCacheService(BlockCacheConfig{CacheMode: "q8"})
	_, _ = cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2}, Labels: map[string]string{"tenant": "a"}})
	_, _ = cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{3, 4}, Labels: map[string]string{"tenant": "b"}})
	stats, _ := cache.ClearCache(context.Background(), map[string]string{"tenant": "a"})
	core.Println(stats.Blocks)
	core.Println(stats.Evictions)
	// Output:
	// 1
	// 1
}

func ExampleBlockCacheService_Close() {
	cache := NewBlockCacheService(BlockCacheConfig{CacheMode: "q8"})
	_, _ = cache.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	_ = cache.Close()
	stats, _ := cache.CacheStats(context.Background())
	core.Println(stats.Blocks)
	// Output: 0
}
