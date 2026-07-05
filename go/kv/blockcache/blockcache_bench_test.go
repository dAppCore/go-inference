// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the block-prefix cache metadata layer.
// Per AX-11 — WarmCache fires per prompt (block-chunked), CacheEntries
// fires per dashboard/status query, the in-memory lookup + hashed
// identity (HashModelParts, blockCacheID) is the inner loop both warm
// and stat paths hit. Memory-only (no disk, no state store) baseline
// covers the hot path; helper sweeps catch per-call overhead under
// big block populations.
//
// Run:    go test -bench='BenchmarkBlockCache|BenchmarkBlockRefMatch|BenchmarkSortCacheBlockRefs|BenchmarkHashModelParts' -benchmem -run='^$' ./go/blockcache

package blockcache

import (
	"context"
	"maps"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	benchSinkWarm    inference.CacheWarmResult
	benchSinkStats   inference.CacheStats
	benchSinkEntries []inference.CacheBlockRef
	benchSinkRef     inference.CacheBlockRef
	benchSinkRefs    []inference.CacheBlockRef
	benchSinkErr     error
	benchSinkString  string
	benchSinkBool    bool
	benchSinkLabels  map[string]string
)

// benchTokens builds a deterministic token slice the warm path can
// chunk into block-sized prefixes. 512 → 1 block at default size,
// 2048 → 4 blocks. Sized to mirror the prompt-class workload the
// block cache fronts on real generation.
func benchTokens(n int) []int32 {
	tokens := make([]int32, n)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	return tokens
}

// benchService constructs a memory-only service with identity hashes
// resolved up-front so block ID computation is deterministic per call.
func benchService(blockSize int) *Service {
	return New(Config{
		BlockSize:     blockSize,
		ModelHash:     "sha256:bench-model",
		AdapterHash:   "sha256:bench-adapter",
		TokenizerHash: "sha256:bench-tokenizer",
	})
}

// --- WarmCache hot path (miss → block insert) ---

func BenchmarkBlockCache_WarmCache_Miss_512Tokens(b *testing.B) {
	tokens := benchTokens(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		service := benchService(DefaultBlockSize)
		b.StartTimer()
		benchSinkWarm, benchSinkErr = service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
	}
}

func BenchmarkBlockCache_WarmCache_Miss_2048Tokens(b *testing.B) {
	tokens := benchTokens(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		service := benchService(DefaultBlockSize)
		b.StartTimer()
		benchSinkWarm, benchSinkErr = service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
	}
}

// --- WarmCache hot path (all hit — every block already present) ---

func BenchmarkBlockCache_WarmCache_AllHit_2048Tokens(b *testing.B) {
	service := benchService(DefaultBlockSize)
	tokens := benchTokens(2048)
	// Prime the cache once so every subsequent warm is pure hit.
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkWarm, benchSinkErr = service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens})
	}
}

// --- CacheStats — fires per dashboard query, scans all blocks ---

func BenchmarkBlockCache_CacheStats_100Blocks(b *testing.B) {
	service := benchService(128)
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: benchTokens(100 * 128)}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkStats, benchSinkErr = service.CacheStats(context.Background())
	}
}

func BenchmarkBlockCache_CacheStats_1000Blocks(b *testing.B) {
	service := benchService(16)
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: benchTokens(1000 * 16)}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkStats, benchSinkErr = service.CacheStats(context.Background())
	}
}

// --- CacheEntries — fires per UI/list query; sorts + clones every block ---

func BenchmarkBlockCache_CacheEntries_Unfiltered_100Blocks(b *testing.B) {
	service := benchService(128)
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: benchTokens(100 * 128)}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkEntries, benchSinkErr = service.CacheEntries(context.Background(), nil)
	}
}

func BenchmarkBlockCache_CacheEntries_FilteredByLabel_100Blocks(b *testing.B) {
	service := benchService(128)
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Tokens: benchTokens(100 * 128),
		Labels: map[string]string{"tenant": "alpha"},
	}); err != nil {
		b.Fatal(err)
	}
	filter := map[string]string{"tenant": "alpha"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkEntries, benchSinkErr = service.CacheEntries(context.Background(), filter)
	}
}

// --- HashModelParts — fires per cache adapter setup; SHA256 + JSON marshal ---

func BenchmarkHashModelParts_Short(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = HashModelParts("qwen3", 151936)
	}
}

func BenchmarkHashModelParts_TypicalParts(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = HashModelParts("qwen3", 151936, 28, 2048, "fp16", "sha256:tokenizer-abcdef")
	}
}

// --- blockCacheID — internal hashing per block; fires per WarmCache block ---

func BenchmarkBlockCacheID_512TokenPrefix(b *testing.B) {
	tokens := benchTokens(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = blockCacheID("sha256:model", "sha256:adapter", "sha256:tokenizer", mode, tokens)
	}
}

func BenchmarkBlockCacheID_2048TokenPrefix(b *testing.B) {
	tokens := benchTokens(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = blockCacheID("sha256:model", "sha256:adapter", "sha256:tokenizer", mode, tokens)
	}
}

// --- blockRefMatchesLabels — fires per ref during filtered CacheEntries / ClearCache ---

func BenchmarkBlockRefMatch_AllMatch(b *testing.B) {
	ref := inference.CacheBlockRef{
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
		Labels: map[string]string{
			"tenant":      "alpha",
			"block_index": "3",
		},
	}
	filter := map[string]string{
		"model_hash":   "sha256:model",
		"adapter_hash": "sha256:adapter",
		"tenant":       "alpha",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBool = blockRefMatchesLabels(ref, filter)
	}
}

func BenchmarkBlockRefMatch_FirstKeyMiss(b *testing.B) {
	ref := inference.CacheBlockRef{
		ModelHash: "sha256:model-a",
		Labels:    map[string]string{"tenant": "alpha"},
	}
	filter := map[string]string{
		"model_hash": "sha256:model-b",
		"tenant":     "alpha",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBool = blockRefMatchesLabels(ref, filter)
	}
}

// --- sortCacheBlockRefs — fires per CacheEntries; insertion sort over N refs ---

func makeBenchRefs(n int) []inference.CacheBlockRef {
	out := make([]inference.CacheBlockRef, n)
	for i := range out {
		// Reverse order to maximise sort work.
		out[i] = inference.CacheBlockRef{
			ID:         "block-" + core.Itoa(n-i),
			TokenStart: n - i,
		}
	}
	return out
}

func BenchmarkSortCacheBlockRefs_16(b *testing.B) {
	template := makeBenchRefs(16)
	work := make([]inference.CacheBlockRef, len(template))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, template)
		sortCacheBlockRefs(work)
	}
}

func BenchmarkSortCacheBlockRefs_256(b *testing.B) {
	template := makeBenchRefs(256)
	work := make([]inference.CacheBlockRef, len(template))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, template)
		sortCacheBlockRefs(work)
	}
}

// --- cloneBlockCacheLabels / cloneCacheBlockRef ---

func BenchmarkCloneBlockCacheLabels_Typical(b *testing.B) {
	labels := map[string]string{
		"tenant":      "alpha",
		"block_index": "3",
		"cache_mode":  mode,
		"block_size":  "512",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkLabels = cloneBlockCacheLabels(labels)
	}
}

func BenchmarkCloneCacheBlockRef_Typical(b *testing.B) {
	ref := inference.CacheBlockRef{
		ID:            "block-abc",
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
		Encoding:      "token-prefix/int32",
		TokenStart:    0,
		TokenCount:    512,
		SizeBytes:     2048,
		Labels: map[string]string{
			"tenant":     "alpha",
			"cache_mode": mode,
			"block_size": "512",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkRef = cloneCacheBlockRef(ref)
	}
}

// --- firstNonEmptyString — fires per blockRefs identity resolution ---

func BenchmarkFirstNonEmptyString_FirstHit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = firstNonEmptyString("sha256:model", "", "")
	}
}

func BenchmarkFirstNonEmptyString_LastHit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = firstNonEmptyString("", "  ", "sha256:model")
	}
}

// --- ClearCache — fires on cache reset; includes cheap in-memory refill ---

func BenchmarkBlockCache_ClearCache_100Blocks(b *testing.B) {
	tokens := benchTokens(100 * 128)
	template := benchService(128)
	if _, err := template.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: tokens}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		service := benchService(128)
		service.blocks = cloneBenchBlockRefs(template.blocks)
		service.misses = uint64(len(service.blocks))
		benchSinkStats, benchSinkErr = service.ClearCache(context.Background(), nil)
	}
}

func cloneBenchBlockRefs(src map[string]inference.CacheBlockRef) map[string]inference.CacheBlockRef {
	if len(src) == 0 {
		return map[string]inference.CacheBlockRef{}
	}
	dst := make(map[string]inference.CacheBlockRef, len(src))
	maps.Copy(dst, src)
	return dst
}
