// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the InMemoryStore backend.
// Per AX-11 — InMemoryStore is the test-and-bench default store and
// the cheapest target for cache-warm-up shapes. Get / Resolve fire
// per chunk on every session load; Put / PutBytes fire per Save.
// ResolveURI is the per-name lookup that backs the URIResolver path
// in the top-level state.ResolveURI helper.
//
// Run:    go test -bench='Benchmark' -benchmem -run='^$' ./state

package state

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE. Distinct names per state-package bench file.
var (
	memorySinkChunk    Chunk
	memorySinkText     string
	memorySinkRef      ChunkRef
	memorySinkErr      error
	memorySinkStorePtr *InMemoryStore
)

// benchMemoryStore builds an InMemoryStore with n text chunks of
// payloadSize bytes each + n URIs registered for ResolveURI lookups.
func benchMemoryStore(tb testing.TB, n, payloadSize int) *InMemoryStore {
	tb.Helper()
	chunks := make(map[int]string, n)
	payload := make([]byte, payloadSize)
	for i := range payload {
		payload[i] = byte('a' + i%26)
	}
	text := string(payload)
	for i := 1; i <= n; i++ {
		chunks[i] = text
	}
	store := NewInMemoryStore(chunks)
	// Register URIs after the fact via Put — keeps the bench helper
	// off the URI-pre-seeding path the test file exercises.
	for i := 1; i <= n; i++ {
		_, err := store.Put(context.Background(), text, PutOptions{
			URI: "state://bench/" + core.Sprintf("chunk-%d", i),
		})
		if err != nil {
			tb.Fatal(err)
		}
	}
	return store
}

// --- NewInMemoryStore (one per session boot) ---

func BenchmarkMemory_NewInMemoryStore_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkStorePtr = NewInMemoryStore(nil)
	}
}

func BenchmarkMemory_NewInMemoryStore_10(b *testing.B) {
	chunks := map[int]string{
		1: "a", 2: "b", 3: "c", 4: "d", 5: "e",
		6: "f", 7: "g", 8: "h", 9: "i", 10: "j",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkStorePtr = NewInMemoryStore(chunks)
	}
}

func BenchmarkMemory_NewInMemoryStore_100(b *testing.B) {
	chunks := make(map[int]string, 100)
	for i := 1; i <= 100; i++ {
		chunks[i] = "chunk"
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkStorePtr = NewInMemoryStore(chunks)
	}
}

func BenchmarkMemory_NewInMemoryStore_1000(b *testing.B) {
	chunks := make(map[int]string, 1000)
	for i := 1; i <= 1000; i++ {
		chunks[i] = "chunk"
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkStorePtr = NewInMemoryStore(chunks)
	}
}

func BenchmarkMemory_NewInMemoryStoreWithManifest_10(b *testing.B) {
	chunks := map[int]string{
		1: "a", 2: "b", 3: "c", 4: "d", 5: "e",
		6: "f", 7: "g", 8: "h", 9: "i", 10: "j",
	}
	refs := map[int]ChunkRef{
		1: {ChunkID: 1, Codec: CodecStateVideo, FrameOffset: 7, HasFrameOffset: true},
		2: {ChunkID: 2, Codec: CodecStateVideo, FrameOffset: 8, HasFrameOffset: true},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkStorePtr = NewInMemoryStoreWithManifest(chunks, refs)
	}
}

// --- Get (text read — Store interface, simplest path) ---

func BenchmarkMemory_Get_Short(b *testing.B) {
	store := benchMemoryStore(b, 1, 16)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkText, memorySinkErr = store.Get(ctx, 1)
	}
}

func BenchmarkMemory_Get_1KB(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkText, memorySinkErr = store.Get(ctx, 1)
	}
}

func BenchmarkMemory_Get_64KB(b *testing.B) {
	store := benchMemoryStore(b, 1, 64*1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkText, memorySinkErr = store.Get(ctx, 1)
	}
}

// --- Resolve (Chunk read — Resolver interface) ---

func BenchmarkMemory_Resolve_1KB(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkChunk, memorySinkErr = store.Resolve(ctx, 1)
	}
}

func BenchmarkMemory_Resolve_64KB(b *testing.B) {
	store := benchMemoryStore(b, 1, 64*1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkChunk, memorySinkErr = store.Resolve(ctx, 1)
	}
}

// --- ResolveBytes (binary read — BinaryResolver path) ---

func BenchmarkMemory_ResolveBytes_1KB(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkChunk, memorySinkErr = store.ResolveBytes(ctx, 1)
	}
}

func BenchmarkMemory_ResolveBytes_64KB(b *testing.B) {
	store := benchMemoryStore(b, 1, 64*1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkChunk, memorySinkErr = store.ResolveBytes(ctx, 1)
	}
}

// --- ResolveURI (name → ID lookup, then Resolve) ---

func BenchmarkMemory_ResolveURI_10Chunks(b *testing.B) {
	store := benchMemoryStore(b, 10, 1024)
	ctx := context.Background()
	uri := "state://bench/chunk-1"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkChunk, memorySinkErr = store.ResolveURI(ctx, uri)
	}
}

func BenchmarkMemory_ResolveURI_1000Chunks(b *testing.B) {
	store := benchMemoryStore(b, 1000, 1024)
	ctx := context.Background()
	uri := "state://bench/chunk-1"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkChunk, memorySinkErr = store.ResolveURI(ctx, uri)
	}
}

// --- Put (text write — fires per text Save) ---

func BenchmarkMemory_Put_1KB(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	text := string(make([]byte, 1024))
	opts := PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkRef, memorySinkErr = store.Put(ctx, text, opts)
	}
}

func BenchmarkMemory_Put_64KB(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	text := string(make([]byte, 64*1024))
	opts := PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkRef, memorySinkErr = store.Put(ctx, text, opts)
	}
}

func BenchmarkMemory_Put_WithURI(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	text := string(make([]byte, 1024))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkRef, memorySinkErr = store.Put(ctx, text, PutOptions{
			Kind: "bench",
			URI:  "state://bench/put",
		})
	}
}

// --- PutBytes (binary write — fires per binary Save) ---

func BenchmarkMemory_PutBytes_1KB(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	data := make([]byte, 1024)
	opts := PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkRef, memorySinkErr = store.PutBytes(ctx, data, opts)
	}
}

func BenchmarkMemory_PutBytes_64KB(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	data := make([]byte, 64*1024)
	opts := PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkRef, memorySinkErr = store.PutBytes(ctx, data, opts)
	}
}

func BenchmarkMemory_PutBytes_1MB(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	data := make([]byte, 1024*1024)
	opts := PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memorySinkRef, memorySinkErr = store.PutBytes(ctx, data, opts)
	}
}
