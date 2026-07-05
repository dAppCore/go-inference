// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for InMemoryStore at larger capacities.
// Per AX-11 — the existing memory bench file covers single-chunk and
// 10/100/1000-entry constructors, plus a 1000-chunk ResolveURI. This
// file extends to the eviction-pressure shapes that matter for the
// Virgil portable-memory thesis: continuous workspaces accumulate
// thousands of chunks before any rollover. Random + sequential read
// patterns expose the map-hash + slice-append cost at scale.
//
// Run:    go test -bench='BenchmarkMemoryCapacity' -benchmem -run='^$' ./state

package state

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE. Distinct names per state-package bench file.
var (
	memCapSinkChunk    Chunk
	memCapSinkText     string
	memCapSinkRef      ChunkRef
	memCapSinkErr      error
	memCapSinkStorePtr *InMemoryStore
)

// memoryStoreNoURI populates n chunks WITHOUT URIs — avoids the
// per-chunk Put loop that would otherwise dominate setup time. URI
// presence is benched separately above; this file targets the bare
// map-driven read path.
func memoryStoreNoURI(tb testing.TB, n, payloadSize int) *InMemoryStore {
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
	return NewInMemoryStore(chunks)
}

// --- Resolve at scale (sequential access) ---
// Walks IDs in registration order — the dominant pattern for a
// session-wake bundle replay (chunk-1, chunk-2, ..., chunk-N).

func BenchmarkMemoryCapacity_Resolve_1k_Seq(b *testing.B) {
	store := memoryStoreNoURI(b, 1000, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := (i % 1000) + 1
		memCapSinkChunk, memCapSinkErr = store.Resolve(ctx, id)
	}
}

func BenchmarkMemoryCapacity_Resolve_10k_Seq(b *testing.B) {
	store := memoryStoreNoURI(b, 10000, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := (i % 10000) + 1
		memCapSinkChunk, memCapSinkErr = store.Resolve(ctx, id)
	}
}

// --- Get at scale ---
// Get is the bare Store.Get contract — the cheapest dispatch.

func BenchmarkMemoryCapacity_Get_1k(b *testing.B) {
	store := memoryStoreNoURI(b, 1000, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := (i % 1000) + 1
		memCapSinkText, memCapSinkErr = store.Get(ctx, id)
	}
}

func BenchmarkMemoryCapacity_Get_10k(b *testing.B) {
	store := memoryStoreNoURI(b, 10000, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := (i % 10000) + 1
		memCapSinkText, memCapSinkErr = store.Get(ctx, id)
	}
}

// --- ResolveBytes at scale (binary-read path) ---

func BenchmarkMemoryCapacity_ResolveBytes_1k(b *testing.B) {
	store := memoryStoreNoURI(b, 1000, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := (i % 1000) + 1
		memCapSinkChunk, memCapSinkErr = store.ResolveBytes(ctx, id)
	}
}

// --- Put growth (repeated insert into existing store) ---
// Models a Save loop on a live, already-warm store. The per-Put cost
// should be dominated by the map-write + ref construction; growing
// past the initial capacity exercises map-grow.

func BenchmarkMemoryCapacity_Put_Repeated_1k(b *testing.B) {
	store := memoryStoreNoURI(b, 1000, 256)
	ctx := context.Background()
	text := "growth"
	opts := PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memCapSinkRef, memCapSinkErr = store.Put(ctx, text, opts)
	}
}

// --- ResolveURI at scale (URI table lookup) ---
// 10k URIs in the lookup table. The existing 1000 bench shows the
// hot path; 10k tests the constant cost claim against larger maps.

func BenchmarkMemoryCapacity_ResolveURI_10k(b *testing.B) {
	store := memoryStoreNoURI(b, 10000, 256)
	ctx := context.Background()
	// Stage URIs via Put so the uri index is populated. Doing this in
	// the helper would slow every other bench in this file.
	for i := 1; i <= 10000; i++ {
		_, err := store.Put(ctx, "x", PutOptions{
			URI: "state://bench/cap-" + core.Sprintf("%d", i),
		})
		if err != nil {
			b.Fatal(err)
		}
	}
	uri := "state://bench/cap-5000"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memCapSinkChunk, memCapSinkErr = store.ResolveURI(ctx, uri)
	}
}

// --- NewInMemoryStore at very large size ---
// One-pass construction over 10k chunks — the seed-load cost for a
// large project bundle.

func BenchmarkMemoryCapacity_NewInMemoryStore_10000(b *testing.B) {
	chunks := make(map[int]string, 10000)
	for i := 1; i <= 10000; i++ {
		chunks[i] = "chunk"
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memCapSinkStorePtr = NewInMemoryStore(chunks)
	}
}
