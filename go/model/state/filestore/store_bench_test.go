// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the filestore state primitives.
// Per AX-11 — state.filestore is the persistence layer behind every
// session checkpoint, every memvid chunk read, every cross-process
// state handoff. Read/Resolve fires per chunk during a session load;
// Put fires per Save during a generation step.
//
// Run:    go test -bench='Benchmark' -benchmem -run='^$' ./state/filestore

package filestore

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/state"
)

// Sinks defeat compiler DCE.
var (
	bSinkChunk state.Chunk
	bSinkRef   state.ChunkRef
	bSinkErr   error
)

// benchStore opens a fresh filestore in a temp dir + populates n chunks
// of the requested size. Returns the store + the IDs in registration
// order so benches can target a known chunk.
func benchStore(tb testing.TB, n, payloadSize int) (*Store, []state.ChunkRef) {
	tb.Helper()
	dir := tb.TempDir()
	path := dir + "/state.bin"
	store, err := Create(context.Background(), path)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { _ = store.Close() })

	payload := make([]byte, payloadSize)
	for i := range payload {
		payload[i] = byte('a' + i%26)
	}
	refs := make([]state.ChunkRef, 0, n)
	for i := 0; i < n; i++ {
		ref, err := store.PutBytes(context.Background(), payload, state.PutOptions{
			Kind:  "bench",
			Title: core.Sprintf("chunk-%d", i),
		})
		if err != nil {
			tb.Fatal(err)
		}
		refs = append(refs, ref)
	}
	return store, refs
}

// --- ResolveBytes (binary read — hot for state load) ---

func BenchmarkFilestore_ResolveBytes_1KB(b *testing.B) {
	store, refs := benchStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bSinkChunk, bSinkErr = store.ResolveBytes(ctx, refs[0].ChunkID)
	}
}

func BenchmarkFilestore_ResolveBytes_64KB(b *testing.B) {
	store, refs := benchStore(b, 1, 64*1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bSinkChunk, bSinkErr = store.ResolveBytes(ctx, refs[0].ChunkID)
	}
}

func BenchmarkFilestore_ResolveBytes_1MB(b *testing.B) {
	store, refs := benchStore(b, 1, 1024*1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bSinkChunk, bSinkErr = store.ResolveBytes(ctx, refs[0].ChunkID)
	}
}

// --- Resolve (text read — exercises the AsString path) ---

func BenchmarkFilestore_Resolve_1KB(b *testing.B) {
	store, refs := benchStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bSinkChunk, bSinkErr = state.Resolve(ctx, store, refs[0].ChunkID)
	}
}

func BenchmarkFilestore_Resolve_64KB(b *testing.B) {
	store, refs := benchStore(b, 1, 64*1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bSinkChunk, bSinkErr = state.Resolve(ctx, store, refs[0].ChunkID)
	}
}

// --- ResolveRefBytes (ref-with-frame-offset — alternate read path) ---

func BenchmarkFilestore_ResolveRefBytes_1KB(b *testing.B) {
	store, refs := benchStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bSinkChunk, bSinkErr = store.ResolveRefBytes(ctx, refs[0])
	}
}

// --- Put (write path — fires per Save during generation) ---

func BenchmarkFilestore_PutBytes_1KB(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/state.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	payload := make([]byte, 1024)
	opts := state.PutOptions{Kind: "bench"}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bSinkRef, bSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

func BenchmarkFilestore_Put_Text_1KB(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/state.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	text := string(make([]byte, 1024))
	opts := state.PutOptions{Kind: "bench"}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bSinkRef, bSinkErr = store.Put(ctx, text, opts)
	}
}
