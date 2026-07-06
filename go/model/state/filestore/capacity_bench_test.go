// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the filestore at larger record counts.
// Per AX-11 — filestore's in-memory index grows linearly with the
// record count. Read paths probe the map directly; reopen replays
// the on-disk records into a fresh index. At 1k+ records the cost
// of index lookups becomes observable, and the reopen path is one
// of the slowest entry points in the cold-start sequence.
//
// Run:    go test -bench='BenchmarkFilestoreCapacity' -benchmem -run='^$' ./state/filestore

package filestore

import (
	"context"
	"strconv"
	"testing"

	state "dappco.re/go/inference/model/state"
)

// Sinks defeat compiler DCE.
var (
	fcSinkChunk state.Chunk
	fcSinkRef   state.ChunkRef
	fcSinkErr   error
)

// --- ResolveBytes at scale ---
// The store_bench_test.go file covers single-record stores. These
// cover 1k+ records — the index map probe should stay constant
// but the bench tracks regressions.

func BenchmarkFilestoreCapacity_ResolveBytes_1000Records(b *testing.B) {
	store, refs := benchStore(b, 1000, 64)
	ctx := context.Background()
	// Read the middle record so the bench isn't penalised by hash
	// ordering on the first/last id.
	id := refs[500].ChunkID
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fcSinkChunk, fcSinkErr = store.ResolveBytes(ctx, id)
	}
}

func BenchmarkFilestoreCapacity_ResolveBytes_10000Records(b *testing.B) {
	store, refs := benchStore(b, 10000, 64)
	ctx := context.Background()
	id := refs[5000].ChunkID
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fcSinkChunk, fcSinkErr = store.ResolveBytes(ctx, id)
	}
}

// --- Resolve (text path) at scale ---

func BenchmarkFilestoreCapacity_Resolve_1000Records(b *testing.B) {
	store, refs := benchStore(b, 1000, 64)
	ctx := context.Background()
	id := refs[500].ChunkID
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fcSinkChunk, fcSinkErr = store.Resolve(ctx, id)
	}
}

// --- ResolveRefBytes at scale (frame-offset path) ---

func BenchmarkFilestoreCapacity_ResolveRefBytes_1000Records(b *testing.B) {
	store, refs := benchStore(b, 1000, 64)
	ctx := context.Background()
	target := refs[500]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fcSinkChunk, fcSinkErr = store.ResolveRefBytes(ctx, target)
	}
}

// --- PutBytes into a warm store ---
// 1000-record store + one more Put. Tracks the per-Put cost when the
// index is not empty.

func BenchmarkFilestoreCapacity_PutBytes_Warm_1000(b *testing.B) {
	store, _ := benchStore(b, 1000, 64)
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fcSinkRef, fcSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

// --- ChunkCount on a large index ---

func BenchmarkFilestoreCapacity_ChunkCount_1000(b *testing.B) {
	store, _ := benchStore(b, 1000, 64)
	var sink int
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = store.ChunkCount()
	}
	_ = sink
}

// --- Reopen + index-rebuild at large scale ---
// Cold-start cost. The 100/1000-chunk variants live in resolveuri_bench_test.go
// (because the URI index is part of rebuildIndex); this adds the 10k variant.

func BenchmarkFilestoreCapacity_Open_10000Records(b *testing.B) {
	dir := b.TempDir()
	path := dir + "/index-10000.bin"
	{
		store, err := Create(context.Background(), path)
		if err != nil {
			b.Fatal(err)
		}
		payload := make([]byte, 64)
		for i := range 10000 {
			if _, err := store.PutBytes(context.Background(), payload, state.PutOptions{
				URI:  "mlx://bench/open-" + strconv.Itoa(i),
				Kind: "bench",
			}); err != nil {
				b.Fatal(err)
			}
		}
		_ = store.Close()
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := Open(ctx, path)
		if err != nil {
			b.Fatal(err)
		}
		_ = s.Close()
	}
}

func BenchmarkFilestoreCapacity_Open_SingleLargePayload(b *testing.B) {
	dir := b.TempDir()
	path := dir + "/single-large.bin"
	{
		store, err := Create(context.Background(), path)
		if err != nil {
			b.Fatal(err)
		}
		payload := make([]byte, indexHintMaxFileBytes+1)
		if _, err := store.PutBytes(context.Background(), payload, state.PutOptions{
			URI:  "mlx://bench/open-large",
			Kind: "kv",
		}); err != nil {
			b.Fatal(err)
		}
		_ = store.Close()
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := Open(ctx, path)
		if err != nil {
			b.Fatal(err)
		}
		_ = s.Close()
	}
}

// --- Open without URIs (no uriIndex population) ---
// Faster path because the URI map stays empty. Confirms the URI map
// writes dominate the rebuildIndex cost.

func BenchmarkFilestoreCapacity_Open_NoURIs_1000(b *testing.B) {
	dir := b.TempDir()
	path := dir + "/noupd.bin"
	{
		store, err := Create(context.Background(), path)
		if err != nil {
			b.Fatal(err)
		}
		payload := make([]byte, 64)
		opts := state.PutOptions{Kind: "bench"}
		for range 1000 {
			if _, err := store.PutBytes(context.Background(), payload, opts); err != nil {
				b.Fatal(err)
			}
		}
		_ = store.Close()
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := Open(ctx, path)
		if err != nil {
			b.Fatal(err)
		}
		_ = s.Close()
	}
}
