// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the error-path dispatchers in the filestore backend.
// Per AX-11 — filestore is the persistence layer behind every disk-backed
// state snapshot. Closed-store paths fire during shutdown drain, cancelled-
// context paths fire when a parent session aborts mid-restore, and
// missing-chunk paths fire when a stale ref points past the live index.
// Coverage here lets us see what the "miss + close + cancel" floor costs.
//
// Run:    go test -bench='BenchmarkFilestoreError' -benchmem -run='^$' ./state/filestore

package filestore

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// Sinks defeat compiler DCE. Distinct names per filestore bench file.
var (
	feSinkChunk state.Chunk
	feSinkRef   state.ChunkRef
	feSinkErr   error
)

// --- Missing-chunk path ---
// ResolveBytes / Resolve return the wrapped ChunkNotFoundError when an
// id is not in the index. Hot path under cache eviction.

func BenchmarkFilestoreError_ResolveBytes_Missing(b *testing.B) {
	store, _ := benchStore(b, 1, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkChunk, feSinkErr = store.ResolveBytes(ctx, 99999)
	}
}

func BenchmarkFilestoreError_Resolve_Missing(b *testing.B) {
	store, _ := benchStore(b, 1, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkChunk, feSinkErr = store.Resolve(ctx, 99999)
	}
}

func BenchmarkFilestoreError_ResolveURI_Missing(b *testing.B) {
	store, _ := benchStore(b, 1, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkChunk, feSinkErr = store.ResolveURI(ctx, "mlx://missing/chunk")
	}
}

// --- Closed-store paths ---
// After Close, every read/write must return a clean error. Fires on
// shutdown-drain when in-flight requests race the close.

func BenchmarkFilestoreError_ResolveBytes_Closed(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/closed.bin")
	if err != nil {
		b.Fatal(err)
	}
	if err := store.Close(); err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkChunk, feSinkErr = store.ResolveBytes(ctx, 1)
	}
}

func BenchmarkFilestoreError_Resolve_Closed(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/closed.bin")
	if err != nil {
		b.Fatal(err)
	}
	if err := store.Close(); err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkChunk, feSinkErr = store.Resolve(ctx, 1)
	}
}

func BenchmarkFilestoreError_PutBytes_Closed(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/closed.bin")
	if err != nil {
		b.Fatal(err)
	}
	if err := store.Close(); err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkRef, feSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

func BenchmarkFilestoreError_ResolveURI_Closed(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/closed.bin")
	if err != nil {
		b.Fatal(err)
	}
	if err := store.Close(); err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkChunk, feSinkErr = store.ResolveURI(ctx, "mlx://x")
	}
}

// --- Cancelled-context paths ---
// All filestore entry points run checkContext first. Cancelled contexts
// fire on session-shutdown drain — every in-flight resolve must early-
// return without doing disk I/O.

func BenchmarkFilestoreError_ResolveBytes_CancelledCtx(b *testing.B) {
	store, refs := benchStore(b, 1, 256)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	id := refs[0].ChunkID
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkChunk, feSinkErr = store.ResolveBytes(ctx, id)
	}
}

func BenchmarkFilestoreError_PutBytes_CancelledCtx(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/cancelled.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	payload := make([]byte, 64)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkRef, feSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

func BenchmarkFilestoreError_ResolveURI_CancelledCtx(b *testing.B) {
	store, _ := benchStore(b, 1, 256)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkChunk, feSinkErr = store.ResolveURI(ctx, "mlx://x")
	}
}

// --- Nil-store paths ---
// (*Store)(nil).PutBytes / ResolveBytes must early-return without a
// nil deref. Cheap guard, but the bench tracks the floor cost.

func BenchmarkFilestoreError_NilStore_ResolveBytes(b *testing.B) {
	var store *Store
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkChunk, feSinkErr = store.ResolveBytes(ctx, 1)
	}
}

func BenchmarkFilestoreError_NilStore_PutBytes(b *testing.B) {
	var store *Store
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkRef, feSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

func BenchmarkFilestoreError_NilStore_ResolveURI(b *testing.B) {
	var store *Store
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feSinkChunk, feSinkErr = store.ResolveURI(ctx, "mlx://x")
	}
}

// --- Open on missing file ---
// Open of a non-existent path should return a clean error from
// core.OpenFile. Fires during the first session-load probe before
// the on-disk store has been created.

func BenchmarkFilestoreError_Open_Missing(b *testing.B) {
	dir := b.TempDir()
	path := core.PathJoin(dir, "does-not-exist.bin")
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, feSinkErr = Open(ctx, path)
	}
}
