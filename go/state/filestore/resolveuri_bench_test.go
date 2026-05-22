// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the filestore ResolveURI variants.
// Per AX-11 — ResolveURI walks the in-memory uriIndex first, then does
// a Resolve by ChunkID. Misses are cheap; hits at scale matter because
// the uriIndex grows linearly with chunk count. The existing bench
// surface covers a typical hit on a fresh store — these cover the
// capacity + URI-shape variants.
//
// Run:    go test -bench='BenchmarkFilestoreURI' -benchmem -run='^$' ./state/filestore

package filestore

import (
	"context"
	"strconv"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// Sinks defeat compiler DCE.
var (
	furiSinkChunk state.Chunk
	furiSinkErr   error
)

// benchStoreWithURIs creates a filestore + populates n chunks of
// payloadSize each, every chunk carrying a unique URI in the form
// "mlx://bench/uri-<i>". Returns the store + the URI list.
func benchStoreWithURIs(tb testing.TB, n, payloadSize int) (*Store, []string) {
	tb.Helper()
	dir := tb.TempDir()
	path := dir + "/uri.bin"
	store, err := Create(context.Background(), path)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { _ = store.Close() })

	payload := make([]byte, payloadSize)
	for i := range payload {
		payload[i] = byte('a' + i%26)
	}
	uris := make([]string, 0, n)
	for i := 0; i < n; i++ {
		uri := "mlx://bench/uri-" + strconv.Itoa(i)
		_, err := store.PutBytes(context.Background(), payload, state.PutOptions{
			URI:  uri,
			Kind: "bench",
		})
		if err != nil {
			tb.Fatal(err)
		}
		uris = append(uris, uri)
	}
	return store, uris
}

// --- ResolveURI hit at various capacities ---

func BenchmarkFilestoreURI_Hit_10(b *testing.B) {
	store, uris := benchStoreWithURIs(b, 10, 256)
	ctx := context.Background()
	target := uris[5]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		furiSinkChunk, furiSinkErr = store.ResolveURI(ctx, target)
	}
}

func BenchmarkFilestoreURI_Hit_100(b *testing.B) {
	store, uris := benchStoreWithURIs(b, 100, 256)
	ctx := context.Background()
	target := uris[50]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		furiSinkChunk, furiSinkErr = store.ResolveURI(ctx, target)
	}
}

func BenchmarkFilestoreURI_Hit_1000(b *testing.B) {
	store, uris := benchStoreWithURIs(b, 1000, 256)
	ctx := context.Background()
	target := uris[500]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		furiSinkChunk, furiSinkErr = store.ResolveURI(ctx, target)
	}
}

// --- ResolveURI miss at various capacities ---
// Miss-path under load — the map probe returns immediately but the
// URIChunkNotFoundError allocates one wrapper.

func BenchmarkFilestoreURI_Miss_10(b *testing.B) {
	store, _ := benchStoreWithURIs(b, 10, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		furiSinkChunk, furiSinkErr = store.ResolveURI(ctx, "mlx://nope/zzz")
	}
}

func BenchmarkFilestoreURI_Miss_1000(b *testing.B) {
	store, _ := benchStoreWithURIs(b, 1000, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		furiSinkChunk, furiSinkErr = store.ResolveURI(ctx, "mlx://nope/zzz")
	}
}

// --- URI string-shape sensitivity ---
// Short URI vs long URI. The uriIndex is a map[string]int — hash cost
// scales with URI length on hit.

func BenchmarkFilestoreURI_Hit_LongURI(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/long.bin")
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = store.Close() })

	longURI := "mlx://lthn/projects/core/go-mlx/snapshots/2026-05-22T12:00:00Z/" +
		"runtime/metal/m3-ultra/model/qwen3-27b-4bit/adapter/lora-1/" +
		"workload/long-context/segment/chunk-00000042/epoch-3/layer/all"
	payload := make([]byte, 256)
	if _, err := store.PutBytes(context.Background(), payload, state.PutOptions{URI: longURI}); err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		furiSinkChunk, furiSinkErr = store.ResolveURI(ctx, longURI)
	}
}

// --- ResolveURI via top-level state dispatcher ---
// state.ResolveURI walks the type-assertion to URIResolver before
// dispatching — the per-call overhead matters on multi-store probes.

func BenchmarkFilestoreURI_TopLevelDispatcher_Hit(b *testing.B) {
	store, uris := benchStoreWithURIs(b, 100, 256)
	ctx := context.Background()
	target := uris[50]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		furiSinkChunk, furiSinkErr = state.ResolveURI(ctx, store, target)
	}
}

// --- ResolveURI after Reopen ---
// Open() rebuilds the uriIndex from the on-disk metadata. Hit-after-
// reopen tests that the index rebuild produces the same observable
// performance as a freshly populated store.

func BenchmarkFilestoreURI_HitAfterReopen(b *testing.B) {
	dir := b.TempDir()
	path := dir + "/reopen.bin"
	store, err := Create(context.Background(), path)
	if err != nil {
		b.Fatal(err)
	}
	payload := make([]byte, 256)
	uri := "mlx://bench/reopen-50"
	for i := 0; i < 100; i++ {
		thisURI := "mlx://bench/reopen-" + strconv.Itoa(i)
		if _, err := store.PutBytes(context.Background(), payload, state.PutOptions{
			URI:  thisURI,
			Kind: "bench",
		}); err != nil {
			b.Fatal(err)
		}
	}
	if err := store.Close(); err != nil {
		b.Fatal(err)
	}
	reopened, err := Open(context.Background(), path)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = reopened.Close() })
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		furiSinkChunk, furiSinkErr = reopened.ResolveURI(ctx, uri)
	}
}

// --- Open with a populated file (rebuildIndex cost) ---
// Open replays the on-disk record headers + metadata into the
// uriIndex. Cost is linear in the chunk count + metadata size.

func BenchmarkFilestoreURI_Open_100Chunks(b *testing.B) {
	dir := b.TempDir()
	path := dir + "/index.bin"
	{
		store, err := Create(context.Background(), path)
		if err != nil {
			b.Fatal(err)
		}
		payload := make([]byte, 64)
		for i := 0; i < 100; i++ {
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

func BenchmarkFilestoreURI_Open_1000Chunks(b *testing.B) {
	dir := b.TempDir()
	path := core.PathJoin(dir, "index-1000.bin")
	{
		store, err := Create(context.Background(), path)
		if err != nil {
			b.Fatal(err)
		}
		payload := make([]byte, 64)
		for i := 0; i < 1000; i++ {
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
