// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the top-level store dispatchers.
// Per AX-11 — Resolve / ResolveBytes / ResolveRefBytes / ResolveURI
// are the front-door API every consumer hits. They route to either
// the Store's native impl (filestore / memvid) or fall back to the
// minimal Store.Get adapter; both paths matter. MergeRef + the error
// formatters fire per chunk on the read-side hot loop.
//
// Run:    go test -bench='Benchmark' -benchmem -run='^$' ./state

package state

import (
	"context"
	"testing"
)

// Sinks defeat compiler DCE. Distinct names per state-package bench file.
var (
	storeSinkChunk    Chunk
	storeSinkRef      ChunkRef
	storeSinkErr      error
	storeSinkErrText  string
	storeSinkChunkRef ChunkRef
)

// --- Resolve (top-level dispatcher) ---
// Routes through the Resolver interface when available — InMemoryStore
// implements it, so this path is the "native dispatcher" cost.

func BenchmarkStore_Resolve_Native_1KB(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = Resolve(ctx, store, 1)
	}
}

// Adapter store implements only the bare Store.Get — exercises the
// fallback branch in Resolve that wraps Get into a Chunk.

func BenchmarkStore_Resolve_GetAdapter_1KB(b *testing.B) {
	store := &benchGetOnlyStore{text: string(make([]byte, 1024))}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = Resolve(ctx, store, 1)
	}
}

func BenchmarkStore_Resolve_NilStore(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = Resolve(ctx, nil, 1)
	}
}

// --- ResolveBytes (binary dispatcher) ---

func BenchmarkStore_ResolveBytes_Native_1KB(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = ResolveBytes(ctx, store, 1)
	}
}

func BenchmarkStore_ResolveBytes_Native_64KB(b *testing.B) {
	store := benchMemoryStore(b, 1, 64*1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = ResolveBytes(ctx, store, 1)
	}
}

// GetAdapter path — Store has no BinaryResolver, so ResolveBytes
// falls back through Resolve and copies Text → Data.

func BenchmarkStore_ResolveBytes_GetAdapter_1KB(b *testing.B) {
	store := &benchGetOnlyStore{text: string(make([]byte, 1024))}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = ResolveBytes(ctx, store, 1)
	}
}

// --- ResolveRefBytes (ChunkRef-with-frame-offset dispatcher) ---

func BenchmarkStore_ResolveRefBytes_Native_1KB(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	ref := ChunkRef{ChunkID: 1, FrameOffset: 1, HasFrameOffset: true, Codec: CodecMemory}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = ResolveRefBytes(ctx, store, ref)
	}
}

// Without RefBinaryResolver — falls back through ResolveBytes by ID.

func BenchmarkStore_ResolveRefBytes_GetAdapter_1KB(b *testing.B) {
	store := &benchGetOnlyStore{text: string(make([]byte, 1024))}
	ctx := context.Background()
	ref := ChunkRef{ChunkID: 1, FrameOffset: 1, HasFrameOffset: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = ResolveRefBytes(ctx, store, ref)
	}
}

// --- ResolveURI (top-level URI dispatcher) ---

func BenchmarkStore_ResolveURI_Native(b *testing.B) {
	store := benchMemoryStore(b, 10, 1024)
	ctx := context.Background()
	uri := "state://bench/chunk-1"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = ResolveURI(ctx, store, uri)
	}
}

func BenchmarkStore_ResolveURI_Empty(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = ResolveURI(ctx, store, "")
	}
}

func BenchmarkStore_ResolveURI_NoResolver(b *testing.B) {
	// benchGetOnlyStore doesn't implement URIResolver — exercises
	// the not-implemented branch that returns URIChunkNotFoundError.
	store := &benchGetOnlyStore{text: "x"}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunk, storeSinkErr = ResolveURI(ctx, store, "state://bench/missing")
	}
}

// --- MergeRef (per-chunk overlay merge) ---
// Fires whenever a fork or restore needs to overlay a manifest ref
// onto a base ref (segment changes between bundle versions).

func BenchmarkStore_MergeRef_OverlayAll(b *testing.B) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory}
	overlay := ChunkRef{
		ChunkID:        7,
		FrameOffset:    42,
		HasFrameOffset: true,
		Codec:          CodecStateVideo,
		Segment:        "epoch-3",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunkRef = MergeRef(base, overlay)
	}
}

func BenchmarkStore_MergeRef_OverlayPartial(b *testing.B) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory}
	overlay := ChunkRef{Codec: CodecStateVideo}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunkRef = MergeRef(base, overlay)
	}
}

func BenchmarkStore_MergeRef_OverlayEmpty(b *testing.B) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory}
	overlay := ChunkRef{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkChunkRef = MergeRef(base, overlay)
	}
}

// --- ChunkNotFoundError / URIChunkNotFoundError formatters ---
// Fire on every miss; the format path crosses through core.Sprintf.

func BenchmarkStore_ChunkNotFoundError_Error(b *testing.B) {
	err := &ChunkNotFoundError{ID: 42}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkErrText = err.Error()
	}
}

func BenchmarkStore_URIChunkNotFoundError_Error(b *testing.B) {
	err := &URIChunkNotFoundError{URI: "state://bench/missing"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkErrText = err.Error()
	}
}

func BenchmarkStore_URIChunkNotFoundError_ErrorEmpty(b *testing.B) {
	err := &URIChunkNotFoundError{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkErrText = err.Error()
	}
}

// --- ChunkRef value construction (the ID-only-shape) ---

func BenchmarkStore_ChunkRef_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storeSinkRef = ChunkRef{
			ChunkID:        7,
			FrameOffset:    42,
			HasFrameOffset: true,
			Codec:          CodecStateVideo,
			Segment:        "epoch-3",
		}
	}
}

// --- Bench helpers ---

// benchGetOnlyStore implements just the bare Store.Get contract so
// the bench can exercise the fallback dispatch path in Resolve /
// ResolveBytes / ResolveRefBytes when a backend only ships text reads.
type benchGetOnlyStore struct {
	text string
}

func (s *benchGetOnlyStore) Get(_ context.Context, _ int) (string, error) {
	return s.text, nil
}
