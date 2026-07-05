// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the error-path dispatchers in the state surface.
// Per AX-11 — error formatting + miss dispatch fires on every cache miss
// during a session load. ChunkNotFound is the dominant hot path under
// memory pressure (eviction → re-read); ResolveRefBytes mismatches fire
// when a stale bundle ref lands against a fresher store. Coverage here
// makes the cost of "miss + format + return" data-driven.
//
// Run:    go test -bench='BenchmarkErrorPath' -benchmem -run='^$' ./state

package state

import (
	"context"
	"testing"
)

// Sinks defeat compiler DCE. Distinct names per state-package bench file.
var (
	errorPathSinkChunk Chunk
	errorPathSinkErr   error
	errorPathSinkText  string
	errorPathSinkBool  bool
)

// --- ChunkNotFound dispatch (miss path) ---
// InMemoryStore returns ChunkNotFoundError on missing id; the wrapper
// chain (Resolve → Get → ChunkNotFoundError) costs ~one alloc per miss.

func BenchmarkErrorPath_Resolve_Miss(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = Resolve(ctx, store, 9999)
	}
}

func BenchmarkErrorPath_ResolveBytes_Miss(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = ResolveBytes(ctx, store, 9999)
	}
}

func BenchmarkErrorPath_Get_Miss(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkText, errorPathSinkErr = store.Get(ctx, 9999)
	}
}

// --- ResolveRefBytes mismatch paths (stale-ref shape) ---
// ResolveRefBytes returns the ChunkNotFoundError when ChunkID == 0 and
// no RefBinaryResolver is present. Fires from cache-miss → seed-restore.

func BenchmarkErrorPath_ResolveRefBytes_NilStore(b *testing.B) {
	ctx := context.Background()
	ref := ChunkRef{ChunkID: 0, Codec: CodecMemory}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = ResolveRefBytes(ctx, nil, ref)
	}
}

func BenchmarkErrorPath_ResolveRefBytes_ZeroIDFallback(b *testing.B) {
	// benchGetOnlyStore implements only Store.Get — exercises the
	// non-RefBinaryResolver branch where ref.ChunkID == 0 returns the
	// formatter-flavoured miss.
	store := &benchGetOnlyStore{text: "x"}
	ctx := context.Background()
	ref := ChunkRef{ChunkID: 0}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = ResolveRefBytes(ctx, store, ref)
	}
}

func BenchmarkErrorPath_ResolveRefBytes_MissingID(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	ref := ChunkRef{ChunkID: 9999, Codec: CodecMemory, HasFrameOffset: true, FrameOffset: 9999}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = ResolveRefBytes(ctx, store, ref)
	}
}

// --- ResolveURI miss paths ---
// Empty URI, missing URI, and a URI against a no-URIResolver store.

func BenchmarkErrorPath_ResolveURI_NilStore(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = ResolveURI(ctx, nil, "state://missing")
	}
}

func BenchmarkErrorPath_ResolveURI_Whitespace(b *testing.B) {
	// core.Trim short-circuits the URIResolver path. Whitespace-only URIs
	// hit the empty-URI early-return without dispatching to the resolver.
	store := benchMemoryStore(b, 1, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = ResolveURI(ctx, store, "   ")
	}
}

func BenchmarkErrorPath_ResolveURI_NotFound(b *testing.B) {
	store := benchMemoryStore(b, 10, 1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = ResolveURI(ctx, store, "state://bench/missing")
	}
}

// --- Cancelled-context paths ---
// All Resolve/Put paths check ctx.Done before doing work. Cancelled
// contexts fire on session-shutdown drain — every in-flight resolve
// must early-return. The early-return path matters because seed restores
// can issue 100+ resolves in one shutdown sweep.

func BenchmarkErrorPath_Memory_Resolve_CancelledCtx(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = store.Resolve(ctx, 1)
	}
}

func BenchmarkErrorPath_Memory_ResolveBytes_CancelledCtx(b *testing.B) {
	store := benchMemoryStore(b, 1, 1024)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = store.ResolveBytes(ctx, 1)
	}
}

func BenchmarkErrorPath_Memory_Put_CancelledCtx(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	text := "x"
	opts := PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, errorPathSinkErr = store.Put(ctx, text, opts)
	}
}

// --- Nil-store path on all dispatchers ---
// Each top-level dispatcher (Resolve, ResolveBytes, ResolveRefBytes,
// ResolveURI) has a nil-store guard. These fire from a partial-init
// codepath where the consumer hasn't yet hydrated its Store handle.

func BenchmarkErrorPath_Resolve_NilStore(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = Resolve(ctx, nil, 7)
	}
}

func BenchmarkErrorPath_ResolveBytes_NilStore(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = ResolveBytes(ctx, nil, 7)
	}
}

// --- Nil-receiver path ---
// (*InMemoryStore)(nil).Resolve must early-return without panic so a
// partially-constructed Session can still drain. Confirms the receiver
// guard cost is bounded.

func BenchmarkErrorPath_Memory_NilReceiver_Resolve(b *testing.B) {
	var store *InMemoryStore
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = store.Resolve(ctx, 7)
	}
}

func BenchmarkErrorPath_Memory_NilReceiver_ResolveBytes(b *testing.B) {
	var store *InMemoryStore
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = store.ResolveBytes(ctx, 7)
	}
}

func BenchmarkErrorPath_Memory_NilReceiver_ResolveURI(b *testing.B) {
	var store *InMemoryStore
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkChunk, errorPathSinkErr = store.ResolveURI(ctx, "state://x")
	}
}

// --- Unwrap chain (errors.Is across the wrapper) ---
// Consumers walk the error chain via `core.Is(err, ErrChunkNotFound)`
// in every cache-miss branch. Confirms the cost of the Unwrap hop.

func BenchmarkErrorPath_ChunkNotFound_Unwrap(b *testing.B) {
	err := &ChunkNotFoundError{ID: 42}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkErr = err.Unwrap()
	}
}

func BenchmarkErrorPath_URIChunkNotFound_Unwrap(b *testing.B) {
	err := &URIChunkNotFoundError{URI: "state://x"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errorPathSinkErr = err.Unwrap()
	}
}
