// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the filestore ResolveRefBytes mismatch shapes.
// Per AX-11 — ResolveRefBytes is the "stale-ref" path: a bundle ref
// arrives with codec / segment / frame-offset metadata that may not
// match the live store. The mismatch branches need cheap rejection
// so the consumer can retry with the right backend. The 1KB happy path
// is already benched in store_bench_test.go — these cover the shapes
// it lacks.
//
// Run:    go test -bench='BenchmarkFilestoreRef' -benchmem -run='^$' ./state/filestore

package filestore

import (
	"context"
	"testing"

	state "dappco.re/go/inference/state"
)

// Sinks defeat compiler DCE.
var (
	frbSinkChunk state.Chunk
	frbSinkErr   error
)

// --- ResolveRefBytes without HasFrameOffset ---
// When HasFrameOffset is false, ResolveRefBytes falls through to
// ResolveBytes by ChunkID. Common shape for refs from non-file
// backends that don't carry a frame offset.

func BenchmarkFilestoreRef_NoFrameOffset_1KB(b *testing.B) {
	store, refs := benchStore(b, 1, 1024)
	ctx := context.Background()
	ref := state.ChunkRef{
		ChunkID:        refs[0].ChunkID,
		HasFrameOffset: false,
		// No Codec / Segment — exercises the bare ID-only path.
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frbSinkChunk, frbSinkErr = store.ResolveRefBytes(ctx, ref)
	}
}

// --- ResolveRefBytes with HasFrameOffset (the bench-light large size) ---

func BenchmarkFilestoreRef_WithFrameOffset_64KB(b *testing.B) {
	store, refs := benchStore(b, 1, 64*1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(64 * 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frbSinkChunk, frbSinkErr = store.ResolveRefBytes(ctx, refs[0])
	}
}

func BenchmarkFilestoreRef_WithFrameOffset_1MB(b *testing.B) {
	store, refs := benchStore(b, 1, 1024*1024)
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(1024 * 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frbSinkChunk, frbSinkErr = store.ResolveRefBytes(ctx, refs[0])
	}
}

// --- Codec mismatch ---
// A ref carrying state/qr-video must not resolve against a file-log
// store — the codec guard returns immediately. Hot path when a
// memvid bundle was migrated and the runtime probed the wrong store.

func BenchmarkFilestoreRef_CodecMismatch(b *testing.B) {
	store, refs := benchStore(b, 1, 1024)
	ctx := context.Background()
	ref := refs[0]
	ref.Codec = state.CodecStateVideo // not CodecFile / CodecMemvidFile
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frbSinkChunk, frbSinkErr = store.ResolveRefBytes(ctx, ref)
	}
}

// --- Segment mismatch ---
// Segment carries the file path. A ref with the wrong segment must
// be rejected without doing disk I/O.

func BenchmarkFilestoreRef_SegmentMismatch(b *testing.B) {
	store, refs := benchStore(b, 1, 1024)
	ctx := context.Background()
	ref := refs[0]
	ref.Segment = ref.Segment + ".other"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frbSinkChunk, frbSinkErr = store.ResolveRefBytes(ctx, ref)
	}
}

// --- ID mismatch on FrameOffset ---
// The ref's ChunkID disagrees with what the on-disk record claims.
// The mismatch is detected mid-read after the header parse — slightly
// more expensive than a pre-read codec/segment reject.

func BenchmarkFilestoreRef_IDMismatch(b *testing.B) {
	store, refs := benchStore(b, 2, 1024)
	ctx := context.Background()
	// Ref claims chunk 1 but points at frame-offset for chunk 2.
	ref := refs[0]
	ref.FrameOffset = refs[1].FrameOffset
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frbSinkChunk, frbSinkErr = store.ResolveRefBytes(ctx, ref)
	}
}

// --- Codec=MemvidFile (legacy header) ---
// CodecMemvidFile is the legacy codec name — the guard explicitly
// accepts both CodecFile and CodecMemvidFile. Benching the legacy
// path makes sure it stays as fast as the canonical one.

func BenchmarkFilestoreRef_CodecLegacyMemvid(b *testing.B) {
	store, refs := benchStore(b, 1, 1024)
	ctx := context.Background()
	ref := refs[0]
	ref.Codec = CodecMemvidFile
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frbSinkChunk, frbSinkErr = store.ResolveRefBytes(ctx, ref)
	}
}

// --- Codec empty (no codec constraint) ---
// A bare ref with no codec passes the guard (codec=="" is permissive).
// Common when refs are constructed from URI-only manifests.

func BenchmarkFilestoreRef_CodecEmpty(b *testing.B) {
	store, refs := benchStore(b, 1, 1024)
	ctx := context.Background()
	ref := refs[0]
	ref.Codec = ""
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frbSinkChunk, frbSinkErr = store.ResolveRefBytes(ctx, ref)
	}
}
