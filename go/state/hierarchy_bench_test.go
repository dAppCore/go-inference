// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the Store interface-dispatch hierarchy.
// Per AX-11 — Store is a layered interface (Store / Resolver / URIResolver /
// BinaryResolver / RefBinaryResolver / Writer / BinaryWriter /
// BinaryStreamWriter). The top-level dispatchers (Resolve, ResolveBytes,
// ResolveRefBytes, ResolveURI) probe each interface in turn. The Wake
// path for a project seed can issue dozens of dispatches per restore;
// the cost of an interface-probe miss compounds in that flow.
//
// Run:    go test -bench='BenchmarkHierarchy' -benchmem -run='^$' ./state

package state

import (
	"context"
	"testing"
)

// Sinks defeat compiler DCE. Distinct names per state-package bench file.
var (
	hierarchySinkChunk Chunk
	hierarchySinkErr   error
	hierarchySinkText  string
	hierarchySinkRef   ChunkRef
)

// --- Interface-probe miss paths ---
// When a Store implements ONLY Store.Get, the top-level dispatcher must
// type-assert against Resolver / BinaryResolver / RefBinaryResolver /
// URIResolver. Each miss costs a runtime probe. The fallback branch
// then synthesises a Chunk.

func BenchmarkHierarchy_GetAdapter_Resolve(b *testing.B) {
	// benchGetOnlyStore is the bare Store.Get adapter — Resolve walks
	// the Resolver-not-implemented branch and constructs a Chunk wrapper
	// around the returned text.
	store := &benchGetOnlyStore{text: string(make([]byte, 256))}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hierarchySinkChunk, hierarchySinkErr = Resolve(ctx, store, 1)
	}
}

func BenchmarkHierarchy_GetAdapter_ResolveBytes(b *testing.B) {
	store := &benchGetOnlyStore{text: string(make([]byte, 256))}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hierarchySinkChunk, hierarchySinkErr = ResolveBytes(ctx, store, 1)
	}
}

// --- Multi-resolver fallback chain ---
// hierarchyResolverShim implements Store + Resolver but NOT
// BinaryResolver. ResolveBytes therefore goes through the Resolve
// fallback that copies chunk.Text → chunk.Data. Common in dappcore
// wrappers that adapt a remote storage backend.

func BenchmarkHierarchy_ResolverOnly_ResolveBytes(b *testing.B) {
	store := &hierarchyResolverShim{
		ref: ChunkRef{ChunkID: 1, FrameOffset: 1, HasFrameOffset: true, Codec: CodecMemory},
		text: string(make([]byte, 1024)),
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hierarchySinkChunk, hierarchySinkErr = ResolveBytes(ctx, store, 1)
	}
}

func BenchmarkHierarchy_ResolverOnly_ResolveRefBytes(b *testing.B) {
	// ResolveRefBytes falls through to ResolveBytes → Resolve when the
	// Store implements neither RefBinaryResolver nor BinaryResolver.
	store := &hierarchyResolverShim{
		ref: ChunkRef{ChunkID: 1, FrameOffset: 1, HasFrameOffset: true, Codec: CodecMemory},
		text: string(make([]byte, 1024)),
	}
	ctx := context.Background()
	ref := ChunkRef{ChunkID: 1, FrameOffset: 1, HasFrameOffset: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hierarchySinkChunk, hierarchySinkErr = ResolveRefBytes(ctx, store, ref)
	}
}

// --- BinaryResolver path without RefBinaryResolver ---
// hierarchyBinaryShim implements Store + BinaryResolver. ResolveRefBytes
// must fall through to ResolveBytes (the BinaryResolver-without-Ref path).

func BenchmarkHierarchy_BinaryOnly_ResolveRefBytes(b *testing.B) {
	store := &hierarchyBinaryShim{
		ref:  ChunkRef{ChunkID: 1, FrameOffset: 1, HasFrameOffset: true, Codec: CodecMemory},
		data: make([]byte, 1024),
	}
	ctx := context.Background()
	ref := ChunkRef{ChunkID: 1, FrameOffset: 1, HasFrameOffset: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hierarchySinkChunk, hierarchySinkErr = ResolveRefBytes(ctx, store, ref)
	}
}

// --- MergeRef shape coverage ---
// MergeRef merges an overlay onto a base ref. The existing bench file
// covers OverlayAll / OverlayPartial / OverlayEmpty. These cover the
// less-typical permutations: same-base (no-op merge), zero-id base,
// codec-only overlay, segment-only overlay, frame-offset only overlay.

func BenchmarkHierarchy_MergeRef_SameBase(b *testing.B) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory, Segment: "seg-a"}
	overlay := base
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hierarchySinkRef = MergeRef(base, overlay)
	}
}

func BenchmarkHierarchy_MergeRef_ZeroBase(b *testing.B) {
	// Zero base — every field on overlay wins, but the no-id branch
	// short-circuits the merge.
	overlay := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hierarchySinkRef = MergeRef(ChunkRef{}, overlay)
	}
}

func BenchmarkHierarchy_MergeRef_CodecOnlyOverlay(b *testing.B) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory}
	overlay := ChunkRef{Codec: CodecStateVideo}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hierarchySinkRef = MergeRef(base, overlay)
	}
}

func BenchmarkHierarchy_MergeRef_SegmentOnlyOverlay(b *testing.B) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory}
	overlay := ChunkRef{Segment: "epoch-9"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hierarchySinkRef = MergeRef(base, overlay)
	}
}

func BenchmarkHierarchy_MergeRef_FrameOffsetOnlyOverlay(b *testing.B) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory}
	overlay := ChunkRef{FrameOffset: 99, HasFrameOffset: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hierarchySinkRef = MergeRef(base, overlay)
	}
}

// --- Shim helpers ---
// One file holds the shim defs to keep the bench surface flat.

// hierarchyResolverShim implements Store.Get + Resolver but not the
// binary interfaces. Forces ResolveBytes/ResolveRefBytes to dispatch
// through the Resolver fallback which copies Text → Data.
type hierarchyResolverShim struct {
	ref  ChunkRef
	text string
}

func (s *hierarchyResolverShim) Get(_ context.Context, _ int) (string, error) {
	return s.text, nil
}

func (s *hierarchyResolverShim) Resolve(_ context.Context, chunkID int) (Chunk, error) {
	ref := s.ref
	ref.ChunkID = chunkID
	return Chunk{Ref: ref, Text: s.text}, nil
}

// hierarchyBinaryShim implements Store.Get + BinaryResolver but not
// RefBinaryResolver. ResolveRefBytes must fall through ResolveBytes.
type hierarchyBinaryShim struct {
	ref  ChunkRef
	data []byte
}

func (s *hierarchyBinaryShim) Get(_ context.Context, _ int) (string, error) {
	return string(s.data), nil
}

func (s *hierarchyBinaryShim) ResolveBytes(_ context.Context, chunkID int) (Chunk, error) {
	ref := s.ref
	ref.ChunkID = chunkID
	return Chunk{Ref: ref, Data: append([]byte(nil), s.data...)}, nil
}
