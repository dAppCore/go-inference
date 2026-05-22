// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"testing"
)

// fakeStore implements only the Store interface (no Resolver upgrade) so the
// top-level Resolve helper exercises its Get fallback path.
type fakeStore struct {
	text string
}

func (f fakeStore) Get(_ context.Context, _ int) (string, error) {
	return f.text, nil
}

func BenchmarkResolve_DirectResolver_Typical(b *testing.B) {
	store := NewInMemoryStore(map[int]string{1: "alpha"})
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len("alpha")))
	for i := 0; i < b.N; i++ {
		if _, err := Resolve(ctx, store, 1); err != nil {
			b.Fatalf("Resolve() error = %v", err)
		}
	}
}

func BenchmarkResolve_GetFallback_Typical(b *testing.B) {
	store := fakeStore{text: "alpha"}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len("alpha")))
	for i := 0; i < b.N; i++ {
		if _, err := Resolve(ctx, store, 1); err != nil {
			b.Fatalf("Resolve() error = %v", err)
		}
	}
}

func BenchmarkResolveBytes_DirectResolver_Typical(b *testing.B) {
	store := NewInMemoryStore(map[int]string{1: "alpha"})
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len("alpha")))
	for i := 0; i < b.N; i++ {
		if _, err := ResolveBytes(ctx, store, 1); err != nil {
			b.Fatalf("ResolveBytes() error = %v", err)
		}
	}
}

func BenchmarkResolveBytes_GetFallback_Typical(b *testing.B) {
	store := fakeStore{text: "alpha"}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len("alpha")))
	for i := 0; i < b.N; i++ {
		if _, err := ResolveBytes(ctx, store, 1); err != nil {
			b.Fatalf("ResolveBytes() error = %v", err)
		}
	}
}

func BenchmarkResolveRefBytes_DirectResolver_Typical(b *testing.B) {
	store := NewInMemoryStore(map[int]string{1: "alpha"})
	ctx := context.Background()
	ref := ChunkRef{ChunkID: 1, FrameOffset: 1, HasFrameOffset: true, Codec: CodecMemory}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := ResolveRefBytes(ctx, store, ref); err != nil {
			b.Fatalf("ResolveRefBytes() error = %v", err)
		}
	}
}

func BenchmarkResolveURI_DirectResolver_Typical(b *testing.B) {
	store := NewInMemoryStore(nil)
	if _, err := store.Put(context.Background(), "alpha", PutOptions{URI: "state://x/1"}); err != nil {
		b.Fatalf("Put() error = %v", err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := ResolveURI(ctx, store, "state://x/1"); err != nil {
			b.Fatalf("ResolveURI() error = %v", err)
		}
	}
}

func BenchmarkMergeRef_Typical(b *testing.B) {
	base := ChunkRef{ChunkID: 1, FrameOffset: 100, HasFrameOffset: true, Codec: CodecMemory}
	overlay := ChunkRef{Codec: "memvid/file-log", Segment: "/tmp/seg"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = MergeRef(base, overlay)
	}
}

func BenchmarkMergeRef_Empty(b *testing.B) {
	base := ChunkRef{ChunkID: 1, FrameOffset: 100, HasFrameOffset: true, Codec: CodecMemory}
	overlay := ChunkRef{}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = MergeRef(base, overlay)
	}
}
