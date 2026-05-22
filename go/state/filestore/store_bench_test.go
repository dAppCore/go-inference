// SPDX-Licence-Identifier: EUPL-1.2

package filestore

import (
	"context"
	stdio "io"
	"strconv"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/state"
)

func newCreatedStore(b *testing.B) *Store {
	b.Helper()
	store, err := Create(context.Background(), core.PathJoin(b.TempDir(), "bench.mvlog"))
	if err != nil {
		b.Fatalf("Create() error = %v", err)
	}
	b.Cleanup(func() { _ = store.Close() })
	return store
}

func newPopulatedFileStore(b *testing.B, n, payloadSize int) (*Store, []state.ChunkRef, []string) {
	b.Helper()
	store := newCreatedStore(b)
	payload := make([]byte, payloadSize)
	for i := range payload {
		payload[i] = byte(i)
	}
	refs := make([]state.ChunkRef, n)
	uris := make([]string, n)
	for i := 0; i < n; i++ {
		uris[i] = "state://bench/" + strconv.Itoa(i)
		ref, err := store.PutBytes(context.Background(), payload, state.PutOptions{URI: uris[i]})
		if err != nil {
			b.Fatalf("PutBytes(seed %d) error = %v", i, err)
		}
		refs[i] = ref
	}
	return store, refs, uris
}

func BenchmarkFileStore_Create_Typical(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		path := core.PathJoin(b.TempDir(), "create-"+strconv.Itoa(i)+".mvlog")
		store, err := Create(context.Background(), path)
		if err != nil {
			b.Fatalf("Create() error = %v", err)
		}
		if err := store.Close(); err != nil {
			b.Fatalf("Close() error = %v", err)
		}
	}
}

func BenchmarkFileStore_Put_Typical(b *testing.B) {
	ctx := context.Background()
	store := newCreatedStore(b)
	payload := "alpha-bravo-charlie-delta"
	b.ReportAllocs()
	b.SetBytes(int64(len(payload)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Put(ctx, payload, state.PutOptions{URI: "state://put/" + strconv.Itoa(i)}); err != nil {
			b.Fatalf("Put() error = %v", err)
		}
	}
}

func BenchmarkFileStore_PutBytes_Typical(b *testing.B) {
	ctx := context.Background()
	store := newCreatedStore(b)
	payload := make([]byte, 1024)
	for i := range payload {
		payload[i] = byte(i)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(payload)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.PutBytes(ctx, payload, state.PutOptions{}); err != nil {
			b.Fatalf("PutBytes() error = %v", err)
		}
	}
}

func BenchmarkFileStore_PutBytes_Scale(b *testing.B) {
	ctx := context.Background()
	store := newCreatedStore(b)
	payload := make([]byte, 10*1024)
	for i := range payload {
		payload[i] = byte(i)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(payload)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.PutBytes(ctx, payload, state.PutOptions{}); err != nil {
			b.Fatalf("PutBytes() error = %v", err)
		}
	}
}

func BenchmarkFileStore_PutBytesStream_Typical(b *testing.B) {
	ctx := context.Background()
	store := newCreatedStore(b)
	chunkA := []byte("alpha-bravo-")
	chunkB := []byte("charlie-delta")
	payloadSize := len(chunkA) + len(chunkB)
	writer := func(w stdio.Writer) error {
		if _, err := w.Write(chunkA); err != nil {
			return err
		}
		_, err := w.Write(chunkB)
		return err
	}
	b.ReportAllocs()
	b.SetBytes(int64(payloadSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.PutBytesStream(ctx, payloadSize, state.PutOptions{}, writer); err != nil {
			b.Fatalf("PutBytesStream() error = %v", err)
		}
	}
}

func BenchmarkFileStore_Resolve_Typical(b *testing.B) {
	ctx := context.Background()
	store, refs, _ := newPopulatedFileStore(b, 64, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := refs[i%len(refs)].ChunkID
		if _, err := store.Resolve(ctx, id); err != nil {
			b.Fatalf("Resolve(%d) error = %v", id, err)
		}
	}
}

func BenchmarkFileStore_ResolveBytes_Typical(b *testing.B) {
	ctx := context.Background()
	store, refs, _ := newPopulatedFileStore(b, 64, 512)
	b.ReportAllocs()
	b.SetBytes(512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := refs[i%len(refs)].ChunkID
		if _, err := store.ResolveBytes(ctx, id); err != nil {
			b.Fatalf("ResolveBytes(%d) error = %v", id, err)
		}
	}
}

func BenchmarkFileStore_ResolveRefBytes_Typical(b *testing.B) {
	ctx := context.Background()
	store, refs, _ := newPopulatedFileStore(b, 64, 512)
	b.ReportAllocs()
	b.SetBytes(512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ref := refs[i%len(refs)]
		if _, err := store.ResolveRefBytes(ctx, ref); err != nil {
			b.Fatalf("ResolveRefBytes() error = %v", err)
		}
	}
}

func BenchmarkFileStore_ResolveURI_Typical(b *testing.B) {
	ctx := context.Background()
	store, _, uris := newPopulatedFileStore(b, 64, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		uri := uris[i%len(uris)]
		if _, err := store.ResolveURI(ctx, uri); err != nil {
			b.Fatalf("ResolveURI(%q) error = %v", uri, err)
		}
	}
}

func BenchmarkFileStore_ResolveBytes_Scale(b *testing.B) {
	ctx := context.Background()
	store, refs, _ := newPopulatedFileStore(b, 1024, 512)
	b.ReportAllocs()
	b.SetBytes(512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := refs[i%len(refs)].ChunkID
		if _, err := store.ResolveBytes(ctx, id); err != nil {
			b.Fatalf("ResolveBytes(%d) error = %v", id, err)
		}
	}
}

func BenchmarkFileStore_Open_Rebuild(b *testing.B) {
	ctx := context.Background()
	path := core.PathJoin(b.TempDir(), "rebuild.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		b.Fatalf("Create() error = %v", err)
	}
	payload := make([]byte, 256)
	for i := range payload {
		payload[i] = byte(i)
	}
	for i := 0; i < 64; i++ {
		if _, err := store.PutBytes(ctx, payload, state.PutOptions{URI: "state://rebuild/" + strconv.Itoa(i)}); err != nil {
			b.Fatalf("PutBytes(seed %d) error = %v", i, err)
		}
	}
	if err := store.Close(); err != nil {
		b.Fatalf("Close() error = %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reopened, err := Open(ctx, path)
		if err != nil {
			b.Fatalf("Open() error = %v", err)
		}
		if err := reopened.Close(); err != nil {
			b.Fatalf("Close() error = %v", err)
		}
	}
}

func BenchmarkEncodeRecordHeader_Typical(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = encodeRecordHeader(123, 1024, 32)
	}
}

func BenchmarkDecodeRecordHeader_Typical(b *testing.B) {
	header := encodeRecordHeader(123, 1024, 32)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := decodeRecordHeader(header); err != nil {
			b.Fatalf("decodeRecordHeader() error = %v", err)
		}
	}
}
