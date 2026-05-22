// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"strconv"
	"testing"
)

func newPopulatedMemoryStore(b *testing.B, n int) *InMemoryStore {
	b.Helper()
	store := NewInMemoryStore(nil)
	for i := 0; i < n; i++ {
		uri := "state://bench/" + strconv.Itoa(i)
		if _, err := store.Put(context.Background(), "payload-"+strconv.Itoa(i), PutOptions{URI: uri}); err != nil {
			b.Fatalf("Put(seed %d) error = %v", i, err)
		}
	}
	return store
}

func BenchmarkInMemoryStore_Put_Typical(b *testing.B) {
	ctx := context.Background()
	store := NewInMemoryStore(nil)
	opts := PutOptions{URI: "state://put/typical"}
	payload := "abcdefghijklmnop"
	b.ReportAllocs()
	b.SetBytes(int64(len(payload)))
	for i := 0; i < b.N; i++ {
		opts.URI = "state://put/" + strconv.Itoa(i)
		if _, err := store.Put(ctx, payload, opts); err != nil {
			b.Fatalf("Put() error = %v", err)
		}
	}
}

func BenchmarkInMemoryStore_PutBytes_Typical(b *testing.B) {
	ctx := context.Background()
	store := NewInMemoryStore(nil)
	payload := make([]byte, 1024)
	for i := range payload {
		payload[i] = byte(i)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(payload)))
	for i := 0; i < b.N; i++ {
		if _, err := store.PutBytes(ctx, payload, PutOptions{}); err != nil {
			b.Fatalf("PutBytes() error = %v", err)
		}
	}
}

func BenchmarkInMemoryStore_PutBytes_Scale(b *testing.B) {
	ctx := context.Background()
	store := NewInMemoryStore(nil)
	payload := make([]byte, 10*1024)
	for i := range payload {
		payload[i] = byte(i)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(payload)))
	for i := 0; i < b.N; i++ {
		if _, err := store.PutBytes(ctx, payload, PutOptions{}); err != nil {
			b.Fatalf("PutBytes() error = %v", err)
		}
	}
}

func BenchmarkInMemoryStore_Get_Typical(b *testing.B) {
	ctx := context.Background()
	store := newPopulatedMemoryStore(b, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := (i % 64) + 1
		if _, err := store.Get(ctx, id); err != nil {
			b.Fatalf("Get(%d) error = %v", id, err)
		}
	}
}

func BenchmarkInMemoryStore_Resolve_Typical(b *testing.B) {
	ctx := context.Background()
	store := newPopulatedMemoryStore(b, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := (i % 64) + 1
		if _, err := store.Resolve(ctx, id); err != nil {
			b.Fatalf("Resolve(%d) error = %v", id, err)
		}
	}
}

func BenchmarkInMemoryStore_ResolveBytes_Typical(b *testing.B) {
	ctx := context.Background()
	store := newPopulatedMemoryStore(b, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := (i % 64) + 1
		if _, err := store.ResolveBytes(ctx, id); err != nil {
			b.Fatalf("ResolveBytes(%d) error = %v", id, err)
		}
	}
}

func BenchmarkInMemoryStore_ResolveURI_Typical(b *testing.B) {
	ctx := context.Background()
	store := newPopulatedMemoryStore(b, 64)
	uris := make([]string, 64)
	for i := range uris {
		uris[i] = "state://bench/" + strconv.Itoa(i)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		uri := uris[i%len(uris)]
		if _, err := store.ResolveURI(ctx, uri); err != nil {
			b.Fatalf("ResolveURI(%q) error = %v", uri, err)
		}
	}
}

func BenchmarkInMemoryStore_ResolveBytes_Scale(b *testing.B) {
	ctx := context.Background()
	store := newPopulatedMemoryStore(b, 1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := (i % 1024) + 1
		if _, err := store.ResolveBytes(ctx, id); err != nil {
			b.Fatalf("ResolveBytes(%d) error = %v", id, err)
		}
	}
}

func BenchmarkInMemoryStore_Resolve_Missing(b *testing.B) {
	ctx := context.Background()
	store := newPopulatedMemoryStore(b, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Resolve(ctx, 1<<20+i); err == nil {
			b.Fatal("Resolve(missing) error = nil")
		}
	}
}

func BenchmarkNewInMemoryStore_FromMap(b *testing.B) {
	seed := map[int]string{}
	for i := 1; i <= 64; i++ {
		seed[i] = "chunk-" + strconv.Itoa(i)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = NewInMemoryStore(seed)
	}
}
