// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for embedded State regions inside a larger container.
// Per AX-11 - .kv wake now opens the State log by payload offset instead of
// materialising a temporary file, so the extra offset arithmetic must remain
// visible in benchmark output.
//
// Run:    go test -bench='BenchmarkFilestoreRegion' -benchmem -run='^$' ./state/filestore

package filestore

import (
	"context"
	"strconv"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
)

var (
	frSinkChunk state.Chunk
	frSinkErr   error
)

func benchRegionStore(tb testing.TB, records int, payloadSize int) (*Store, []state.ChunkRef) {
	tb.Helper()
	source, refs := benchStore(tb, records, payloadSize)
	sourcePath := source.Path()
	if err := source.Close(); err != nil {
		tb.Fatal(err)
	}
	read := core.ReadFile(sourcePath)
	if !read.OK {
		tb.Fatalf("read source store: %s", read.Error())
	}
	prefix := []byte("KVST-bench-header")
	suffix := []byte("KVST-bench-tail")
	sourceBytes := read.Value.([]byte)
	container := make([]byte, 0, len(prefix)+len(sourceBytes)+len(suffix))
	container = append(container, prefix...)
	container = append(container, sourceBytes...)
	container = append(container, suffix...)
	containerPath := core.PathJoin(core.PathDir(sourcePath), "session.kv")
	if write := core.WriteFile(containerPath, container, 0o600); !write.OK {
		tb.Fatalf("write region container: %s", write.Error())
	}
	region, err := OpenRegionWithSegmentAlias(context.Background(), containerPath, int64(len(prefix)), int64(len(sourceBytes)), sourcePath)
	if err != nil {
		tb.Fatalf("open region store: %v", err)
	}
	tb.Cleanup(func() { _ = region.Close() })
	return region, refs
}

func BenchmarkFilestoreRegion_ResolveRefBytes_64KB(b *testing.B) {
	store, refs := benchRegionStore(b, 1, 64*1024)
	ctx := context.Background()
	target := refs[0]
	b.ReportAllocs()
	b.SetBytes(64 * 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frSinkChunk, frSinkErr = store.ResolveRefBytes(ctx, target)
	}
}

func BenchmarkFilestoreRegion_BorrowRefBytes_64KB(b *testing.B) {
	store, refs := benchRegionStore(b, 1, 64*1024)
	ctx := context.Background()
	target := refs[0]
	b.ReportAllocs()
	b.SetBytes(64 * 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		borrowed, err := state.BorrowRefBytes(ctx, store, target)
		frSinkChunk = state.Chunk{Ref: borrowed.Ref, Data: borrowed.Data}
		frSinkErr = err
	}
}

func BenchmarkFilestoreRegion_ResolveRefBytes_1000Records(b *testing.B) {
	store, refs := benchRegionStore(b, 1000, 64)
	ctx := context.Background()
	target := refs[500]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frSinkChunk, frSinkErr = store.ResolveRefBytes(ctx, target)
	}
}

func BenchmarkFilestoreRegion_BorrowRefBytes_1000Records(b *testing.B) {
	store, refs := benchRegionStore(b, 1000, 64)
	ctx := context.Background()
	target := refs[500]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		borrowed, err := state.BorrowRefBytes(ctx, store, target)
		frSinkChunk = state.Chunk{Ref: borrowed.Ref, Data: borrowed.Data}
		frSinkErr = err
	}
}

func BenchmarkFilestoreRegion_Open_10000Records(b *testing.B) {
	dir := b.TempDir()
	sourcePath := core.PathJoin(dir, "index-10000.mvlog")
	{
		store, err := Create(context.Background(), sourcePath)
		if err != nil {
			b.Fatal(err)
		}
		payload := make([]byte, 64)
		for i := range 10000 {
			if _, err := store.PutBytes(context.Background(), payload, state.PutOptions{
				URI:  "mlx://bench/region-open-" + strconv.Itoa(i),
				Kind: "bench",
			}); err != nil {
				b.Fatal(err)
			}
		}
		_ = store.Close()
	}
	read := core.ReadFile(sourcePath)
	if !read.OK {
		b.Fatalf("read source store: %s", read.Error())
	}
	prefix := []byte("KVST-bench-header")
	sourceBytes := read.Value.([]byte)
	containerPath := core.PathJoin(dir, "session.kv")
	container := make([]byte, 0, len(prefix)+len(sourceBytes))
	container = append(container, prefix...)
	container = append(container, sourceBytes...)
	if write := core.WriteFile(containerPath, container, 0o600); !write.OK {
		b.Fatalf("write region container: %s", write.Error())
	}

	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store, err := OpenRegionWithSegmentAlias(ctx, containerPath, int64(len(prefix)), int64(len(sourceBytes)), sourcePath)
		if err != nil {
			b.Fatal(err)
		}
		_ = store.Close()
	}
}
