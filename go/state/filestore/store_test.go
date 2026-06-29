// SPDX-Licence-Identifier: EUPL-1.2

// filestore lifecycle tests: append+reopen round-trips, Open variants, Close gates and context cancellation.
package filestore

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

func TestFileStore_Good_AppendsAndReopens(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "kv-blocks.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if store.Path() != path {
		t.Fatalf("Path() = %q, want %q", store.Path(), path)
	}

	first, err := store.Put(ctx, "alpha", state.PutOptions{URI: "mlx://kv/0", Title: "first"})
	if err != nil {
		t.Fatalf("Put(first) error = %v", err)
	}
	second, err := store.Put(ctx, "bravo", state.PutOptions{URI: "mlx://kv/1", Title: "second"})
	if err != nil {
		t.Fatalf("Put(second) error = %v", err)
	}
	if first.ChunkID != 1 || second.ChunkID != 2 || second.Codec != CodecFile || second.Segment != path {
		t.Fatalf("refs = %+v/%+v, want sequential file refs", first, second)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	stat := core.Stat(path)
	if !stat.OK {
		t.Fatalf("Stat(%q): %s", path, stat.Error())
	}
	if stat.Value.(interface{ Size() int64 }).Size() <= int64(len("alphabravo")) {
		t.Fatalf("file size = %d, want framed payload on disk", stat.Value.(interface{ Size() int64 }).Size())
	}

	reopened, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reopened.Close()
	if reopened.ChunkCount() != 2 {
		t.Fatalf("ChunkCount() = %d, want 2", reopened.ChunkCount())
	}
	chunk, err := reopened.Resolve(ctx, 2)
	if err != nil {
		t.Fatalf("Resolve(2) error = %v", err)
	}
	if chunk.Text != "bravo" || chunk.Ref.ChunkID != 2 || chunk.Ref.Codec != CodecFile || chunk.Ref.Segment != path {
		t.Fatalf("chunk = %+v, want second chunk from file", chunk)
	}
	byURI, err := state.ResolveURI(ctx, reopened, "mlx://kv/1")
	if err != nil {
		t.Fatalf("ResolveURI() error = %v", err)
	}
	if byURI.Text != "bravo" || byURI.Ref.ChunkID != 2 {
		t.Fatalf("ResolveURI() chunk = %+v, want second chunk", byURI)
	}
}

func TestFileStore_Good_OpensLegacyStateHeader(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "legacy.mvlog")
	meta := []byte(core.JSONMarshalString(recordMeta{URI: "mlx://legacy/1"}))
	payload := []byte("legacy payload")
	data := append([]byte(nil), legacyFileMagic...)
	var hdrBuf [recordHeaderLen]byte
	encodeRecordHeader(hdrBuf[:], 1, len(payload), len(meta))
	data = append(data, hdrBuf[:]...)
	data = append(data, meta...)
	data = append(data, payload...)
	if result := core.WriteFile(path, data, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}

	store, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open(legacy) error = %v", err)
	}
	defer store.Close()

	chunk, err := state.ResolveURI(ctx, store, "mlx://legacy/1")
	if err != nil {
		t.Fatalf("ResolveURI(legacy) error = %v", err)
	}
	if chunk.Text != "legacy payload" || chunk.Ref.FrameOffset != uint64(len(legacyFileMagic)) {
		t.Fatalf("legacy chunk = %+v, want payload and legacy frame offset", chunk)
	}
}

func TestFileStore_Good_OpenWithSegmentAlias(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	sourcePath := core.PathJoin(dir, "source.mvlog")
	relocatedPath := core.PathJoin(dir, "relocated.mvlog")
	source, err := Create(ctx, sourcePath)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ref, err := source.PutBytes(ctx, []byte("relocated payload"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	read := core.ReadFile(sourcePath)
	if !read.OK {
		t.Fatalf("ReadFile(source) error = %s", read.Error())
	}
	if write := core.WriteFile(relocatedPath, read.Value.([]byte), 0o600); !write.OK {
		t.Fatalf("WriteFile(relocated) error = %s", write.Error())
	}

	strict, err := Open(ctx, relocatedPath)
	if err != nil {
		t.Fatalf("Open(relocated) error = %v", err)
	}
	if _, err := state.ResolveRefBytes(ctx, strict, ref); err == nil {
		t.Fatal("strict ResolveRefBytes(source segment) error = nil")
	}
	if err := strict.Close(); err != nil {
		t.Fatalf("strict Close() error = %v", err)
	}

	aliased, err := OpenWithSegmentAlias(ctx, relocatedPath, sourcePath)
	if err != nil {
		t.Fatalf("OpenWithSegmentAlias() error = %v", err)
	}
	defer aliased.Close()
	chunk, err := state.ResolveRefBytes(ctx, aliased, ref)
	if err != nil {
		t.Fatalf("ResolveRefBytes(alias) error = %v", err)
	}
	if string(chunk.Data) != "relocated payload" {
		t.Fatalf("alias payload = %q, want relocated payload", string(chunk.Data))
	}
	physicalRef := ref
	physicalRef.Segment = relocatedPath
	if _, err := state.ResolveRefBytes(ctx, aliased, physicalRef); err != nil {
		t.Fatalf("ResolveRefBytes(physical segment) error = %v", err)
	}
	wrongRef := ref
	wrongRef.Segment = sourcePath + ".wrong"
	if _, err := state.ResolveRefBytes(ctx, aliased, wrongRef); err == nil {
		t.Fatal("ResolveRefBytes(wrong segment) error = nil")
	}
}

func TestFileStore_Good_OpenRegionWithSegmentAlias(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	sourcePath := core.PathJoin(dir, "source.mvlog")
	containerPath := core.PathJoin(dir, "session.kv")
	source, err := Create(ctx, sourcePath)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	first, err := source.PutBytes(ctx, []byte("first region payload"), state.PutOptions{URI: "mlx://region/first"})
	if err != nil {
		t.Fatalf("PutBytes(first) error = %v", err)
	}
	second, err := source.PutBytes(ctx, []byte("second region payload"), state.PutOptions{URI: "mlx://region/second"})
	if err != nil {
		t.Fatalf("PutBytes(second) error = %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	read := core.ReadFile(sourcePath)
	if !read.OK {
		t.Fatalf("ReadFile(source) error = %s", read.Error())
	}
	prefix := []byte("KVST-test-header")
	suffix := []byte("not-state-log-tail")
	sourceBytes := read.Value.([]byte)
	container := append(append(append([]byte(nil), prefix...), sourceBytes...), suffix...)
	if write := core.WriteFile(containerPath, container, 0o600); !write.OK {
		t.Fatalf("WriteFile(container) error = %s", write.Error())
	}

	store, err := OpenRegionWithSegmentAlias(ctx, containerPath, int64(len(prefix)), int64(len(sourceBytes)), sourcePath)
	if err != nil {
		t.Fatalf("OpenRegionWithSegmentAlias() error = %v", err)
	}
	defer store.Close()
	if store.Path() != containerPath {
		t.Fatalf("Path() = %q, want container path", store.Path())
	}
	if store.ChunkCount() != 2 {
		t.Fatalf("ChunkCount() = %d, want 2", store.ChunkCount())
	}
	chunk, err := state.ResolveRefBytes(ctx, store, second)
	if err != nil {
		t.Fatalf("ResolveRefBytes(alias region) error = %v", err)
	}
	if string(chunk.Data) != "second region payload" || chunk.Ref.FrameOffset != second.FrameOffset {
		t.Fatalf("region chunk = %+v, want second payload at original frame offset", chunk)
	}
	borrowed, err := state.BorrowRefBytes(ctx, store, second)
	if err != nil {
		t.Fatalf("BorrowRefBytes(alias region) error = %v", err)
	}
	if string(borrowed.Data) != "second region payload" || borrowed.Ref.FrameOffset != second.FrameOffset {
		t.Fatalf("borrowed region chunk = %+v, want second payload at original frame offset", borrowed)
	}
	byURI, err := state.ResolveURI(ctx, store, "mlx://region/first")
	if err != nil {
		t.Fatalf("ResolveURI(region) error = %v", err)
	}
	if byURI.Text != "first region payload" || byURI.Ref.FrameOffset != first.FrameOffset {
		t.Fatalf("ResolveURI(region) = %+v, want first payload with relative offset", byURI)
	}
	physicalRef := second
	physicalRef.Segment = containerPath
	if _, err := state.ResolveRefBytes(ctx, store, physicalRef); err != nil {
		t.Fatalf("ResolveRefBytes(physical region) error = %v", err)
	}
	wrongRef := second
	wrongRef.Segment = sourcePath + ".wrong"
	if _, err := state.ResolveRefBytes(ctx, store, wrongRef); err == nil {
		t.Fatal("ResolveRefBytes(wrong region segment) error = nil")
	}
	if _, err := state.BorrowRefBytes(ctx, store, wrongRef); err == nil {
		t.Fatal("BorrowRefBytes(wrong region segment) error = nil")
	}
	if _, err := store.PutBytes(ctx, []byte("blocked"), state.PutOptions{}); err == nil {
		t.Fatal("PutBytes(read-only region) error = nil")
	}
}

func TestFileStore_Bad_ClosedStore(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "closed.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close(second) error = %v", err)
	}
	if _, err := store.Put(context.Background(), "payload", state.PutOptions{}); err == nil {
		t.Fatal("Put(closed) error = nil")
	}
	if _, err := store.Resolve(context.Background(), 1); err == nil {
		t.Fatal("Resolve(closed) error = nil")
	}
	if _, err := store.ResolveBytes(context.Background(), 1); err == nil {
		t.Fatal("ResolveBytes(closed) error = nil")
	}
	if _, err := store.ResolveURI(context.Background(), "mlx://missing"); err == nil {
		t.Fatal("ResolveURI(closed) error = nil")
	}
}

func TestFileStore_Ugly_CancelledContext(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "cancelled.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = store.Put(ctx, "payload", state.PutOptions{})

	if !core.Is(err, context.Canceled) {
		t.Fatalf("Put(cancelled) error = %v, want context.Canceled", err)
	}
	if _, err := store.Resolve(context.Background(), 1); !core.Is(err, state.ErrChunkNotFound) {
		t.Fatalf("Resolve(after cancelled put) error = %v, want missing chunk", err)
	}
}
