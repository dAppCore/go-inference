// SPDX-Licence-Identifier: EUPL-1.2

package filestore

import (
	"context"
	stdio "io"
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

func TestFileStore_Good_BinaryPayload(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "binary.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	payload := []byte{0, 1, 2, 255}
	ref, err := store.PutBytes(ctx, payload, state.PutOptions{URI: "mlx://binary/1"})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	payload[1] = 99
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	reopened, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reopened.Close()
	chunk, err := state.ResolveBytes(ctx, reopened, ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes() error = %v", err)
	}
	if len(chunk.Data) != 4 || chunk.Data[0] != 0 || chunk.Data[1] != 1 || chunk.Data[3] != 255 {
		t.Fatalf("ResolveBytes() data = %v, want original binary payload", chunk.Data)
	}
	chunk.Data[2] = 88
	again, err := state.ResolveBytes(ctx, reopened, ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(second) error = %v", err)
	}
	if again.Data[2] != 2 {
		t.Fatalf("ResolveBytes() returned aliased payload = %v", again.Data)
	}
	byURI, err := state.ResolveURI(ctx, reopened, "mlx://binary/1")
	if err != nil {
		t.Fatalf("ResolveURI(binary) error = %v", err)
	}
	if byURI.Text != string([]byte{0, 1, 2, 255}) {
		t.Fatalf("ResolveURI(binary) text = %q, want binary-compatible text fallback", byURI.Text)
	}
}

func TestFileStore_Good_ResolveRefBytesUsesFrameOffset(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "offset.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	first, err := store.PutBytes(ctx, []byte("first"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes(first) error = %v", err)
	}
	second, err := store.PutBytes(ctx, []byte("second"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes(second) error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	reopened, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reopened.Close()

	chunk, err := state.ResolveRefBytes(ctx, reopened, state.ChunkRef{
		ChunkID:        second.ChunkID,
		FrameOffset:    second.FrameOffset,
		HasFrameOffset: true,
		Codec:          CodecFile,
		Segment:        path,
	})

	if err != nil {
		t.Fatalf("ResolveRefBytes(offset) error = %v", err)
	}
	if string(chunk.Data) != "second" || chunk.Ref.FrameOffset != second.FrameOffset {
		t.Fatalf("ResolveRefBytes(offset) chunk = %+v, want second payload by frame offset", chunk)
	}
	if _, err := state.ResolveRefBytes(ctx, reopened, state.ChunkRef{ChunkID: first.ChunkID, FrameOffset: second.FrameOffset, HasFrameOffset: true, Codec: CodecFile, Segment: path}); err == nil {
		t.Fatal("ResolveRefBytes(id mismatch) error = nil")
	}
	if _, err := state.ResolveRefBytes(ctx, reopened, state.ChunkRef{ChunkID: second.ChunkID, FrameOffset: second.FrameOffset, HasFrameOffset: true, Codec: CodecFile, Segment: path + ".other"}); err == nil {
		t.Fatal("ResolveRefBytes(segment mismatch) error = nil")
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

func TestFileStore_Good_StreamPayload(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "stream.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ref, err := store.PutBytesStream(ctx, 5, state.PutOptions{URI: "mlx://stream/1"}, func(writer stdio.Writer) error {
		if _, err := writer.Write([]byte("he")); err != nil {
			return err
		}
		_, err := writer.Write([]byte("llo"))
		return err
	})
	if err != nil {
		t.Fatalf("PutBytesStream() error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	reopened, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reopened.Close()
	chunk, err := state.ResolveBytes(ctx, reopened, ref.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(stream) error = %v", err)
	}
	if string(chunk.Data) != "hello" {
		t.Fatalf("streamed payload = %q, want hello", string(chunk.Data))
	}
}

func TestFileStore_Bad_MissingChunk(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "empty.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()

	_, err = store.Get(context.Background(), 99)

	if !core.Is(err, state.ErrChunkNotFound) {
		t.Fatalf("Get(missing) error = %v, want ErrChunkNotFound", err)
	}
}

func TestFileStore_Bad_InvalidInputs(t *testing.T) {
	if _, err := Create(context.Background(), ""); err == nil {
		t.Fatal("Create(empty) error = nil, want path error")
	}
	if _, err := Open(context.Background(), ""); err == nil {
		t.Fatal("Open(empty) error = nil, want path error")
	}
	if _, err := (*Store)(nil).PutBytes(context.Background(), []byte("x"), state.PutOptions{}); err == nil {
		t.Fatal("PutBytes(nil store) error = nil")
	}
	if _, err := (*Store)(nil).ResolveBytes(context.Background(), 1); !core.Is(err, state.ErrChunkNotFound) {
		t.Fatalf("ResolveBytes(nil store) error = %v, want ErrChunkNotFound", err)
	}
	streamPath := core.PathJoin(t.TempDir(), "invalid-stream.mvlog")
	store, err := Create(context.Background(), streamPath)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	if _, err := store.PutBytesStream(context.Background(), -1, state.PutOptions{}, func(writer stdio.Writer) error {
		return nil
	}); err == nil {
		t.Fatal("PutBytesStream(negative size) error = nil")
	}
	if _, err := store.PutBytesStream(context.Background(), 1, state.PutOptions{}, nil); err == nil {
		t.Fatal("PutBytesStream(nil writer) error = nil")
	}
	if _, err := store.PutBytesStream(context.Background(), 2, state.PutOptions{}, func(writer stdio.Writer) error {
		_, err := writer.Write([]byte("x"))
		return err
	}); err == nil {
		t.Fatal("PutBytesStream(short payload) error = nil")
	}
	if _, err := store.PutBytesStream(context.Background(), 1, state.PutOptions{}, func(writer stdio.Writer) error {
		_, err := writer.Write([]byte("too long"))
		return err
	}); err == nil {
		t.Fatal("PutBytesStream(oversized payload) error = nil")
	}
	if store.ChunkCount() != 0 {
		t.Fatalf("ChunkCount() = %d after failed streams, want 0", store.ChunkCount())
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	reopened, err := Open(context.Background(), streamPath)
	if err != nil {
		t.Fatalf("Open(after failed streams) error = %v", err)
	}
	defer reopened.Close()
	if reopened.ChunkCount() != 0 {
		t.Fatalf("reopened ChunkCount() = %d after failed streams, want 0", reopened.ChunkCount())
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

func TestFileStore_Bad_InvalidFile(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "invalid.mvlog")
	if result := core.WriteFile(path, []byte("not a state log"), 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	if _, err := Open(context.Background(), path); err == nil {
		t.Fatal("Open(invalid header) error = nil")
	}
}

func TestFileStore_Bad_CorruptRecords(t *testing.T) {
	cases := []struct {
		name string
		data []byte
	}{
		{
			name: "truncated-record-header",
			data: append(append([]byte(nil), fileMagic...), recordMagic[:2]...),
		},
		{
			name: "invalid-record-header",
			data: append(append([]byte(nil), fileMagic...), make([]byte, recordHeaderLen)...),
		},
		{
			name: "truncated-payload",
			data: append(append(append([]byte(nil), fileMagic...), testHeader(1, 4, 0)...), []byte{1, 2}...),
		},
		{
			name: "invalid-metadata",
			data: append(append(append([]byte(nil), fileMagic...), testHeader(1, 0, 1)...), []byte("{")...),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := core.PathJoin(t.TempDir(), tc.name+".mvlog")
			if result := core.WriteFile(path, tc.data, 0o600); !result.OK {
				t.Fatalf("WriteFile() error = %s", result.Error())
			}
			if _, err := Open(context.Background(), path); err == nil {
				t.Fatalf("Open(%s) error = nil, want corruption error", tc.name)
			}
		})
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

// testHeader is a test-only wrapper that returns a fresh []byte built
// via encodeRecordHeader's in-place API. Production callers should use
// encodeRecordHeader directly with a stack-allocated [recordHeaderLen]byte.
func testHeader(chunkID, payloadSize, metaSize int) []byte {
	buf := make([]byte, recordHeaderLen)
	encodeRecordHeader(buf, chunkID, payloadSize, metaSize)
	return buf
}

// TestFileStore_Good_RebuildIndexPreservesIndexShape pins the index
// shape across rebuildIndex changes — Wave 8 perf rewrites can alter
// how the meta JSON is parsed, but the resulting index entries (per
// chunk id) must match a Put-built index 1:1 in ref + payload offset.
// The uriIndex must contain exactly the URIs that were Put with a
// non-empty URI, mapped to the same chunk ids.
func TestFileStore_Good_RebuildIndexPreservesIndexShape(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "rebuild-shape.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	// Mix records with URI, without URI, with tag-maps + label-slices,
	// with empty meta — covers every branch rebuildIndex touches.
	cases := []state.PutOptions{
		{URI: "mlx://kv/0", Title: "with-uri", Kind: "bench"},
		{}, // empty meta
		{URI: "mlx://kv/2", Tags: map[string]string{"a": "1", "b": "2"}, Labels: []string{"x", "y"}},
		{Kind: "no-uri", Track: "tr"},
		{URI: "mlx://kv/4", Title: "another", Tags: map[string]string{}},
	}
	payloads := [][]byte{
		[]byte("alpha"),
		[]byte("beta"),
		[]byte("gamma"),
		[]byte("delta"),
		[]byte("epsilon"),
	}
	var putRefs []state.ChunkRef
	for i, opts := range cases {
		ref, err := store.PutBytes(ctx, payloads[i], opts)
		if err != nil {
			t.Fatalf("PutBytes(%d) error = %v", i, err)
		}
		putRefs = append(putRefs, ref)
	}
	// Snapshot the live index built by Put for later comparison.
	store.mu.Lock()
	putIndex := make(map[int]fileIndexEntry, len(store.index))
	for id, entry := range store.index {
		putIndex[id] = entry
	}
	putURIIndex := make(map[string]int, len(store.uriIndex))
	for uri, id := range store.uriIndex {
		putURIIndex[uri] = id
	}
	putNextID := store.nextID
	putWriteAt := store.writeAt
	store.mu.Unlock()
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	reopened, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reopened.Close()

	reopened.mu.Lock()
	defer reopened.mu.Unlock()

	if reopened.nextID != putNextID {
		t.Fatalf("rebuilt nextID = %d, want %d", reopened.nextID, putNextID)
	}
	if reopened.writeAt != putWriteAt {
		t.Fatalf("rebuilt writeAt = %d, want %d", reopened.writeAt, putWriteAt)
	}
	if len(reopened.index) != len(putIndex) {
		t.Fatalf("rebuilt index size = %d, want %d", len(reopened.index), len(putIndex))
	}
	for id, want := range putIndex {
		got, ok := reopened.index[id]
		if !ok {
			t.Fatalf("rebuilt index missing chunk id %d", id)
		}
		if got.ref != want.ref {
			t.Fatalf("rebuilt entry[%d].ref = %+v, want %+v", id, got.ref, want.ref)
		}
		if got.payloadAt != want.payloadAt {
			t.Fatalf("rebuilt entry[%d].payloadAt = %d, want %d", id, got.payloadAt, want.payloadAt)
		}
		if got.payloadSize != want.payloadSize {
			t.Fatalf("rebuilt entry[%d].payloadSize = %d, want %d", id, got.payloadSize, want.payloadSize)
		}
	}
	if len(reopened.uriIndex) != len(putURIIndex) {
		t.Fatalf("rebuilt uriIndex size = %d, want %d", len(reopened.uriIndex), len(putURIIndex))
	}
	for uri, wantID := range putURIIndex {
		gotID, ok := reopened.uriIndex[uri]
		if !ok {
			t.Fatalf("rebuilt uriIndex missing %q", uri)
		}
		if gotID != wantID {
			t.Fatalf("rebuilt uriIndex[%q] = %d, want %d", uri, gotID, wantID)
		}
	}
	_ = putRefs
}

// TestEncodeRecordMeta_RoundTrip locks the hand-rolled encoder to
// encoding/json's deserialisation contract. The encoder is the
// canonical PutBytesStream meta serialiser — every record we write
// passes through it, so its output must round-trip cleanly through
// json.Unmarshal back into recordMeta with no field loss or value
// drift. Mixed shapes (empty, single string, tag map, label slice,
// escape-sensitive characters) cover the branches the encoder
// walks.
func TestEncodeRecordMeta_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		meta recordMeta
	}{
		{"empty", recordMeta{}},
		{"uri-only", recordMeta{URI: "mlx://kv/0"}},
		{"all-strings", recordMeta{
			URI:   "mlx://kv/1",
			Title: "training-checkpoint",
			Kind:  "kv",
			Track: "primary",
		}},
		{"tags-1", recordMeta{
			URI:  "mlx://kv/2",
			Tags: map[string]string{"epoch": "3"},
		}},
		{"tags-many", recordMeta{
			URI: "mlx://kv/3",
			Tags: map[string]string{
				"epoch": "3", "track": "primary",
				"branch": "dev", "runner": "homelab",
			},
		}},
		{"labels", recordMeta{
			URI:    "mlx://kv/4",
			Labels: []string{"k0:v0", "k1:v1"},
		}},
		{"full", recordMeta{
			URI: "mlx://kv/5", Title: "bench", Kind: "training",
			Track: "primary", Tags: map[string]string{"a": "1"},
			Labels: []string{"x"},
		}},
		{"escapes", recordMeta{
			Title: `quote " and backslash \ and slash /`,
			Kind:  "tabs\tand\nnewlines",
			Tags:  map[string]string{"control": "\x01\x02"},
		}},
		{"unicode", recordMeta{
			Title:  "ünïcödé",
			Labels: []string{"日本", "🐦"},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := encodeRecordMeta(&tc.meta)
			var decoded recordMeta
			if result := core.JSONUnmarshal(encoded, &decoded); !result.OK {
				t.Fatalf("JSONUnmarshal(%s) error: %v\nencoded: %s", tc.name, result.Value, encoded)
			}
			if decoded.URI != tc.meta.URI {
				t.Fatalf("URI = %q, want %q", decoded.URI, tc.meta.URI)
			}
			if decoded.Title != tc.meta.Title {
				t.Fatalf("Title = %q, want %q", decoded.Title, tc.meta.Title)
			}
			if decoded.Kind != tc.meta.Kind {
				t.Fatalf("Kind = %q, want %q", decoded.Kind, tc.meta.Kind)
			}
			if decoded.Track != tc.meta.Track {
				t.Fatalf("Track = %q, want %q", decoded.Track, tc.meta.Track)
			}
			if len(decoded.Tags) != len(tc.meta.Tags) {
				t.Fatalf("Tags len = %d, want %d", len(decoded.Tags), len(tc.meta.Tags))
			}
			for k, v := range tc.meta.Tags {
				if decoded.Tags[k] != v {
					t.Fatalf("Tags[%q] = %q, want %q", k, decoded.Tags[k], v)
				}
			}
			if len(decoded.Labels) != len(tc.meta.Labels) {
				t.Fatalf("Labels len = %d, want %d", len(decoded.Labels), len(tc.meta.Labels))
			}
			for i, v := range tc.meta.Labels {
				if decoded.Labels[i] != v {
					t.Fatalf("Labels[%d] = %q, want %q", i, decoded.Labels[i], v)
				}
			}
			// extractRecordURI must also accept the encoder output.
			uri, err := extractRecordURI(encoded)
			if err != nil {
				t.Fatalf("extractRecordURI: %v\nencoded: %s", err, encoded)
			}
			if uri != tc.meta.URI {
				t.Fatalf("extractRecordURI URI = %q, want %q", uri, tc.meta.URI)
			}
		})
	}
}
