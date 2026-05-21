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
	data = append(data, encodeRecordHeader(1, len(payload), len(meta))...)
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
			data: append(append(append([]byte(nil), fileMagic...), encodeRecordHeader(1, 4, 0)...), []byte{1, 2}...),
		},
		{
			name: "invalid-metadata",
			data: append(append(append([]byte(nil), fileMagic...), encodeRecordHeader(1, 0, 1)...), []byte("{")...),
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
