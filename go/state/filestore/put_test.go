// SPDX-Licence-Identifier: EUPL-1.2

// filestore write-path tests: streamed payloads and PutBytesStream input validation / rollback.
package filestore

import (
	"context"
	stdio "io"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

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

func TestPutBytesStream_Bad_SeekError(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "seek-fail.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	// Close the underlying OS file out from under the Store while the
	// Store.file field stays non-nil, so PutBytesStream's own
	// `s.file == nil` gate passes but the following Seek syscall
	// fails deterministically on the closed descriptor.
	if err := store.file.Close(); err != nil {
		t.Fatalf("underlying Close() error = %v", err)
	}
	_, err = store.PutBytesStream(ctx, 1, state.PutOptions{}, func(w stdio.Writer) error {
		_, err := w.Write([]byte("x"))
		return err
	})
	if err == nil {
		t.Fatal("PutBytesStream(closed fd) error = nil, want seek error")
	}
}

func TestPutBytesStream_Bad_WriteHeaderError(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "write-header-fail.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.file.Close()
	// Swap in a read-only handle on the same path: Seek still
	// succeeds (repositioning needs no write permission) but the
	// following header+meta Write fails — this exercises both the
	// writeAll-failure branch in PutBytesStream and rollbackWriteLocked's
	// real Truncate/Seek body (as opposed to its early-return guards).
	result := core.OpenFile(path, core.O_RDONLY, 0o600)
	if !result.OK {
		t.Fatalf("OpenFile(read-only) error = %s", result.Error())
	}
	readOnlyFile := result.Value.(*core.OSFile)
	defer readOnlyFile.Close()
	store.file = readOnlyFile

	_, err = store.PutBytesStream(ctx, 1, state.PutOptions{}, func(w stdio.Writer) error {
		_, err := w.Write([]byte("x"))
		return err
	})
	if err == nil {
		t.Fatal("PutBytesStream(read-only fd) error = nil, want write error")
	}
}

func TestPutBytesStream_Bad_PhysicalOffsetError(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "region-write.mvlog")
	if result := core.WriteFile(path, fileMagic, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	result := core.OpenFile(path, core.O_RDWR, 0o600)
	if !result.OK {
		t.Fatalf("OpenFile() error = %s", result.Error())
	}
	file := result.Value.(*core.OSFile)
	defer file.Close()
	// A writable, region-bounded Store is unreachable through the
	// public API — OpenRegionWithSegmentAlias always opens read-only —
	// but PutBytesStream still defends s.writeAt against s.region.
	// Construct that combination directly to exercise the guard.
	s := &Store{
		file:     file,
		region:   4,
		writeAt:  100,
		index:    map[int]fileIndexEntry{},
		uriIndex: map[string]int{},
		nextID:   1,
	}
	_, err := s.PutBytesStream(ctx, 1, state.PutOptions{}, func(w stdio.Writer) error {
		_, err := w.Write([]byte("x"))
		return err
	})
	if err == nil {
		t.Fatal("PutBytesStream(writeAt beyond region) error = nil")
	}
}

func TestRollbackWriteLocked_Good_NilReceiver(t *testing.T) {
	var s *Store
	s.rollbackWriteLocked(0)
}

func TestRollbackWriteLocked_Good_NilFile(t *testing.T) {
	s := &Store{}
	s.rollbackWriteLocked(0)
}

func TestRollbackWriteLocked_Bad_PhysicalOffsetError(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "rollback-offset.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	before := store.writeAt
	// physicalOffset(-1) rejects before any Seek/Truncate is
	// attempted — rollbackWriteLocked must return early rather than
	// touch the file or any Store state.
	store.rollbackWriteLocked(-1)
	if store.writeAt != before {
		t.Fatalf("writeAt changed after rollbackWriteLocked(-1): got %d, want %d", store.writeAt, before)
	}
}
