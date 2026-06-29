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
