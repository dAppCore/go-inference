// SPDX-Licence-Identifier: EUPL-1.2

// filestore read-path tests: binary payload round-trip, ResolveRefBytes frame-offset resolution and missing-chunk errors.
package filestore

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

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
