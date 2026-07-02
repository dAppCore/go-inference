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

func TestGet_Good_ExistingChunk(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "get-existing.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	if _, err := store.Put(ctx, "hello", state.PutOptions{}); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	got, err := store.Get(ctx, 1)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if got != "hello" {
		t.Fatalf("Get() = %q, want %q", got, "hello")
	}
}

func TestResolve_Bad_CancelledContext(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "resolve-cancelled.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.Resolve(ctx, 1); !core.Is(err, context.Canceled) {
		t.Fatalf("Resolve(cancelled) error = %v, want context.Canceled", err)
	}
}

func TestResolve_Bad_NilStore(t *testing.T) {
	if _, err := (*Store)(nil).Resolve(context.Background(), 1); !core.Is(err, state.ErrChunkNotFound) {
		t.Fatalf("Resolve(nil store) error = %v, want ErrChunkNotFound", err)
	}
}

func TestResolveURI_Bad_CancelledContext(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "resolveuri-cancelled.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.ResolveURI(ctx, "mlx://x"); !core.Is(err, context.Canceled) {
		t.Fatalf("ResolveURI(cancelled) error = %v, want context.Canceled", err)
	}
}

func TestResolveURI_Bad_NilStore(t *testing.T) {
	if _, err := (*Store)(nil).ResolveURI(context.Background(), "mlx://x"); err == nil {
		t.Fatal("ResolveURI(nil store) error = nil")
	}
}

func TestResolveURI_Bad_NotFoundOnOpenStore(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "resolveuri-miss.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	if _, err := store.Put(ctx, "hello", state.PutOptions{URI: "mlx://present"}); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if _, err := store.ResolveURI(ctx, "mlx://truly-missing"); !core.Is(err, state.ErrChunkNotFound) {
		t.Fatalf("ResolveURI(miss on open store) error = %v, want ErrChunkNotFound", err)
	}
}

func TestResolveBytes_Bad_CancelledContext(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "resolvebytes-cancelled.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.ResolveBytes(ctx, 1); !core.Is(err, context.Canceled) {
		t.Fatalf("ResolveBytes(cancelled) error = %v, want context.Canceled", err)
	}
}

func TestResolveBytesLocked_Bad_ReadAtError(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "poke-index.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	// Poke the index entry's payload offset out past real EOF —
	// simulates on-disk/index inconsistency without needing a
	// genuinely corrupt file, and lands squarely on resolveBytesLocked's
	// ReadAt error path.
	store.mu.Lock()
	entry := store.index[ref.ChunkID]
	entry.payloadAt = 999999
	store.index[ref.ChunkID] = entry
	store.mu.Unlock()

	if _, err := store.ResolveBytes(ctx, ref.ChunkID); err == nil {
		t.Fatal("ResolveBytes(payload offset beyond EOF) error = nil")
	}
}

func TestBorrowBytes_Good_WritableStore(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "borrow-writable.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	borrowed, err := store.BorrowBytes(ctx, ref.ChunkID)
	if err != nil {
		t.Fatalf("BorrowBytes() error = %v", err)
	}
	if string(borrowed.Data) != "hello" || borrowed.Ref.ChunkID != ref.ChunkID {
		t.Fatalf("BorrowBytes() = %+v, want hello payload for chunk %d", borrowed, ref.ChunkID)
	}
}

func TestBorrowBytes_Good_ReadOnlyRegionStore(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "borrow-readonly.mvlog")
	source, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ref, err := source.PutBytes(ctx, []byte("region payload"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	store, err := OpenRegionWithSegmentAlias(ctx, path, 0, 0, "")
	if err != nil {
		t.Fatalf("OpenRegionWithSegmentAlias() error = %v", err)
	}
	defer store.Close()

	borrowed, err := store.BorrowBytes(ctx, ref.ChunkID)
	if err != nil {
		t.Fatalf("BorrowBytes(readOnly) error = %v", err)
	}
	if string(borrowed.Data) != "region payload" {
		t.Fatalf("BorrowBytes(readOnly) = %+v, want region payload", borrowed)
	}
}

func TestBorrowBytes_Bad_NilStore(t *testing.T) {
	if _, err := (*Store)(nil).BorrowBytes(context.Background(), 1); !core.Is(err, state.ErrChunkNotFound) {
		t.Fatalf("BorrowBytes(nil store) error = %v, want ErrChunkNotFound", err)
	}
}

func TestBorrowBytes_Bad_CancelledContext(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "borrow-cancelled.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.BorrowBytes(ctx, 1); !core.Is(err, context.Canceled) {
		t.Fatalf("BorrowBytes(cancelled) error = %v, want context.Canceled", err)
	}
}

func TestBorrowBytes_Bad_ClosedStore(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "borrow-closed.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if _, err := store.BorrowBytes(context.Background(), 1); err == nil {
		t.Fatal("BorrowBytes(closed) error = nil")
	}
}

func TestBorrowBytes_Bad_MissingChunk(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "borrow-missing.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	if _, err := store.BorrowBytes(context.Background(), 42); !core.Is(err, state.ErrChunkNotFound) {
		t.Fatalf("BorrowBytes(missing) error = %v, want ErrChunkNotFound", err)
	}
}

func TestBorrowBytes_Bad_BorrowPayloadLockedError(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "borrow-payload-error.mvlog")
	source, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ref, err := source.PutBytes(ctx, []byte("payload"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	store, err := OpenRegionWithSegmentAlias(ctx, path, 0, 0, "")
	if err != nil {
		t.Fatalf("OpenRegionWithSegmentAlias() error = %v", err)
	}
	defer store.Close()

	store.mu.Lock()
	entry := store.index[ref.ChunkID]
	entry.payloadAt = -1
	store.index[ref.ChunkID] = entry
	store.mu.Unlock()

	if _, err := store.BorrowBytes(ctx, ref.ChunkID); err == nil {
		t.Fatal("BorrowBytes(poisoned payload offset) error = nil")
	}
}

func TestResolveRefBytes_Good_DelegatesWhenNoFrameOffset(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "delegate-resolveref.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	chunk, err := store.ResolveRefBytes(ctx, state.ChunkRef{ChunkID: ref.ChunkID})
	if err != nil {
		t.Fatalf("ResolveRefBytes(no frame offset) error = %v", err)
	}
	if string(chunk.Data) != "hello" {
		t.Fatalf("ResolveRefBytes(no frame offset) = %+v, want hello payload", chunk)
	}
}

func TestResolveRefBytes_Bad_CancelledContext(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "resolveref-cancelled.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.ResolveRefBytes(ctx, state.ChunkRef{}); !core.Is(err, context.Canceled) {
		t.Fatalf("ResolveRefBytes(cancelled) error = %v, want context.Canceled", err)
	}
}

func TestResolveRefBytes_Bad_NilStore(t *testing.T) {
	if _, err := (*Store)(nil).ResolveRefBytes(context.Background(), state.ChunkRef{}); err == nil {
		t.Fatal("ResolveRefBytes(nil store) error = nil")
	}
}

func TestResolveRefBytes_Bad_NonFileCodec(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "resolveref-badcodec.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	_, err = store.ResolveRefBytes(context.Background(), state.ChunkRef{HasFrameOffset: true, Codec: "other/codec"})
	if !core.Is(err, errRefNonFileCodec) {
		t.Fatalf("ResolveRefBytes(non-file codec) error = %v, want errRefNonFileCodec", err)
	}
}

func TestResolveRefBytes_Bad_ClosedStore(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "resolveref-closed.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	path := store.Path()
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	_, err = store.ResolveRefBytes(context.Background(), state.ChunkRef{HasFrameOffset: true, Codec: CodecFile, Segment: path})
	if err == nil {
		t.Fatal("ResolveRefBytes(closed) error = nil")
	}
}

func TestBorrowRefBytes_Good_DelegatesWhenNoFrameOffset(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "delegate-borrowref.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	borrowed, err := store.BorrowRefBytes(ctx, state.ChunkRef{ChunkID: ref.ChunkID})
	if err != nil {
		t.Fatalf("BorrowRefBytes(no frame offset) error = %v", err)
	}
	if string(borrowed.Data) != "hello" {
		t.Fatalf("BorrowRefBytes(no frame offset) = %+v, want hello payload", borrowed)
	}
}

func TestBorrowRefBytes_Good_WritableStoreSuccess(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "writable-borrowref.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	borrowed, err := store.BorrowRefBytes(ctx, ref)
	if err != nil {
		t.Fatalf("BorrowRefBytes(writable store) error = %v", err)
	}
	if string(borrowed.Data) != "hello" {
		t.Fatalf("BorrowRefBytes(writable store) = %+v, want hello payload", borrowed)
	}
}

func TestBorrowRefBytes_Bad_WritableStoreResolveError(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "writable-borrowref-bad.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	first, err := store.PutBytes(ctx, []byte("first"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes(first) error = %v", err)
	}
	second, err := store.PutBytes(ctx, []byte("second"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes(second) error = %v", err)
	}
	badRef := state.ChunkRef{ChunkID: first.ChunkID, FrameOffset: second.FrameOffset, HasFrameOffset: true, Codec: CodecFile, Segment: store.Path()}
	if _, err := store.BorrowRefBytes(ctx, badRef); err == nil {
		t.Fatal("BorrowRefBytes(writable store, id mismatch) error = nil")
	}
}

func TestBorrowRefBytes_Bad_CancelledContext(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "borrowref-cancelled.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := store.BorrowRefBytes(ctx, state.ChunkRef{}); !core.Is(err, context.Canceled) {
		t.Fatalf("BorrowRefBytes(cancelled) error = %v, want context.Canceled", err)
	}
}

func TestBorrowRefBytes_Bad_NilStore(t *testing.T) {
	if _, err := (*Store)(nil).BorrowRefBytes(context.Background(), state.ChunkRef{}); err == nil {
		t.Fatal("BorrowRefBytes(nil store) error = nil")
	}
}

func TestBorrowRefBytes_Bad_NonFileCodec(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "borrowref-badcodec.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	_, err = store.BorrowRefBytes(context.Background(), state.ChunkRef{HasFrameOffset: true, Codec: "other/codec"})
	if !core.Is(err, errRefNonFileCodec) {
		t.Fatalf("BorrowRefBytes(non-file codec) error = %v, want errRefNonFileCodec", err)
	}
}

func TestBorrowRefBytes_Bad_ClosedStore(t *testing.T) {
	store, err := Create(context.Background(), core.PathJoin(t.TempDir(), "borrowref-closed.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	path := store.Path()
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	_, err = store.BorrowRefBytes(context.Background(), state.ChunkRef{HasFrameOffset: true, Codec: CodecFile, Segment: path})
	if err == nil {
		t.Fatal("BorrowRefBytes(closed) error = nil")
	}
}

func TestResolveRefBytesLocked_Bad_FrameOffsetTooBig(t *testing.T) {
	s := &Store{}
	_, err := s.resolveRefBytesLocked(state.ChunkRef{FrameOffset: ^uint64(0)})
	if !core.Is(err, errRefFrameOffsetTooBig) {
		t.Fatalf("resolveRefBytesLocked(huge frame offset) error = %v, want errRefFrameOffsetTooBig", err)
	}
}

func TestResolveRefBytesLocked_Bad_PhysicalOffsetError(t *testing.T) {
	s := &Store{region: 5}
	_, err := s.resolveRefBytesLocked(state.ChunkRef{FrameOffset: 100})
	if !core.Is(err, errRegionInvalid) {
		t.Fatalf("resolveRefBytesLocked(offset beyond region) error = %v, want errRegionInvalid", err)
	}
}

func TestResolveRefBytesLocked_Bad_ReadAtHeaderError(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "resolverefbytes-eof.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	if _, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if _, err := store.resolveRefBytesLocked(state.ChunkRef{FrameOffset: 999999}); err == nil {
		t.Fatal("resolveRefBytesLocked(offset beyond EOF) error = nil")
	}
}

func TestResolveRefBytesLocked_Bad_DecodeHeaderError(t *testing.T) {
	// FrameOffset 0 points at the file's magic bytes, not a record —
	// decodeRecordHeader's 4-byte magic check must reject it.
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "resolverefbytes-badmagic.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	if _, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if _, err := store.resolveRefBytesLocked(state.ChunkRef{FrameOffset: 0}); err == nil {
		t.Fatal("resolveRefBytesLocked(offset 0, file magic) error = nil")
	}
}

// buildHandCraftedStore writes fileMagic followed by the given raw
// record bytes (each already including its own header+meta+payload)
// and returns a live, directly-constructed *Store over it — bypassing
// Open()/rebuildIndex entirely. Some overflow-field fault injection
// (chunk id / payload size beyond maxInt()) cannot go through the
// public Open() path at all: rebuildIndex would reject the very same
// malformed record while scanning during cold open, before a caller
// ever gets a handle to call resolveRefBytesLocked/borrowRefBytesLocked
// on it directly.
func buildHandCraftedStore(t *testing.T, name string, records ...[]byte) *Store {
	t.Helper()
	data := append([]byte(nil), fileMagic...)
	for _, r := range records {
		data = append(data, r...)
	}
	path := core.PathJoin(t.TempDir(), name)
	if result := core.WriteFile(path, data, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	file := openFileOrFatal(t, path, core.O_RDWR)
	t.Cleanup(func() { file.Close() })
	return &Store{file: file, index: map[int]fileIndexEntry{}, uriIndex: map[string]int{}, nextID: 1}
}

func TestResolveRefBytesLocked_Bad_ChunkIDOverflow(t *testing.T) {
	record := append(append(rawHeader(uint64(1)<<63, 1, 2), []byte("{}")...), []byte("x")...)
	s := buildHandCraftedStore(t, "resolveref-chunkid-overflow.mvlog", record)
	if _, err := s.resolveRefBytesLocked(state.ChunkRef{FrameOffset: uint64(len(fileMagic))}); err == nil {
		t.Fatal("resolveRefBytesLocked(chunk id overflow) error = nil")
	}
}

func TestResolveRefBytesLocked_Bad_PayloadSizeOverflow(t *testing.T) {
	record := rawHeader(1, uint64(1)<<63, 0)
	s := buildHandCraftedStore(t, "resolveref-payloadsize-overflow.mvlog", record)
	if _, err := s.resolveRefBytesLocked(state.ChunkRef{FrameOffset: uint64(len(fileMagic))}); err == nil {
		t.Fatal("resolveRefBytesLocked(payload size overflow) error = nil")
	}
}

func TestBorrowRefBytesLocked_Bad_FrameOffsetTooBig(t *testing.T) {
	s := &Store{}
	_, err := s.borrowRefBytesLocked(state.ChunkRef{FrameOffset: ^uint64(0)})
	if !core.Is(err, errRefFrameOffsetTooBig) {
		t.Fatalf("borrowRefBytesLocked(huge frame offset) error = %v, want errRefFrameOffsetTooBig", err)
	}
}

func TestBorrowRefBytesLocked_Bad_PhysicalOffsetError(t *testing.T) {
	// s.file is nil, so ensureMappedRegionLocked fails fast
	// (errStoreClosed) and the fallback path's own physicalOffset
	// check is what must reject the out-of-region offset.
	s := &Store{region: 5}
	_, err := s.borrowRefBytesLocked(state.ChunkRef{FrameOffset: 100})
	if !core.Is(err, errRegionInvalid) {
		t.Fatalf("borrowRefBytesLocked(offset beyond region) error = %v, want errRegionInvalid", err)
	}
}

func TestBorrowRefBytesLocked_Good_NonMmapFallback(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "borrowref-fallback.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	store.mu.Lock()
	// region=-1 makes ensureMappedRegionLocked's own regionSize call
	// fail without touching the fd's access mode, forcing
	// borrowRefBytesLocked onto its non-mmap ReadAt header-view path
	// (the existing region-store tests only ever exercise the mmap
	// path, since OpenRegionWithSegmentAlias always succeeds at mmap
	// setup for a real O_RDONLY fd).
	store.region = -1
	borrowed, err := store.borrowRefBytesLocked(state.ChunkRef{ChunkID: ref.ChunkID, FrameOffset: ref.FrameOffset, HasFrameOffset: true})
	store.mu.Unlock()
	if err != nil {
		t.Fatalf("borrowRefBytesLocked(non-mmap fallback) error = %v", err)
	}
	if string(borrowed.Data) != "hello" {
		t.Fatalf("borrowRefBytesLocked(non-mmap fallback) = %+v, want hello payload", borrowed)
	}
}

func TestBorrowRefBytesLocked_Bad_ChunkIDOverflow(t *testing.T) {
	record := append(append(rawHeader(uint64(1)<<63, 1, 2), []byte("{}")...), []byte("x")...)
	s := buildHandCraftedStore(t, "borrowref-chunkid-overflow.mvlog", record)
	s.region = -1
	if _, err := s.borrowRefBytesLocked(state.ChunkRef{FrameOffset: uint64(len(fileMagic))}); err == nil {
		t.Fatal("borrowRefBytesLocked(chunk id overflow) error = nil")
	}
}

func TestBorrowRefBytesLocked_Bad_PayloadSizeOverflowPropagatesFromBorrowPayloadLocked(t *testing.T) {
	// A declared payload size that is large but still within int
	// range: intFromUint64 accepts it, so the failure must come from
	// borrowPayloadLocked's own ReadAt against the (much smaller)
	// real file — this is the distinct call-site branch inside
	// borrowRefBytesLocked, not borrowPayloadLocked's own bounds
	// checks (covered separately).
	record := rawHeader(1, 10000, 0)
	s := buildHandCraftedStore(t, "borrowref-payload-readat-fail.mvlog", record)
	s.region = -1
	if _, err := s.borrowRefBytesLocked(state.ChunkRef{FrameOffset: uint64(len(fileMagic))}); err == nil {
		t.Fatal("borrowRefBytesLocked(payload beyond EOF) error = nil")
	}
}

func TestBorrowPayloadLocked_Bad_NegativeInputs(t *testing.T) {
	s := &Store{}
	if _, err := s.borrowPayloadLocked(-1, 5); !core.Is(err, errRegionInvalid) {
		t.Fatalf("borrowPayloadLocked(negative payloadAt) error = %v, want errRegionInvalid", err)
	}
	if _, err := s.borrowPayloadLocked(0, -1); !core.Is(err, errRegionInvalid) {
		t.Fatalf("borrowPayloadLocked(negative payloadSize) error = %v, want errRegionInvalid", err)
	}
}

func TestBorrowPayloadLocked_Good_FallbackReadAt(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "borrowpayload-fallback-ok.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	store.mu.Lock()
	entry := store.index[ref.ChunkID]
	store.region = -1
	data, err := store.borrowPayloadLocked(entry.payloadAt, entry.payloadSize)
	store.mu.Unlock()
	if err != nil {
		t.Fatalf("borrowPayloadLocked(fallback) error = %v", err)
	}
	if string(data) != "hello" {
		t.Fatalf("borrowPayloadLocked(fallback) = %q, want %q", data, "hello")
	}
}

func TestBorrowPayloadLocked_Bad_FallbackReadAtError(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "borrowpayload-fallback-bad.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	if _, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	store.mu.Lock()
	store.region = -1
	_, err = store.borrowPayloadLocked(999999, 5)
	store.mu.Unlock()
	if err == nil {
		t.Fatal("borrowPayloadLocked(fallback, beyond EOF) error = nil")
	}
}

func TestBorrowPayloadLocked_Bad_FallbackPhysicalOffsetError(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "borrowpayload-wronly.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if _, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	// A write-only fd makes mmap(PROT_READ) fail deterministically
	// (EACCES) without altering regionSize's own bookkeeping, forcing
	// ensureMappedRegionLocked to fail for a reason unrelated to
	// region math — so the fallback's own physicalOffset bound check
	// is what must reject the out-of-region payloadAt.
	woFile := openFileOrFatal(t, path, core.O_WRONLY)
	defer woFile.Close()
	s := &Store{file: woFile, region: 3}
	if _, err := s.borrowPayloadLocked(100, 2); err == nil {
		t.Fatal("borrowPayloadLocked(write-only fd, offset beyond region) error = nil")
	}
}

func TestBorrowBytes_Bad_WritableResolveError(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "borrow-writable-bad.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	store.mu.Lock()
	entry := store.index[ref.ChunkID]
	entry.payloadAt = 999999
	store.index[ref.ChunkID] = entry
	store.mu.Unlock()

	if _, err := store.BorrowBytes(ctx, ref.ChunkID); err == nil {
		t.Fatal("BorrowBytes(writable, poisoned payload offset) error = nil")
	}
}

func TestResolveRefBytesLocked_Bad_ReadAtPayloadError(t *testing.T) {
	// payloadSize (10000) is well within int range — intFromUint64
	// accepts it — but the hand-crafted file has no payload bytes at
	// all, so the payload ReadAt itself must fail. Distinct from the
	// PayloadSizeOverflow case above, which never reaches this read.
	record := rawHeader(1, 10000, 0)
	s := buildHandCraftedStore(t, "resolveref-payload-readat-fail.mvlog", record)
	if _, err := s.resolveRefBytesLocked(state.ChunkRef{FrameOffset: uint64(len(fileMagic))}); err == nil {
		t.Fatal("resolveRefBytesLocked(payload beyond EOF) error = nil")
	}
}

func TestBorrowRefBytesLocked_Bad_MmapOffsetOutOfBounds(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "mmap-oob.mvlog")
	source, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if _, err := source.PutBytes(ctx, []byte("hello"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	store, err := OpenRegionWithSegmentAlias(ctx, path, 0, 0, "")
	if err != nil {
		t.Fatalf("OpenRegionWithSegmentAlias() error = %v", err)
	}
	defer store.Close()

	store.mu.Lock()
	_, err = store.borrowRefBytesLocked(state.ChunkRef{FrameOffset: 999999, HasFrameOffset: true})
	store.mu.Unlock()
	if err == nil {
		t.Fatal("borrowRefBytesLocked(offset beyond mapped region) error = nil")
	}
}

func TestBorrowRefBytesLocked_Bad_NonMmapReadAtHeaderError(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "borrowref-fallback-readat-fail.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	if _, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	store.mu.Lock()
	store.region = -1
	_, err = store.borrowRefBytesLocked(state.ChunkRef{FrameOffset: 999999, HasFrameOffset: true})
	store.mu.Unlock()
	if err == nil {
		t.Fatal("borrowRefBytesLocked(non-mmap fallback, offset beyond EOF) error = nil")
	}
}

func TestBorrowRefBytesLocked_Bad_DecodeHeaderError(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "borrowref-badmagic.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	if _, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	store.mu.Lock()
	store.region = -1
	_, err = store.borrowRefBytesLocked(state.ChunkRef{FrameOffset: 0, HasFrameOffset: true})
	store.mu.Unlock()
	if err == nil {
		t.Fatal("borrowRefBytesLocked(offset 0, file magic) error = nil")
	}
}

func TestBorrowRefBytesLocked_Bad_ChunkIDMismatch(t *testing.T) {
	ctx := context.Background()
	store, err := Create(ctx, core.PathJoin(t.TempDir(), "borrowref-idmismatch.mvlog"))
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	first, err := store.PutBytes(ctx, []byte("first"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes(first) error = %v", err)
	}
	second, err := store.PutBytes(ctx, []byte("second"), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes(second) error = %v", err)
	}

	store.mu.Lock()
	store.region = -1
	_, err = store.borrowRefBytesLocked(state.ChunkRef{ChunkID: first.ChunkID, FrameOffset: second.FrameOffset, HasFrameOffset: true})
	store.mu.Unlock()
	if err == nil {
		t.Fatal("borrowRefBytesLocked(chunk id mismatch) error = nil")
	}
}

func TestBorrowRefBytesLocked_Bad_PayloadSizeOverflow(t *testing.T) {
	record := rawHeader(1, uint64(1)<<63, 0)
	s := buildHandCraftedStore(t, "borrowref-payloadsize-overflow.mvlog", record)
	s.region = -1
	if _, err := s.borrowRefBytesLocked(state.ChunkRef{FrameOffset: uint64(len(fileMagic))}); err == nil {
		t.Fatal("borrowRefBytesLocked(payload size overflow) error = nil")
	}
}

func TestBorrowPayloadLocked_Bad_MmapBoundsExceeded(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "mmap-payload-oob.mvlog")
	source, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if _, err := source.PutBytes(ctx, []byte("hello"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	store, err := OpenRegionWithSegmentAlias(ctx, path, 0, 0, "")
	if err != nil {
		t.Fatalf("OpenRegionWithSegmentAlias() error = %v", err)
	}
	defer store.Close()

	store.mu.Lock()
	_, err = store.borrowPayloadLocked(0, 999999)
	store.mu.Unlock()
	if err == nil {
		t.Fatal("borrowPayloadLocked(mmap bounds exceeded) error = nil")
	}
}
