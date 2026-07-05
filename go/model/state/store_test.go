// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the package-level Store dispatchers in store.go — the
// nil-ctx/nil-store guards and interface-probe fallback branches that
// state_test.go's native-Resolver-path tests never reach, plus the
// error formatters (ChunkNotFoundError / URIChunkNotFoundError) and
// MergeRef, none of which any non-benchmark test previously exercised.

package state

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// --- fakes ---------------------------------------------------------------

// storeGetErr implements only Store.Get, returning a configurable text
// or error. Exercises the Get-adapter fallback branch the top-level
// dispatchers fall through to when a store implements nothing richer
// than the bare Store contract.
type storeGetErr struct {
	text string
	err  error
}

func (s storeGetErr) Get(_ context.Context, _ int) (string, error) {
	return s.text, s.err
}

// storeBinaryOnly implements Store.Get + BinaryResolver but not
// RefBinaryResolver/URIResolver — exercises the BinaryResolver branch in
// the package-level ResolveBytes, including the Text->Data backfill when
// the resolver hands back a text-only chunk.
type storeBinaryOnly struct {
	chunk Chunk
	err   error
}

func (s storeBinaryOnly) Get(_ context.Context, _ int) (string, error) {
	return s.chunk.Text, nil
}

func (s storeBinaryOnly) ResolveBytes(_ context.Context, chunkID int) (Chunk, error) {
	if s.err != nil {
		return Chunk{}, s.err
	}
	chunk := s.chunk
	chunk.Ref.ChunkID = chunkID
	return chunk, nil
}

// storeRefBinaryOnly implements Store.Get + RefBinaryResolver — exercises
// the RefBinaryResolver branch in the package-level ResolveRefBytes,
// including the Text->Data backfill.
type storeRefBinaryOnly struct {
	chunk Chunk
	err   error
}

func (s storeRefBinaryOnly) Get(_ context.Context, _ int) (string, error) {
	return s.chunk.Text, nil
}

func (s storeRefBinaryOnly) ResolveRefBytes(_ context.Context, ref ChunkRef) (Chunk, error) {
	if s.err != nil {
		return Chunk{}, s.err
	}
	chunk := s.chunk
	chunk.Ref.ChunkID = ref.ChunkID
	return chunk, nil
}

// --- Resolve ---------------------------------------------------------------

func TestResolve_Good(t *testing.T) {
	// A nil context is normalised to context.Background() rather than
	// panicking.
	store := NewInMemoryStore(map[int]string{1: "hello"})
	chunk, err := Resolve(nil, store, 1)
	if err != nil {
		t.Fatalf("Resolve(nil ctx) error = %v", err)
	}
	if chunk.Text != "hello" {
		t.Fatalf("Resolve(nil ctx) = %+v, want hello", chunk)
	}

	// A store implementing only Store.Get falls back to the Get-adapter
	// path and wraps the text into a minimal Chunk.
	chunk, err = Resolve(context.Background(), storeGetErr{text: "adapter text"}, 7)
	if err != nil {
		t.Fatalf("Resolve(get-adapter) error = %v", err)
	}
	if chunk.Ref.ChunkID != 7 || chunk.Text != "adapter text" {
		t.Fatalf("Resolve(get-adapter) = %+v, want id 7 + adapter text", chunk)
	}
}

func TestResolve_Bad(t *testing.T) {
	// A nil store is rejected without dereferencing.
	if _, err := Resolve(context.Background(), nil, 5); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Resolve(nil store) error = %v, want ErrChunkNotFound", err)
	}

	// A Get-adapter store whose Get fails propagates the error verbatim.
	getErr := core.NewError("get failed")
	if _, err := Resolve(context.Background(), storeGetErr{err: getErr}, 1); !core.Is(err, getErr) {
		t.Fatalf("Resolve(get error) error = %v, want %v", err, getErr)
	}
}

// --- ResolveBytes ------------------------------------------------------------

func TestResolveBytes_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte{1, 2, 3}, PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if _, err := ResolveBytes(nil, store, ref.ChunkID); err != nil {
		t.Fatalf("ResolveBytes(nil ctx) error = %v", err)
	}

	// A BinaryResolver that returns a text-only chunk gets backfilled
	// into Data by the dispatcher.
	chunk, err := ResolveBytes(context.Background(), storeBinaryOnly{chunk: Chunk{Text: "plain text"}}, 3)
	if err != nil {
		t.Fatalf("ResolveBytes(text-only resolver) error = %v", err)
	}
	if string(chunk.Data) != "plain text" {
		t.Fatalf("ResolveBytes(text-only resolver) Data = %q, want backfilled text", chunk.Data)
	}

	// A store implementing only Store.Get falls through Resolve and gets
	// the same Text->Data backfill at the ResolveBytes layer.
	chunk, err = ResolveBytes(context.Background(), storeGetErr{text: "fallback text"}, 9)
	if err != nil {
		t.Fatalf("ResolveBytes(get-adapter) error = %v", err)
	}
	if string(chunk.Data) != "fallback text" || chunk.Ref.ChunkID != 9 {
		t.Fatalf("ResolveBytes(get-adapter) = %+v, want backfilled fallback text", chunk)
	}
}

func TestResolveBytes_Bad(t *testing.T) {
	if _, err := ResolveBytes(context.Background(), nil, 1); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveBytes(nil store) error = %v, want ErrChunkNotFound", err)
	}

	// A BinaryResolver whose ResolveBytes fails propagates the error.
	resolveErr := core.NewError("resolve bytes failed")
	if _, err := ResolveBytes(context.Background(), storeBinaryOnly{err: resolveErr}, 1); !core.Is(err, resolveErr) {
		t.Fatalf("ResolveBytes(resolver error) error = %v, want %v", err, resolveErr)
	}

	// A Get-adapter store whose Get fails propagates through the Resolve
	// fallback.
	getErr := core.NewError("get failed")
	if _, err := ResolveBytes(context.Background(), storeGetErr{err: getErr}, 1); !core.Is(err, getErr) {
		t.Fatalf("ResolveBytes(get error) error = %v, want %v", err, getErr)
	}
}

// --- ResolveRefBytes ---------------------------------------------------------

func TestResolveRefBytes_Good(t *testing.T) {
	if _, err := ResolveRefBytes(nil, NewInMemoryStore(map[int]string{1: "x"}), ChunkRef{ChunkID: 1}); err != nil {
		t.Fatalf("ResolveRefBytes(nil ctx) error = %v", err)
	}

	// RefBinaryResolver dispatch, including the Text->Data backfill.
	chunk, err := ResolveRefBytes(context.Background(), storeRefBinaryOnly{chunk: Chunk{Text: "ref text"}}, ChunkRef{ChunkID: 4})
	if err != nil {
		t.Fatalf("ResolveRefBytes(ref-resolver) error = %v", err)
	}
	if string(chunk.Data) != "ref text" {
		t.Fatalf("ResolveRefBytes(ref-resolver) Data = %q, want backfilled ref text", chunk.Data)
	}

	// A store without RefBinaryResolver falls through to ResolveBytes by
	// ChunkID (InMemoryStore implements BinaryResolver, not
	// RefBinaryResolver).
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte("fallback"), PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	chunk, err = ResolveRefBytes(context.Background(), store, ref)
	if err != nil {
		t.Fatalf("ResolveRefBytes(fallback) error = %v", err)
	}
	if string(chunk.Data) != "fallback" {
		t.Fatalf("ResolveRefBytes(fallback) = %+v, want fallback payload", chunk)
	}
}

func TestResolveRefBytes_Bad(t *testing.T) {
	if _, err := ResolveRefBytes(context.Background(), nil, ChunkRef{ChunkID: 1}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveRefBytes(nil store) error = %v, want ErrChunkNotFound", err)
	}

	refErr := core.NewError("resolve ref bytes failed")
	if _, err := ResolveRefBytes(context.Background(), storeRefBinaryOnly{err: refErr}, ChunkRef{ChunkID: 1}); !core.Is(err, refErr) {
		t.Fatalf("ResolveRefBytes(resolver error) error = %v, want %v", err, refErr)
	}

	// Zero ChunkID with no RefBinaryResolver is rejected without a
	// downstream lookup.
	if _, err := ResolveRefBytes(context.Background(), storeGetErr{text: "x"}, ChunkRef{ChunkID: 0}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveRefBytes(zero id) error = %v, want ErrChunkNotFound", err)
	}
}

// --- BorrowBytes ---------------------------------------------------------

func TestBorrowBytes_Good(t *testing.T) {
	if _, err := BorrowBytes(nil, NewInMemoryStore(map[int]string{1: "x"}), 1); err != nil {
		t.Fatalf("BorrowBytes(nil ctx) error = %v", err)
	}

	// BinaryBorrower dispatch — InMemoryStore implements it natively.
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte{9, 8, 7}, PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	borrowed, err := BorrowBytes(context.Background(), store, ref.ChunkID)
	if err != nil {
		t.Fatalf("BorrowBytes(native) error = %v", err)
	}
	if len(borrowed.Data) != 3 || borrowed.Data[0] != 9 {
		t.Fatalf("BorrowBytes(native) = %+v, want borrowed payload", borrowed)
	}

	// A store without BinaryBorrower falls back to ResolveBytes and
	// wraps the result.
	borrowed, err = BorrowBytes(context.Background(), storeGetErr{text: "fallback"}, 3)
	if err != nil {
		t.Fatalf("BorrowBytes(fallback) error = %v", err)
	}
	if string(borrowed.Data) != "fallback" {
		t.Fatalf("BorrowBytes(fallback) = %+v, want fallback payload", borrowed)
	}
}

func TestBorrowBytes_Bad(t *testing.T) {
	if _, err := BorrowBytes(context.Background(), nil, 1); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowBytes(nil store) error = %v, want ErrChunkNotFound", err)
	}

	getErr := core.NewError("get failed")
	if _, err := BorrowBytes(context.Background(), storeGetErr{err: getErr}, 1); !core.Is(err, getErr) {
		t.Fatalf("BorrowBytes(fallback error) error = %v, want %v", err, getErr)
	}
}

// --- BorrowRefBytes ---------------------------------------------------------

func TestBorrowRefBytes_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte("x"), PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if _, err := BorrowRefBytes(nil, store, ref); err != nil {
		t.Fatalf("BorrowRefBytes(nil ctx) error = %v", err)
	}

	// A store without RefBinaryBorrower falls through to BorrowBytes by
	// ChunkID.
	borrowed, err := BorrowRefBytes(context.Background(), storeGetErr{text: "delegate"}, ChunkRef{ChunkID: 5})
	if err != nil {
		t.Fatalf("BorrowRefBytes(delegate) error = %v", err)
	}
	if string(borrowed.Data) != "delegate" {
		t.Fatalf("BorrowRefBytes(delegate) = %+v, want delegated payload", borrowed)
	}
}

func TestBorrowRefBytes_Ugly(t *testing.T) {
	// Zero ChunkID with no RefBinaryBorrower is rejected before any
	// delegate lookup runs.
	if _, err := BorrowRefBytes(context.Background(), storeGetErr{text: "x"}, ChunkRef{ChunkID: 0}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowRefBytes(zero id) error = %v, want ErrChunkNotFound", err)
	}
}

// --- ResolveURI ------------------------------------------------------------

func TestResolveURI_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	if _, err := store.Put(context.Background(), "hi", PutOptions{URI: "state://x/1"}); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if _, err := ResolveURI(nil, store, "state://x/1"); err != nil {
		t.Fatalf("ResolveURI(nil ctx) error = %v", err)
	}
}

func TestResolveURI_Bad(t *testing.T) {
	if _, err := ResolveURI(context.Background(), nil, "state://x"); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveURI(nil store) error = %v, want ErrChunkNotFound", err)
	}

	store := NewInMemoryStore(map[int]string{1: "x"})
	if _, err := ResolveURI(context.Background(), store, ""); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveURI(empty uri) error = %v, want ErrChunkNotFound", err)
	}

	// A store that doesn't implement URIResolver falls to the terminal
	// not-found branch instead of dispatching.
	if _, err := ResolveURI(context.Background(), storeGetErr{text: "x"}, "state://missing"); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("ResolveURI(no resolver) error = %v, want ErrChunkNotFound", err)
	}
}

// --- MergeRef ---------------------------------------------------------------

func TestMergeRef_Good(t *testing.T) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory, Segment: "epoch-1"}
	overlay := ChunkRef{ChunkID: 9, FrameOffset: 42, HasFrameOffset: true, Codec: CodecStateVideo, Segment: "epoch-3"}

	if merged := MergeRef(base, overlay); merged != overlay {
		t.Fatalf("MergeRef(full overlay) = %+v, want overlay to fully win = %+v", merged, overlay)
	}
}

func TestMergeRef_Ugly(t *testing.T) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory, Segment: "epoch-1"}

	// An empty overlay changes nothing — base.ChunkID is non-zero so the
	// zero-base OR-branch never fires either.
	if same := MergeRef(base, ChunkRef{}); same != base {
		t.Fatalf("MergeRef(empty overlay) = %+v, want unchanged base %+v", same, base)
	}

	// overlay.ChunkID==0 with base.ChunkID==0 still assigns — the
	// "base.ChunkID == 0" side of the OR condition.
	if zeroBase := MergeRef(ChunkRef{}, ChunkRef{Codec: CodecStateVideo}); zeroBase.ChunkID != 0 || zeroBase.Codec != CodecStateVideo {
		t.Fatalf("MergeRef(zero base) = %+v, want zero id + overlay codec", zeroBase)
	}

	// Partial overlays touch only the field they set.
	if codecOnly := MergeRef(base, ChunkRef{Codec: CodecStateVideo}); codecOnly.ChunkID != base.ChunkID || codecOnly.Codec != CodecStateVideo || codecOnly.Segment != base.Segment || codecOnly.FrameOffset != base.FrameOffset {
		t.Fatalf("MergeRef(codec-only overlay) = %+v, want only Codec changed from %+v", codecOnly, base)
	}

	if segmentOnly := MergeRef(base, ChunkRef{Segment: "epoch-9"}); segmentOnly.Codec != base.Codec || segmentOnly.Segment != "epoch-9" {
		t.Fatalf("MergeRef(segment-only overlay) = %+v, want only Segment changed", segmentOnly)
	}

	if frameOnly := MergeRef(base, ChunkRef{FrameOffset: 99, HasFrameOffset: true}); frameOnly.FrameOffset != 99 || !frameOnly.HasFrameOffset || frameOnly.Codec != base.Codec {
		t.Fatalf("MergeRef(frame-only overlay) = %+v, want only FrameOffset changed", frameOnly)
	}
}

// --- ChunkNotFoundError / URIChunkNotFoundError -----------------------------

func TestChunkNotFoundError_Error(t *testing.T) {
	err := &ChunkNotFoundError{ID: 42}
	if got := err.Error(); got != "state chunk 42 not found" {
		t.Fatalf("Error() = %q, want %q", got, "state chunk 42 not found")
	}
	if !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("Is(ErrChunkNotFound) = false, want true")
	}
}

func TestURIChunkNotFoundError_Error(t *testing.T) {
	empty := &URIChunkNotFoundError{}
	if got := empty.Error(); got != "state chunk URI not found" {
		t.Fatalf("Error() (empty URI) = %q, want %q", got, "state chunk URI not found")
	}

	withURI := &URIChunkNotFoundError{URI: "state://missing"}
	if got := withURI.Error(); got != `state chunk URI "state://missing" not found` {
		t.Fatalf("Error() (with URI) = %q, want %q", got, `state chunk URI "state://missing" not found`)
	}
	if !core.Is(withURI, ErrChunkNotFound) {
		t.Fatalf("Is(ErrChunkNotFound) = false, want true")
	}
}
