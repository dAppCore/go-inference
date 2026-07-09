// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the package-level Store dispatchers in store.go — the
// nil-ctx/nil-store guards, the interface-probe fallback branches, the
// pass-through-vs-sanitise asymmetry each dispatcher has on its resolver
// error path, plus the error formatters (ChunkNotFoundError /
// URIChunkNotFoundError) and MergeRef.

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

// storeDirtyResolver implements Store.Get + Resolver and always returns a
// non-zero Chunk ALONGSIDE a non-nil error — used to prove whether a
// dispatcher passes the resolver's return value through verbatim or
// sanitises it to the zero value on error.
type storeDirtyResolver struct {
	err error
}

func (s storeDirtyResolver) Get(_ context.Context, _ int) (string, error) {
	return "unused", nil
}

func (s storeDirtyResolver) Resolve(_ context.Context, chunkID int) (Chunk, error) {
	return Chunk{Ref: ChunkRef{ChunkID: chunkID}, Text: "dirty"}, s.err
}

// storeDirtyBinaryResolver implements BinaryResolver and always returns a
// non-zero Chunk alongside a non-nil error.
type storeDirtyBinaryResolver struct {
	err error
}

func (s storeDirtyBinaryResolver) Get(_ context.Context, _ int) (string, error) {
	return "unused", nil
}

func (s storeDirtyBinaryResolver) ResolveBytes(_ context.Context, chunkID int) (Chunk, error) {
	return Chunk{Ref: ChunkRef{ChunkID: chunkID}, Data: []byte("dirty")}, s.err
}

// storeDirtyRefBinaryResolver implements RefBinaryResolver and always
// returns a non-zero Chunk alongside a non-nil error.
type storeDirtyRefBinaryResolver struct {
	err error
}

func (s storeDirtyRefBinaryResolver) Get(_ context.Context, _ int) (string, error) {
	return "unused", nil
}

func (s storeDirtyRefBinaryResolver) ResolveRefBytes(_ context.Context, ref ChunkRef) (Chunk, error) {
	return Chunk{Ref: ref, Data: []byte("dirty")}, s.err
}

// storeDirtyBinaryBorrower implements BinaryBorrower and always returns a
// non-zero BorrowedChunk alongside a non-nil error.
type storeDirtyBinaryBorrower struct {
	err error
}

func (s storeDirtyBinaryBorrower) Get(_ context.Context, _ int) (string, error) {
	return "unused", nil
}

func (s storeDirtyBinaryBorrower) BorrowBytes(_ context.Context, chunkID int) (BorrowedChunk, error) {
	return BorrowedChunk{Ref: ChunkRef{ChunkID: chunkID}, Data: []byte("dirty")}, s.err
}

// storeDirtyURIResolver implements URIResolver and always returns a
// non-zero Chunk alongside a non-nil error.
type storeDirtyURIResolver struct {
	err error
}

func (s storeDirtyURIResolver) Get(_ context.Context, _ int) (string, error) {
	return "unused", nil
}

func (s storeDirtyURIResolver) ResolveURI(_ context.Context, uri string) (Chunk, error) {
	return Chunk{Ref: ChunkRef{Segment: uri}, Text: "dirty"}, s.err
}

// storeRefBorrowerErr implements RefBinaryBorrower and always fails.
type storeRefBorrowerErr struct {
	err error
}

func (s storeRefBorrowerErr) Get(_ context.Context, _ int) (string, error) {
	return "unused", nil
}

func (s storeRefBorrowerErr) BorrowRefBytes(_ context.Context, ref ChunkRef) (BorrowedChunk, error) {
	return BorrowedChunk{}, s.err
}

// --- Resolve ---------------------------------------------------------------

func TestStore_Resolve_Good(t *testing.T) {
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

func TestStore_Resolve_Bad(t *testing.T) {
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

// TestStore_Resolve_Ugly proves Resolve's Resolver-interface branch passes
// the resolver's return value through verbatim — unlike ResolveBytes and
// ResolveRefBytes below, it does not sanitise a non-zero chunk to the zero
// value when the resolver also returns an error.
func TestStore_Resolve_Ugly(t *testing.T) {
	resolveErr := core.NewError("dirty resolve")
	chunk, err := Resolve(context.Background(), storeDirtyResolver{err: resolveErr}, 3)
	if !core.Is(err, resolveErr) {
		t.Fatalf("Resolve(dirty resolver) error = %v, want %v", err, resolveErr)
	}
	if chunk.Ref.ChunkID != 3 || chunk.Text != "dirty" {
		t.Fatalf("Resolve(dirty resolver) = %+v, want the resolver's chunk passed through unsanitised", chunk)
	}
}

// --- ResolveBytes ------------------------------------------------------------

func TestStore_ResolveBytes_Good(t *testing.T) {
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

func TestStore_ResolveBytes_Bad(t *testing.T) {
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

// TestStore_ResolveBytes_Ugly proves the opposite of Resolve's pass-through:
// ResolveBytes sanitises a dirty BinaryResolver return to the zero Chunk on
// error rather than leaking the resolver's partial payload.
func TestStore_ResolveBytes_Ugly(t *testing.T) {
	resolveErr := core.NewError("dirty resolve bytes")
	chunk, err := ResolveBytes(context.Background(), storeDirtyBinaryResolver{err: resolveErr}, 3)
	if !core.Is(err, resolveErr) {
		t.Fatalf("ResolveBytes(dirty resolver) error = %v, want %v", err, resolveErr)
	}
	if chunk.Ref != (ChunkRef{}) || chunk.Text != "" || chunk.Data != nil {
		t.Fatalf("ResolveBytes(dirty resolver) = %+v, want zero Chunk on error", chunk)
	}
}

// --- ResolveRefBytes ---------------------------------------------------------

func TestStore_ResolveRefBytes_Good(t *testing.T) {
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

func TestStore_ResolveRefBytes_Bad(t *testing.T) {
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

// TestStore_ResolveRefBytes_Ugly mirrors ResolveBytes' sanitise-on-error
// behaviour: a dirty RefBinaryResolver return is zeroed rather than leaked.
func TestStore_ResolveRefBytes_Ugly(t *testing.T) {
	refErr := core.NewError("dirty resolve ref bytes")
	chunk, err := ResolveRefBytes(context.Background(), storeDirtyRefBinaryResolver{err: refErr}, ChunkRef{ChunkID: 3})
	if !core.Is(err, refErr) {
		t.Fatalf("ResolveRefBytes(dirty resolver) error = %v, want %v", err, refErr)
	}
	if chunk.Ref != (ChunkRef{}) || chunk.Text != "" || chunk.Data != nil {
		t.Fatalf("ResolveRefBytes(dirty resolver) = %+v, want zero Chunk on error", chunk)
	}
}

// --- BorrowBytes ---------------------------------------------------------

func TestStore_BorrowBytes_Good(t *testing.T) {
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

func TestStore_BorrowBytes_Bad(t *testing.T) {
	if _, err := BorrowBytes(context.Background(), nil, 1); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowBytes(nil store) error = %v, want ErrChunkNotFound", err)
	}

	getErr := core.NewError("get failed")
	if _, err := BorrowBytes(context.Background(), storeGetErr{err: getErr}, 1); !core.Is(err, getErr) {
		t.Fatalf("BorrowBytes(fallback error) error = %v, want %v", err, getErr)
	}
}

// TestStore_BorrowBytes_Ugly proves the BinaryBorrower branch passes a
// dirty return through verbatim — the opposite of ResolveBytes' sanitising
// behaviour, because BorrowBytes forwards the borrower's tuple directly
// instead of re-wrapping it.
func TestStore_BorrowBytes_Ugly(t *testing.T) {
	borrowErr := core.NewError("dirty borrow bytes")
	borrowed, err := BorrowBytes(context.Background(), storeDirtyBinaryBorrower{err: borrowErr}, 5)
	if !core.Is(err, borrowErr) {
		t.Fatalf("BorrowBytes(dirty borrower) error = %v, want %v", err, borrowErr)
	}
	if borrowed.Ref.ChunkID != 5 || string(borrowed.Data) != "dirty" {
		t.Fatalf("BorrowBytes(dirty borrower) = %+v, want the borrower's chunk passed through unsanitised", borrowed)
	}
}

// --- BorrowRefBytes ---------------------------------------------------------

func TestStore_BorrowRefBytes_Good(t *testing.T) {
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

// TestStore_BorrowRefBytes_Bad proves the nil-store guard fires before the
// ref is even inspected, and a failing RefBinaryBorrower propagates its
// error through verbatim (the pass-through branch, mirroring BorrowBytes).
func TestStore_BorrowRefBytes_Bad(t *testing.T) {
	if _, err := BorrowRefBytes(context.Background(), nil, ChunkRef{ChunkID: 9}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowRefBytes(nil store) error = %v, want ErrChunkNotFound", err)
	}

	borrowErr := core.NewError("ref borrower failed")
	if _, err := BorrowRefBytes(context.Background(), storeRefBorrowerErr{err: borrowErr}, ChunkRef{ChunkID: 1}); !core.Is(err, borrowErr) {
		t.Fatalf("BorrowRefBytes(borrower error) error = %v, want %v", err, borrowErr)
	}
}

func TestStore_BorrowRefBytes_Ugly(t *testing.T) {
	// Zero ChunkID with no RefBinaryBorrower is rejected before any
	// delegate lookup runs.
	if _, err := BorrowRefBytes(context.Background(), storeGetErr{text: "x"}, ChunkRef{ChunkID: 0}); !core.Is(err, ErrChunkNotFound) {
		t.Fatalf("BorrowRefBytes(zero id) error = %v, want ErrChunkNotFound", err)
	}
}

// --- ResolveURI ------------------------------------------------------------

func TestStore_ResolveURI_Good(t *testing.T) {
	store := NewInMemoryStore(nil)
	if _, err := store.Put(context.Background(), "hi", PutOptions{URI: "state://x/1"}); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if _, err := ResolveURI(nil, store, "state://x/1"); err != nil {
		t.Fatalf("ResolveURI(nil ctx) error = %v", err)
	}
}

func TestStore_ResolveURI_Bad(t *testing.T) {
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

// TestStore_ResolveURI_Ugly proves the URIResolver branch passes its return
// value through verbatim (like Resolve, unlike ResolveBytes/ResolveRefBytes)
// — a dirty non-zero chunk survives alongside a resolver error.
func TestStore_ResolveURI_Ugly(t *testing.T) {
	uriErr := core.NewError("dirty resolve uri")
	chunk, err := ResolveURI(context.Background(), storeDirtyURIResolver{err: uriErr}, "state://dirty")
	if !core.Is(err, uriErr) {
		t.Fatalf("ResolveURI(dirty resolver) error = %v, want %v", err, uriErr)
	}
	if chunk.Ref.Segment != "state://dirty" || chunk.Text != "dirty" {
		t.Fatalf("ResolveURI(dirty resolver) = %+v, want the resolver's chunk passed through unsanitised", chunk)
	}
}

// --- MergeRef ---------------------------------------------------------------

func TestStore_MergeRef_Good(t *testing.T) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true, Codec: CodecMemory, Segment: "epoch-1"}
	overlay := ChunkRef{ChunkID: 9, FrameOffset: 42, HasFrameOffset: true, Codec: CodecStateVideo, Segment: "epoch-3"}

	if merged := MergeRef(base, overlay); merged != overlay {
		t.Fatalf("MergeRef(full overlay) = %+v, want overlay to fully win = %+v", merged, overlay)
	}
}

// TestStore_MergeRef_Bad feeds MergeRef an internally-inconsistent overlay —
// HasFrameOffset false but FrameOffset non-zero — and proves the flag, not
// the stray numeric value, gates the merge: base's FrameOffset survives.
func TestStore_MergeRef_Bad(t *testing.T) {
	base := ChunkRef{ChunkID: 7, FrameOffset: 7, HasFrameOffset: true}
	overlay := ChunkRef{FrameOffset: 999, HasFrameOffset: false}

	merged := MergeRef(base, overlay)
	if merged.FrameOffset != base.FrameOffset || !merged.HasFrameOffset {
		t.Fatalf("MergeRef(inconsistent overlay) = %+v, want base FrameOffset preserved when HasFrameOffset is false", merged)
	}
}

func TestStore_MergeRef_Ugly(t *testing.T) {
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

// --- ChunkNotFoundError ------------------------------------------------------

func TestStore_ChunkNotFoundError_Error_Good(t *testing.T) {
	err := &ChunkNotFoundError{ID: 42}
	if got := err.Error(); got != "state chunk 42 not found" {
		t.Fatalf("Error() = %q, want %q", got, "state chunk 42 not found")
	}
}

// TestStore_ChunkNotFoundError_Error_Bad feeds the degenerate zero ID —
// still formats cleanly, no special-cased message.
func TestStore_ChunkNotFoundError_Error_Bad(t *testing.T) {
	err := &ChunkNotFoundError{}
	if got := err.Error(); got != "state chunk 0 not found" {
		t.Fatalf("Error() (zero ID) = %q, want %q", got, "state chunk 0 not found")
	}
}

// TestStore_ChunkNotFoundError_Error_Ugly feeds a negative ID — an
// out-of-domain value the type never validates against, formatted as-is.
func TestStore_ChunkNotFoundError_Error_Ugly(t *testing.T) {
	err := &ChunkNotFoundError{ID: -1}
	if got := err.Error(); got != "state chunk -1 not found" {
		t.Fatalf("Error() (negative ID) = %q, want %q", got, "state chunk -1 not found")
	}
}

func TestStore_ChunkNotFoundError_Unwrap_Good(t *testing.T) {
	err := &ChunkNotFoundError{ID: 1}
	if unwrapped := err.Unwrap(); unwrapped != ErrChunkNotFound {
		t.Fatalf("Unwrap() = %v, want ErrChunkNotFound", unwrapped)
	}
}

// TestStore_ChunkNotFoundError_Unwrap_Bad calls Unwrap on a nil receiver —
// the method never dereferences its fields, so it must still return the
// sentinel rather than panicking.
func TestStore_ChunkNotFoundError_Unwrap_Bad(t *testing.T) {
	var err *ChunkNotFoundError
	if unwrapped := err.Unwrap(); unwrapped != ErrChunkNotFound {
		t.Fatalf("Unwrap(nil receiver) = %v, want ErrChunkNotFound", unwrapped)
	}
}

// TestStore_ChunkNotFoundError_Unwrap_Ugly proves the Unwrap chain survives
// interface type-erasure — errors.Is (via core.Is) still walks through to
// the sentinel once the concrete *ChunkNotFoundError is boxed as `error`.
func TestStore_ChunkNotFoundError_Unwrap_Ugly(t *testing.T) {
	var boxed error = &ChunkNotFoundError{ID: 5}
	if !core.Is(boxed, ErrChunkNotFound) {
		t.Fatalf("core.Is(boxed ChunkNotFoundError, ErrChunkNotFound) = false, want true")
	}
}

// --- URIChunkNotFoundError ---------------------------------------------------

func TestStore_URIChunkNotFoundError_Error_Good(t *testing.T) {
	err := &URIChunkNotFoundError{URI: "state://missing"}
	if got := err.Error(); got != `state chunk URI "state://missing" not found` {
		t.Fatalf("Error() = %q, want %q", got, `state chunk URI "state://missing" not found`)
	}
}

// TestStore_URIChunkNotFoundError_Error_Bad feeds the empty-URI branch,
// which formats a distinct message rather than an empty quoted string.
func TestStore_URIChunkNotFoundError_Error_Bad(t *testing.T) {
	err := &URIChunkNotFoundError{}
	if got := err.Error(); got != "state chunk URI not found" {
		t.Fatalf("Error() (empty URI) = %q, want %q", got, "state chunk URI not found")
	}
}

// TestStore_URIChunkNotFoundError_Error_Ugly feeds a URI containing a
// double quote, proving %q escapes it rather than corrupting the message.
func TestStore_URIChunkNotFoundError_Error_Ugly(t *testing.T) {
	err := &URIChunkNotFoundError{URI: `state://"quoted"`}
	want := `state chunk URI "state://\"quoted\"" not found`
	if got := err.Error(); got != want {
		t.Fatalf("Error() (quoted URI) = %q, want %q", got, want)
	}
}

func TestStore_URIChunkNotFoundError_Unwrap_Good(t *testing.T) {
	err := &URIChunkNotFoundError{URI: "state://x"}
	if unwrapped := err.Unwrap(); unwrapped != ErrChunkNotFound {
		t.Fatalf("Unwrap() = %v, want ErrChunkNotFound", unwrapped)
	}
}

// TestStore_URIChunkNotFoundError_Unwrap_Bad calls Unwrap on a nil
// receiver — no field dereference occurs, so it must still return the
// sentinel.
func TestStore_URIChunkNotFoundError_Unwrap_Bad(t *testing.T) {
	var err *URIChunkNotFoundError
	if unwrapped := err.Unwrap(); unwrapped != ErrChunkNotFound {
		t.Fatalf("Unwrap(nil receiver) = %v, want ErrChunkNotFound", unwrapped)
	}
}

// TestStore_URIChunkNotFoundError_Unwrap_Ugly proves the Unwrap chain
// survives interface type-erasure, mirroring ChunkNotFoundError's case.
func TestStore_URIChunkNotFoundError_Unwrap_Ugly(t *testing.T) {
	var boxed error = &URIChunkNotFoundError{URI: "state://x"}
	if !core.Is(boxed, ErrChunkNotFound) {
		t.Fatalf("core.Is(boxed URIChunkNotFoundError, ErrChunkNotFound) = false, want true")
	}
}
