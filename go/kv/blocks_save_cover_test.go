// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	stdio "io"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// failingBinaryStore implements BinaryWriter (PutBytes) and Writer (Put) and
// fails both — SaveStateBlocks reaches PutBytes (the BinaryWriter branch of
// saveKVSnapshotStateBlock) and the error propagates back up the walk.
type failingBinaryStore struct{}

func (failingBinaryStore) Get(context.Context, int) (string, error) {
	return "", core.NewError("get refused")
}

func (failingBinaryStore) Put(context.Context, string, state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, core.NewError("put refused")
}

func (failingBinaryStore) PutBytes(context.Context, []byte, state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, core.NewError("put bytes refused")
}

// failingPlainStore implements only Writer (Put) and fails it — the JSON-base64
// fallback branch of saveKVSnapshotStateBlock, and SaveStateBlockBundle's write.
type failingPlainStore struct{}

func (failingPlainStore) Get(context.Context, int) (string, error) {
	return "", core.NewError("get refused")
}

func (failingPlainStore) Put(context.Context, string, state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, core.NewError("put refused")
}

// TestBlocksSaveCover_NilContextDefaults drives the ctx == nil → Background()
// fallbacks of SaveStateBlocks, SaveStateBlocksFromStream and
// SaveStateBlockBundle.
func TestBlocksSaveCover_NilContextDefaults(t *testing.T) {
	store := state.NewInMemoryStore(nil)

	if _, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(nil, store, StateBlockOptions{BlockSize: 2}); err != nil { //nolint:staticcheck
		t.Fatalf("SaveStateBlocks(nil ctx) error = %v", err)
	}
	bundle, err := SaveStateBlocksFromStream(nil, store, StateBlockOptions{BlockSize: 2}, func(yield func(Block) (bool, error)) error { //nolint:staticcheck
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()})
		return err
	})
	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream(nil ctx) error = %v", err)
	}
	if _, err := SaveStateBlockBundle(nil, store, bundle, "mlx://nilctx-bundle"); err != nil { //nolint:staticcheck
		t.Fatalf("SaveStateBlockBundle(nil ctx) error = %v", err)
	}
}

// TestBlocksSaveCover_FailingStores drives the store-write error arms: a
// BinaryWriter that fails PutBytes (SaveStateBlocks + FromStream), a plain
// Writer that fails Put (JSON-base64 save path), and SaveStateBlockBundle's
// bundle-write failure.
func TestBlocksSaveCover_FailingStores(t *testing.T) {
	ctx := context.Background()
	snapshot := kvSnapshotBlocksTestSnapshot()

	// BinaryWriter PutBytes failure through SaveStateBlocks.
	if _, err := snapshot.SaveStateBlocks(ctx, failingBinaryStore{}, StateBlockOptions{BlockSize: 2}); err == nil {
		t.Fatal("SaveStateBlocks(failing binary store) error = nil, want write error")
	}
	// And through the stream entry point.
	_, err := SaveStateBlocksFromStream(ctx, failingBinaryStore{}, StateBlockOptions{BlockSize: 2}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()})
		return err
	})
	if err == nil {
		t.Fatal("SaveStateBlocksFromStream(failing binary store) error = nil, want write error")
	}

	// Plain Writer Put failure through the JSON-base64 save path.
	if _, err := snapshot.SaveStateBlocks(ctx, failingPlainStore{}, StateBlockOptions{BlockSize: 2}); err == nil {
		t.Fatal("SaveStateBlocks(failing plain store) error = nil, want write error")
	}

	// SaveStateBlockBundle write failure.
	_, bundle := kvSnapshotBlocksTestBundle(t)
	if _, err := SaveStateBlockBundle(ctx, failingPlainStore{}, bundle, "mlx://fail-bundle"); err == nil {
		t.Fatal("SaveStateBlockBundle(failing store) error = nil, want write error")
	}
}

// TestBlocksSaveCover_DirectHelpers drives the directly-callable helper guards
// that the save paths never trip with valid input: applyKVSnapshotStateBundleBlock
// with a nil bundle/snapshot, kvSnapshotStateBlockBundleHash(nil), and
// hashStateBlockPayload with a nil snapshot and with a bad encoding.
func TestBlocksSaveCover_DirectHelpers(t *testing.T) {
	// applyKVSnapshotStateBundleBlock early returns on nil bundle / snapshot.
	applyKVSnapshotStateBundleBlock(nil, Block{})
	applyKVSnapshotStateBundleBlock(&StateBlockBundle{}, Block{Snapshot: nil})

	// kvSnapshotStateBlockBundleHash(nil) returns "".
	if got := kvSnapshotStateBlockBundleHash(nil); got != "" {
		t.Fatalf("kvSnapshotStateBlockBundleHash(nil) = %q, want empty", got)
	}

	// hashStateBlockPayload with a nil snapshot.
	if _, err := hashStateBlockPayload(Block{Snapshot: nil}, KVSnapshotEncodingFloat32); err == nil {
		t.Fatal("hashStateBlockPayload(nil snapshot) error = nil, want block error")
	}
	// hashStateBlockPayload where writeWithOptions fails (raw-only head under a
	// non-native encoding → errRawTensorNeedsNative).
	rawOnly := testSnapshot()
	rawOnly.Layers = []LayerSnapshot{{Heads: []HeadSnapshot{{KeyBytes: cvtRawF16(2, 2), KeyDType: "float16"}}}}
	if _, err := hashStateBlockPayload(Block{Snapshot: rawOnly}, EncodingQ8); err == nil {
		t.Fatal("hashStateBlockPayload(write failure) error = nil, want encode error")
	}
}

// TestBlocksSaveCover_ReuseEncodingMismatch drives the parent-encoding mismatch
// arm of reusableKVSnapshotStateBlockRef: a child save whose encoding differs
// from the parent bundle's, so the prefix is not reused.
func TestBlocksSaveCover_ReuseEncodingMismatch(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent := kvSnapshotBlocksTestSnapshot()
	parentBundle, err := parent.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://reuse-enc-parent",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}

	child := kvSnapshotBlocksTestSnapshot()
	// Child uses Q8 while the parent recorded Native → encoding mismatch path.
	childBundle, err := child.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:         2,
		KVEncoding:        EncodingQ8,
		URI:               "mlx://reuse-enc-child",
		ReusePrefix:       parentBundle,
		ReusePrefixTokens: 2,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(child mismatch) error = %v", err)
	}
	if childBundle.ReusedBlocks != 0 {
		t.Fatalf("child reused blocks = %d, want 0 (encoding mismatch)", childBundle.ReusedBlocks)
	}
}

// TestBlocksSaveCover_TrustedReuseMatch drives the ReusePrefixTrusted match
// loop of reusableKVSnapshotStateBlockRef via SaveStateBlocks: a trusted parent
// whose first block is adopted by range alone (no re-hash).
func TestBlocksSaveCover_TrustedReuseMatch(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent := kvSnapshotBlocksTestSnapshot()
	parentBundle, err := parent.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://trusted-parent",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}

	child := kvSnapshotBlocksTestSnapshot()
	child.Tokens[2] = 9
	child.Tokens[3] = 10
	childBundle, err := child.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:          2,
		KVEncoding:         EncodingNative,
		URI:                "mlx://trusted-child",
		ReusePrefix:        parentBundle,
		ReusePrefixTokens:  2,
		ReusePrefixTrusted: true,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(trusted child) error = %v", err)
	}
	if childBundle.ReusedBlocks != 1 {
		t.Fatalf("trusted child reused blocks = %d, want 1", childBundle.ReusedBlocks)
	}
}

// TestBlocksSaveCover_StreamBadEncoding drives the bad-encoding guard of
// SaveStateBlocksFromStream.
func TestBlocksSaveCover_StreamBadEncoding(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	_, err := SaveStateBlocksFromStream(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: "q2"}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 4, Snapshot: kvSnapshotBlocksTestSnapshot()})
		return err
	})
	if err == nil {
		t.Fatal("SaveStateBlocksFromStream(bad encoding) error = nil, want encoding error")
	}
}

// TestBlocksSaveCover_BinaryEncodeError drives the bytesWithOptions error arm of
// saveKVSnapshotStateBlock's BinaryWriter branch: a raw-only head under a
// non-native encoding fails to serialise before the store write. InMemoryStore
// implements BinaryWriter (PutBytes) but not the stream interface, so this
// exercises the non-stream binary path's encode error.
func TestBlocksSaveCover_BinaryEncodeError(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	rawOnly := testSnapshot()
	rawOnly.SeqLen = 2
	rawOnly.Layers = []LayerSnapshot{{Heads: []HeadSnapshot{{KeyBytes: cvtRawF16(2, 2), KeyDType: "float16"}}}}
	if _, err := rawOnly.SaveStateBlocks(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8}); err == nil {
		t.Fatal("SaveStateBlocks(raw-only head, Q8) error = nil, want encode error")
	}
}

// TestBlocksSaveCover_UntrustedReuseHashError drives the hash-error arm of
// reusableKVSnapshotStateBlockRef (and its propagation through
// saveOrReuseKVSnapshotStateBlock) plus the non-matching-range continue: an
// untrusted parent whose block range overlaps but whose child block fails to
// hash under a raw-only / non-native encoding.
func TestBlocksSaveCover_UntrustedReuseHashError(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	// A parent bundle with a single in-range block, recorded under Native.
	parent := kvSnapshotBlocksTestSnapshot()
	parentBundle, err := parent.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://untrusted-parent",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}

	// A child whose first block carries a raw-only head: under Native reuse the
	// hash walk (hashStateBlockPayload) runs and succeeds (Native passes raw),
	// so instead force a hash failure by reusing under a Q8-declared parent.
	q8Parent := &StateBlockBundle{
		Version:    StateBlockVersion,
		Kind:       StateBlockBundleKind,
		KVEncoding: EncodingQ8,
		TokenCount: parentBundle.TokenCount,
		BlockSize:  2,
		Blocks:     parentBundle.Blocks,
	}
	child := testSnapshot()
	child.SeqLen = 2
	child.Tokens = []int32{1, 2}
	child.Layers = []LayerSnapshot{{Heads: []HeadSnapshot{{KeyBytes: cvtRawF16(2, 2), KeyDType: "float16"}}}}
	if _, err := child.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:         2,
		KVEncoding:        EncodingQ8,
		URI:               "mlx://untrusted-child",
		ReusePrefix:       q8Parent,
		ReusePrefixTokens: 2,
	}); err == nil {
		t.Fatal("SaveStateBlocks(untrusted reuse hash error) error = nil, want encode error")
	}
}

// TestBlocksSaveCover_TrustedReuseNonMatchingRange drives the range-mismatch
// continue inside the trusted-reuse loop: a trusted parent whose only block
// covers a different token range than the child block, so no graft happens.
func TestBlocksSaveCover_TrustedReuseNonMatchingRange(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	// Parent with blocks at [0,2) and [2,4).
	parent := kvSnapshotBlocksTestSnapshot()
	parentBundle, err := parent.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://trusted-range-parent",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}

	// Synthesise a trusted parent whose recorded block ranges do not line up
	// with a block-size-2 child (shift every TokenStart by 1) so the trusted
	// match loop iterates and `continue`s without finding a range match.
	shifted := &StateBlockBundle{
		Version:    StateBlockVersion,
		Kind:       StateBlockBundleKind,
		KVEncoding: EncodingNative,
		TokenCount: parentBundle.TokenCount,
		BlockSize:  2,
		Blocks:     append([]StateBlockRef(nil), parentBundle.Blocks...),
	}
	shifted.Blocks[0].TokenStart = 1 // no longer matches a [0,2) child block

	child := kvSnapshotBlocksTestSnapshot()
	if _, err := child.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:          2,
		KVEncoding:         EncodingNative,
		URI:                "mlx://trusted-range-child",
		ReusePrefix:        shifted,
		ReusePrefixTokens:  2,
		ReusePrefixTrusted: true,
	}); err != nil {
		t.Fatalf("SaveStateBlocks(trusted non-matching range) error = %v", err)
	}
}

// streamFailEncodeStore implements BinaryStreamWriter but its write callback is
// driven against a snapshot that fails to encode, so the stream save path's
// size/encode error arm fires.
type streamFailEncodeStore struct {
	store *state.InMemoryStore
}

func (s streamFailEncodeStore) Get(ctx context.Context, id int) (string, error) {
	return s.store.Get(ctx, id)
}

func (s streamFailEncodeStore) Put(ctx context.Context, text string, opts state.PutOptions) (state.ChunkRef, error) {
	return s.store.Put(ctx, text, opts)
}

func (s streamFailEncodeStore) PutBytesStream(ctx context.Context, size int, opts state.PutOptions, write func(stdio.Writer) error) (state.ChunkRef, error) {
	writer := &streamRecordingWriter{data: make([]byte, 0, size)}
	if err := write(writer); err != nil {
		return state.ChunkRef{}, err
	}
	return s.store.PutBytes(ctx, writer.data, opts)
}

// TestBlocksSaveCover_StreamEncodeError drives the encode/size error arm of the
// stream save path (saveKVSnapshotStateBlock's BinaryStreamWriter branch): a
// raw-only head under a non-native encoding fails encodedSizeWithOptions.
func TestBlocksSaveCover_StreamEncodeError(t *testing.T) {
	ctx := context.Background()
	store := streamFailEncodeStore{store: state.NewInMemoryStore(nil)}

	rawOnly := testSnapshot()
	rawOnly.SeqLen = 2
	rawOnly.Layers = []LayerSnapshot{{Heads: []HeadSnapshot{{KeyBytes: cvtRawF16(2, 2), KeyDType: "float16"}}}}
	_, err := SaveStateBlocksFromStream(ctx, store, StateBlockOptions{BlockSize: 2, KVEncoding: EncodingQ8}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: rawOnly})
		return err
	})
	if err == nil {
		t.Fatal("SaveStateBlocksFromStream(raw-only Q8 stream) error = nil, want encode error")
	}
}
