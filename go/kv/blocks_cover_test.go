// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"errors"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// TestBlocksCover_RangeBlocks_NilYield drives the nil-yield guard of
// RangeBlocks.
func TestBlocksCover_RangeBlocks_NilYield(t *testing.T) {
	if err := kvSnapshotBlocksTestSnapshot().RangeBlocks(2, nil); !errors.Is(err, errBlockYieldNil) {
		t.Fatalf("RangeBlocks(nil yield) error = %v, want errBlockYieldNil", err)
	}
}

// TestBlocksCover_BoundaryInsert drives kvBoundaryInsert's three arms: an
// insert before an existing boundary, a dedupe of an existing value, and an
// append past the end.
func TestBlocksCover_BoundaryInsert(t *testing.T) {
	base := []int{0, 4, 8}

	// Insert 2 → lands between 0 and 4.
	got := kvBoundaryInsert(append([]int(nil), base...), 2)
	if len(got) != 4 || got[1] != 2 {
		t.Fatalf("insert middle = %v, want 2 spliced in", got)
	}
	// Dedupe 4 → unchanged.
	got = kvBoundaryInsert(append([]int(nil), base...), 4)
	if len(got) != 3 {
		t.Fatalf("dedupe = %v, want unchanged length", got)
	}
	// Append 9 → past the end.
	got = kvBoundaryInsert(append([]int(nil), base...), 9)
	if len(got) != 4 || got[3] != 9 {
		t.Fatalf("append = %v, want 9 at end", got)
	}
}

// TestBlocksCover_BlockPayloadSlices drives the empty-input early return of
// kvBlockPayloadSlices and a clone vs share comparison.
func TestBlocksCover_BlockPayloadSlices(t *testing.T) {
	if got := kvBlockPayloadSlices(nil, true); got != nil {
		t.Fatalf("kvBlockPayloadSlices(nil) = %v, want nil", got)
	}
	src := [][]byte{{1, 2}, {3, 4}}
	cloned := kvBlockPayloadSlices(src, true)
	cloned[0][0] = 9
	if src[0][0] == 9 {
		t.Fatal("kvBlockPayloadSlices(clone) shared backing array")
	}
	shared := kvBlockPayloadSlices(src, false)
	if &shared[0][0] != &src[0][0] {
		t.Fatal("kvBlockPayloadSlices(share) cloned instead of sharing")
	}
}

// TestBlocksCover_DecodeEnvelope_Errors drives every validation error arm of
// decodeKVSnapshotStateBlockEnvelope by constructing envelopes directly.
func TestBlocksCover_DecodeEnvelope_Errors(t *testing.T) {
	good := func() kvSnapshotStateBlockEnvelope {
		payload := []byte{1, 2, 3, 4}
		return kvSnapshotStateBlockEnvelope{
			Version:          StateBlockVersion,
			Kind:             KVSnapshotStateBlockKind,
			BinaryEncoding:   "base64",
			PayloadByteCount: len(payload),
			KVHash:           core.SHA256Hex(payload),
			Data:             core.Base64Encode(payload),
		}
	}

	// Bad version.
	e := good()
	e.Version = StateBlockVersion + 1
	if _, err := decodeKVSnapshotStateBlockEnvelope(e, ""); !errors.Is(err, errUnsupportedBlockVersion) {
		t.Fatalf("decode(bad version) error = %v, want errUnsupportedBlockVersion", err)
	}
	// Bad kind.
	e = good()
	e.Kind = "not-a-block"
	if _, err := decodeKVSnapshotStateBlockEnvelope(e, ""); !errors.Is(err, errBlockKindInvalid) {
		t.Fatalf("decode(bad kind) error = %v, want errBlockKindInvalid", err)
	}
	// Bad binary encoding.
	e = good()
	e.BinaryEncoding = "hex"
	if _, err := decodeKVSnapshotStateBlockEnvelope(e, ""); !errors.Is(err, errUnsupportedBlockEncoding) {
		t.Fatalf("decode(bad encoding) error = %v, want errUnsupportedBlockEncoding", err)
	}
	// Payload byte-count mismatch.
	e = good()
	e.PayloadByteCount = 999
	if _, err := decodeKVSnapshotStateBlockEnvelope(e, ""); !errors.Is(err, errBlockPayloadLenMismatch) {
		t.Fatalf("decode(byte mismatch) error = %v, want errBlockPayloadLenMismatch", err)
	}
	// Stored-hash mismatch.
	e = good()
	e.KVHash = "deadbeef"
	if _, err := decodeKVSnapshotStateBlockEnvelope(e, ""); !errors.Is(err, errBlockHashMismatch) {
		t.Fatalf("decode(hash mismatch) error = %v, want errBlockHashMismatch", err)
	}
	// Expected-hash (caller-supplied) mismatch.
	e = good()
	if _, err := decodeKVSnapshotStateBlockEnvelope(e, "deadbeef"); err == nil {
		t.Fatal("decode(expected hash mismatch) error = nil, want hash error")
	}
}

// TestBlocksCover_RawBlockPayload_Errors drives loadRawStateBlockPayload's
// payload-length and hash mismatch arms by mutating the ref of a real raw
// block so its declared byte count / hash no longer match the stored bytes.
func TestBlocksCover_RawBlockPayload_Errors(t *testing.T) {
	ctx := context.Background()
	store, bundle := kvSnapshotBlocksTestBundle(t)

	// Byte-count mismatch.
	lenRef := bundle.Blocks[0]
	lenRef.PayloadByteCount = 999999
	if _, err := LoadStateBlockWithOptions(ctx, store, lenRef, LoadOptions{}); !errors.Is(err, errRawBlockPayloadLenMismatch) {
		t.Fatalf("load(byte mismatch) error = %v, want errRawBlockPayloadLenMismatch", err)
	}

	// Hash mismatch.
	hashRef := bundle.Blocks[0]
	hashRef.KVHash = "deadbeefdeadbeef"
	if _, err := LoadStateBlockWithOptions(ctx, store, hashRef, LoadOptions{}); !errors.Is(err, errRawBlockHashMismatch) {
		t.Fatalf("load(hash mismatch) error = %v, want errRawBlockHashMismatch", err)
	}

	// Same two arms through the token-only raw loader.
	if _, err := LoadStateBlockTokensWithOptions(ctx, store, lenRef, LoadOptions{}); !errors.Is(err, errRawBlockPayloadLenMismatch) {
		t.Fatalf("token load(byte mismatch) error = %v, want errRawBlockPayloadLenMismatch", err)
	}
}

// TestBlocksCover_JSONBlock_LoadAndErrors drives the JSON-base64 block load
// path (the branch taken when PayloadEncoding is not "raw"): a clean round trip
// plus the resolve-failure and parse-failure arms.
func TestBlocksCover_JSONBlock_LoadAndErrors(t *testing.T) {
	ctx := context.Background()
	// textOnlyStateStore implements only Put + Resolve, so SaveStateBlocks
	// falls back to the JSON-base64 envelope payload encoding.
	store := &textOnlyStateStore{store: state.NewInMemoryStore(nil)}
	bundle, err := kvSnapshotBlocksTestSnapshot().SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://json-blocks",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(json) error = %v", err)
	}
	if bundle.Blocks[0].PayloadEncoding == kvSnapshotStatePayloadRaw {
		t.Fatalf("expected JSON payload encoding, got %q", bundle.Blocks[0].PayloadEncoding)
	}

	// Clean full-block load through the JSON path.
	block, err := LoadStateBlockWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{})
	if err != nil || block.Snapshot == nil {
		t.Fatalf("LoadStateBlockWithOptions(json) = %+v, err = %v", block, err)
	}
	// Token-only load through the JSON path.
	if _, err := LoadStateBlockTokensWithOptions(ctx, store, bundle.Blocks[0], LoadOptions{}); err != nil {
		t.Fatalf("LoadStateBlockTokensWithOptions(json) error = %v", err)
	}

	// Resolve failure: a ref whose chunk ID does not exist in the store.
	missing := bundle.Blocks[0]
	missing.PayloadEncoding = "" // force the resolve path
	missing.State = state.ChunkRef{ChunkID: 999999}
	missing.Memvid = state.ChunkRef{ChunkID: 999999}
	if _, err := LoadStateBlockWithOptions(ctx, store, missing, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockWithOptions(missing chunk) error = nil, want resolve error")
	}
	if _, err := LoadStateBlockTokensWithOptions(ctx, store, missing, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockTokensWithOptions(missing chunk) error = nil, want resolve error")
	}
}

// TestBlocksCover_JSONBlock_EnvelopeErrors drives the JSON-envelope parse and
// decode error arms of LoadStateBlockWithOptions / LoadStateBlockTokensWithOptions
// by storing a chunk whose text is malformed, then pointing a non-raw ref at it.
func TestBlocksCover_JSONBlock_EnvelopeErrors(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	// A chunk of non-JSON text → the envelope parse fails.
	garbageRef, err := store.Put(ctx, "not json at all", state.PutOptions{URI: "mlx://garbage"})
	if err != nil {
		t.Fatalf("Put(garbage) error = %v", err)
	}
	parseRef := StateBlockRef{Index: 0, TokenStart: 0, TokenCount: 2, State: garbageRef}
	if _, err := LoadStateBlockWithOptions(ctx, store, parseRef, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockWithOptions(garbage envelope) error = nil, want parse error")
	}
	if _, err := LoadStateBlockTokensWithOptions(ctx, store, parseRef, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockTokensWithOptions(garbage envelope) error = nil, want parse error")
	}

	// Valid JSON but an envelope that fails decode validation (bad version).
	badEnvelope := core.JSONMarshalString(kvSnapshotStateBlockEnvelope{
		Version:        StateBlockVersion + 1,
		Kind:           KVSnapshotStateBlockKind,
		BinaryEncoding: "base64",
	})
	decodeChunk, err := store.Put(ctx, badEnvelope, state.PutOptions{URI: "mlx://bad-envelope"})
	if err != nil {
		t.Fatalf("Put(bad envelope) error = %v", err)
	}
	decodeRef := StateBlockRef{Index: 0, TokenStart: 0, TokenCount: 2, State: decodeChunk}
	if _, err := LoadStateBlockWithOptions(ctx, store, decodeRef, LoadOptions{}); !errors.Is(err, errUnsupportedBlockVersion) {
		t.Fatalf("LoadStateBlockWithOptions(bad version envelope) error = %v, want errUnsupportedBlockVersion", err)
	}
	if _, err := LoadStateBlockTokensWithOptions(ctx, store, decodeRef, LoadOptions{}); !errors.Is(err, errUnsupportedBlockVersion) {
		t.Fatalf("LoadStateBlockTokensWithOptions(bad version envelope) error = %v, want errUnsupportedBlockVersion", err)
	}
}

// TestBlocksCover_SliceBlock_TensorShapeError drives the layer-window error arm
// of sliceBlockInternal: a head whose Key length is inconsistent with the
// snapshot's sequence length, which the window validator rejects.
func TestBlocksCover_SliceBlock_TensorShapeError(t *testing.T) {
	bad := &Snapshot{
		Architecture: "gemma4_text",
		Tokens:       []int32{1, 2},
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       2,
		HeadDim:      2,
		Layers: []LayerSnapshot{{
			Heads: []HeadSnapshot{{
				// 3 values cannot tile a seqLen-2 / headDim-2 head → window error.
				Key: []float32{1, 2, 3},
			}},
		}},
	}
	if _, err := bad.SliceBlock(0, 1, 0, false); err == nil {
		t.Fatal("SliceBlock(malformed head) error = nil, want shape error")
	}
}
