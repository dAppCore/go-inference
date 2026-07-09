// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"bytes"
	"errors"
	"testing"
)

// TestBlocksWindowCover_LayerWindowLen_ErrorBranches drives the error and
// continue branches of kvSnapshotLayerWindowLen that the happy-path block
// callers never trip: a malformed layer-raw shape, mixed window lengths
// across the layer-raw K/V pair, and mixed lengths across head tensors.
func TestBlocksWindowCover_LayerWindowLen_ErrorBranches(t *testing.T) {
	// A layer-raw KeyBytes with a non-4D shape yields length -1 →
	// errLayerRawShapeMismatch (the `length < 0` guard at 16).
	bad := LayerSnapshot{
		KeyBytes: cvtRawF16(2, 2),
		KeyDType: "float16",
		KeyShape: []int32{2, 2}, // not 4-D → -1
	}
	if _, err := kvSnapshotLayerWindowLen(bad, 2, 2); !errors.Is(err, errLayerRawShapeMismatch) {
		t.Fatalf("layer-raw bad shape err = %v, want errLayerRawShapeMismatch", err)
	}

	// KeyBytes implies L=2, ValueBytes implies L=4 → the layer-raw pair
	// mixes window lengths → errLayerMixesWindowLens (the `windowLen != length`
	// guard at 26).
	mixedRaw := LayerSnapshot{
		KeyBytes:   cvtRawF16(2, 2),
		KeyDType:   "float16",
		KeyShape:   []int32{1, 1, 2, 2},
		ValueBytes: cvtRawF16(4, 2),
		ValueDType: "float16",
		ValueShape: []int32{1, 1, 4, 2},
	}
	if _, err := kvSnapshotLayerWindowLen(mixedRaw, 4, 2); !errors.Is(err, errLayerMixesWindowLens) {
		t.Fatalf("layer-raw mixed lens err = %v, want errLayerMixesWindowLens", err)
	}

	// Two heads with different sequence lengths → the head loop trips
	// errLayerMixesWindowLens (the `windowLen != length` guard at 47).
	mixedHeads := LayerSnapshot{
		Heads: []HeadSnapshot{
			{Key: cvtF32(2, 2)}, // seqLen 2
			{Key: cvtF32(3, 2)}, // seqLen 3
		},
	}
	if _, err := kvSnapshotLayerWindowLen(mixedHeads, 0, 2); !errors.Is(err, errLayerMixesWindowLens) {
		t.Fatalf("head mixed lens err = %v, want errLayerMixesWindowLens", err)
	}
}

// TestBlocksWindowCover_RawTensorWindowLen drives the -1 byte-length guard
// inside kvSnapshotRawTensorWindowLen (an odd byte count for an f16 dtype).
func TestBlocksWindowCover_RawTensorWindowLen(t *testing.T) {
	// 3 bytes is not a whole number of f16 values → -1.
	if got := kvSnapshotRawTensorWindowLen([]byte{1, 2, 3}, "float16", 2, 2); got != -1 {
		t.Fatalf("kvSnapshotRawTensorWindowLen(odd bytes) = %d, want -1", got)
	}
	// Empty raw → 0 (skipped branch).
	if got := kvSnapshotRawTensorWindowLen(nil, "float16", 2, 2); got != 0 {
		t.Fatalf("kvSnapshotRawTensorWindowLen(empty) = %d, want 0", got)
	}
}

// TestBlocksWindowCover_LayerRawWindowLen drives every -1 guard inside
// kvSnapshotLayerRawWindowLen: bad dtype/shape, a non-positive dim, a byte
// length that disagrees with the shape, and a seqLen smaller than shape[2].
func TestBlocksWindowCover_LayerRawWindowLen(t *testing.T) {
	raw := cvtRawF16(2, 2) // 4 f16 values → 8 bytes

	// Unsupported 2-D shape → -1.
	if got := kvSnapshotLayerRawWindowLen(raw, "float16", []int32{2, 2}, 2); got != -1 {
		t.Fatalf("layer-raw 2D shape = %d, want -1", got)
	}
	// A zero dimension → -1.
	if got := kvSnapshotLayerRawWindowLen(raw, "float16", []int32{1, 1, 0, 2}, 2); got != -1 {
		t.Fatalf("layer-raw zero dim = %d, want -1", got)
	}
	// Byte length disagrees with the shape's element count → -1.
	if got := kvSnapshotLayerRawWindowLen(raw, "float16", []int32{1, 1, 4, 2}, 4); got != -1 {
		t.Fatalf("layer-raw byte mismatch = %d, want -1", got)
	}
	// shape[2] (L=2) exceeds seqLen (1) → -1.
	if got := kvSnapshotLayerRawWindowLen(raw, "float16", []int32{1, 1, 2, 2}, 1); got != -1 {
		t.Fatalf("layer-raw L>seqLen = %d, want -1", got)
	}
}

func TestBlocksWindowCover_TokenMajorLayerRaw3D_Good(t *testing.T) {
	raw := []byte{
		1, 0, 2, 0,
		3, 0, 4, 0,
		5, 0, 6, 0,
	}
	shape := []int32{3, 2, 1}
	if got := kvSnapshotLayerRawWindowLen(raw, "bfloat16", shape, 3); got != 3 {
		t.Fatalf("token-major layer window len = %d, want 3", got)
	}
	sliced, slicedShape, err := sliceKVSnapshotLayerRawTensorOpt(raw, "bfloat16", shape, 1, 3, false)
	if err != nil {
		t.Fatalf("slice token-major layer raw tensor: %v", err)
	}
	if !bytes.Equal(sliced, raw[4:12]) {
		t.Fatalf("token-major slice = %v, want %v", sliced, raw[4:12])
	}
	if len(slicedShape) != 3 || slicedShape[0] != 2 || slicedShape[1] != 2 || slicedShape[2] != 1 {
		t.Fatalf("token-major slice shape = %v, want [2 2 1]", slicedShape)
	}
	if len(sliced) > 0 && &sliced[0] != &raw[4] {
		t.Fatal("token-major no-clone slice copied bytes; want borrowed contiguous range")
	}
}

// TestBlocksWindowCover_SliceTensorOpt_ErrorBranches trips the two error
// branches of sliceKVSnapshotTensorOpt the wrappers don't reach: a value
// count not divisible by seqLen, and an inverted [start,end) range.
func TestBlocksWindowCover_SliceTensorOpt_ErrorBranches(t *testing.T) {
	values := cvtF32(4, 2) // 8 values

	// headDim 0 forces the `len(values)%seqLen` path; 8 % 3 != 0 → error.
	if _, err := sliceKVSnapshotTensorOpt(values, 0, 1, 0, 3, false); !errors.Is(err, errTensorShapeSeqHead) {
		t.Fatalf("slice tensor non-divisible seqLen err = %v, want errTensorShapeSeqHead", err)
	}
	// begin >= finish (start == end) → range invalid.
	if _, err := sliceKVSnapshotTensorOpt(values, 2, 2, 2, 4, false); !errors.Is(err, errTensorBlockRangeInvalid) {
		t.Fatalf("slice tensor inverted range err = %v, want errTensorBlockRangeInvalid", err)
	}
}

// TestBlocksWindowCover_SliceRawTensorOpt_ErrorBranches trips the two error
// branches of sliceKVSnapshotRawTensorOpt: an odd byte length with an
// inferred valueCount, and a raw length that disagrees with the shape.
func TestBlocksWindowCover_SliceRawTensorOpt_ErrorBranches(t *testing.T) {
	// valueCount 0 + odd byte length (3 bytes, f16) → byte-len invalid.
	if _, err := sliceKVSnapshotRawTensorOpt([]byte{1, 2, 3}, "float16", 0, 1, 2, 0, false); !errors.Is(err, errRawTensorByteLenInvalid) {
		t.Fatalf("slice raw odd bytes err = %v, want errRawTensorByteLenInvalid", err)
	}
	// valueCount 4 but seqLen 3 → 4 % 3 != 0 → raw shape-seq error.
	raw := cvtRawF16(2, 2) // 8 bytes, 4 f16 values
	if _, err := sliceKVSnapshotRawTensorOpt(raw, "float16", 0, 1, 3, 4, false); !errors.Is(err, errRawTensorShapeSeq) {
		t.Fatalf("slice raw shape-seq err = %v, want errRawTensorShapeSeq", err)
	}
}

// TestBlocksWindowCover_SliceLayerRawTensorOpt_ErrorBranches trips the two
// error branches of sliceKVSnapshotLayerRawTensorOpt: an out-of-bounds
// sequence range, and a byte length that disagrees with the [B,H,L,D] shape.
func TestBlocksWindowCover_SliceLayerRawTensorOpt_ErrorBranches(t *testing.T) {
	raw := cvtRawF16(4, 2) // 8 f16 values → 16 bytes
	shape := []int32{1, 1, 4, 2}

	// end (5) > L (4) → range invalid.
	if _, _, err := sliceKVSnapshotLayerRawTensorOpt(raw, "float16", shape, 0, 5, false); !errors.Is(err, errLayerRawTensorRangeInvalid) {
		t.Fatalf("layer-raw out-of-range err = %v, want errLayerRawTensorRangeInvalid", err)
	}
	// Shape claims L=8 (32 bytes) but raw is only 16 → byte-len mismatch.
	if _, _, err := sliceKVSnapshotLayerRawTensorOpt(raw, "float16", []int32{1, 1, 8, 2}, 0, 1, false); !errors.Is(err, errLayerRawByteLenMismatch) {
		t.Fatalf("layer-raw byte mismatch err = %v, want errLayerRawByteLenMismatch", err)
	}
}
