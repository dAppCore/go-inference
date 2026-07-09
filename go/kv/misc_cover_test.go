// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"errors"
	"testing"

	state "dappco.re/go/inference/model/state"
)

// TestMiscCover_LayerLookupZeroFallback drives the `Layer == 0` positional
// fallback of (*Snapshot).layer: a snapshot whose layers all carry Layer 0 but
// sit at different positions, so a lookup by a non-zero index misses the exact
// and scan matches and lands on the positional zero fallback. Driven through
// Analyze, which looks up each layer by index.
func TestMiscCover_LayerLookupZeroFallback(t *testing.T) {
	head := []float32{1, 0, 1, 0}
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "test",
		Tokens:        make([]int32, 2),
		NumLayers:     2,
		NumHeads:      2,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{
			// Both layers carry Layer 0 (only CacheIndex differs) so layer(1)
			// falls through to the positional Layer == 0 fallback.
			{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{
				{Key: append([]float32(nil), head...), Value: append([]float32(nil), head...)},
				{Key: append([]float32(nil), head...), Value: append([]float32(nil), head...)},
			}},
			{Layer: 0, CacheIndex: 1, Heads: []HeadSnapshot{
				{Key: append([]float32(nil), head...), Value: append([]float32(nil), head...)},
				{Key: append([]float32(nil), head...), Value: append([]float32(nil), head...)},
			}},
		},
	}
	if result := Analyze(snapshot); result == nil {
		t.Fatal("Analyze(zero-layer fallback) = nil")
	}
}

// TestMiscCover_DirectNilGuards drives the directly-callable nil guards:
// kvSharedCacheLayerGroups(nil) and preSizeAssembledRawBytesFromFirst with a
// non-positive block count.
func TestMiscCover_DirectNilGuards(t *testing.T) {
	if got := kvSharedCacheLayerGroups(nil); got == nil {
		t.Fatal("kvSharedCacheLayerGroups(nil) = nil, want empty map")
	}
	// blockCount <= 0 → early return without touching assembled/first.
	preSizeAssembledRawBytesFromFirst(&Snapshot{}, &Snapshot{}, 0)
	preSizeAssembledRawBytesFromFirst(nil, nil, 5)
}

// TestMiscCover_PreSizeFromFirst_BoundsGuards drives the layerIndex / headIndex
// bounds `continue` guards of preSizeAssembledRawBytesFromFirst by calling it
// with an assembled skeleton wider (more layers, more heads) than the first
// block, so the per-layer and per-head index checks skip the missing entries.
func TestMiscCover_PreSizeFromFirst_BoundsGuards(t *testing.T) {
	// assembled has two layers; the first layer has two heads.
	assembled := &Snapshot{Layers: []LayerSnapshot{
		{Heads: make([]HeadSnapshot, 2)},
		{Heads: make([]HeadSnapshot, 1)},
	}}
	// first has only one layer carrying one head → assembled's second layer and
	// the first layer's second head exceed first's bounds.
	first := &Snapshot{Layers: []LayerSnapshot{
		{Heads: []HeadSnapshot{{KeyBytes: cvtRawF16(1, 2), ValueBytes: cvtRawF16(1, 2), Key: []float32{1, 2}, Value: []float32{3, 4}}}},
	}}
	preSizeAssembledRawBytesFromFirst(assembled, first, 2)
}

// TestMiscCover_QuantizeQ8_LowerClamp drives the lower (-127) clamp of
// quantizeKVSnapshotQ8WithMaxAbs by supplying a maxAbs smaller than the actual
// value magnitude — the helper takes maxAbs as a parameter, so an undersized
// scale pushes the quantised value past -127.
func TestMiscCover_QuantizeQ8_LowerClamp(t *testing.T) {
	// maxAbs 1 → scale 1/127; -2/scale ≈ -254, clamped up to -127.
	scale, quantized := quantizeKVSnapshotQ8WithMaxAbs([]float32{-2}, 1)
	if scale <= 0 {
		t.Fatalf("scale = %v, want > 0", scale)
	}
	if int8(quantized[0]) != -127 {
		t.Fatalf("clamped value = %d, want -127", int8(quantized[0]))
	}
}

// TestMiscCover_PairCoherence_NoPairs drives the pairs == 0 arm of
// kvAnalysisPairCoherence: a single vector yields no i<j pair, so the mean is
// returned as zero.
func TestMiscCover_PairCoherence_NoPairs(t *testing.T) {
	mean, locked, pairs := kvAnalysisPairCoherence([][]float32{{1, 2}}, nil)
	if pairs != 0 || locked != 0 || mean != 0 {
		t.Fatalf("kvAnalysisPairCoherence(single vector) = (%v, %d, %d), want (0, 0, 0)", mean, locked, pairs)
	}
}

// TestMiscCover_RawTokenHashMismatch drives the raw-path load error arm of
// LoadStateBlockTokensWithOptions: a raw token-block ref whose declared KVHash
// disagrees with the stored payload.
func TestMiscCover_RawTokenHashMismatch(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	payload, err := kvSnapshotBlocksTestSnapshot().MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary() error = %v", err)
	}
	chunk, err := store.PutBytes(ctx, payload, state.PutOptions{URI: "mlx://raw-token-hash"})
	if err != nil {
		t.Fatalf("PutBytes error = %v", err)
	}
	ref := StateBlockRef{
		Index: 0, TokenStart: 0, TokenCount: 4,
		PayloadEncoding:  kvSnapshotStatePayloadRaw,
		PayloadByteCount: len(payload),
		KVHash:           "deadbeefdeadbeef", // disagrees with the stored bytes
		State:            chunk,
	}
	if _, err := LoadStateBlockTokensWithOptions(ctx, store, ref, LoadOptions{}); !errors.Is(err, errRawBlockHashMismatch) {
		t.Fatalf("LoadStateBlockTokensWithOptions(raw hash mismatch) error = %v, want errRawBlockHashMismatch", err)
	}
}

// TestMiscCover_SliceRawTensorOpt_RangeInvalid drives the range-invalid arm of
// sliceKVSnapshotRawTensorOpt on the inferred-valueCount path: a valid f16
// payload with an inverted [start,end) range.
func TestMiscCover_SliceRawTensorOpt_RangeInvalid(t *testing.T) {
	raw := cvtRawF16(4, 2) // 8 f16 values → 16 bytes, valueCount inferred to 8
	// seqLen 4, start == end → begin >= finish → range invalid.
	if _, err := sliceKVSnapshotRawTensorOpt(raw, "float16", 2, 2, 4, 0, false); !errors.Is(err, errRawTensorBlockRangeInvalid) {
		t.Fatalf("sliceKVSnapshotRawTensorOpt(inverted range) error = %v, want errRawTensorBlockRangeInvalid", err)
	}
}

// TestMiscCover_SliceBlock_HeadSlabFallback drives the per-layer head-slab
// fallback (make path) of sliceBlockInternal: a layer carrying more heads than
// NumHeads exhausts the slab sized to NumHeads.
func TestMiscCover_SliceBlock_HeadSlabFallback(t *testing.T) {
	headKey := []float32{1, 2, 3, 4} // seqLen 2 × headDim 2
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "test",
		Tokens:        []int32{1, 2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1, // slab sized to 1 head per layer
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			// Two heads though NumHeads is 1 → the slab is exhausted and the
			// layer falls back to its own make([]HeadSnapshot, 2).
			Heads: []HeadSnapshot{
				{Key: append([]float32(nil), headKey...), Value: append([]float32(nil), headKey...)},
				{Key: append([]float32(nil), headKey...), Value: append([]float32(nil), headKey...)},
			},
		}},
	}
	slice, err := snapshot.SliceBlock(0, 2, 0, false)
	if err != nil {
		t.Fatalf("SliceBlock(slab fallback) error = %v", err)
	}
	if len(slice.Layers[0].Heads) != 2 {
		t.Fatalf("sliced heads = %d, want 2", len(slice.Layers[0].Heads))
	}
}

// TestMiscCover_AssembleBlocks_EmptyLayers drives the preSizeAssembledRawBytes
// empty-layers early return via AssembleBlocks over blocks whose snapshots have
// no layers at all.
func TestMiscCover_AssembleBlocks_EmptyLayers(t *testing.T) {
	block := func(index, start int, token int32) Block {
		return Block{
			Index:      index,
			TokenStart: start,
			TokenCount: 1,
			Snapshot: &Snapshot{
				Version:      SnapshotVersion,
				Architecture: "test",
				Tokens:       []int32{token},
				TokenOffset:  start + 1,
				NumLayers:    0,
				NumHeads:     0,
				SeqLen:       1,
				HeadDim:      2,
				Layers:       nil, // no layers → preSize empty-layers early return
			},
		}
	}
	assembled, err := AssembleBlocks([]Block{block(0, 0, 1), block(1, 1, 2)})
	if err != nil {
		t.Fatalf("AssembleBlocks(empty layers) error = %v", err)
	}
	if len(assembled.Tokens) != 2 {
		t.Fatalf("assembled tokens = %d, want 2", len(assembled.Tokens))
	}
}
