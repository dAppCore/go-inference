// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"errors"
	"testing"
)

// rawF16Bytes builds n float16 little-endian values (value = i) as a byte
// slice — the raw-tensor payload shape the assemble helpers expect.
func rawF16Bytes(n int) []byte {
	out := make([]byte, 0, n*2)
	for i := range n {
		out = appendUint16LE(out, float32ToFloat16(float32(i)))
	}
	return out
}

// TestBlocksAssembleCover_AssembleBlocks_Guards drives the up-front guards of
// AssembleBlocks: an empty slice, a nil first-block snapshot, an out-of-order
// index, and a non-contiguous token range.
func TestBlocksAssembleCover_AssembleBlocks_Guards(t *testing.T) {
	if _, err := AssembleBlocks(nil); !errors.Is(err, errBlocksEmpty) {
		t.Fatalf("AssembleBlocks(nil) error = %v, want errBlocksEmpty", err)
	}

	// validateKVSnapshotBlockOrder passes only when block.Snapshot is set, so a
	// nil first-snapshot trips the token-count guard inside validation first.
	if _, err := AssembleBlocks([]Block{{Index: 0, TokenStart: 0, TokenCount: 1}}); err == nil {
		t.Fatal("AssembleBlocks(nil first snapshot) error = nil")
	}

	good := stateTokenOnlyTestSnapshot([]int32{1, 2}, 2, 2)
	// Out-of-order index.
	if _, err := AssembleBlocks([]Block{{Index: 7, TokenStart: 0, TokenCount: 2, Snapshot: good}}); !errors.Is(err, errBlocksOutOfOrder) {
		t.Fatalf("AssembleBlocks(out of order) error = %v, want errBlocksOutOfOrder", err)
	}
	// Non-contiguous (TokenStart != 0 for the first block).
	if _, err := AssembleBlocks([]Block{{Index: 0, TokenStart: 5, TokenCount: 2, Snapshot: good}}); !errors.Is(err, errBlocksNotContiguous) {
		t.Fatalf("AssembleBlocks(not contiguous) error = %v, want errBlocksNotContiguous", err)
	}
	// Token count disagrees with the snapshot's token slice.
	if _, err := AssembleBlocks([]Block{{Index: 0, TokenStart: 0, TokenCount: 9, Snapshot: good}}); !errors.Is(err, errBlockTokenCountMismatch) {
		t.Fatalf("AssembleBlocks(token count mismatch) error = %v, want errBlockTokenCountMismatch", err)
	}
}

// TestBlocksAssembleCover_AppendBlock_ShapeGuards drives the shape/arch/layer
// mismatch guards of appendKVSnapshotBlock via two snapshots whose geometry
// disagrees.
func TestBlocksAssembleCover_AppendBlock_ShapeGuards(t *testing.T) {
	dst := &Snapshot{Architecture: "gemma4_text", HeadDim: 2, NumHeads: 1, NumLayers: 1, Layers: []LayerSnapshot{{}}}

	// Architecture mismatch.
	if err := appendKVSnapshotBlock(dst, &Snapshot{Architecture: "other", HeadDim: 2, NumHeads: 1, NumLayers: 1, Layers: []LayerSnapshot{{}}}); !errors.Is(err, errBlockArchMismatch) {
		t.Fatalf("appendKVSnapshotBlock(arch) error = %v, want errBlockArchMismatch", err)
	}
	// HeadDim / NumHeads / NumLayers mismatch.
	if err := appendKVSnapshotBlock(dst, &Snapshot{Architecture: "gemma4_text", HeadDim: 9, NumHeads: 1, NumLayers: 1, Layers: []LayerSnapshot{{}}}); !errors.Is(err, errBlockShapeMismatch) {
		t.Fatalf("appendKVSnapshotBlock(shape) error = %v, want errBlockShapeMismatch", err)
	}
	// Layer-count mismatch (geometry agrees but layer slices differ in length).
	if err := appendKVSnapshotBlock(dst, &Snapshot{Architecture: "gemma4_text", HeadDim: 2, NumHeads: 1, NumLayers: 1, Layers: nil}); !errors.Is(err, errBlockLayerCountMismatch) {
		t.Fatalf("appendKVSnapshotBlock(layer count) error = %v, want errBlockLayerCountMismatch", err)
	}
}

// TestBlocksAssembleCover_AppendBlock_MetadataGuards drives the per-layer
// cache-mode and max-size mismatch guards plus the head-count mismatch guard
// of appendKVSnapshotBlock.
func TestBlocksAssembleCover_AppendBlock_MetadataGuards(t *testing.T) {
	// Cache-mode mismatch: dst already carries a mode, block carries another.
	cacheDst := &Snapshot{HeadDim: 2, NumHeads: 1, NumLayers: 1, Layers: []LayerSnapshot{{CacheMode: "a"}}}
	if err := appendKVSnapshotBlock(cacheDst, &Snapshot{HeadDim: 2, NumHeads: 1, NumLayers: 1, Layers: []LayerSnapshot{{CacheMode: "b"}}}); !errors.Is(err, errBlockMetadataMismatch) {
		t.Fatalf("appendKVSnapshotBlock(cache mode) error = %v, want errBlockMetadataMismatch", err)
	}

	// Max-size mismatch.
	maxDst := &Snapshot{HeadDim: 2, NumHeads: 1, NumLayers: 1, Layers: []LayerSnapshot{{MaxSize: 4}}}
	if err := appendKVSnapshotBlock(maxDst, &Snapshot{HeadDim: 2, NumHeads: 1, NumLayers: 1, Layers: []LayerSnapshot{{MaxSize: 8}}}); !errors.Is(err, errBlockMetadataMismatch) {
		t.Fatalf("appendKVSnapshotBlock(max size) error = %v, want errBlockMetadataMismatch", err)
	}

	// Head-count mismatch: dst layer already has one head, block has two.
	headDst := &Snapshot{HeadDim: 2, NumHeads: 1, NumLayers: 1, Layers: []LayerSnapshot{{Heads: []HeadSnapshot{{Key: []float32{1, 2}}}}}}
	block := &Snapshot{HeadDim: 2, NumHeads: 1, NumLayers: 1, Layers: []LayerSnapshot{{Heads: []HeadSnapshot{
		{Key: []float32{1, 2}}, {Key: []float32{3, 4}},
	}}}}
	if err := appendKVSnapshotBlock(headDst, block); !errors.Is(err, errBlockHeadCountMismatch) {
		t.Fatalf("appendKVSnapshotBlock(head count) error = %v, want errBlockHeadCountMismatch", err)
	}
}

// TestBlocksAssembleCover_LayerRawBlock_SingleHead drives the single-head
// (B*H==1) fast path of appendKVSnapshotLayerRawBlock, including the
// first-arrival clone and the in-place append on the second block.
func TestBlocksAssembleCover_LayerRawBlock_SingleHead(t *testing.T) {
	var dstDType string
	var dstBytes []byte
	var dstShape []int32

	// First arrival: shape {1,1,2,2} → 4 f16 values → 8 bytes. Clones shape.
	if err := appendKVSnapshotLayerRawBlock(&dstDType, &dstBytes, &dstShape, "float16", rawF16Bytes(4), []int32{1, 1, 2, 2}); err != nil {
		t.Fatalf("layer-raw first arrival error = %v", err)
	}
	if dstShape[2] != 2 || len(dstBytes) != 8 {
		t.Fatalf("after first arrival shape = %v, bytes = %d", dstShape, len(dstBytes))
	}
	// Second block: same B/H/D, another L=2 → in-place append, L grows to 4.
	if err := appendKVSnapshotLayerRawBlock(&dstDType, &dstBytes, &dstShape, "float16", rawF16Bytes(4), []int32{1, 1, 2, 2}); err != nil {
		t.Fatalf("layer-raw second block error = %v", err)
	}
	if dstShape[2] != 4 || len(dstBytes) != 16 {
		t.Fatalf("after second block shape = %v, bytes = %d, want L=4 / 16 bytes", dstShape, len(dstBytes))
	}
}

// TestBlocksAssembleCover_LayerRawBlock_MultiHead drives the B*H>1 row-major
// merge path of appendKVSnapshotLayerRawBlock — the densest uncovered block.
// Shape {1,2,L,D}: two K/V heads, so the merge interleaves rows rather than
// taking the single-head append shortcut.
func TestBlocksAssembleCover_LayerRawBlock_MultiHead(t *testing.T) {
	var dstDType string
	var dstBytes []byte
	var dstShape []int32

	// {1,2,2,2} → 8 f16 values → 16 bytes.
	if err := appendKVSnapshotLayerRawBlock(&dstDType, &dstBytes, &dstShape, "float16", rawF16Bytes(8), []int32{1, 2, 2, 2}); err != nil {
		t.Fatalf("multi-head first arrival error = %v", err)
	}
	// Second {1,2,2,2} block → merge path: L grows to 4, byte count doubles.
	if err := appendKVSnapshotLayerRawBlock(&dstDType, &dstBytes, &dstShape, "float16", rawF16Bytes(8), []int32{1, 2, 2, 2}); err != nil {
		t.Fatalf("multi-head merge error = %v", err)
	}
	if dstShape[2] != 4 {
		t.Fatalf("after merge shape = %v, want L=4", dstShape)
	}
	if len(dstBytes) != 1*2*4*2*2 {
		t.Fatalf("after merge bytes = %d, want %d", len(dstBytes), 1*2*4*2*2)
	}
}

// TestBlocksAssembleCover_LayerRawBlock_Errors drives the validation error arms
// of appendKVSnapshotLayerRawBlock.
func TestBlocksAssembleCover_LayerRawBlock_Errors(t *testing.T) {
	var dstDType string
	var dstBytes []byte
	var dstShape []int32

	// Unsupported dtype.
	if err := appendKVSnapshotLayerRawBlock(&dstDType, &dstBytes, &dstShape, "nonsense", rawF16Bytes(4), []int32{1, 1, 2, 2}); !errors.Is(err, errUnsupportedLayerRawTensor) {
		t.Fatalf("layer-raw bad dtype error = %v, want errUnsupportedLayerRawTensor", err)
	}
	// Byte length disagrees with the shape's element count.
	if err := appendKVSnapshotLayerRawBlock(&dstDType, &dstBytes, &dstShape, "float16", rawF16Bytes(2), []int32{1, 1, 2, 2}); !errors.Is(err, errLayerRawTensorShape) {
		t.Fatalf("layer-raw byte mismatch error = %v, want errLayerRawTensorShape", err)
	}

	// Dtype mismatch on the second block (dst already float16, block bfloat16).
	var d2 string
	var b2 []byte
	var s2 []int32
	if err := appendKVSnapshotLayerRawBlock(&d2, &b2, &s2, "float16", rawF16Bytes(4), []int32{1, 1, 2, 2}); err != nil {
		t.Fatalf("layer-raw seed error = %v", err)
	}
	if err := appendKVSnapshotLayerRawBlock(&d2, &b2, &s2, "bfloat16", rawF16Bytes(4), []int32{1, 1, 2, 2}); !errors.Is(err, errLayerRawDtypeMismatch) {
		t.Fatalf("layer-raw dtype mismatch error = %v, want errLayerRawDtypeMismatch", err)
	}
	// Second block with a divergent B/H/D shape → tensor-shape error.
	if err := appendKVSnapshotLayerRawBlock(&d2, &b2, &s2, "float16", rawF16Bytes(8), []int32{1, 2, 2, 2}); !errors.Is(err, errLayerRawTensorShape) {
		t.Fatalf("layer-raw merge shape mismatch error = %v, want errLayerRawTensorShape", err)
	}
}

// nativeRawBlock builds a single-token native-raw block snapshot whose layer
// and head both carry float16 raw payloads, with TokenOffset 0 so the assembled
// snapshot exercises the len(Tokens) fallback.
func nativeRawBlock(index, tokenStart int, token int32) Block {
	return Block{
		Index:      index,
		TokenStart: tokenStart,
		TokenCount: 1,
		Snapshot: &Snapshot{
			Architecture:  "gemma4_text",
			Tokens:        []int32{token},
			TokenOffset:   0, // → len(Tokens) fallback in AssembleBlocks
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        1,
			HeadDim:       2,
			NumQueryHeads: 1,
			Layers: []LayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				KeyDType:   "float16",
				KeyBytes:   rawF16Bytes(2), // {1,1,1,2} → 2 f16 values
				KeyShape:   []int32{1, 1, 1, 2},
				ValueDType: "float16",
				ValueBytes: rawF16Bytes(2),
				ValueShape: []int32{1, 1, 1, 2},
				Heads: []HeadSnapshot{{
					KeyDType:   "float16",
					KeyBytes:   rawF16Bytes(2),
					ValueDType: "float16",
					ValueBytes: rawF16Bytes(2),
				}},
			}},
		},
	}
}

// TestBlocksAssembleCover_AssembleNativeRaw drives AssembleBlocks end-to-end
// over native-raw blocks: this is the path that exercises the head-level loops
// of preSizeAssembledRawBytes, the layer/head raw append arms of
// appendKVSnapshotBlock, and the TokenOffset == 0 → len(Tokens) fallback.
func TestBlocksAssembleCover_AssembleNativeRaw(t *testing.T) {
	blocks := []Block{
		nativeRawBlock(0, 0, 1),
		nativeRawBlock(1, 1, 2),
	}
	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		t.Fatalf("AssembleBlocks(native raw) error = %v", err)
	}
	if len(assembled.Tokens) != 2 {
		t.Fatalf("assembled tokens = %d, want 2", len(assembled.Tokens))
	}
	if assembled.TokenOffset != 2 {
		t.Fatalf("assembled TokenOffset = %d, want len(Tokens)=2", assembled.TokenOffset)
	}
	// Layer KeyBytes should have grown across both blocks (L=1 + L=1 = 2).
	if assembled.Layers[0].KeyShape[2] != 2 {
		t.Fatalf("assembled layer KeyShape = %v, want L=2", assembled.Layers[0].KeyShape)
	}
}

// TestBlocksAssembleCover_AppendBlock_RawErrors drives the layer/head raw
// append error arms of appendKVSnapshotBlock: the second block's layer (then
// head) raw tensor carries a malformed shape so the append helper rejects it.
func TestBlocksAssembleCover_AppendBlock_RawErrors(t *testing.T) {
	// Layer key append error: dst seeded by block 0, block 1's layer KeyBytes
	// has a byte length that disagrees with its shape.
	dst, err := AssembleBlocks([]Block{nativeRawBlock(0, 0, 1)})
	if err != nil {
		t.Fatalf("seed AssembleBlocks error = %v", err)
	}
	badLayer := nativeRawBlock(1, 1, 2).Snapshot
	badLayer.Layers[0].KeyBytes = rawF16Bytes(1) // shape says 2 values, give 1
	if err := appendKVSnapshotBlock(dst, badLayer); err == nil {
		t.Fatal("appendKVSnapshotBlock(bad layer key) error = nil, want raw error")
	}

	// Head value append error: a fresh dst, block with a malformed head dtype.
	dst2, err := AssembleBlocks([]Block{nativeRawBlock(0, 0, 1)})
	if err != nil {
		t.Fatalf("seed AssembleBlocks(2) error = %v", err)
	}
	badHead := nativeRawBlock(1, 1, 2).Snapshot
	badHead.Layers[0].Heads[0].ValueDType = "nonsense"
	if err := appendKVSnapshotBlock(dst2, badHead); err == nil {
		t.Fatal("appendKVSnapshotBlock(bad head value) error = nil, want raw error")
	}
}

// TestBlocksAssembleCover_EmptyLayersSlabFallback drives the per-layer make
// fallback in emptyKVSnapshotLayers, where a layer's head count exceeds the
// uniform slab size derived from the widest layer is not the case — instead a
// later layer carries more heads than an earlier one, forcing slab exhaustion.
func TestBlocksAssembleCover_EmptyLayersSlabFallback(t *testing.T) {
	// Two layers: the slab is sized to the widest (2 heads) × 2 layers = 4.
	// A layout where the first layer claims 2 heads and the second also claims
	// 2 fills the slab exactly; to force the fallback, make the first layer
	// wide (3 heads) so slabHeadsPerLayer=3, slab=6, then both fit — instead we
	// rely on emptyKVSnapshotLayers being called with layers whose cumulative
	// head count exceeds the slab. Use three layers of widths 2,2,2 with slab
	// sized to 2 → 6, exact. The fallback fires when a single layer's headCount
	// exceeds the remaining slab, which a width spike after the max achieves.
	layers := []LayerSnapshot{
		{Heads: make([]HeadSnapshot, 1)},
		{Heads: make([]HeadSnapshot, 2)},
		{Heads: make([]HeadSnapshot, 2)},
	}
	out := emptyKVSnapshotLayers(layers)
	if len(out) != 3 {
		t.Fatalf("emptyKVSnapshotLayers len = %d, want 3", len(out))
	}
	if len(out[2].Heads) != 2 {
		t.Fatalf("layer 2 heads = %d, want 2", len(out[2].Heads))
	}
}

// TestBlocksAssembleCover_PreSizeBoundsGuards drives the layer/head bounds
// `continue` guards of preSizeAssembledRawBytes. preSizeAssembledRawBytes runs
// before the append validation, so a block whose later layers/heads are
// narrower than the assembled skeleton (built from the first block) exercises
// the `layerIndex >= len(Layers)` and `headIndex >= len(Heads)` skips.
func TestBlocksAssembleCover_PreSizeBoundsGuards(t *testing.T) {
	// First block: two native-raw layers, each with one head. The assembled
	// skeleton therefore has two layers.
	first := nativeRawBlock(0, 0, 1).Snapshot
	first.NumLayers = 2
	first.Layers = []LayerSnapshot{first.Layers[0], cloneNativeRawLayer(first.Layers[0])}

	// Second block: only one layer (narrower than the skeleton) and that layer
	// carries no heads. Order validation ignores layer/head counts, so preSize
	// reaches the second block and skips its missing layer/heads.
	second := nativeRawBlock(1, 1, 2).Snapshot
	second.NumLayers = 2 // keep geometry checks happy in append (won't be reached)
	second.Layers = []LayerSnapshot{{Layer: 0, CacheIndex: 0}}

	// AssembleBlocks fails at the append validation (layer count), but
	// preSizeAssembledRawBytes runs first and must skip the second block's
	// out-of-bounds layer without panicking — assert the specific error so a
	// regression that silently swallowed the mismatch (or panicked before
	// reaching it) would be caught.
	if _, err := AssembleBlocks([]Block{
		{Index: 0, TokenStart: 0, TokenCount: 1, Snapshot: first},
		{Index: 1, TokenStart: 1, TokenCount: 1, Snapshot: second},
	}); !errors.Is(err, errBlockLayerCountMismatch) {
		t.Fatalf("AssembleBlocks(narrower second block) error = %v, want errBlockLayerCountMismatch", err)
	}

	// Also drive the head-level skip: both blocks have one layer, but the
	// second block's layer carries zero heads while the skeleton's does. This
	// combination does NOT error — appendKVSnapshotBlock's `len(layer.Heads)
	// == 0` guard just skips head-level folding for that block — so assert the
	// resulting shape: the layer-level raw slab grows to cover both blocks'
	// tokens (preSizeAssembledRawBytes summed both), but the per-head raw
	// bytes stay at the first block's contribution only (second block never
	// reached the head loop).
	firstHeads := nativeRawBlock(0, 0, 1).Snapshot
	firstHeads.NumLayers = 1
	secondHeads := nativeRawBlock(1, 1, 2).Snapshot
	secondHeads.NumLayers = 1
	secondHeads.Layers[0].Heads = nil // narrower head count than the skeleton
	assembled, err := AssembleBlocks([]Block{
		{Index: 0, TokenStart: 0, TokenCount: 1, Snapshot: firstHeads},
		{Index: 1, TokenStart: 1, TokenCount: 1, Snapshot: secondHeads},
	})
	if err != nil {
		t.Fatalf("AssembleBlocks(head-level skip) error = %v, want success", err)
	}
	if len(assembled.Tokens) != 2 || assembled.TokenOffset != 2 {
		t.Fatalf("assembled = tokens %v offset %d, want [1 2] / offset 2", assembled.Tokens, assembled.TokenOffset)
	}
	if len(assembled.Layers[0].KeyShape) != 4 || assembled.Layers[0].KeyShape[2] != 2 {
		t.Fatalf("assembled layer KeyShape = %v, want L=2 (layer slab folds both blocks)", assembled.Layers[0].KeyShape)
	}
	if got := len(assembled.Layers[0].Heads[0].KeyBytes); got != 4 {
		t.Fatalf("assembled head KeyBytes = %d bytes, want 4 (only the first block's single f16 token; the headless second block never reached the head loop)", got)
	}
}

// cloneNativeRawLayer deep-copies a native-raw layer so two skeleton layers do
// not alias the same backing slices.
func cloneNativeRawLayer(layer LayerSnapshot) LayerSnapshot {
	out := layer
	out.KeyBytes = append([]byte(nil), layer.KeyBytes...)
	out.KeyShape = append([]int32(nil), layer.KeyShape...)
	out.ValueBytes = append([]byte(nil), layer.ValueBytes...)
	out.ValueShape = append([]int32(nil), layer.ValueShape...)
	out.Heads = append([]HeadSnapshot(nil), layer.Heads...)
	return out
}

// TestBlocksAssembleCover_AppendBlock_ValueRawError drives the layer/head VALUE
// raw append error arms (the value-side mirrors of the key arms) of
// appendKVSnapshotBlock plus the byte-length mismatch guard.
func TestBlocksAssembleCover_AppendBlock_ValueRawError(t *testing.T) {
	// Layer value append error.
	dst, err := AssembleBlocks([]Block{nativeRawBlock(0, 0, 1)})
	if err != nil {
		t.Fatalf("seed AssembleBlocks error = %v", err)
	}
	badLayerValue := nativeRawBlock(1, 1, 2).Snapshot
	badLayerValue.Layers[0].ValueBytes = rawF16Bytes(1) // shape says 2 values
	if err := appendKVSnapshotBlock(dst, badLayerValue); err == nil {
		t.Fatal("appendKVSnapshotBlock(bad layer value) error = nil, want raw error")
	}

	// Head key append error (the value-side head error is covered elsewhere).
	dst2, err := AssembleBlocks([]Block{nativeRawBlock(0, 0, 1)})
	if err != nil {
		t.Fatalf("seed AssembleBlocks(2) error = %v", err)
	}
	badHeadKey := nativeRawBlock(1, 1, 2).Snapshot
	badHeadKey.Layers[0].Heads[0].KeyDType = "nonsense"
	if err := appendKVSnapshotBlock(dst2, badHeadKey); err == nil {
		t.Fatal("appendKVSnapshotBlock(bad head key) error = nil, want raw error")
	}

	// Layer-raw byte-length mismatch on a second append (282): seed a layer slab
	// then append a block whose stored dst byte length no longer matches.
	var dDType string
	var dBytes []byte
	var dShape []int32
	if err := appendKVSnapshotLayerRawBlock(&dDType, &dBytes, &dShape, "float16", rawF16Bytes(4), []int32{1, 1, 2, 2}); err != nil {
		t.Fatalf("seed layer-raw error = %v", err)
	}
	// Corrupt the recorded byte buffer so the oldLen byte-count check fails.
	dBytes = dBytes[:len(dBytes)-2]
	if err := appendKVSnapshotLayerRawBlock(&dDType, &dBytes, &dShape, "float16", rawF16Bytes(4), []int32{1, 1, 2, 2}); !errors.Is(err, errLayerRawByteLenMismatch) {
		t.Fatalf("layer-raw byte-len mismatch error = %v, want errLayerRawByteLenMismatch", err)
	}
}

// TestBlocksAssembleCover_RawBlock drives appendKVSnapshotRawBlock: the
// happy append, the unsupported-dtype guard, and the dtype-mismatch guard.
func TestBlocksAssembleCover_RawBlock(t *testing.T) {
	var dstDType string
	var dstBytes []byte

	if err := appendKVSnapshotRawBlock(&dstDType, &dstBytes, "float16", rawF16Bytes(2)); err != nil {
		t.Fatalf("raw-block first append error = %v", err)
	}
	if dstDType != "float16" || len(dstBytes) != 4 {
		t.Fatalf("raw-block dtype = %q, bytes = %d", dstDType, len(dstBytes))
	}
	// Unsupported dtype.
	var d2 string
	var b2 []byte
	if err := appendKVSnapshotRawBlock(&d2, &b2, "nonsense", rawF16Bytes(2)); !errors.Is(err, errUnsupportedRawTensorDtype) {
		t.Fatalf("raw-block bad dtype error = %v, want errUnsupportedRawTensorDtype", err)
	}
	// Dtype mismatch on a second append.
	if err := appendKVSnapshotRawBlock(&dstDType, &dstBytes, "bfloat16", rawF16Bytes(2)); !errors.Is(err, errRawTensorDtypeMismatch) {
		t.Fatalf("raw-block dtype mismatch error = %v, want errRawTensorDtypeMismatch", err)
	}
}

// tokenBytes builds one token of width bytes, each byte set to fill — a
// recognisable pattern so a compaction test can assert exactly which bytes
// survived and which gap bytes were dropped.
func tokenBytes(fill byte, width int) []byte {
	out := make([]byte, width)
	for i := range out {
		out[i] = fill
	}
	return out
}

// TestBlocksAssembleCover_CompactLayerRawGaps_Good drives the gapped-placement
// compaction path (the multi-head assembly fast path leaves a [B,H,capL,D]
// buffer with capL > filledL when its token estimate overshoots). Compaction
// must rewrite it to packed [B,H,filledL,D] form byte-for-byte, dropping only
// the trailing gap tokens — a torn-state edge the single-head packed benches
// never reach.
func TestBlocksAssembleCover_CompactLayerRawGaps_Good(t *testing.T) {
	// float32 → bytesPerValue 4; B=1, H=2 (bh=2), D=1, filledL=2, capL=4.
	const bpv, bh, d, filledL = 4, 2, 1, 2
	// row0: real tokens 1,2 then gap tokens (9); row1: real 3,4 then gap.
	buf := make([]byte, 0, bh*4*d*bpv)
	buf = append(buf, tokenBytes(1, bpv)...) // row0 t0 (real)
	buf = append(buf, tokenBytes(2, bpv)...) // row0 t1 (real)
	buf = append(buf, tokenBytes(9, bpv)...) // row0 t2 (gap)
	buf = append(buf, tokenBytes(9, bpv)...) // row0 t3 (gap)
	buf = append(buf, tokenBytes(3, bpv)...) // row1 t0 (real)
	buf = append(buf, tokenBytes(4, bpv)...) // row1 t1 (real)
	buf = append(buf, tokenBytes(9, bpv)...) // row1 t2 (gap)
	buf = append(buf, tokenBytes(9, bpv)...) // row1 t3 (gap)

	shape := []int32{1, 2, filledL, 1} // dstShape[2] carries the fill cursor
	compactLayerRawGaps(&buf, shape, bh, d, bpv)

	want := make([]byte, 0, bh*filledL*d*bpv)
	want = append(want, tokenBytes(1, bpv)...)
	want = append(want, tokenBytes(2, bpv)...)
	want = append(want, tokenBytes(3, bpv)...)
	want = append(want, tokenBytes(4, bpv)...)
	if len(buf) != len(want) {
		t.Fatalf("compacted length = %d, want %d (packed [B,H,filledL,D])", len(buf), len(want))
	}
	for i := range want {
		if buf[i] != want[i] {
			t.Fatalf("compacted byte %d = %d, want %d — gap tokens not dropped cleanly", i, buf[i], want[i])
		}
	}
}

// TestBlocksAssembleCover_CompactLayerRawGaps_NoOp pins the no-op guards: a
// non-4D shape, an already-packed buffer (capL == filledL), and an undersized
// buffer (reads as packed, not gapped) must all leave the bytes untouched.
func TestBlocksAssembleCover_CompactLayerRawGaps_NoOp(t *testing.T) {
	const bpv, bh, d = 4, 2, 1

	// non-4D shape: returns immediately.
	nb := append([]byte(nil), tokenBytes(7, bpv*4)...)
	before := append([]byte(nil), nb...)
	compactLayerRawGaps(&nb, []int32{2, 3, 4}, bh, d, bpv)
	if len(nb) != len(before) {
		t.Fatalf("non-4D shape mutated the buffer (len %d -> %d)", len(before), len(nb))
	}

	// packed: len == bh*filledL*d*bpv exactly, so layerRawGapStride reads capL ==
	// filledL and compaction is skipped.
	packed := make([]byte, bh*2*d*bpv) // filledL = 2
	for i := range packed {
		packed[i] = byte(i)
	}
	packedCopy := append([]byte(nil), packed...)
	compactLayerRawGaps(&packed, []int32{1, 2, 2, 1}, bh, d, bpv)
	if len(packed) != len(packedCopy) {
		t.Fatalf("packed buffer truncated (len %d -> %d)", len(packedCopy), len(packed))
	}
	for i := range packedCopy {
		if packed[i] != packedCopy[i] {
			t.Fatalf("packed buffer byte %d mutated", i)
		}
	}
}
