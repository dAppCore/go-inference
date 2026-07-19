// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"math"
	"testing"
)

// TestVisionRotaryTable_Good replays vision_block_golden.json's rot_pos_emb — the REAL
// DotsVisionTransformer.rot_pos_emb(grid_thw) RAW angle table (radians, pre-cos/sin) for the same
// 2×2 synthetic grid — by taking cos/sin of the golden's raw angles and comparing against
// visionRotaryTable's cosHalf/sinHalf directly (visionRotaryTable returns the already-evaluated
// tables since that is what applyRotaryHalf actually consumes; the golden captures the reference's
// own intermediate tensor faithfully rather than a value already transformed for comparison).
func TestVisionRotaryTable_Good(t *testing.T) {
	g := readVisionBlockGolden(t)
	headDim := g.EmbedDim / 12 // NumAttentionHeads=12, confirmed against the real vision_config
	cosHalf, sinHalf := visionRotaryTable(g.GridH, g.GridW, 2, headDim)
	n := g.GridH * g.GridW
	half := headDim / 2
	if len(g.RotPosEmb) != n*half {
		t.Fatalf("golden rot_pos_emb has %d elements, want %d (%d patches × %d)", len(g.RotPosEmb), n*half, n, half)
	}
	for i := range n {
		for j := range half {
			angle := float64(g.RotPosEmb[i*half+j])
			wantCos, wantSin := float32(math.Cos(angle)), float32(math.Sin(angle))
			if d := math.Abs(float64(cosHalf[i][j] - wantCos)); d > 1e-4 {
				t.Fatalf("cosHalf[%d][%d] = %v, want %v (diff %v)", i, j, cosHalf[i][j], wantCos, d)
			}
			if d := math.Abs(float64(sinHalf[i][j] - wantSin)); d > 1e-4 {
				t.Fatalf("sinHalf[%d][%d] = %v, want %v (diff %v)", i, j, sinHalf[i][j], wantSin, d)
			}
		}
	}
}

// TestVisionAttentionForward_Bad proves a head count that doesn't divide the embedding dim
// refuses cleanly.
func TestVisionAttentionForward_Bad(t *testing.T) {
	w := VisionAttnWeights{
		Q:    LinearWeights{Weight: make([]float32, 9), In: 3, Out: 3},
		K:    LinearWeights{Weight: make([]float32, 9), In: 3, Out: 3},
		V:    LinearWeights{Weight: make([]float32, 9), In: 3, Out: 3},
		Proj: LinearWeights{Weight: make([]float32, 9), In: 3, Out: 3},
	}
	cosHalf := [][]float32{{1}}
	sinHalf := [][]float32{{0}}
	if _, err := visionAttentionForward(make([]float32, 3), 1, 3, 2, w, cosHalf, sinHalf); err == nil {
		t.Fatal("visionAttentionForward accepted embed_dim=3 with heads=2")
	}
}

// TestPatchMerger_Good hand-verifies the merge-group reshape + LayerNorm + GELU-MLP on a tiny
// (embedDim=2, mergeSize=2 -> mergedDim=8) identity-weighted case: LayerNorm's weight=1/bias=0
// leaves a CONSTANT row at exactly zero post-norm (matching layerNormForward's own _Bad case),
// so the GELU-gated MLP's output is driven entirely by its own bias terms — set to a known
// constant to make the expected output exact.
func TestPatchMerger_Good(t *testing.T) {
	embedDim, merge := 2, 2
	mergedDim := embedDim * merge * merge // 8
	w := &VisionWeights{
		MergerLNQ: LayerNormWeights{Weight: []float32{1, 1}, Bias: []float32{0, 0}},
		MergerFC1: LinearWeights{
			Weight: make([]float32, mergedDim*mergedDim), // all-zero: fc1 output = bias only
			Bias:   fill(mergedDim, 1),                   // gelu(1) constant per channel
			In:     mergedDim, Out: mergedDim,
		},
		MergerFC2: LinearWeights{
			Weight: make([]float32, embedDim*mergedDim), // all-zero: fc2 output = bias only
			Bias:   []float32{5, 7},
			In:     mergedDim, Out: embedDim,
		},
	}
	// 4 constant rows (one merge group) -> LayerNorm(constant row, weight=1,bias=0) = 0 regardless
	// of the row's own value (mirrors whisper's TestLayerNormForward_Bad).
	x := []float32{3, 3, 3, 3, 3, 3, 3, 3}
	got, err := patchMerger(x, 4, embedDim, merge, w)
	if err != nil {
		t.Fatalf("patchMerger: %v", err)
	}
	want := []float32{5, 7} // fc2's bias, since both fc1 and fc2 weights are zero
	if d := maxAbsDiff32(t, got, want); d > 1e-4 {
		t.Fatalf("patchMerger = %v, want %v (diff %v)", got, want, d)
	}
}

func fill(n int, v float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// TestPatchMerger_Bad proves a patch count that isn't a multiple of merge_size² refuses.
func TestPatchMerger_Bad(t *testing.T) {
	w := &VisionWeights{
		MergerLNQ: LayerNormWeights{Weight: []float32{1, 1}, Bias: []float32{0, 0}},
		MergerFC1: LinearWeights{Weight: make([]float32, 64), In: 8, Out: 8},
		MergerFC2: LinearWeights{Weight: make([]float32, 16), In: 8, Out: 2},
	}
	if _, err := patchMerger(make([]float32, 6), 3, 2, 2, w); err == nil {
		t.Fatal("patchMerger accepted 3 patches with merge_size=2 (not a multiple of 4)")
	}
}

// TestEncodeImage_Bad proves nil weights/config refuses.
func TestEncodeImage_Bad(t *testing.T) {
	if _, err := EncodeImage(nil, 1, 2, 2, nil, nil); err == nil {
		t.Fatal("EncodeImage accepted nil weights/config")
	}
}

// TestEncodeImage_Ugly proves grid_t != 1 (video/multi-frame batching, out of scope — see
// EncodeImage's doc comment) refuses distinctly from the nil-argument case above.
func TestEncodeImage_Ugly(t *testing.T) {
	cfg := &Config{VisionConfig: realVisionConfig()}
	w := &Weights{}
	if _, err := EncodeImage(nil, 2, 2, 2, w, cfg); err == nil {
		t.Fatal("EncodeImage accepted grid_t=2")
	}
}
