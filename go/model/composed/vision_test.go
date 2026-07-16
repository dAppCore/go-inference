// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"
)

// identityLinear builds an n×n identity-weight visionLinear (no bias) — the "pass-through" projection
// used to isolate one stage of the tower/merger maths from its neighbours in a test.
func identityLinear(n int) visionLinear {
	w := make([]float32, n*n)
	for i := range n {
		w[i*n+i] = 1
	}
	return visionLinear{W: w, Out: n, In: n}
}

// mkVisionTower builds a synthetic nBlocks-deep Qwen-VL-shaped vision tower: Hidden=8, NumHeads=2,
// HeadDim=4 (headDim%4==0, so visionRope2D's row/col quarters are exact), NumKVHeads=numKVHeads (2 ⇒ MHA,
// 1 ⇒ GQA), PatchSize=2, InChannels=3 (PatchDim=12), FF=16, MergeSize=2, TextHidden=textHidden, no
// QK-norm — deterministic synthetic weights via the package's own syn() helper (composed_test.go).
func mkVisionTower(nBlocks, numKVHeads, textHidden, seed int) *visionTower {
	const hidden, numHeads, headDim, ff, patchDim = 8, 2, 4, 16, 12
	blocks := make([]visionBlock, nBlocks)
	for i := range blocks {
		s := seed + i*100
		blocks[i] = visionBlock{
			Norm1W: syn(hidden, s+1), Norm1B: syn(hidden, s+2),
			Norm2W: syn(hidden, s+3), Norm2B: syn(hidden, s+4),
			Attn: visionAttnWeights{
				Q: visionLinear{W: syn(numHeads*headDim*hidden, s+5), Out: numHeads * headDim, In: hidden},
				K: visionLinear{W: syn(numKVHeads*headDim*hidden, s+6), Out: numKVHeads * headDim, In: hidden},
				V: visionLinear{W: syn(numKVHeads*headDim*hidden, s+7), Out: numKVHeads * headDim, In: hidden},
				O: visionLinear{W: syn(hidden*numHeads*headDim, s+8), Out: hidden, In: numHeads * headDim},
			},
			MLP: visionMLPWeights{
				Gate: visionLinear{W: syn(ff*hidden, s+9), Out: ff, In: hidden},
				Up:   visionLinear{W: syn(ff*hidden, s+10), Out: ff, In: hidden},
				Down: visionLinear{W: syn(hidden*ff, s+11), Out: hidden, In: ff},
			},
		}
	}
	const mergeSize = 2
	mergedIn := hidden * mergeSize * mergeSize
	return &visionTower{
		Patch:  visionLinear{W: syn(hidden*patchDim, seed+1000), Out: hidden, In: patchDim},
		Blocks: blocks,
		Merger: visionMerger{
			NormW: syn(hidden, seed+2000), NormB: syn(hidden, seed+2001),
			L1: visionLinear{W: syn(mergedIn*mergedIn, seed+2002), Out: mergedIn, In: mergedIn},
			L2: visionLinear{W: syn(textHidden*mergedIn, seed+2003), Out: textHidden, In: mergedIn},
		},
		Cfg: visionTowerCfg{
			Hidden: hidden, PatchDim: patchDim,
			NumHeads: numHeads, NumKVHeads: numKVHeads, HeadDim: headDim,
			PatchSize: 2, InChannels: 3, TemporalPatchSize: 1,
			MergeSize: mergeSize, TextHidden: textHidden,
			RopeTheta: 10000, Eps: 1e-6,
		},
	}
}

// mkSyntheticPNG encodes a deterministic w×h RGBA PNG in memory — never a downloaded/checkpoint asset,
// just a generated fixture — so imageToPatchGrid has real PNG bytes to decode.
func mkSyntheticPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, color.RGBA{R: uint8((x*37 + 11) % 256), G: uint8((y*53 + 5) % 256), B: uint8((x + y*7) % 256), A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("mkSyntheticPNG: encode: %v", err)
	}
	return buf.Bytes()
}

func TestIsqrt_Good(t *testing.T) {
	cases := map[int]int{0: 0, 1: 1, 4: 2, 9: 3, 15: 3, 16: 4, 25: 5, 99: 9, 100: 10}
	for n, want := range cases {
		if got := isqrt(n); got != want {
			t.Fatalf("isqrt(%d) = %d, want %d", n, got, want)
		}
	}
}

func TestIsqrt_Bad(t *testing.T) {
	if got := isqrt(-1); got != -1 {
		t.Fatalf("isqrt(-1) = %d, want -1", got)
	}
}

func TestGeluTanh_Good(t *testing.T) {
	if got := geluTanh(0); got != 0 {
		t.Fatalf("geluTanh(0) = %v, want exactly 0", got)
	}
	if got := geluTanh(10); math.Abs(float64(got)-10) > 1e-3 {
		t.Fatalf("geluTanh(10) = %v, want ≈10 (saturated positive tail)", got)
	}
	if got := geluTanh(-10); math.Abs(float64(got)) > 1e-3 {
		t.Fatalf("geluTanh(-10) = %v, want ≈0 (saturated negative tail)", got)
	}
}

func TestVisionRotaryTable_Good(t *testing.T) {
	inv := visionRotaryTable(8, 10000) // dim=4, n=2
	if len(inv) != 2 {
		t.Fatalf("len(inv) = %d, want 2 (headDim/4)", len(inv))
	}
	if inv[0] != 1 {
		t.Fatalf("inv[0] = %v, want 1 (theta^0)", inv[0])
	}
	if math.Abs(inv[1]-0.01) > 1e-9 {
		t.Fatalf("inv[1] = %v, want 0.01 (10000^-0.5)", inv[1])
	}
}

func TestVisionRope2D_Good(t *testing.T) {
	invFreq := visionRotaryTable(8, 10000)
	// Zero coordinate ⇒ every angle is 0 ⇒ exact identity.
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	got := append([]float32(nil), x...)
	visionRope2D(got, 0, 0, 8, invFreq)
	for i := range x {
		if got[i] != x[i] {
			t.Fatalf("visionRope2D at (row=0,col=0)[%d] = %v, want %v (zero-angle identity)", i, got[i], x[i])
		}
	}
	// A non-zero coordinate rotates (changes) the vector but preserves its norm.
	rotated := append([]float32(nil), x...)
	visionRope2D(rotated, 3, 5, 8, invFreq)
	var before, after float64
	for i := range x {
		before += float64(x[i]) * float64(x[i])
		after += float64(rotated[i]) * float64(rotated[i])
	}
	if math.Abs(before-after) > 1e-3 {
		t.Fatalf("visionRope2D norm not preserved: before=%v after=%v", before, after)
	}
	same := true
	for i := range x {
		if rotated[i] != x[i] {
			same = false
		}
	}
	if same {
		t.Fatal("visionRope2D at a non-zero coordinate left the vector unchanged")
	}
}

func TestLayerNormRowsWithBias_Good(t *testing.T) {
	x := []float32{1, 3}
	w := []float32{1, 1}
	got := layerNormRowsWithBias(x, w, nil, 1, 2, 0)
	want := []float32{-1, 1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("layerNormRowsWithBias(no bias)[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	gotB := layerNormRowsWithBias(x, w, []float32{10, 20}, 1, 2, 0)
	wantB := []float32{9, 21}
	for i := range wantB {
		if gotB[i] != wantB[i] {
			t.Fatalf("layerNormRowsWithBias(bias)[%d] = %v, want %v", i, gotB[i], wantB[i])
		}
	}
}

func TestLinearForward_Good(t *testing.T) {
	w := identityLinear(2)
	w.B = []float32{1, 1}
	out := linearForward([]float32{2, 3}, &w, 1)
	want := []float32{3, 4}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("linearForward[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestAddRows_Good(t *testing.T) {
	got := addRows([]float32{1, 2, 3}, []float32{10, 20, 30})
	want := []float32{11, 22, 33}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("addRows[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestVisionAttentionForward_Bidirectional proves the vision tower's attention is FULL (non-causal),
// unlike the text stack's causal attnMixer: perturbing only the LATER token's input still changes the
// EARLIER token's output, because the earlier token attends forward to it.
func TestVisionAttentionForward_Bidirectional(t *testing.T) {
	const hidden, gridW = 4, 2
	w := &visionAttnWeights{Q: identityLinear(hidden), K: identityLinear(hidden), V: identityLinear(hidden), O: identityLinear(hidden)}
	cfg := visionTowerCfg{NumHeads: 1, NumKVHeads: 1, HeadDim: hidden, RopeTheta: 10000, Eps: 1e-6}

	x1 := []float32{1, 0, 0, 0, 2, 0, 0, 0} // token 0, token 1
	out1, err := visionAttentionForward(append([]float32(nil), x1...), w, 2, gridW, cfg)
	if err != nil {
		t.Fatalf("visionAttentionForward: %v", err)
	}
	x2 := append([]float32(nil), x1...)
	x2[4] = 99 // perturb token 1 ONLY
	out2, err := visionAttentionForward(x2, w, 2, gridW, cfg)
	if err != nil {
		t.Fatalf("visionAttentionForward (perturbed): %v", err)
	}
	same := true
	for i := range hidden {
		if out1[i] != out2[i] {
			same = false
		}
	}
	if same {
		t.Fatal("token 0's output did not change when token 1 changed — attention is not bidirectional")
	}
}

func TestVisionAttentionForward_Bad(t *testing.T) {
	w := &visionAttnWeights{Q: identityLinear(4), K: identityLinear(4), V: identityLinear(4), O: identityLinear(4)}
	cfg := visionTowerCfg{NumHeads: 1, NumKVHeads: 0, HeadDim: 4} // NumKVHeads=0 is invalid (division by zero guard)
	if _, err := visionAttentionForward(make([]float32, 8), w, 2, 2, cfg); err == nil {
		t.Fatal("visionAttentionForward: want an error for NumKVHeads=0, got nil")
	}
}

func TestVisionMLPForward_Good(t *testing.T) {
	w := &visionMLPWeights{
		Gate: visionLinear{W: []float32{1}, Out: 1, In: 1},
		Up:   visionLinear{W: []float32{1}, Out: 1, In: 1},
		Down: visionLinear{W: []float32{1}, Out: 1, In: 1},
	}
	got := visionMLPForward([]float32{2}, w, 1)
	want := float32(silu(2) * 2)
	if got[0] != want {
		t.Fatalf("visionMLPForward = %v, want %v (silu(2)*2, identity down-proj)", got[0], want)
	}
}

func TestVisionBlockForward_Good(t *testing.T) {
	tower := mkVisionTower(1, 2, 8, 1)
	block := &tower.Blocks[0]
	x := syn(2*8, 500) // L=2 rows, Hidden=8
	out, err := block.forward(append([]float32(nil), x...), 2, 2, tower.Cfg)
	if err != nil {
		t.Fatalf("visionBlock.forward: %v", err)
	}
	if len(out) != len(x) {
		t.Fatalf("visionBlock.forward output len = %d, want %d", len(out), len(x))
	}
	same := true
	for i := range x {
		if out[i] != x[i] {
			same = false
		}
	}
	if same {
		t.Fatal("visionBlock.forward returned the input unchanged")
	}
	out2, err := block.forward(append([]float32(nil), x...), 2, 2, tower.Cfg)
	if err != nil {
		t.Fatalf("visionBlock.forward (rerun): %v", err)
	}
	for i := range out {
		if out[i] != out2[i] {
			t.Fatalf("visionBlock.forward is not deterministic: [%d] %v != %v", i, out[i], out2[i])
		}
	}
}

func TestMergeSpatialBlocks_Good(t *testing.T) {
	// A 2x2 grid of hidden=1 "patches" merges into ONE row, gathered (0,0),(0,1),(1,0),(1,1).
	x := []float32{10, 20, 30, 40} // raster order: (0,0)=10 (0,1)=20 (1,0)=30 (1,1)=40
	got := mergeSpatialBlocks(x, 2, 2, 1, 2)
	want := []float32{10, 20, 30, 40}
	if len(got) != len(want) {
		t.Fatalf("mergeSpatialBlocks len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("mergeSpatialBlocks[%d] = %v, want %v (gather order)", i, got[i], want[i])
		}
	}

	// A 4x4 grid merges into 4 rows (2x2 merge-blocks); the top-right block (by=0,bx=2) gathers
	// patches (0,2),(0,3),(1,2),(1,3).
	x4 := make([]float32, 16)
	for i := range x4 {
		x4[i] = float32(i)
	}
	got4 := mergeSpatialBlocks(x4, 4, 4, 1, 2)
	if len(got4) != 16 {
		t.Fatalf("mergeSpatialBlocks(4x4) len = %d, want 16 (4 rows x 4 cols)", len(got4))
	}
	// block index 1 = (by=0,bx=2) — row-major over the 2x2 grid of merge-blocks.
	wantBlock1 := []float32{2, 3, 6, 7} // raster indices (0,2)=2 (0,3)=3 (1,2)=6 (1,3)=7
	for i, w := range wantBlock1 {
		if got4[1*4+i] != w {
			t.Fatalf("mergeSpatialBlocks(4x4) block 1[%d] = %v, want %v", i, got4[1*4+i], w)
		}
	}
}

// TestVisionMergerForward_BlockIsolation proves the merger's per-merge-block independence with the REAL
// (non-toy) LayerNorm→gather→Linear1→GELU→Linear2 pipeline: perturbing ONE input patch's value changes
// ONLY the output row for the merge-block that patch belongs to — every other output row stays
// bit-identical, since LayerNorm is per-row and the gather never mixes across merge-blocks.
func TestVisionMergerForward_BlockIsolation(t *testing.T) {
	tower := mkVisionTower(1, 2, 8, 7)
	const gridH, gridW = 4, 4
	x := syn(gridH*gridW*tower.Cfg.Hidden, 42)

	out1, softTokens, err := tower.Merger.forward(append([]float32(nil), x...), gridH, gridW, tower.Cfg)
	if err != nil {
		t.Fatalf("visionMerger.forward: %v", err)
	}
	if softTokens != 4 {
		t.Fatalf("softTokens = %d, want 4 ((4/2)*(4/2))", softTokens)
	}
	if len(out1) != softTokens*tower.Cfg.TextHidden {
		t.Fatalf("features len = %d, want %d (softTokens*TextHidden)", len(out1), softTokens*tower.Cfg.TextHidden)
	}

	// Perturb patch (0,2) — raster index 0*gridW+2=2 — which belongs to merge-block (by=0,bx=2), the
	// SECOND output row (block index 1, matching TestMergeSpatialBlocks_Good's indexing).
	x2 := append([]float32(nil), x...)
	patchOff := 2 * tower.Cfg.Hidden
	x2[patchOff] += 5
	out2, _, err := tower.Merger.forward(x2, gridH, gridW, tower.Cfg)
	if err != nil {
		t.Fatalf("visionMerger.forward (perturbed): %v", err)
	}

	TH := tower.Cfg.TextHidden
	for block := range softTokens {
		row1 := out1[block*TH : (block+1)*TH]
		row2 := out2[block*TH : (block+1)*TH]
		changed := false
		for i := range row1 {
			if row1[i] != row2[i] {
				changed = true
			}
		}
		if block == 1 && !changed {
			t.Fatal("perturbing patch (0,2) did not change merge-block 1's output row")
		}
		if block != 1 && changed {
			t.Fatalf("perturbing patch (0,2) changed merge-block %d's output row — merge blocks are not independent", block)
		}
	}
}

func TestVisionMergerForward_Bad(t *testing.T) {
	tower := mkVisionTower(1, 2, 8, 3)
	// gridH=3 is not divisible by MergeSize=2.
	if _, _, err := tower.Merger.forward(syn(3*4*8, 9), 3, 4, tower.Cfg); err == nil {
		t.Fatal("visionMerger.forward: want an error for a non-divisible grid, got nil")
	}
}

func TestVisionTowerForward_Good(t *testing.T) {
	tower := mkVisionTower(2, 2, 8, 11)
	const gridH, gridW = 4, 4 // 16 patches, PatchDim=12
	patches := syn(gridH*gridW*tower.Cfg.PatchDim, 77)

	features, softTokens, err := visionTowerForward(patches, gridH, gridW, tower)
	if err != nil {
		t.Fatalf("visionTowerForward: %v", err)
	}
	wantSoft := (gridH / tower.Cfg.MergeSize) * (gridW / tower.Cfg.MergeSize)
	if softTokens != wantSoft {
		t.Fatalf("softTokens = %d, want %d", softTokens, wantSoft)
	}
	if len(features) != softTokens*tower.Cfg.TextHidden {
		t.Fatalf("features len = %d, want %d ([softTokens x TextHidden])", len(features), softTokens*tower.Cfg.TextHidden)
	}
	for i, v := range features {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("features[%d] = %v (NaN/Inf)", i, v)
		}
	}

	features2, _, err := visionTowerForward(append([]float32(nil), patches...), gridH, gridW, tower)
	if err != nil {
		t.Fatalf("visionTowerForward (rerun): %v", err)
	}
	for i := range features {
		if features[i] != features2[i] {
			t.Fatalf("visionTowerForward is not deterministic: [%d] %v != %v", i, features[i], features2[i])
		}
	}
}

func TestVisionTowerForward_Bad(t *testing.T) {
	tower := mkVisionTower(1, 2, 8, 1)
	if _, _, err := visionTowerForward(syn(5, 1), 4, 4, tower); err == nil {
		t.Fatal("visionTowerForward: want an error for a patch buffer size mismatch, got nil")
	}
}

func TestImageToPatchGrid_Good(t *testing.T) {
	data := mkSyntheticPNG(t, 8, 8)
	cfg := visionTowerCfg{PatchSize: 2, InChannels: 3, TemporalPatchSize: 1, MergeSize: 2}
	patches, gridH, gridW, err := imageToPatchGrid(data, cfg)
	if err != nil {
		t.Fatalf("imageToPatchGrid: %v", err)
	}
	if gridH != 4 || gridW != 4 {
		t.Fatalf("grid = %dx%d, want 4x4 (8px / 2px patch)", gridH, gridW)
	}
	wantLen := gridH * gridW * (3 * 2 * 2) * 1
	if len(patches) != wantLen {
		t.Fatalf("len(patches) = %d, want %d", len(patches), wantLen)
	}
	for i, v := range patches {
		if v < 0 || v > 1 {
			t.Fatalf("patches[%d] = %v, want in [0,1]", i, v)
		}
	}
}

func TestImageToPatchGrid_TemporalReplication(t *testing.T) {
	data := mkSyntheticPNG(t, 4, 4)
	cfg := visionTowerCfg{PatchSize: 2, InChannels: 3, TemporalPatchSize: 2, MergeSize: 1}
	patches, gridH, gridW, err := imageToPatchGrid(data, cfg)
	if err != nil {
		t.Fatalf("imageToPatchGrid: %v", err)
	}
	perFrame := 3 * 2 * 2
	wantLen := gridH * gridW * perFrame * 2
	if len(patches) != wantLen {
		t.Fatalf("len(patches) = %d, want %d (TemporalPatchSize=2)", len(patches), wantLen)
	}
	frame0 := patches[0:perFrame]
	frame1 := patches[perFrame : 2*perFrame]
	for i := range frame0 {
		if frame0[i] != frame1[i] {
			t.Fatalf("temporal replication: frame1[%d]=%v != frame0[%d]=%v", i, frame1[i], i, frame0[i])
		}
	}
}

func TestImageToPatchGrid_CropsToMultiple(t *testing.T) {
	data := mkSyntheticPNG(t, 7, 7) // not a multiple of patch(2)*merge(1)... use patch=2 so floor(7/2)=3
	cfg := visionTowerCfg{PatchSize: 2, InChannels: 3, TemporalPatchSize: 1, MergeSize: 1}
	_, gridH, gridW, err := imageToPatchGrid(data, cfg)
	if err != nil {
		t.Fatalf("imageToPatchGrid: %v", err)
	}
	if gridH != 3 || gridW != 3 {
		t.Fatalf("grid = %dx%d, want 3x3 (7px cropped down to 6px / 2px patch)", gridH, gridW)
	}
}

func TestImageToPatchGrid_Bad(t *testing.T) {
	data := mkSyntheticPNG(t, 2, 2)
	cfg := visionTowerCfg{PatchSize: 4, InChannels: 3, TemporalPatchSize: 1, MergeSize: 1}
	if _, _, _, err := imageToPatchGrid(data, cfg); err == nil {
		t.Fatal("imageToPatchGrid: want an error for an image smaller than one patch, got nil")
	}
}
