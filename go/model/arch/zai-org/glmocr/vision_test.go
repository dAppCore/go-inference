// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"fmt"
	"testing"
)

// buildVisionWeightsFromGolden maps testdata/block_goldens.json's "vision".state_dict (captured
// straight off a REAL torch GlmOcrVisionModel's state_dict()) onto this package's VisionWeights
// — the same tensor names/shapes LoadWeights reads from a real checkpoint's safetensors, just
// sourced from JSON instead of an mmap.
func buildVisionWeightsFromGolden(t *testing.T, g visionBlockGolden) *VisionWeights {
	t.Helper()
	sd := g.StateDict
	get := func(name string) []float32 {
		v, ok := sd[name]
		if !ok {
			t.Fatalf("golden vision state_dict missing %q", name)
		}
		return v
	}
	hidden, ff := g.Config.HiddenSize, g.Config.IntermediateSize
	patchDim := g.Config.InChannels * g.Config.TemporalPatchSize * g.Config.PatchSize * g.Config.PatchSize
	outHidden := g.Config.OutHiddenSize
	merge := g.Config.SpatialMergeSize

	block := func(i int) VisionBlockWeights {
		p := func(s string) string { return fmt.Sprintf("blocks.%d.%s", i, s) }
		qkvW, qkvB := get(p("attn.qkv.weight")), get(p("attn.qkv.bias"))
		return VisionBlockWeights{
			Norm1: RMSNormWeights{Weight: get(p("norm1.weight"))},
			Norm2: RMSNormWeights{Weight: get(p("norm2.weight"))},
			Attn: VisionAttnWeights{
				Q:     LinearWeights{Weight: qkvW[0 : hidden*hidden], Bias: qkvB[0:hidden], In: hidden, Out: hidden},
				K:     LinearWeights{Weight: qkvW[hidden*hidden : 2*hidden*hidden], Bias: qkvB[hidden : 2*hidden], In: hidden, Out: hidden},
				V:     LinearWeights{Weight: qkvW[2*hidden*hidden : 3*hidden*hidden], Bias: qkvB[2*hidden : 3*hidden], In: hidden, Out: hidden},
				Proj:  LinearWeights{Weight: get(p("attn.proj.weight")), Bias: get(p("attn.proj.bias")), In: hidden, Out: hidden},
				QNorm: RMSNormWeights{Weight: get(p("attn.q_norm.weight"))},
				KNorm: RMSNormWeights{Weight: get(p("attn.k_norm.weight"))},
			},
			MLP: VisionMLPWeights{
				Gate: LinearWeights{Weight: get(p("mlp.gate_proj.weight")), Bias: get(p("mlp.gate_proj.bias")), In: hidden, Out: ff},
				Up:   LinearWeights{Weight: get(p("mlp.up_proj.weight")), Bias: get(p("mlp.up_proj.bias")), In: hidden, Out: ff},
				Down: LinearWeights{Weight: get(p("mlp.down_proj.weight")), Bias: get(p("mlp.down_proj.bias")), In: ff, Out: hidden},
			},
		}
	}
	blocks := make([]VisionBlockWeights, g.Config.Depth)
	for i := range blocks {
		blocks[i] = block(i)
	}
	return &VisionWeights{
		PatchEmbed:    LinearWeights{Weight: get("patch_embed.proj.weight"), Bias: get("patch_embed.proj.bias"), In: patchDim, Out: hidden},
		Blocks:        blocks,
		PostLayernorm: RMSNormWeights{Weight: get("post_layernorm.weight")},
		Downsample:    LinearWeights{Weight: get("downsample.weight"), Bias: get("downsample.bias"), In: hidden * merge * merge, Out: outHidden},
		Merger: VisionMergerWeights{
			Proj:               LinearWeights{Weight: get("merger.proj.weight"), In: outHidden, Out: outHidden},
			PostProjectionNorm: LayerNormWeights{Weight: get("merger.post_projection_norm.weight"), Bias: get("merger.post_projection_norm.bias")},
			Gate:               LinearWeights{Weight: get("merger.gate_proj.weight"), In: outHidden, Out: outHidden * g.Config.InChannels},
			Up:                 LinearWeights{Weight: get("merger.up_proj.weight"), In: outHidden, Out: outHidden * g.Config.InChannels},
			Down:               LinearWeights{Weight: get("merger.down_proj.weight"), In: outHidden * g.Config.InChannels, Out: outHidden},
		},
	}
}

func TestVisionBlockForward_BlockGoldens_Good(t *testing.T) {
	g := readBlockGoldens(t).Vision
	w := buildVisionWeightsFromGolden(t, g)
	vc := &VisionConfig{
		HiddenSize: g.Config.HiddenSize, NumHeads: g.Config.NumHeads, Depth: g.Config.Depth,
		IntermediateSize: g.Config.IntermediateSize, RMSNormEps: g.Config.RMSNormEps,
	}
	T := len(g.PixelValues) / (g.Config.InChannels * g.Config.TemporalPatchSize * g.Config.PatchSize * g.Config.PatchSize)
	headDim := vc.HiddenSize / vc.NumHeads

	patchOut := linearForward(g.PixelValues, w.PatchEmbed, T)
	if d := maxAbsDiff32(t, patchOut, g.PatchEmbedOut); d > 1e-4 {
		t.Fatalf("patch_embed maxAbsDiff = %v, want < 1e-4", d)
	}

	gridT, gridH, gridW := g.GridTHW[0][0], g.GridTHW[0][1], g.GridTHW[0][2]
	hpos, wpos := visionPosIDs(gridT, gridH, gridW, g.Config.SpatialMergeSize)
	cos, sin := visionCosSin(hpos, wpos, headDim, 10000.0)

	block0 := visionBlockForward(patchOut, T, vc.HiddenSize, vc.IntermediateSize, vc.NumHeads, headDim, w.Blocks[0], cos, sin, vc.RMSNormEps)
	if d := maxAbsDiff32(t, block0, g.Block0Out); d > 1e-3 {
		t.Fatalf("block0 maxAbsDiff = %v, want < 1e-3", d)
	}

	block1 := visionBlockForward(block0, T, vc.HiddenSize, vc.IntermediateSize, vc.NumHeads, headDim, w.Blocks[1], cos, sin, vc.RMSNormEps)
	if d := maxAbsDiff32(t, block1, g.Block1Out); d > 1e-3 {
		t.Fatalf("block1 maxAbsDiff = %v, want < 1e-3", d)
	}
}

func TestVisionBlockForward_BlockGoldens_Bad(t *testing.T) {
	// a single-block stack must NOT reproduce the two-block golden — proves the test itself can
	// fail (not a tautology) and that each block genuinely transforms its input.
	g := readBlockGoldens(t).Vision
	w := buildVisionWeightsFromGolden(t, g)
	vc := &VisionConfig{HiddenSize: g.Config.HiddenSize, NumHeads: g.Config.NumHeads, IntermediateSize: g.Config.IntermediateSize, RMSNormEps: g.Config.RMSNormEps}
	T := len(g.PixelValues) / (g.Config.InChannels * g.Config.TemporalPatchSize * g.Config.PatchSize * g.Config.PatchSize)
	headDim := vc.HiddenSize / vc.NumHeads
	gridT, gridH, gridW := g.GridTHW[0][0], g.GridTHW[0][1], g.GridTHW[0][2]
	hpos, wpos := visionPosIDs(gridT, gridH, gridW, g.Config.SpatialMergeSize)
	cos, sin := visionCosSin(hpos, wpos, headDim, 10000.0)
	patchOut := linearForward(g.PixelValues, w.PatchEmbed, T)
	block0 := visionBlockForward(patchOut, T, vc.HiddenSize, vc.IntermediateSize, vc.NumHeads, headDim, w.Blocks[0], cos, sin, vc.RMSNormEps)
	if d := maxAbsDiff32(t, block0, g.Block1Out); d < 1e-3 {
		t.Fatalf("block0 output unexpectedly matches the block1 golden (d=%v) — the two blocks have different weights and must diverge", d)
	}
}

func TestDownsampleForward_BlockGoldens_Good(t *testing.T) {
	g := readBlockGoldens(t).Vision
	w := buildVisionWeightsFromGolden(t, g)
	T := len(g.PostLayernormOut) / g.Config.HiddenSize
	got := downsampleForward(g.PostLayernormOut, T, g.Config.HiddenSize, g.Config.SpatialMergeSize, w.Downsample)
	if d := maxAbsDiff32(t, got, g.DownsampleOut); d > 1e-3 {
		t.Fatalf("downsampleForward maxAbsDiff = %v, want < 1e-3", d)
	}
}

func TestDownsampleForward_Bad(t *testing.T) {
	// a uniform input (every merged position identical) must produce IDENTICAL output rows —
	// downsample has no cross-block mixing.
	hiddenV, merge, outHidden := 2, 2, 3
	w := LinearWeights{Weight: []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
	}, Bias: []float32{0, 0, 0}, In: hiddenV * merge * merge, Out: outHidden}
	// two identical 2x2 blocks (T=8 rows, hiddenV=2 each)
	block := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	hidden := append(append([]float32{}, block...), block...)
	got := downsampleForward(hidden, 8, hiddenV, merge, w)
	if len(got) != 2*outHidden {
		t.Fatalf("downsampleForward output length = %d, want %d", len(got), 2*outHidden)
	}
	for i := range outHidden {
		if got[i] != got[outHidden+i] {
			t.Fatalf("downsampleForward identical blocks produced different outputs: %v vs %v", got[:outHidden], got[outHidden:])
		}
	}
}

func TestDownsampleForward_Ugly(t *testing.T) {
	// the gather must NOT simply concatenate the 4 sub-patch rows in sequence order — it groups
	// by CHANNEL first (weight layout [out,in,mh,mw]). Prove this by checking a downsample whose
	// weight only reads channel 0's four (mh,mw) values ignores channel 1 entirely.
	hiddenV, merge := 2, 2
	w := LinearWeights{Weight: []float32{1, 1, 1, 1, 0, 0, 0, 0}, Bias: []float32{0}, In: hiddenV * merge * merge, Out: 1}
	hidden := []float32{10, 999, 20, 999, 30, 999, 40, 999} // channel0=10,20,30,40 channel1=999 (ignored by this weight)
	got := downsampleForward(hidden, 4, hiddenV, merge, w)
	want := float32(10 + 20 + 30 + 40)
	if d := absDiff32(got[0], want); d > 1e-4 {
		t.Fatalf("downsampleForward channel gather = %v, want %v (sum of channel-0 values only)", got[0], want)
	}
}

func TestVisionMergerForward_BlockGoldens_Good(t *testing.T) {
	g := readBlockGoldens(t).Vision
	w := buildVisionWeightsFromGolden(t, g)
	numMerged := len(g.DownsampleOut) / g.Config.OutHiddenSize
	got := visionMergerForward(g.DownsampleOut, numMerged, w.Merger)
	if d := maxAbsDiff32(t, got, g.MergerOut); d > 1e-3 {
		t.Fatalf("visionMergerForward maxAbsDiff = %v, want < 1e-3", d)
	}
	if d := maxAbsDiff32(t, got, g.PoolerOutput); d > 1e-3 {
		t.Fatalf("visionMergerForward vs pooler_output maxAbsDiff = %v, want < 1e-3", d)
	}
}

func TestVisionMergerForward_Bad(t *testing.T) {
	// zero gate/up weights -> silu(0)*0=0 -> down(0)=bias-free zero, regardless of proj/norm
	dim := 2
	w := VisionMergerWeights{
		Proj:               LinearWeights{Weight: []float32{1, 0, 0, 1}, In: dim, Out: dim},
		PostProjectionNorm: LayerNormWeights{Weight: []float32{1, 1}, Bias: []float32{0, 0}},
		Gate:               LinearWeights{Weight: []float32{0, 0, 0, 0}, In: dim, Out: dim},
		Up:                 LinearWeights{Weight: []float32{0, 0, 0, 0}, In: dim, Out: dim},
		Down:               LinearWeights{Weight: []float32{1, 0, 0, 1}, In: dim, Out: dim},
	}
	got := visionMergerForward([]float32{5, -5}, 1, w)
	if got[0] != 0 || got[1] != 0 {
		t.Fatalf("visionMergerForward with zero gate/up = %v, want [0 0]", got)
	}
}

func TestVisionMergerForward_Ugly(t *testing.T) {
	// T=2 rows must be processed independently (no cross-row mixing anywhere in the merger).
	// dim=3 (neither 1 nor 2): a single-feature LayerNorm always self-normalises to exactly 0
	// (zero self-variance), and ANY two-feature row normalises to exactly [-1,+1] or [+1,-1]
	// regardless of its actual values (a 2-vector's z-score is a pure sign) — both would make
	// this test pass vacuously even if rows leaked into each other. row0/row1 below also differ
	// in SHAPE, not just an additive/multiplicative transform of each other (LayerNorm is
	// invariant to both), so their normalised — and therefore final — outputs must differ.
	dim := 3
	identity := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
	w := VisionMergerWeights{
		Proj:               LinearWeights{Weight: identity, In: dim, Out: dim},
		PostProjectionNorm: LayerNormWeights{Weight: []float32{1, 1, 1}, Bias: []float32{0, 0, 0}},
		Gate:               LinearWeights{Weight: identity, In: dim, Out: dim},
		Up:                 LinearWeights{Weight: identity, In: dim, Out: dim},
		Down:               LinearWeights{Weight: identity, In: dim, Out: dim},
	}
	got := visionMergerForward([]float32{1, 2, 4, 1, 2, 9}, 2, w)
	row0, row1 := got[:dim], got[dim:]
	same := true
	for i := range row0 {
		if absDiff32(row0[i], row1[i]) > 1e-4 {
			same = false
		}
	}
	if same {
		t.Fatalf("visionMergerForward T=2 rows collapsed: row0=%v row1=%v", row0, row1)
	}
}

func TestVisionForward_BlockGoldens_Good(t *testing.T) {
	g := readBlockGoldens(t).Vision
	w := buildVisionWeightsFromGolden(t, g)
	vc := &VisionConfig{
		HiddenSize: g.Config.HiddenSize, NumHeads: g.Config.NumHeads, Depth: g.Config.Depth,
		IntermediateSize: g.Config.IntermediateSize, RMSNormEps: g.Config.RMSNormEps,
		SpatialMergeSize: g.Config.SpatialMergeSize, OutHiddenSize: g.Config.OutHiddenSize, InChannels: g.Config.InChannels,
	}
	patchDim := g.Config.InChannels * g.Config.TemporalPatchSize * g.Config.PatchSize * g.Config.PatchSize
	patches := &PatchGrid{
		Patches: g.PixelValues, PatchDim: patchDim,
		GridT: g.GridTHW[0][0], GridH: g.GridTHW[0][1], GridW: g.GridTHW[0][2],
	}
	got, numMerged, err := VisionForward(patches, w, vc)
	if err != nil {
		t.Fatalf("VisionForward: %v", err)
	}
	wantMerged := len(g.PoolerOutput) / vc.OutHiddenSize
	if numMerged != wantMerged {
		t.Fatalf("VisionForward numMerged = %d, want %d", numMerged, wantMerged)
	}
	if d := maxAbsDiff32(t, got, g.PoolerOutput); d > 1e-3 {
		t.Fatalf("VisionForward maxAbsDiff vs pooler_output = %v, want < 1e-3", d)
	}
}

func TestVisionForward_Bad(t *testing.T) {
	if _, _, err := VisionForward(nil, &VisionWeights{}, &VisionConfig{}); err == nil {
		t.Fatal("VisionForward accepted nil patches")
	}
}

func TestVisionForward_Ugly(t *testing.T) {
	// a patch grid whose element count doesn't divide evenly by PatchDim must refuse, not
	// silently truncate/panic
	patches := &PatchGrid{Patches: make([]float32, 5), PatchDim: 3, GridT: 1, GridH: 2, GridW: 2}
	if _, _, err := VisionForward(patches, &VisionWeights{}, &VisionConfig{HiddenSize: 2, NumHeads: 1}); err == nil {
		t.Fatal("VisionForward accepted a patch grid whose length is not a multiple of PatchDim")
	}
}
