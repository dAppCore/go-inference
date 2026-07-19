// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import "testing"

// vision_sam_test.go gates SAM's ported ops against vision_toy_golden.json — captured from the
// REAL deepencoder.PatchEmbed/Block/nn.Conv2d/LayerNorm2d classes, run directly at toy dims (see
// golden_test.go's doc comment).

func samBlockWeightsFromGolden(t *testing.T, g samBlockGolden) SAMBlockWeights {
	t.Helper()
	relPosLen := 2*g.WindowSize - 1
	if g.WindowSize == 0 {
		relPosLen = 2*g.Grid[0] - 1
	}
	w := g.Weights
	return SAMBlockWeights{
		Norm1W: w.mustGet(t, "norm1.weight"), Norm1B: w.mustGet(t, "norm1.bias"),
		Attn: SAMAttnWeights{
			QKVWeight: w.mustGet(t, "attn.qkv.weight"), QKVBias: w.mustGet(t, "attn.qkv.bias"),
			ProjWeight: w.mustGet(t, "attn.proj.weight"), ProjBias: w.mustGet(t, "attn.proj.bias"),
			RelPosH: w.mustGet(t, "attn.rel_pos_h"), RelPosW: w.mustGet(t, "attn.rel_pos_w"),
			RelPosLen: relPosLen,
		},
		Norm2W: w.mustGet(t, "norm2.weight"), Norm2B: w.mustGet(t, "norm2.bias"),
		MLPLin1W: w.mustGet(t, "mlp.lin1.weight"), MLPLin1B: w.mustGet(t, "mlp.lin1.bias"),
		MLPLin2W: w.mustGet(t, "mlp.lin2.weight"), MLPLin2B: w.mustGet(t, "mlp.lin2.bias"),
		WindowSize: g.WindowSize,
	}
}

// TestSAMPatchEmbed_Golden_Good pins conv2D + the pos_embed add (via get_abs_pos_sam's
// direct-match no-interpolation branch, since the golden's grid matches its pos_embed exactly)
// against the real deepencoder.PatchEmbed class.
func TestSAMPatchEmbed_Golden_Good(t *testing.T) {
	g := readVisionToyGolden(t).PatchEmbed
	out, outH, outW := conv2D(g.Input, g.InputShape[1], g.InputShape[2], g.InChans, g.Weight, g.Bias, g.EmbedDim, g.Kernel, g.Kernel, 0)
	if outH != g.OutputShape[1] || outW != g.OutputShape[2] {
		t.Fatalf("conv2D grid = %dx%d, want %dx%d", outH, outW, g.OutputShape[1], g.OutputShape[2])
	}
	got := addRows(out, g.PosEmbed)
	if d := maxAbsDiff32(t, got, g.Output); d > 1e-4 {
		t.Fatalf("patch embed + pos_embed max abs diff %g, want <=1e-4", d)
	}
}

// TestSAMBlockForward_Windowed_Golden_Good pins samBlockForward's window-partition path
// (window_size=4 over an 8x8 grid — 4 windows of 4x4, exercising deepencoder's
// window_partition/window_unpartition AND the windowed rel_pos_h/w table, len 2*4-1=7) against
// the real deepencoder.Block class.
func TestSAMBlockForward_Windowed_Golden_Good(t *testing.T) {
	g := readVisionToyGolden(t).SAMBlockWindowed
	b := samBlockWeightsFromGolden(t, g)
	got := samBlockForward(g.Input, g.Grid[0], g.Grid[1], g.NumHeads, b)
	if d := maxAbsDiff32(t, got, g.Output); d > 1e-3 {
		t.Fatalf("windowed SAM block max abs diff %g, want <=1e-3", d)
	}
}

// TestSAMBlockForward_Global_Golden_Good pins samBlockForward's global (window_size=0) path
// (full 8x8-grid attention, rel_pos_h/w table len 2*8-1=15 — a DIFFERENT table size from the
// windowed test above, proving both branches of weights_sam.go's RelPosLen selection) against the
// real deepencoder.Block class.
func TestSAMBlockForward_Global_Golden_Good(t *testing.T) {
	g := readVisionToyGolden(t).SAMBlockGlobal
	if g.WindowSize != 0 {
		t.Fatalf("golden fixture WindowSize = %d, want 0 (global block)", g.WindowSize)
	}
	b := samBlockWeightsFromGolden(t, g)
	got := samBlockForward(g.Input, g.Grid[0], g.Grid[1], g.NumHeads, b)
	if d := maxAbsDiff32(t, got, g.Output); d > 1e-3 {
		t.Fatalf("global SAM block max abs diff %g, want <=1e-3", d)
	}
}

// TestSAMNeckDownsample_Golden_Good pins the neck (Conv1x1->LayerNorm2d->Conv3x3pad1->
// LayerNorm2d) + net_2/net_3 (two stride-2 Conv3x3pad1) chain — deepencoder.ImageEncoderViT.
// forward's tail — against the real nn.Conv2d/LayerNorm2d classes, at toy channel counts
// (8->6->6 neck, 6->8 net_2, 8->10 net_3) since the checkpoint's real 768->256->256->512->1024
// progression is hardcoded and untestable at toy scale (see weights_sam.go's doc comment) — the
// live E2E gate proves the real channel counts load and run (live_test.go).
func TestSAMNeckDownsample_Golden_Good(t *testing.T) {
	g := readVisionToyGolden(t).SAMNeckDownsample
	w := g.Weights
	gh, gw := g.InputShape[1], g.InputShape[2]
	inC := g.InputShape[3]
	neck1, gh, gw := conv2D(g.Input, gh, gw, inC, w.mustGet(t, "0.weight"), nil, len(w.mustGet(t, "1.bias")), 1, 1, 0)
	neck1 = layerNorm2D(neck1, w.mustGet(t, "1.weight"), w.mustGet(t, "1.bias"), len(w.mustGet(t, "1.bias")), 1e-6)
	neck2, gh, gw := conv2D(neck1, gh, gw, len(w.mustGet(t, "1.bias")), w.mustGet(t, "2.weight"), nil, len(w.mustGet(t, "3.bias")), 3, 1, 1)
	neck2 = layerNorm2D(neck2, w.mustGet(t, "3.weight"), w.mustGet(t, "3.bias"), len(w.mustGet(t, "3.bias")), 1e-6)
	if d := maxAbsDiff32(t, neck2, g.NeckOut); d > 1e-3 {
		t.Fatalf("neck output max abs diff %g, want <=1e-3", d)
	}

	net2W := w.mustGet(t, "net_2.weight")
	net2Out := len(net2W) / (len(w.mustGet(t, "3.bias")) * 3 * 3)
	net2, gh, gw := conv2D(neck2, gh, gw, len(w.mustGet(t, "3.bias")), net2W, nil, net2Out, 3, 2, 1)
	net3W := w.mustGet(t, "net_3.weight")
	net3Out := len(net3W) / (net2Out * 3 * 3)
	net3, _, _ := conv2D(net2, gh, gw, net2Out, net3W, nil, net3Out, 3, 2, 1)
	if d := maxAbsDiff32(t, net3, g.Output); d > 1e-3 {
		t.Fatalf("net_2/net_3 output max abs diff %g, want <=1e-3", d)
	}
}

// TestSAMForward_Bad proves the pixel-buffer shape guard fires before any tower weight is even
// touched (a nil/zero-value SAMWeights is enough — the length check is the FIRST thing SAMForward
// does).
func TestSAMForward_Bad(t *testing.T) {
	_, err := SAMForward(make([]float32, 10), SAMWeights{})
	if err == nil {
		t.Fatal("SAMForward accepted a pixel buffer far shorter than 1024x1024x3")
	}
}
