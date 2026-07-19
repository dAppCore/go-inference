// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import "testing"

// vision_clip_test.go gates CLIP's ported ops against vision_toy_golden.json — captured from the
// REAL deepencoder.CLIPVisionEmbeddings/NoTPTransformerBlock classes, run directly at toy dims
// (see golden_test.go's doc comment).

// TestClipEmbeddings_Golden_Good pins the CLS-prepend + direct (no-interpolation) position-table
// add against the real deepencoder.CLIPVisionEmbeddings class, fed an external patch_embeds
// tensor (the checkpoint's actual wiring — see vision_clip.go's doc comment).
func TestClipEmbeddings_Golden_Good(t *testing.T) {
	g := readVisionToyGolden(t).CLIPEmbeddings
	w := CLIPWeights{ClassEmbedding: g.ClassEmbedding, PositionEmbedding: g.PositionEmbeddingWeight}
	got, err := clipEmbeddings(g.PatchEmbeds, w)
	if err != nil {
		t.Fatalf("clipEmbeddings: %v", err)
	}
	if d := maxAbsDiff32(t, got, g.Output); d > 1e-4 {
		t.Fatalf("CLIP embeddings max abs diff %g, want <=1e-4", d)
	}
}

// TestClipEmbeddings_Bad proves a patch-token count that doesn't leave exactly one CLS-plus-grid
// short of the loaded position table is refused (the general resize/interpolated-position-table
// path this v1 lane does not implement — vision_clip.go's doc comment).
func TestClipEmbeddings_Bad(t *testing.T) {
	g := readVisionToyGolden(t).CLIPEmbeddings
	w := CLIPWeights{ClassEmbedding: g.ClassEmbedding, PositionEmbedding: g.PositionEmbeddingWeight}
	short := g.PatchEmbeds[:len(g.PatchEmbeds)-len(g.ClassEmbedding)] // one patch token fewer
	if _, err := clipEmbeddings(short, w); err == nil {
		t.Fatal("clipEmbeddings accepted a patch-token count that does not match the loaded position table")
	}
}

// TestClipBlockForward_Golden_Good pins one full pre-norm block (combined-QKV attention,
// quick-GELU MLP) against the real deepencoder.NoTPTransformerBlock class.
func TestClipBlockForward_Golden_Good(t *testing.T) {
	g := readVisionToyGolden(t).CLIPBlock
	w := g.Weights
	b := CLIPBlockWeights{
		Norm1W: w.mustGet(t, "layer_norm1.weight"), Norm1B: w.mustGet(t, "layer_norm1.bias"),
		QKVWeight: w.mustGet(t, "self_attn.qkv_proj.weight"), QKVBias: w.mustGet(t, "self_attn.qkv_proj.bias"),
		OutWeight: w.mustGet(t, "self_attn.out_proj.weight"), OutBias: w.mustGet(t, "self_attn.out_proj.bias"),
		Norm2W: w.mustGet(t, "layer_norm2.weight"), Norm2B: w.mustGet(t, "layer_norm2.bias"),
		FC1Weight: w.mustGet(t, "mlp.fc1.weight"), FC1Bias: w.mustGet(t, "mlp.fc1.bias"),
		FC2Weight: w.mustGet(t, "mlp.fc2.weight"), FC2Bias: w.mustGet(t, "mlp.fc2.bias"),
	}
	got := clipBlockForward(g.Input, g.NumHeads, b)
	if d := maxAbsDiff32(t, got, g.Output); d > 1e-3 {
		t.Fatalf("CLIP block max abs diff %g, want <=1e-3", d)
	}
}

// TestCLIPForward_Bad proves clipEmbeddings' shape refusal propagates through the top-level
// CLIPForward entry point (a nil CLIPWeights' zero-value ClassEmbedding/PositionEmbedding makes
// every patch count "wrong").
func TestCLIPForward_Bad(t *testing.T) {
	_, err := CLIPForward(make([]float32, 16), CLIPWeights{})
	if err == nil {
		t.Fatal("CLIPForward accepted patch embeds against a zero-value (empty position table) CLIPWeights")
	}
}
