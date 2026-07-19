// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// f32Tensor builds a synthetic F32 safetensors.Tensor of the given shape, values counting up
// from 0.1 — deterministic, non-degenerate (never all-zero, so a shape/wiring bug that swaps two
// same-sized tensors is still observable), never needing a real checkpoint on disk.
func f32Tensor(shape ...int) safetensors.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i)*0.01 + 0.1
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: safetensors.EncodeFloat32(v)}
}

// syntheticCheckpoint builds a MINIMAL but complete tensor map covering every name LoadWeights
// reads (vision depth=1, text 1 layer) — small enough to hand-construct, large enough to prove
// every prefix/shape LoadWeights expects actually resolves against a real safetensors.Tensor
// map (not just this package's own types).
func syntheticCheckpoint() (map[string]safetensors.Tensor, *Config) {
	const hidden, ff, heads, headDim = 4, 4, 2, 2
	const inCh, patch, temporal, merge, outHidden = 3, 2, 2, 2, 4
	const tHidden, tFF, tHeads, tKVHeads, tHeadDim, vocab = 4, 4, 2, 1, 2, 5
	patchDim := inCh * temporal * patch * patch

	tensors := map[string]safetensors.Tensor{
		"model.visual.patch_embed.proj.weight":            f32Tensor(hidden, patchDim),
		"model.visual.patch_embed.proj.bias":              f32Tensor(hidden),
		"model.visual.blocks.0.norm1.weight":              f32Tensor(hidden),
		"model.visual.blocks.0.norm2.weight":              f32Tensor(hidden),
		"model.visual.blocks.0.attn.qkv.weight":           f32Tensor(3*hidden, hidden),
		"model.visual.blocks.0.attn.qkv.bias":             f32Tensor(3 * hidden),
		"model.visual.blocks.0.attn.proj.weight":          f32Tensor(hidden, hidden),
		"model.visual.blocks.0.attn.proj.bias":            f32Tensor(hidden),
		"model.visual.blocks.0.attn.q_norm.weight":        f32Tensor(headDim),
		"model.visual.blocks.0.attn.k_norm.weight":        f32Tensor(headDim),
		"model.visual.blocks.0.mlp.gate_proj.weight":      f32Tensor(ff, hidden),
		"model.visual.blocks.0.mlp.gate_proj.bias":        f32Tensor(ff),
		"model.visual.blocks.0.mlp.up_proj.weight":        f32Tensor(ff, hidden),
		"model.visual.blocks.0.mlp.up_proj.bias":          f32Tensor(ff),
		"model.visual.blocks.0.mlp.down_proj.weight":      f32Tensor(hidden, ff),
		"model.visual.blocks.0.mlp.down_proj.bias":        f32Tensor(hidden),
		"model.visual.post_layernorm.weight":              f32Tensor(hidden),
		"model.visual.downsample.weight":                  f32Tensor(outHidden, hidden*merge*merge),
		"model.visual.downsample.bias":                    f32Tensor(outHidden),
		"model.visual.merger.proj.weight":                 f32Tensor(outHidden, outHidden),
		"model.visual.merger.post_projection_norm.weight": f32Tensor(outHidden),
		"model.visual.merger.post_projection_norm.bias":   f32Tensor(outHidden),
		"model.visual.merger.gate_proj.weight":            f32Tensor(outHidden*inCh, outHidden),
		"model.visual.merger.up_proj.weight":              f32Tensor(outHidden*inCh, outHidden),
		"model.visual.merger.down_proj.weight":            f32Tensor(outHidden, outHidden*inCh),

		"model.language_model.embed_tokens.weight":                      f32Tensor(vocab, tHidden),
		"model.language_model.layers.0.input_layernorm.weight":          f32Tensor(tHidden),
		"model.language_model.layers.0.post_attention_layernorm.weight": f32Tensor(tHidden),
		"model.language_model.layers.0.post_self_attn_layernorm.weight": f32Tensor(tHidden),
		"model.language_model.layers.0.post_mlp_layernorm.weight":       f32Tensor(tHidden),
		"model.language_model.layers.0.self_attn.q_proj.weight":         f32Tensor(tHeads*tHeadDim, tHidden),
		"model.language_model.layers.0.self_attn.k_proj.weight":         f32Tensor(tKVHeads*tHeadDim, tHidden),
		"model.language_model.layers.0.self_attn.v_proj.weight":         f32Tensor(tKVHeads*tHeadDim, tHidden),
		"model.language_model.layers.0.self_attn.o_proj.weight":         f32Tensor(tHidden, tHeads*tHeadDim),
		"model.language_model.layers.0.mlp.gate_up_proj.weight":         f32Tensor(2*tFF, tHidden),
		"model.language_model.layers.0.mlp.down_proj.weight":            f32Tensor(tHidden, tFF),
		"model.language_model.norm.weight":                              f32Tensor(tHidden),
		"lm_head.weight":                                                f32Tensor(vocab, tHidden),

		// the MTP layer — LoadWeights must never need this (its bounded NumHiddenLayers loop
		// never reaches index 1 here, matching the real checkpoint's index-16 MTP layer it
		// never reaches either); present to prove that.
		"model.language_model.layers.1.input_layernorm.weight": f32Tensor(tHidden),
	}
	cfg := &Config{
		ModelType: "glm_ocr",
		VisionConfig: &VisionConfig{
			HiddenSize: hidden, Depth: 1, NumHeads: heads, PatchSize: patch, TemporalPatchSize: temporal,
			SpatialMergeSize: merge, OutHiddenSize: outHidden, InChannels: inCh, IntermediateSize: ff, RMSNormEps: 1e-5,
		},
		TextConfig: &TextConfig{
			HiddenSize: tHidden, IntermediateSize: tFF, NumHiddenLayers: 1, NumAttentionHeads: tHeads,
			NumKeyValueHeads: tKVHeads, HeadDim: tHeadDim, VocabSize: vocab, RMSNormEps: 1e-5,
			RopeParameters: &RopeParameters{RopeTheta: 10000, MropeSection: []int{1}, PartialRotaryFactor: 1},
		},
	}
	return tensors, cfg
}

func TestLoadWeights_Good(t *testing.T) {
	tensors, cfg := syntheticCheckpoint()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	if len(w.Vision.Blocks) != 1 || len(w.Text.Layers) != 1 {
		t.Fatalf("LoadWeights loaded %d vision blocks / %d text layers, want 1/1", len(w.Vision.Blocks), len(w.Text.Layers))
	}
	if len(w.Text.EmbedTokens) != 5*4 {
		t.Fatalf("LoadWeights EmbedTokens has %d elements, want %d", len(w.Text.EmbedTokens), 5*4)
	}
	if len(w.Text.LMHead.Weight) != 5*4 || w.Text.LMHead.Bias != nil {
		t.Fatalf("LoadWeights LMHead = %+v, want a 5x4 weight and no bias (untied, no-bias projection)", w.Text.LMHead)
	}
}

func TestLoadWeights_Bad(t *testing.T) {
	if _, err := LoadWeights(nil, nil); err == nil {
		t.Fatal("LoadWeights accepted a nil config")
	}
}

func TestLoadWeights_Ugly(t *testing.T) {
	// a missing tensor is reported BY NAME, not a generic failure
	tensors, cfg := syntheticCheckpoint()
	delete(tensors, "lm_head.weight")
	_, err := LoadWeights(tensors, cfg)
	if err == nil {
		t.Fatal("LoadWeights accepted a checkpoint missing lm_head.weight")
	}
}

func TestLMHeadForward_BlockGoldens_Good(t *testing.T) {
	g := readBlockGoldens(t).LMHead
	w := LinearWeights{Weight: g.Weight, In: g.In, Out: g.Out}
	T := len(g.Input) / g.In
	got := linearForward(g.Input, w, T)
	if d := maxAbsDiff32(t, got, g.Output); d > 1e-3 {
		t.Fatalf("lm_head linearForward maxAbsDiff = %v, want < 1e-3", d)
	}
}
