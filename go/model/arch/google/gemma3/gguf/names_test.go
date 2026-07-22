// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"slices"
	"testing"
)

func TestGemma3Names_gemma3CanonicalTensorName_TopLevel(t *testing.T) {
	cases := map[string]string{
		"model.embed_tokens.weight": "token_embd.weight",
		"lm_head.weight":            "output.weight",
		"model.norm.weight":         "output_norm.weight",
	}
	for src, want := range cases {
		got, err := gemma3CanonicalTensorName(src)
		if err != nil {
			t.Errorf("%s: %v", src, err)
			continue
		}
		if got != want {
			t.Errorf("%s -> %s, want %s", src, got, want)
		}
	}
}

func TestGemma3Names_gemma3CanonicalTensorName_PerLayer(t *testing.T) {
	cases := map[string]string{
		"model.layers.7.self_attn.q_proj.weight":           "blk.7.attn_q.weight",
		"model.layers.7.self_attn.k_proj.weight":           "blk.7.attn_k.weight",
		"model.layers.7.self_attn.v_proj.weight":           "blk.7.attn_v.weight",
		"model.layers.7.self_attn.o_proj.weight":           "blk.7.attn_output.weight",
		"model.layers.7.self_attn.q_norm.weight":           "blk.7.attn_q_norm.weight",
		"model.layers.7.self_attn.k_norm.weight":           "blk.7.attn_k_norm.weight",
		"model.layers.7.input_layernorm.weight":            "blk.7.attn_norm.weight",
		"model.layers.7.post_attention_layernorm.weight":   "blk.7.post_attention_norm.weight",
		"model.layers.7.pre_feedforward_layernorm.weight":  "blk.7.ffn_norm.weight",
		"model.layers.7.post_feedforward_layernorm.weight": "blk.7.post_ffw_norm.weight",
		"model.layers.7.mlp.gate_proj.weight":              "blk.7.ffn_gate.weight",
		"model.layers.7.mlp.up_proj.weight":                "blk.7.ffn_up.weight",
		"model.layers.7.mlp.down_proj.weight":              "blk.7.ffn_down.weight",
	}
	for src, want := range cases {
		got, err := gemma3CanonicalTensorName(src)
		if err != nil {
			t.Errorf("%s: %v", src, err)
			continue
		}
		if got != want {
			t.Errorf("%s -> %s, want %s", src, got, want)
		}
	}
}

func TestGemma3Names_gemma3CanonicalTensorName_LanguageModelWrapper(t *testing.T) {
	// mlx-community-style gemma-3 packs nest the text stack under a
	// "language_model." wrapper (the same layout model.NormalizeWrapperNames
	// handles generically for weight loading); the GGUF export lane must
	// resolve the wrapped name to the identical canonical tensor as the flat
	// (unwrapped) form.
	cases := map[string]string{
		"language_model.model.embed_tokens.weight":              "token_embd.weight",
		"language_model.model.norm.weight":                      "output_norm.weight",
		"language_model.model.layers.7.self_attn.q_proj.weight": "blk.7.attn_q.weight",
		"language_model.model.layers.7.input_layernorm.weight":  "blk.7.attn_norm.weight",
	}
	for src, want := range cases {
		got, err := gemma3CanonicalTensorName(src)
		if err != nil {
			t.Errorf("%s: %v", src, err)
			continue
		}
		if got != want {
			t.Errorf("%s -> %s, want %s", src, got, want)
		}
	}
}

func TestGemma3Names_gemma3CanonicalTensorName_Unmapped(t *testing.T) {
	// A vision-tower tensor gemma3_text never carries — must fail loudly rather
	// than silently drop a weight.
	if _, err := gemma3CanonicalTensorName("vision_tower.encoder.layer.0.weight"); err == nil {
		t.Fatal("gemma3CanonicalTensorName accepted an unmapped tensor, want error")
	}
}

func TestGemma3Names_gemma3GGUFShape_TwoDim(t *testing.T) {
	got := gemma3GGUFShape([]uint64{1024, 1152})
	if !slices.Equal(got, []uint64{1152, 1024}) {
		t.Errorf("gemma3GGUFShape([1024 1152]) = %v, want [1152 1024]", got)
	}
}

func TestGemma3Names_gemma3GGUFShape_OneDim(t *testing.T) {
	got := gemma3GGUFShape([]uint64{1152})
	if !slices.Equal(got, []uint64{1152}) {
		t.Errorf("gemma3GGUFShape([1152]) = %v, want [1152]", got)
	}
}

func TestGemma3Names_gemma3TensorRowLength(t *testing.T) {
	cases := []struct {
		shape []uint64
		want  uint64
	}{
		{[]uint64{1024, 1152}, 1152}, // 2-D: inner (last) source dim
		{[]uint64{1152}, 1152},       // 1-D: the single dim
		{nil, 0},                     // empty: unquantisable
	}
	for _, c := range cases {
		if got := gemma3TensorRowLength(c.shape); got != c.want {
			t.Errorf("gemma3TensorRowLength(%v) = %d, want %d", c.shape, got, c.want)
		}
	}
}
