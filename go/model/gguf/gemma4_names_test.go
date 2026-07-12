// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// TestGemma4Names_gemma4CanonicalTensorName_TopLevel checks every whole-model
// (non per-layer) source tensor maps to the canonical GGUF name.
func TestGemma4Names_gemma4CanonicalTensorName_TopLevel(t *testing.T) {
	cases := map[string]string{
		"language_model.model.embed_tokens.weight":               "token_embd.weight",
		"language_model.model.embed_tokens_per_layer.weight":     "per_layer_token_embd.weight",
		"language_model.model.norm.weight":                       "output_norm.weight",
		"language_model.model.per_layer_model_projection.weight": "per_layer_model_proj.weight",
		"language_model.model.per_layer_projection_norm.weight":  "per_layer_proj_norm.weight",
	}
	for src, want := range cases {
		got, err := gemma4CanonicalTensorName(src)
		if err != nil {
			t.Fatalf("gemma4CanonicalTensorName(%q) error: %v", src, err)
		}
		if got != want {
			t.Errorf("gemma4CanonicalTensorName(%q) = %q, want %q", src, got, want)
		}
	}
}

// TestGemma4Names_gemma4CanonicalTensorName_PerLayer checks every per-layer
// source suffix maps to blk.<N>.<canonical>, including the .weight-less
// layer_scalar special case, across a non-zero layer index.
func TestGemma4Names_gemma4CanonicalTensorName_PerLayer(t *testing.T) {
	cases := map[string]string{
		"language_model.model.layers.7.input_layernorm.weight":            "blk.7.attn_norm.weight",
		"language_model.model.layers.7.self_attn.q_proj.weight":           "blk.7.attn_q.weight",
		"language_model.model.layers.7.self_attn.k_proj.weight":           "blk.7.attn_k.weight",
		"language_model.model.layers.7.self_attn.v_proj.weight":           "blk.7.attn_v.weight",
		"language_model.model.layers.7.self_attn.o_proj.weight":           "blk.7.attn_output.weight",
		"language_model.model.layers.7.self_attn.q_norm.weight":           "blk.7.attn_q_norm.weight",
		"language_model.model.layers.7.self_attn.k_norm.weight":           "blk.7.attn_k_norm.weight",
		"language_model.model.layers.7.mlp.gate_proj.weight":              "blk.7.ffn_gate.weight",
		"language_model.model.layers.7.mlp.up_proj.weight":                "blk.7.ffn_up.weight",
		"language_model.model.layers.7.mlp.down_proj.weight":              "blk.7.ffn_down.weight",
		"language_model.model.layers.7.pre_feedforward_layernorm.weight":  "blk.7.ffn_norm.weight",
		"language_model.model.layers.7.post_feedforward_layernorm.weight": "blk.7.post_ffw_norm.weight",
		"language_model.model.layers.7.post_attention_layernorm.weight":   "blk.7.post_attention_norm.weight",
		"language_model.model.layers.7.post_per_layer_input_norm.weight":  "blk.7.post_norm.weight",
		"language_model.model.layers.7.per_layer_input_gate.weight":       "blk.7.inp_gate.weight",
		"language_model.model.layers.7.per_layer_projection.weight":       "blk.7.proj.weight",
		"language_model.model.layers.7.layer_scalar":                      "blk.7.layer_output_scale.weight",
	}
	for src, want := range cases {
		got, err := gemma4CanonicalTensorName(src)
		if err != nil {
			t.Fatalf("gemma4CanonicalTensorName(%q) error: %v", src, err)
		}
		if got != want {
			t.Errorf("gemma4CanonicalTensorName(%q) = %q, want %q", src, got, want)
		}
	}
}

// TestGemma4Names_gemma4CanonicalTensorName_LayerZero checks the mapper carries
// the layer index verbatim (no zero-padding or renumbering) for layer 0.
func TestGemma4Names_gemma4CanonicalTensorName_LayerZero(t *testing.T) {
	got, err := gemma4CanonicalTensorName("language_model.model.layers.0.self_attn.q_proj.weight")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "blk.0.attn_q.weight" {
		t.Errorf("got %q, want blk.0.attn_q.weight", got)
	}
}

// TestGemma4Names_gemma4CanonicalTensorName_Unmapped checks an unrecognised
// text-stack name is a loud error rather than a silent drop.
func TestGemma4Names_gemma4CanonicalTensorName_Unmapped(t *testing.T) {
	for _, src := range []string{
		"language_model.model.layers.7.self_attn.novel_proj.weight", // unknown per-layer suffix
		"language_model.model.some_new_top_level.weight",            // unknown top-level
		"language_model.model.layers.notanumber.mlp.up_proj.weight", // non-numeric layer index
		"audio_tower.output_proj.weight",                            // tower tensor must be excluded upstream, not mapped
		"",
	} {
		if got, err := gemma4CanonicalTensorName(src); err == nil {
			t.Errorf("gemma4CanonicalTensorName(%q) = %q, want error", src, got)
		}
	}
}

// TestGemma4Names_gemma4GGUFShape_TwoDim checks a 2-D safetensors shape reverses
// to GGUF ne[] order (inner/contiguous dimension first).
func TestGemma4Names_gemma4GGUFShape_TwoDim(t *testing.T) {
	got := gemma4GGUFShape([]uint64{2048, 1536})
	if len(got) != 2 || got[0] != 1536 || got[1] != 2048 {
		t.Errorf("gemma4GGUFShape([2048 1536]) = %v, want [1536 2048]", got)
	}
}

// TestGemma4Names_gemma4GGUFShape_OneDim checks a 1-D shape (norms, scales) is
// unchanged by reversal.
func TestGemma4Names_gemma4GGUFShape_OneDim(t *testing.T) {
	got := gemma4GGUFShape([]uint64{1536})
	if len(got) != 1 || got[0] != 1536 {
		t.Errorf("gemma4GGUFShape([1536]) = %v, want [1536]", got)
	}
}

// TestGemma4Names_gemma4GGUFShape_Empty checks an empty (scalar) shape reverses
// to an empty shape without panicking.
func TestGemma4Names_gemma4GGUFShape_Empty(t *testing.T) {
	if got := gemma4GGUFShape(nil); len(got) != 0 {
		t.Errorf("gemma4GGUFShape(nil) = %v, want empty", got)
	}
}

// TestGemma4Names_gemma4GGUFShape_NoAlias checks reversal returns a fresh slice
// and does not mutate the caller's shape.
func TestGemma4Names_gemma4GGUFShape_NoAlias(t *testing.T) {
	src := []uint64{6144, 1536}
	_ = gemma4GGUFShape(src)
	if src[0] != 6144 || src[1] != 1536 {
		t.Errorf("gemma4GGUFShape mutated its argument: %v", src)
	}
}
