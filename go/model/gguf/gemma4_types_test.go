// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// TestGemma4Types_gemma4UseMoreBits_Gemma4E2B checks the selector reproduces the
// exact Q6_K layer set the oracle carries for the 35-layer gemma-4-E2B model.
func TestGemma4Types_gemma4UseMoreBits_Gemma4E2B(t *testing.T) {
	const layerCount = 35
	want := map[int]bool{}
	for _, i := range []int{0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31, 32, 33, 34} {
		want[i] = true
	}
	for i := 0; i < layerCount; i++ {
		if got := gemma4UseMoreBits(i, layerCount); got != want[i] {
			t.Errorf("gemma4UseMoreBits(%d, %d) = %v, want %v", i, layerCount, got, want[i])
		}
	}
}

// TestGemma4Types_gemma4TensorType_WholeModel checks the fixed types of the
// whole-model tensors (embeddings, output norm, the BF16 per-layer projection).
func TestGemma4Types_gemma4TensorType_WholeModel(t *testing.T) {
	cases := map[string]uint32{
		"token_embd.weight":           ggufTensorTypeQ4K,
		"per_layer_token_embd.weight": ggufTensorTypeQ5K,
		"per_layer_model_proj.weight": ggufTensorTypeBF16,
		"output_norm.weight":          ggufTensorTypeF32,
		"per_layer_proj_norm.weight":  ggufTensorTypeF32,
		"rope_freqs.weight":           ggufTensorTypeF32,
	}
	for name, want := range cases {
		if got := gemma4TensorType(name, -1, 35); got != want {
			t.Errorf("gemma4TensorType(%q) = %d, want %d", name, got, want)
		}
	}
}

// TestGemma4Types_gemma4TensorType_PerLayerF32 checks the per-layer tensors the
// oracle keeps in full precision (RMS norms, per-layer input gate/projection,
// the layer output scale) resolve to F32.
func TestGemma4Types_gemma4TensorType_PerLayerF32(t *testing.T) {
	for _, name := range []string{
		"blk.7.attn_norm.weight",
		"blk.7.attn_q_norm.weight",
		"blk.7.attn_k_norm.weight",
		"blk.7.ffn_norm.weight",
		"blk.7.post_attention_norm.weight",
		"blk.7.post_ffw_norm.weight",
		"blk.7.post_norm.weight",
		"blk.7.inp_gate.weight",
		"blk.7.proj.weight",
		"blk.7.layer_output_scale.weight",
	} {
		if got := gemma4TensorType(name, 7, 35); got != ggufTensorTypeF32 {
			t.Errorf("gemma4TensorType(%q) = %d, want F32(%d)", name, got, ggufTensorTypeF32)
		}
	}
}

// TestGemma4Types_gemma4TensorType_PerLayerQ4K checks the bulk projection
// weights that are always Q4_K under q4_k_m.
func TestGemma4Types_gemma4TensorType_PerLayerQ4K(t *testing.T) {
	for _, name := range []string{
		"blk.5.attn_q.weight",
		"blk.5.attn_k.weight",
		"blk.5.attn_output.weight",
		"blk.5.ffn_gate.weight",
		"blk.5.ffn_up.weight",
	} {
		if got := gemma4TensorType(name, 5, 35); got != ggufTensorTypeQ4K {
			t.Errorf("gemma4TensorType(%q) = %d, want Q4_K(%d)", name, got, ggufTensorTypeQ4K)
		}
	}
}

// TestGemma4Types_gemma4TensorType_AttnVFfnDownBump checks attn_v and ffn_down
// bump to Q6_K on a use_more_bits layer and stay Q4_K on a non-bump layer.
func TestGemma4Types_gemma4TensorType_AttnVFfnDownBump(t *testing.T) {
	for _, name := range []string{"blk.6.attn_v.weight", "blk.6.ffn_down.weight"} {
		if got := gemma4TensorType(name, 6, 35); got != ggufTensorTypeQ6K {
			t.Errorf("gemma4TensorType(%q, layer 6) = %d, want Q6_K(%d)", name, got, ggufTensorTypeQ6K)
		}
	}
	for _, name := range []string{"blk.5.attn_v.weight", "blk.5.ffn_down.weight"} {
		if got := gemma4TensorType(name, 5, 35); got != ggufTensorTypeQ4K {
			t.Errorf("gemma4TensorType(%q, layer 5) = %d, want Q4_K(%d)", name, got, ggufTensorTypeQ4K)
		}
	}
}
