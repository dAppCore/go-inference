// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"

	core "dappco.re/go"
)

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
// whole-model tensors (embeddings, output norm, the BF16 per-layer projection)
// under q4_k_m — the unsloth oracle-matched policy.
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
		if got := gemma4TensorType(QuantizeQ4_K_M, name, -1, 35); got != want {
			t.Errorf("gemma4TensorType(q4_k_m, %q) = %d, want %d", name, got, want)
		}
	}
}

// TestGemma4Types_gemma4TensorType_PerLayerF32 checks the per-layer tensors the
// oracle keeps in full precision (RMS norms, per-layer input gate/projection,
// the layer output scale) resolve to F32 regardless of the requested format —
// llama.cpp's tensor_allows_quantization excludes these unconditionally.
func TestGemma4Types_gemma4TensorType_PerLayerF32(t *testing.T) {
	names := []string{
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
	}
	for _, format := range []QuantizeFormat{QuantizeQ4_K_M, QuantizeQ8_0, QuantizeQ6_K, QuantizeQ5_K_M, QuantizeQ3_K_M} {
		for _, name := range names {
			if got := gemma4TensorType(format, name, 7, 35); got != ggufTensorTypeF32 {
				t.Errorf("gemma4TensorType(%s, %q) = %d, want F32(%d)", format, name, got, ggufTensorTypeF32)
			}
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
		if got := gemma4TensorType(QuantizeQ4_K_M, name, 5, 35); got != ggufTensorTypeQ4K {
			t.Errorf("gemma4TensorType(q4_k_m, %q) = %d, want Q4_K(%d)", name, got, ggufTensorTypeQ4K)
		}
	}
}

// TestGemma4Types_gemma4TensorType_AttnVFfnDownBump checks attn_v and ffn_down
// bump to Q6_K on a use_more_bits layer and stay Q4_K on a non-bump layer.
func TestGemma4Types_gemma4TensorType_AttnVFfnDownBump(t *testing.T) {
	for _, name := range []string{"blk.6.attn_v.weight", "blk.6.ffn_down.weight"} {
		if got := gemma4TensorType(QuantizeQ4_K_M, name, 6, 35); got != ggufTensorTypeQ6K {
			t.Errorf("gemma4TensorType(q4_k_m, %q, layer 6) = %d, want Q6_K(%d)", name, got, ggufTensorTypeQ6K)
		}
	}
	for _, name := range []string{"blk.5.attn_v.weight", "blk.5.ffn_down.weight"} {
		if got := gemma4TensorType(QuantizeQ4_K_M, name, 5, 35); got != ggufTensorTypeQ4K {
			t.Errorf("gemma4TensorType(q4_k_m, %q, layer 5) = %d, want Q4_K(%d)", name, got, ggufTensorTypeQ4K)
		}
	}
}

// TestGemma4Types_gemma4TensorType_Q8_0 checks q8_0's pure/uniform policy:
// every quantisable tensor (whole-model embeddings and every per-layer
// projection, including attn_v/ffn_down/attn_output on every layer) is Q8_0 —
// oracle-confirmed against the on-disk unsloth gemma-4-E2B-it-Q8_0.gguf,
// whose tensor-type histogram is F32:353, Q8_0:247, BF16:1 with no other
// type present.
func TestGemma4Types_gemma4TensorType_Q8_0(t *testing.T) {
	wholeModel := map[string]uint32{
		"token_embd.weight":           TensorTypeQ8_0,
		"per_layer_token_embd.weight": TensorTypeQ8_0,
		"per_layer_model_proj.weight": ggufTensorTypeBF16,
		"output_norm.weight":          ggufTensorTypeF32,
	}
	for name, want := range wholeModel {
		if got := gemma4TensorType(QuantizeQ8_0, name, -1, 35); got != want {
			t.Errorf("gemma4TensorType(q8_0, %q) = %d, want %d", name, got, want)
		}
	}
	perLayer := []string{
		"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
		"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
	}
	// Every layer, including the use_more_bits/first-two/first-1-16 layers
	// q4_k_m and q3_k_m bump — q8_0 has no override table entry, so nothing
	// ever moves off the pure bulk type.
	for _, layer := range []int{0, 1, 2, 6, 17, 34} {
		for _, suffix := range perLayer {
			name := core.Concat("blk.", core.Itoa(layer), ".", suffix)
			if got := gemma4TensorType(QuantizeQ8_0, name, layer, 35); got != TensorTypeQ8_0 {
				t.Errorf("gemma4TensorType(q8_0, %q) = %d, want Q8_0(%d)", name, got, TensorTypeQ8_0)
			}
		}
	}
}

// TestGemma4Types_gemma4TensorType_Q6_K checks q6_k's pure/uniform policy —
// like q8_0, llama.cpp's quantize table carries no per-tensor override for
// Q6_K, so every quantisable tensor is Q6_K on every layer.
func TestGemma4Types_gemma4TensorType_Q6_K(t *testing.T) {
	wholeModel := map[string]uint32{
		"token_embd.weight":           ggufTensorTypeQ6K,
		"per_layer_token_embd.weight": ggufTensorTypeQ6K,
	}
	for name, want := range wholeModel {
		if got := gemma4TensorType(QuantizeQ6_K, name, -1, 35); got != want {
			t.Errorf("gemma4TensorType(q6_k, %q) = %d, want %d", name, got, want)
		}
	}
	perLayer := []string{"attn_q.weight", "attn_v.weight", "attn_output.weight", "ffn_down.weight"}
	for _, layer := range []int{0, 1, 6, 34} {
		for _, suffix := range perLayer {
			name := core.Concat("blk.", core.Itoa(layer), ".", suffix)
			if got := gemma4TensorType(QuantizeQ6_K, name, layer, 35); got != ggufTensorTypeQ6K {
				t.Errorf("gemma4TensorType(q6_k, %q) = %d, want Q6_K(%d)", name, got, ggufTensorTypeQ6K)
			}
		}
	}
}

// TestGemma4Types_gemma4TensorType_Q5_K_M checks q5_k_m's mixed policy:
// bulk Q5_K, attn_v/ffn_down bumped to Q6_K on the same use_more_bits layers
// q4_k_m bumps (llama.cpp groups Q4_K_M and Q5_K_M under one use_more_bits
// condition), everything else — including attn_output and the embeddings —
// stays at the Q5_K bulk (no override table entry for those categories).
func TestGemma4Types_gemma4TensorType_Q5_K_M(t *testing.T) {
	wholeModel := map[string]uint32{
		"token_embd.weight":           ggufTensorTypeQ5K,
		"per_layer_token_embd.weight": ggufTensorTypeQ5K,
	}
	for name, want := range wholeModel {
		if got := gemma4TensorType(QuantizeQ5_K_M, name, -1, 35); got != want {
			t.Errorf("gemma4TensorType(q5_k_m, %q) = %d, want %d", name, got, want)
		}
	}
	for _, name := range []string{"blk.6.attn_v.weight", "blk.6.ffn_down.weight"} {
		if got := gemma4TensorType(QuantizeQ5_K_M, name, 6, 35); got != ggufTensorTypeQ6K {
			t.Errorf("gemma4TensorType(q5_k_m, %q, layer 6) = %d, want Q6_K(%d)", name, got, ggufTensorTypeQ6K)
		}
	}
	for _, name := range []string{"blk.5.attn_v.weight", "blk.5.ffn_down.weight", "blk.5.attn_output.weight", "blk.5.ffn_gate.weight"} {
		if got := gemma4TensorType(QuantizeQ5_K_M, name, 5, 35); got != ggufTensorTypeQ5K {
			t.Errorf("gemma4TensorType(q5_k_m, %q, layer 5) = %d, want Q5_K(%d)", name, got, ggufTensorTypeQ5K)
		}
	}
}

// TestGemma4Types_gemma4TensorType_Q3_K_M checks q3_k_m's mixed policy: bulk
// Q3_K, attn_v Q5_K on the first two layers else Q4_K, ffn_down Q5_K on the
// first layerCount/16 layers else Q4_K, attn_output always Q4_K, everything
// else (attn_q/attn_k/ffn_gate/ffn_up, the embeddings) at the Q3_K bulk.
func TestGemma4Types_gemma4TensorType_Q3_K_M(t *testing.T) {
	const layerCount = 35 // 35/16 == 2: layers 0,1 get ffn_down's Q5_K bump.

	wholeModel := map[string]uint32{
		"token_embd.weight":           ggufTensorTypeQ3K,
		"per_layer_token_embd.weight": ggufTensorTypeQ3K,
	}
	for name, want := range wholeModel {
		if got := gemma4TensorType(QuantizeQ3_K_M, name, -1, layerCount); got != want {
			t.Errorf("gemma4TensorType(q3_k_m, %q) = %d, want %d", name, got, want)
		}
	}

	// attn_v: hardcoded "first two layers" rule, independent of layerCount.
	for _, layer := range []int{0, 1} {
		name := core.Concat("blk.", core.Itoa(layer), ".attn_v.weight")
		if got := gemma4TensorType(QuantizeQ3_K_M, name, layer, layerCount); got != ggufTensorTypeQ5K {
			t.Errorf("gemma4TensorType(q3_k_m, %q) = %d, want Q5_K(%d)", name, got, ggufTensorTypeQ5K)
		}
	}
	for _, layer := range []int{2, 17, 34} {
		name := core.Concat("blk.", core.Itoa(layer), ".attn_v.weight")
		if got := gemma4TensorType(QuantizeQ3_K_M, name, layer, layerCount); got != ggufTensorTypeQ4K {
			t.Errorf("gemma4TensorType(q3_k_m, %q) = %d, want Q4_K(%d)", name, got, ggufTensorTypeQ4K)
		}
	}

	// ffn_down: proportional "< layerCount/16" rule — 35/16 == 2.
	for _, layer := range []int{0, 1} {
		name := core.Concat("blk.", core.Itoa(layer), ".ffn_down.weight")
		if got := gemma4TensorType(QuantizeQ3_K_M, name, layer, layerCount); got != ggufTensorTypeQ5K {
			t.Errorf("gemma4TensorType(q3_k_m, %q) = %d, want Q5_K(%d)", name, got, ggufTensorTypeQ5K)
		}
	}
	for _, layer := range []int{2, 17, 34} {
		name := core.Concat("blk.", core.Itoa(layer), ".ffn_down.weight")
		if got := gemma4TensorType(QuantizeQ3_K_M, name, layer, layerCount); got != ggufTensorTypeQ4K {
			t.Errorf("gemma4TensorType(q3_k_m, %q) = %d, want Q4_K(%d)", name, got, ggufTensorTypeQ4K)
		}
	}

	// attn_output: always bumped to Q4_K, every layer.
	for _, layer := range []int{0, 5, 34} {
		name := core.Concat("blk.", core.Itoa(layer), ".attn_output.weight")
		if got := gemma4TensorType(QuantizeQ3_K_M, name, layer, layerCount); got != ggufTensorTypeQ4K {
			t.Errorf("gemma4TensorType(q3_k_m, %q) = %d, want Q4_K(%d)", name, got, ggufTensorTypeQ4K)
		}
	}

	// attn_q/attn_k/ffn_gate/ffn_up: no override, stay at the Q3_K bulk.
	for _, suffix := range []string{"attn_q.weight", "attn_k.weight", "ffn_gate.weight", "ffn_up.weight"} {
		name := "blk.10." + suffix
		if got := gemma4TensorType(QuantizeQ3_K_M, name, 10, layerCount); got != ggufTensorTypeQ3K {
			t.Errorf("gemma4TensorType(q3_k_m, %q) = %d, want Q3_K(%d)", name, got, ggufTensorTypeQ3K)
		}
	}
}
