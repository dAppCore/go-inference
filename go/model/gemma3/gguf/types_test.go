// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"

	basegguf "dappco.re/go/inference/model/gguf"
)

func TestGemma3Types_gemma3TensorType_Norm(t *testing.T) {
	// Every *norm.weight stays F32 regardless of format or row length.
	for _, name := range []string{"blk.0.attn_norm.weight", "blk.0.attn_q_norm.weight", "output_norm.weight", "blk.0.post_ffw_norm.weight"} {
		if got := gemma3TensorType(basegguf.QuantizeQ4_K_M, name, 1152); got != basegguf.TensorTypeF32 {
			t.Errorf("gemma3TensorType(q4_k_m, %s) = %d, want F32(%d)", name, got, basegguf.TensorTypeF32)
		}
	}
}

func TestGemma3Types_gemma3TensorType_KQuantWhenDivisible(t *testing.T) {
	// A 256-divisible row under q4_k_m takes the K-quant.
	if got := gemma3TensorType(basegguf.QuantizeQ4_K_M, "blk.0.ffn_down.weight", 6912); got != basegguf.TensorTypeQ4K {
		t.Errorf("ffn_down (ne0=6912) = %d, want Q4_K(%d)", got, basegguf.TensorTypeQ4K)
	}
	if got := gemma3TensorType(basegguf.QuantizeQ6_K, "blk.0.attn_output.weight", 1024); got != basegguf.TensorTypeQ6K {
		t.Errorf("attn_output (ne0=1024, q6_k) = %d, want Q6_K(%d)", got, basegguf.TensorTypeQ6K)
	}
}

func TestGemma3Types_gemma3TensorType_Q8FallbackWhenNot256(t *testing.T) {
	// gemma-3-1B's hidden size 1152 is not a 256-multiple → a K-quant request
	// falls back to Q8_0 (1152 is a 32-multiple).
	if got := gemma3TensorType(basegguf.QuantizeQ4_K_M, "blk.0.attn_q.weight", 1152); got != basegguf.TensorTypeQ8_0 {
		t.Errorf("attn_q (ne0=1152, q4_k_m) = %d, want Q8_0(%d)", got, basegguf.TensorTypeQ8_0)
	}
	if got := gemma3TensorType(basegguf.QuantizeQ4_K_M, "token_embd.weight", 1152); got != basegguf.TensorTypeQ8_0 {
		t.Errorf("token_embd (ne0=1152, q4_k_m) = %d, want Q8_0(%d)", got, basegguf.TensorTypeQ8_0)
	}
}

func TestGemma3Types_gemma3TensorType_Q8FormatUniform(t *testing.T) {
	// A q8_0 request quantises every 32-divisible weight to Q8_0.
	if got := gemma3TensorType(basegguf.QuantizeQ8_0, "blk.0.ffn_down.weight", 6912); got != basegguf.TensorTypeQ8_0 {
		t.Errorf("ffn_down (q8_0) = %d, want Q8_0(%d)", got, basegguf.TensorTypeQ8_0)
	}
}

func TestGemma3Types_gemma3TensorType_F32WhenNeither(t *testing.T) {
	// A row divisible by neither 256 nor 32 cannot be block-quantised → F32.
	if got := gemma3TensorType(basegguf.QuantizeQ4_K_M, "blk.0.attn_q.weight", 100); got != basegguf.TensorTypeF32 {
		t.Errorf("odd row (ne0=100) = %d, want F32(%d)", got, basegguf.TensorTypeF32)
	}
}

func TestGemma3Types_gemma3KQuantType(t *testing.T) {
	cases := []struct {
		format basegguf.QuantizeFormat
		want   uint32
		isK    bool
	}{
		{basegguf.QuantizeQ4_K_M, basegguf.TensorTypeQ4K, true},
		{basegguf.QuantizeQ5_K_M, basegguf.TensorTypeQ5K, true},
		{basegguf.QuantizeQ6_K, basegguf.TensorTypeQ6K, true},
		{basegguf.QuantizeQ3_K_M, basegguf.TensorTypeQ3K, true},
		{basegguf.QuantizeQ2_K_M, basegguf.TensorTypeQ2K, true},
		{basegguf.QuantizeQ8_0, 0, false},
	}
	for _, c := range cases {
		got, isK := gemma3KQuantType(c.format)
		if isK != c.isK || (isK && got != c.want) {
			t.Errorf("gemma3KQuantType(%s) = (%d,%v), want (%d,%v)", c.format, got, isK, c.want, c.isK)
		}
	}
}
