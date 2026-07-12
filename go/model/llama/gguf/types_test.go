// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"

	basegguf "dappco.re/go/inference/model/gguf"
)

func TestTypes_llamaTensorType_Q8_0(t *testing.T) {
	if got := llamaTensorType(basegguf.QuantizeQ8_0, "blk.0.attn_q.weight", 0, 16, false); got != basegguf.TensorTypeQ8_0 {
		t.Fatalf("attn_q type = %d, want Q8_0", got)
	}
	if got := llamaTensorType(basegguf.QuantizeQ8_0, "blk.0.attn_norm.weight", 0, 16, false); got != basegguf.TensorTypeF32 {
		t.Fatalf("attn_norm type = %d, want F32", got)
	}
}

func TestTypes_llamaTensorType_Q4_K_M(t *testing.T) {
	if got := llamaTensorType(basegguf.QuantizeQ4_K_M, "blk.0.attn_q.weight", 0, 16, false); got != basegguf.TensorTypeQ4K {
		t.Fatalf("attn_q type = %d, want Q4_K", got)
	}
	if got := llamaTensorType(basegguf.QuantizeQ4_K_M, "blk.0.ffn_down.weight", 0, 16, false); got != basegguf.TensorTypeQ6K {
		t.Fatalf("first-layer ffn_down type = %d, want Q6_K", got)
	}
	if got := llamaTensorType(basegguf.QuantizeQ4_K_M, "blk.5.ffn_up.weight", 5, 16, false); got != basegguf.TensorTypeQ4K {
		t.Fatalf("ffn_up type = %d, want Q4_K", got)
	}
	if got := llamaTensorType(basegguf.QuantizeQ4_K_M, "token_embd.weight", -1, 16, true); got != basegguf.TensorTypeQ6K {
		t.Fatalf("tied token_embd type = %d, want Q6_K", got)
	}
}
