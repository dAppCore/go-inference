// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import basegguf "dappco.re/go/inference/model/gguf"

var qwen3Spec = basegguf.TransformerLaneSpec{
	// Pinned to Qwen/Qwen3-0.6B config revision
	// 9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439 and its bf16 index map.
	Architecture: "qwen3", ModelTypes: []string{"qwen3"}, TokenizerPre: "qwen2",
	TopLevel: map[string]string{"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight", "lm_head.weight": "output.weight"},
	Layer:    map[string]string{"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight", "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight", "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight", "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight"},
}

func init() { basegguf.RegisterQuantizeLane("qwen3", basegguf.NewTransformerQuantizeLane(qwen3Spec)) }
