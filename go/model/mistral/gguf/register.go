// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import basegguf "dappco.re/go/inference/model/gguf"

var mistralSpec = basegguf.TransformerLaneSpec{
	// Pinned to mistralai/Ministral-3-3B-Instruct-2512 config.json revision
	// b4b0163a32c9867d2424ac10b40fe0db6fa95110 and llama.cpp constants.py.
	Architecture: "mistral3", ModelTypes: []string{"mistral3", "ministral3"}, TokenizerPre: "mistral-bpe",
	TopLevel: map[string]string{"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight", "lm_head.weight": "output.weight"},
	Layer:    map[string]string{"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight", "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight", "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight"},
}

func init() {
	basegguf.RegisterQuantizeLane("mistral3", basegguf.NewTransformerQuantizeLane(mistralSpec))
}
