// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import basegguf "dappco.re/go/inference/model/gguf"

var phiSpec = basegguf.TransformerLaneSpec{
	// Pinned to the checked-in Microsoft Phi-3 Mini config and safetensors
	// index fixtures in ../testdata and llama.cpp constants.py.
	Architecture: "phi3", ModelTypes: []string{"phi3"}, TokenizerPre: "llama-bpe",
	TopLevel: map[string]string{"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight", "lm_head.weight": "output.weight"},
	Layer:    map[string]string{"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight", "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight", "self_attn.qkv_proj.weight": "attn_qkv.weight", "self_attn.o_proj.weight": "attn_output.weight", "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight", "mlp.gate_up_proj.weight": "ffn_gate_up.weight", "mlp.down_proj.weight": "ffn_down.weight"},
}

func init() { basegguf.RegisterQuantizeLane("phi3", basegguf.NewTransformerQuantizeLane(phiSpec)) }
