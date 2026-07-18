// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import "dappco.re/go/inference/model"

// WeightNames returns the Qwen 3.6 hybrid tensor layout for model.Assemble — StandardWeightNames with the
// qwen overrides. It is the (norm-mapping + MoE-name) half of retiring model/composed: the reactive
// assembler reacts to this data instead of composed's hand-rolled tensor reads.
//
// The load-bearing override is the NORM MAPPING. qwen is the llama/mistral 2-norm shape, not gemma's
// sandwich: the FFN's input norm IS post_attention_layernorm, and there is no gemma-style post-attention
// norm on the attention output. Mapping post_attention_layernorm → MLPNorm (leaving PostAttnNorm "") is what
// keeps the gated-delta mixer correct — the metal decode applies a post-mixer norm ONLY when PostAttnNorm is
// present, and for qwen it must NOT (post_attention_layernorm is the FFN's pre-norm, applied in the FFN
// block, never to the mixer output). RMSNorm is plain (NormBiasOne stays false — qwen is not gemma).
//
// The gated-delta (linear_attention) layers carry no q/k/v/norm override here: model.Assemble routes a
// MixerGatedDelta layer through assembleGatedDelta (the linear_attn.* weights) by the layer's Mixer kind,
// not by these names — so the attention names below apply to the full_attention layers, and the FFN names
// (dense + MoE) apply to every layer's feed-forward.
func WeightNames() model.WeightNames {
	w := model.StandardWeightNames()

	// qwen norm layout: the MLP/FFN pre-norm is post_attention_layernorm; no gemma post-attention norm.
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	w.NormBiasOne = false // plain RMSNorm (no gemma "+1" fold)

	// MoE (qwen3_5_moe): the batched switch_mlp routed experts, the mlp.gate router, and the always-on
	// shared expert with its sigmoid gate. The MoE FFN's pre-norm is the same post_attention_layernorm.
	w.MoE = model.MoEWeightNames{
		PreFFNorm:     ".post_attention_layernorm.weight",
		Router:        ".mlp.gate",
		ExpGate:       ".mlp.switch_mlp.gate_proj",
		ExpUp:         ".mlp.switch_mlp.up_proj",
		ExpDown:       ".mlp.switch_mlp.down_proj",
		SharedGate:    ".mlp.shared_expert.gate_proj",
		SharedUp:      ".mlp.shared_expert.up_proj",
		SharedDown:    ".mlp.shared_expert.down_proj",
		SharedSigmoid: ".mlp.shared_expert_gate",
	}
	return w
}
