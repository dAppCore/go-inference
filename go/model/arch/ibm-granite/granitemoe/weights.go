// SPDX-Licence-Identifier: EUPL-1.2

package granitemoe

import "dappco.re/go/inference/model"

// FactoryWeightNames returns the GraniteMoE tensor layout for model.Assemble: StandardWeightNames'
// attention projections as-is (GraniteMoE is llama-shaped there — self_attn.{q,k,v,o}_proj,
// input_layernorm, no QK-norm, no partial rotary), the llama/mistral 2-norm FFN override
// (post_attention_layernorm is the pre-MoE norm; there is no gemma-style post-attention sandwich norm —
// mirrors mixtral.FactoryWeightNames and gptoss.WeightNames exactly, same architecture family), and the
// MoE block pointed straight at the checkpoint's OWN already-packed tensors.
//
// Unlike Mixtral/Llama4 (which ship one 2-D matrix PER EXPERT and need packExperts/NormalizeWeights to
// synthesise the packed convention), GraniteMoE ships input_linear.weight/output_linear.weight ALREADY
// packed as one 3-D [experts, …, …] tensor per layer covering every expert (see NormalizeWeights' doc,
// and the real ibm-granite/granite-3.1-1b-a400m-base index fixture in testdata/) — row-major safetensors
// data has no strides, so that 3-D tensor is byte-identical to the [experts·outDim, inDim] shape
// model.Assemble's MoE loader already expects (the same "3-D is byte-identical to the flattened 2-D"
// fact gptoss.WeightNames documents for its own native per-layer expert tensors). No NormalizeConfig
// synthesis step is needed here — just naming the checkpoint's real tensors, zero-copy.
//
// input_linear packs [gate‖up] per expert — NormalizeWeights' existing split (ib[:gateBytes] is gate,
// ib[gateBytes:] is up) confirms gate occupies the first half of each expert's rows, up the second —
// exactly the per-expert [gate‖up] order engine/metal's fused ExpGateUp kernel path expects
// (fuseExpertGateUpQuant lays out separately-shipped gate/up checkpoints in that identical order), so
// ExpGateUp names input_linear directly rather than splitting it into separate ExpGate/ExpUp roles.
//
// GraniteMoE has no shared expert: SharedGate/Up/Down/SharedSigmoid stay "" (nil-safe — see
// MoEWeightNames), matching Config.Arch's unconditional SharedExperts: 0.
func FactoryWeightNames() model.WeightNames {
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	w.MoE = model.MoEWeightNames{
		PreFFNorm: ".post_attention_layernorm.weight",
		Router:    ".block_sparse_moe.router.layer",
		ExpGateUp: ".block_sparse_moe.input_linear",
		ExpDown:   ".block_sparse_moe.output_linear",
	}
	return w
}
