// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import "dappco.re/go/inference/model"

// WeightNames returns the GPT-OSS tensor layout for model.Assemble — StandardWeightNames with the MoE
// overrides. Names verified against the REAL checkpoint's tensor index (never guessed): both
// InferenceIllusionist/gpt-oss-20b-MLX-4bit's model.safetensors.index.json (MLX affine 4-bit) and the
// native openai/gpt-oss-20b config's declared layout share this convention.
//
// Attention (Q/K/V/O) and the top-level Embed/LMHead/FinalNorm/AttnNorm names are ALL already
// StandardWeightNames' defaults (model.layers.%d.self_attn.{q,k,v,o}_proj, model.embed_tokens, lm_head,
// model.norm.weight, .input_layernorm.weight) — gpt_oss is llama-shaped there, so no override needed.
//
// The load-bearing overrides are the FFN pre-norm (llama/qwen 2-norm shape: post_attention_layernorm is
// the FFN's pre-norm, there is no gemma-style post-attention sandwich norm — NormBiasOne stays false,
// plain RMSNorm) and the MoE block: gpt_oss has NO dense variant (every layer is MoE, see
// Config.buildArch) and NO shared expert (SharedGate/Up/Down/SharedSigmoid all correctly left ""). The
// routed experts are ONE BATCHED tensor per layer per role (model.layers.%d.mlp.experts.{gate,up,down}_proj,
// shape [num_local_experts, ...] in the checkpoint) — the same "one tensor spans every expert" shape
// qwen35's switch_mlp convention uses, just gpt_oss's own top-level "experts" name rather than
// "switch_mlp"/"switch_glu". model.Assemble's MoE loader reads the RAW BYTES of this tensor via
// LoadLinear and lets engine/metal's moe_batch.go re-derive the per-expert row stride from
// arch.ExpertFF/NumExperts (not from the tensor's OWN declared shape/rank), so a 3-D-shaped safetensors
// tensor ([experts, outDim, packedInDim], row-major) is byte-identical to the 2-D-flattened
// [experts*outDim, packedInDim] shape the engine already assumes — verified by reading the real shard
// header (python struct-unpack of the safetensors JSON header, not by loading the model).
//
// Every attention/router/expert projection here ALSO carries a plain additive ".bias" tensor beside its
// ".weight" (attention_bias=true reaches beyond q/k/v to o_proj, mlp.router, and every expert
// gate/up/down_proj) — model.LoadLinear auto-probes prefix+".bias" for ANY named weight unconditionally,
// so WeightNames needs no separate bias fields; the loading side already captures every one of these
// biases into Linear.Bias. What's still missing is downstream CONSUMPTION for o_proj/router/expert (see
// config.go's Arch doc) — q/k/v's biases DO already flow end-to-end via the existing BQ/BK/BV mechanism.
func WeightNames() model.WeightNames {
	w := model.StandardWeightNames()

	// llama/qwen 2-norm layout: the FFN's pre-norm is post_attention_layernorm; no gemma post-attn norm.
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	w.NormBiasOne = false // plain RMSNorm (no gemma "+1" fold)

	// Attention sinks: every layer ships self_attn.sinks (bf16 [num_attention_heads] = [64] in the real
	// gpt-oss-20b MLX-4bit checkpoint's shard header) — the learned per-head softmax-
	// denominator logit model.Assemble loads RAW into LoadedLayer.Sinks (never through the NormBiasOne
	// fold; a sink is a logit, not a norm) and engine/metal binds to the sdpa_vector kernels'
	// has_sinks(25) function-constant lane (sinks buffer(16)/buffer(18) — MLX v0.32.0
	// mlx/backend/metal/kernels/sdpa_vector.h, the exact source of the shipped metallib).
	w.Sinks = ".self_attn.sinks"

	w.MoE = model.MoEWeightNames{
		PreFFNorm: ".post_attention_layernorm.weight",
		Router:    ".mlp.router",
		ExpGate:   ".mlp.experts.gate_proj",
		ExpUp:     ".mlp.experts.up_proj",
		ExpDown:   ".mlp.experts.down_proj",
		// No RouterScale/PerExpertScale (gpt_oss's router carries no separate scale weight) and no shared
		// expert (SharedGate/Up/Down/SharedSigmoid) — gpt_oss routes to experts only, never an always-on
		// dense branch, unlike qwen3_5_moe.
	}
	return w
}
