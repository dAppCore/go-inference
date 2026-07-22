// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import core "dappco.re/go"

// weights_decoder.go reads the DeepSeek-V2-lite MoE decoder (modeling_deepseekv2.py's
// DeepseekV2Model/DeepseekV2ForCausalLM, unmodified by the OCR checkpoint — DeepseekOCRModel only
// ADDS the vision merge ahead of it, see vision.go). Tensor names read verbatim off the real
// checkpoint, prefix model.layers.*/model.embed_tokens.weight/model.norm.weight/lm_head.weight.
// Every dimension below (layer count, expert count, per-expert width, which layers are dense vs
// MoE) is read from cfg — the ACTUAL config.json fields, confirmed to be what the checkpoint's
// own custom_code class reads (Config's doc comment) — never hardcoded, unlike the vision tower.

// DecoderExpertWeights is one routed expert's SwiGLU MLP (DeepseekV2MLP with
// intermediate_size=moe_intermediate_size): down(SiLU(gate(x))*up(x)).
type DecoderExpertWeights struct {
	GateW, UpW, DownW []float32 // [moeIntermediate,hidden] / [moeIntermediate,hidden] / [hidden,moeIntermediate]
}

// DecoderLayerWeights is one DeepseekV2DecoderLayer: pre-norm causal self-attention (plain
// rotary MHA — use_mla=false resolves to LlamaAttention, see decoder.go's doc comment) then a
// pre-norm MLP that is EITHER dense (layer_idx < cfg.FirstKDenseReplace — DenseGateW/DenseUpW/
// DenseDownW set, Experts nil) OR a mixture of NRoutedExperts routed experts plus one combined
// shared-expert MLP (layer_idx >= cfg.FirstKDenseReplace — GateWeight/Experts/Shared* set,
// Dense* nil) — IsMoE selects which, mirroring DeepseekV2DecoderLayer.__init__'s own
// "DeepseekV2MoE if … else DeepseekV2MLP" branch exactly (config.go's doc comment).
type DecoderLayerWeights struct {
	InputNormW     []float32
	QW, KW, VW, OW []float32 // each [hidden,hidden] — num_attention_heads == num_key_value_heads in this checkpoint (plain MHA, no GQA repeat)
	PostAttnNormW  []float32

	IsMoE bool

	// Dense MLP (layer_idx < FirstKDenseReplace): down(SiLU(gate(x))*up(x)), intermediate_size wide.
	DenseGateW, DenseUpW, DenseDownW []float32

	// MoE MLP (layer_idx >= FirstKDenseReplace).
	GateWeight                          []float32 // router: [n_routed_experts, hidden]
	Experts                             []DecoderExpertWeights
	SharedGateW, SharedUpW, SharedDownW []float32 // one combined MLP, intermediate_size = moe_intermediate_size * n_shared_experts
}

// DecoderWeights is the whole loaded MoE decoder: token embedding, cfg.NumHiddenLayers layers,
// a final RMSNorm, and the (untied — a separate lm_head.weight tensor, confirmed: no
// tie_word_embeddings override and the checkpoint carries both embed_tokens.weight AND
// lm_head.weight as distinct tensors) output projection.
type DecoderWeights struct {
	EmbedTokens  []float32 // [vocab,hidden]
	Layers       []DecoderLayerWeights
	FinalNormW   []float32
	LMHeadWeight []float32 // [vocab,hidden]
}

func loadDecoderWeights(l weightLoader, cfg *Config) (DecoderWeights, error) {
	var w DecoderWeights
	var err error
	h := cfg.HiddenSize
	if w.EmbedTokens, err = l.f32shaped("model.embed_tokens.weight", cfg.VocabSize*h); err != nil {
		return w, err
	}

	w.Layers = make([]DecoderLayerWeights, cfg.NumHiddenLayers)
	for i := range w.Layers {
		p := core.Sprintf("model.layers.%d", i)
		var ly DecoderLayerWeights
		if ly.InputNormW, err = l.f32shaped(p+".input_layernorm.weight", h); err != nil {
			return w, err
		}
		if ly.PostAttnNormW, err = l.f32shaped(p+".post_attention_layernorm.weight", h); err != nil {
			return w, err
		}
		if ly.QW, _, err = l.linearW(p+".self_attn.q_proj", h, h, false); err != nil {
			return w, err
		}
		if ly.KW, _, err = l.linearW(p+".self_attn.k_proj", h, h, false); err != nil {
			return w, err
		}
		if ly.VW, _, err = l.linearW(p+".self_attn.v_proj", h, h, false); err != nil {
			return w, err
		}
		if ly.OW, _, err = l.linearW(p+".self_attn.o_proj", h, h, false); err != nil {
			return w, err
		}

		ly.IsMoE = i >= cfg.FirstKDenseReplace && cfg.NRoutedExperts > 0
		if !ly.IsMoE {
			if ly.DenseGateW, _, err = l.linearW(p+".mlp.gate_proj", h, cfg.IntermediateSize, false); err != nil {
				return w, err
			}
			if ly.DenseUpW, _, err = l.linearW(p+".mlp.up_proj", h, cfg.IntermediateSize, false); err != nil {
				return w, err
			}
			if ly.DenseDownW, _, err = l.linearW(p+".mlp.down_proj", cfg.IntermediateSize, h, false); err != nil {
				return w, err
			}
			w.Layers[i] = ly
			continue
		}

		if ly.GateWeight, _, err = l.linearW(p+".mlp.gate", h, cfg.NRoutedExperts, false); err != nil {
			return w, err
		}
		ly.Experts = make([]DecoderExpertWeights, cfg.NRoutedExperts)
		for e := range ly.Experts {
			ep := core.Sprintf("%s.mlp.experts.%d", p, e)
			var ex DecoderExpertWeights
			if ex.GateW, _, err = l.linearW(ep+".gate_proj", h, cfg.MoEIntermediateSize, false); err != nil {
				return w, err
			}
			if ex.UpW, _, err = l.linearW(ep+".up_proj", h, cfg.MoEIntermediateSize, false); err != nil {
				return w, err
			}
			if ex.DownW, _, err = l.linearW(ep+".down_proj", cfg.MoEIntermediateSize, h, false); err != nil {
				return w, err
			}
			ly.Experts[e] = ex
		}
		sharedIntermediate := cfg.MoEIntermediateSize * cfg.NSharedExperts
		if ly.SharedGateW, _, err = l.linearW(p+".mlp.shared_experts.gate_proj", h, sharedIntermediate, false); err != nil {
			return w, err
		}
		if ly.SharedUpW, _, err = l.linearW(p+".mlp.shared_experts.up_proj", h, sharedIntermediate, false); err != nil {
			return w, err
		}
		if ly.SharedDownW, _, err = l.linearW(p+".mlp.shared_experts.down_proj", sharedIntermediate, h, false); err != nil {
			return w, err
		}
		w.Layers[i] = ly
	}

	if w.FinalNormW, err = l.f32shaped("model.norm.weight", h); err != nil {
		return w, err
	}
	if w.LMHeadWeight, err = l.f32shaped("lm_head.weight", cfg.VocabSize*h); err != nil {
		return w, err
	}
	return w, nil
}
