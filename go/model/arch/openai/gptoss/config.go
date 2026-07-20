// SPDX-Licence-Identifier: EUPL-1.2

// Package gptoss declares openai/gpt-oss-* (GptOssForCausalLM) to the backend-neutral reactive model
// loader. Unlike its openai/ siblings (privacyfilter is a token-classifier, whisper is an ASR
// encoder-decoder), GPT-OSS IS a generative MoE causal-LM — a decoder-only transformer alternating sliding-
// window and full-attention layers, YaRN long-context rope, and a routed MoE MLP — the same decode shape
// lem's other MoE arches (e.g. Qwen MoE) already fit. This package registers the model_type, parses its real
// config.json, resolves the full neutral model.Arch geometry (#18 factory pattern: DeriveLayers + WeightNames
// in weights.go), and implements gpt_oss's two arch-specific engine primitives — the clamped-sigmoid SwiGLU
// expert activation (engine/metal MoEExpertsQuantClampedSiLU) and YaRN-corrected rope frequencies (yarn.go).
//
// Config.Arch resolves fully — the three engine gaps the original registration refused over are all
// consumed (tracker #37): attention sinks bind the sdpa_vector has_sinks(25) lane (WeightNames.Sinks
// → LoadedLayer.Sinks → the engine's sinks-routed attention halves), the additive biases beyond
// BQ/BK/BV all land (o_proj via the projector BO seam; the router bias before top-k and the
// per-expert gate/up/down biases via encGptOssMoEHalf + MoEExpertsQuantClampedSiLU), and the YaRN
// attention_factor folds into AttnScale as mscale²/√headDim (exact for gpt_oss's full-rotary heads —
// see buildArch's derivation with both lineage sources). The serving lane is the linear-KV
// sequential arch session (paged KV and the batched attention fold decline sinks layers — named
// perf follow-ups); the live generation gate is live_generate_test.go.
//
// See model/quant/jang/jang.go for the separate GGUF quant-scheme name mapping already registered for
// "gpt_oss" (a quant concern, not this arch registration).
//
// Source: https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json
package gptoss

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// defaultRopeTheta/defaultRMSNormEps are defensive fallbacks for a config that omits them — every real
// gpt_oss config.json carries both explicitly (rope_theta 150000, rms_norm_eps 1e-05), so these only guard
// a hand-built/malformed document, matching the qwen35.Config fallback convention.
const (
	defaultRopeTheta  float32 = 10_000
	defaultRMSNormEps float32 = 1e-5
)

// RopeScaling is GPT-OSS's YaRN long-context extension. Unlike privacyfilter's "rope_parameters", GPT-OSS
// carries rope_theta at Config's top level and nests only the scaling factors here under "rope_scaling".
type RopeScaling struct {
	RopeType                      string  `json:"rope_type"`
	Factor                        float32 `json:"factor"`
	BetaFast                      float32 `json:"beta_fast"`
	BetaSlow                      float32 `json:"beta_slow"`
	OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
	Truncate                      bool    `json:"truncate"`
}

// Config is the architecture-relevant subset of a GptOssForCausalLM config.json.
type Config struct {
	ModelType             string      `json:"model_type"`
	HiddenAct             string      `json:"hidden_act"`
	HiddenSize            int         `json:"hidden_size"`
	IntermediateSize      int         `json:"intermediate_size"`
	NumHiddenLayers       int         `json:"num_hidden_layers"`
	NumAttentionHeads     int         `json:"num_attention_heads"`
	NumKeyValueHeads      int         `json:"num_key_value_heads"`
	HeadDim               int         `json:"head_dim"`
	VocabSize             int         `json:"vocab_size"`
	RMSNormEps            float32     `json:"rms_norm_eps"`
	RopeTheta             float32     `json:"rope_theta"`
	RopeScaling           RopeScaling `json:"rope_scaling"`
	MaxPositionEmbeddings int         `json:"max_position_embeddings"`
	SlidingWindow         int         `json:"sliding_window"`
	LayerTypes            []string    `json:"layer_types"`
	NumLocalExperts       int         `json:"num_local_experts"`
	NumExpertsPerTok      int         `json:"num_experts_per_tok"`
	ExpertsPerToken       int         `json:"experts_per_token"`
	SwigluLimit           float32     `json:"swiglu_limit"`
	TieWordEmbeddings     *bool       `json:"tie_word_embeddings"`
	// AttentionBias declares GPT-OSS's additive linear biases on every attention/router/expert projection
	// (q/k/v/o_proj, mlp.router, mlp.experts.{gate,up,down}_proj all carry a ".bias" tensor beside the
	// weight). Captured for documentation/completeness; Arch's refusal names this gap unconditionally
	// (every real gpt_oss checkpoint sets it true) rather than branching on this field — see config.go doc.
	AttentionBias bool `json:"attention_bias"`
}

// resolvedExpertsPerTok picks the router's top-k width: num_experts_per_tok is the field every real
// checkpoint sets; experts_per_token is a synonym some conversions also carry (the InferenceIllusionist
// MLX-4bit conversion sets both, identically, to 4). Prefers num_experts_per_tok, falls back to
// experts_per_token when only that is present — never guesses a third value when they disagree.
func (c *Config) resolvedExpertsPerTok() int {
	if c.NumExpertsPerTok > 0 {
		return c.NumExpertsPerTok
	}
	return c.ExpertsPerToken
}

// ParseConfig parses a GPT-OSS (GptOssForCausalLM) Hugging Face config.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("gptoss.ParseConfig: config.json parse failed")
	}
	return &cfg, nil
}

// InferFromWeights is a no-op: GPT-OSS's config.json declares every geometry field this package captures
// (vocab_size, head_dim, num_key_value_heads are all explicit in every real checkpoint, including the
// InferenceIllusionist MLX-4bit conversion's config.json), so there is nothing to resolve from the
// checkpoint's weight shapes (the don't-guess rule; see qwen2.Config.InferFromWeights for the pattern a
// checkpoint that DID omit a dimension would need). Arch's refusal is unconditional (see config.go doc),
// not weight-derived, so this stays inert even once serving lands.
func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// buildArch resolves the neutral model.Arch geometry from a real gpt_oss config: attention dims, the
// sliding/full layer schedule (DeriveLayers reacts to layer_types directly — gpt_oss's config always
// carries it explicitly, unlike qwen35's full_attention_interval fallback), MoE dims for the softmax
// top-k router (GptOssTopKRouter computes softmax(topk(logits)), which is mathematically IDENTICAL to
// softmax-over-all-experts + renormalise-the-selected-top-k — the same MoEGatingSoftmax+NormaliseMoETopK
// contract mixtral.Config already declares unconditionally; see router_test.go for the identity proof),
// and YaRN-corrected rope frequencies (yarn.go). Split from Arch() so the geometry math is unit-testable
// independent of Arch()'s unconditional serving refusal (see Arch doc) — mirrors qwen35.Config.Arch's
// shape, but qwen35 hands its build straight back; gpt_oss's Arch wraps this with the boundary message.
func (c *Config) buildArch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 {
		return model.Arch{}, core.NewError("gptoss.Config.Arch: hidden_size, num_hidden_layers, num_attention_heads must be > 0")
	}
	headDim := c.HeadDim
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("gptoss.Config.Arch: head_dim absent and hidden_size not divisible by num_attention_heads")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if kvHeads <= 0 || c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("gptoss.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	if len(c.LayerTypes) != c.NumHiddenLayers {
		return model.Arch{}, core.NewError(core.Sprintf(
			"gptoss.Config.Arch: layer_types length %d != num_hidden_layers %d", len(c.LayerTypes), c.NumHiddenLayers))
	}
	numExperts := c.NumLocalExperts
	if numExperts <= 0 {
		return model.Arch{}, core.NewError("gptoss.Config.Arch: num_local_experts must be > 0 — gpt_oss has no dense variant")
	}
	topK := c.resolvedExpertsPerTok()
	if topK <= 0 || topK > numExperts {
		return model.Arch{}, core.NewError("gptoss.Config.Arch: num_experts_per_tok/experts_per_token must be in (0, num_local_experts]")
	}
	eps := c.RMSNormEps
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	ropeBase := c.RopeTheta
	if ropeBase == 0 {
		ropeBase = defaultRopeTheta
	}
	rotaryDim := headDim // gpt_oss carries no partial_rotary_factor: every rotary layer is full rotary

	// swiglu_limit defaults to 7.0 when the config omits it — transformers' GptOssConfig default
	// (modeling/configuration_gpt_oss: limit=7.0; every published checkpoint sets it explicitly).
	swigluLimit := c.SwigluLimit
	if swigluLimit == 0 {
		swigluLimit = 7.0
	}

	freqs, err := c.yarnRopeFreqs(rotaryDim, ropeBase)
	if err != nil {
		return model.Arch{}, err
	}

	// YaRN attention_factor (mscale), folded into the SDPA softmax scale as mscale²/√headDim. The
	// application point, verified against BOTH lineage references, fetched from source (not recalled):
	//
	//	transformers modeling_gpt_oss.py, GptOssRotaryEmbedding.forward:
	//	  "cos = emb.cos() * self.attention_scaling; sin = emb.sin() * self.attention_scaling"
	//	  with attention_scaling = _compute_yarn_parameters' attention_factor default
	//	  0.1·ln(factor)+1 (the config sets neither attention_factor nor mscale), and
	//	  "self.scaling = self.head_dim**-0.5" as the SDPA scale.
	//	mlx-lm rope_utils.py, YarnRoPE.__call__ (the checkpoint's own lineage):
	//	  "x[..., : self.dims] = self.mscale * x[..., : self.dims]" before mx.fast.rope, with
	//	  mscale = yarn_get_mscale(scale, 1)/yarn_get_mscale(scale, 0) = 0.1·ln(scale)+1 — the
	//	  same factor applied to the pre-rope input instead of cos/sin (rotation is linear, so
	//	  the two are identical).
	//
	// Either way BOTH q and k emerge scaled by mscale, so the attention LOGITS carry mscale² — and
	// because gpt_oss is FULL-rotary (rotaryDim == headDim, no un-roped tail), folding mscale² into
	// the SDPA scale is algebraically EXACT, not an approximation: score = scale·(q·k) with
	// scale = mscale²/√headDim reproduces (mscale·q_rot)·(mscale·k_rot)/√headDim termwise. The
	// attention sinks stay OUTSIDE the fold, exactly as in the references: the kernel's scale
	// multiplies q only, and the sink seeds the softmax raw (see engine/metal sdpa_sinks.go).
	// A non-YaRN config keeps mscale = 1 (yarnAttentionFactor's factor<=1 guard) — the plain
	// 1/√headDim, byte-identical to every other arch's scale resolution.
	mscale := float64(1)
	if c.RopeScaling.RopeType == "yarn" {
		mscale = float64(yarnAttentionFactor(c.RopeScaling.Factor))
	}
	attnScale := float32(mscale * mscale / math.Sqrt(float64(headDim)))

	// DeriveLayers maps "sliding_attention"/"full_attention" straight to MixerAttention (gemma3 proves this
	// vocabulary already; gpt_oss has no "linear_attention" layers, so every layer owns its own KV cache —
	// numKVShared 0). MoE is unconditional: gpt_oss has no dense FFN layer at all.
	layers := model.DeriveLayers(c.LayerTypes, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
		layers[i].MoE = true
	}

	return model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads,
		HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: eps,
		AttnScale: attnScale, EmbedScale: 1,
		RopeBase: ropeBase, RopeLocalBase: ropeBase, RopeScale: 1,
		RotaryDim: rotaryDim, RotaryDimLocal: rotaryDim,
		RopeFreqs: freqs,
		// Activation deliberately does NOT say "silu": engine/metal's ffnUsesSiLU does an EXACT match on
		// "silu"/"swish" to select the plain-SiLU MoE expert kernel, and gpt_oss's gate is the CLAMPED
		// sigmoid-gated form (glu = clip(gate,,7)·sigmoid(1.702·clip(gate,,7)); out = glu·(clip(up,-7,7)+1) —
		// see engine/metal's MoEExpertsQuantClampedSiLU, sourced in that file's doc). A plain string here
		// would silently mis-route a future serving pass onto the WRONG (uncapped, unshifted) activation —
		// exactly the coherent-but-wrong GELU/SiLU trap this repo's house rule warns against.
		Activation:        "gpt_oss_clamped_swiglu",
		SwigluLimit:       swigluLimit,
		SlidingWindow:     c.SlidingWindow,
		Experts:           numExperts,
		TopK:              topK,
		ExpertFF:          c.IntermediateSize,
		MoEGating:         model.MoEGatingSoftmax,
		NormaliseMoETopK:  true,
		TieWordEmbeddings: c.TieWordEmbeddings,
		Layer:             layers,
	}, nil
}

// Arch resolves the FULL neutral geometry (buildArch — attention dims, MoE dims, the sliding/full
// schedule, YaRN-corrected rope, the mscale²-folded attention scale, the clamped-SwiGLU declaration)
// and returns it: the three engine gaps the former refusal named are all consumed now — attention
// sinks (engine/metal's sdpa_vector has_sinks lane, loaded via WeightNames.Sinks), the o_proj/
// router/per-expert additive biases (the projector BO seam + encGptOssMoEHalf's router-bias-before-
// top-k + MoEExpertsQuantClampedSiLU's bias adds), and the YaRN attention_factor (folded into
// AttnScale as mscale²/√headDim — see buildArch's derivation and sources). A structurally invalid
// config still surfaces buildArch's precise error.
func (c *Config) Arch() (model.Arch, error) {
	return c.buildArch()
}
