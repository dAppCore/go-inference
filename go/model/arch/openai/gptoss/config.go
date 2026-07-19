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
// Config.Arch STILL refuses — but now only for a precisely named, LOOKED-UP (not guessed) reason: every real
// gpt_oss checkpoint carries per-layer attention sinks (self_attn.sinks, a learned per-head softmax-
// denominator bias — see the eager_attention_forward reference cited in config_test.go) and additive biases
// on self_attn.o_proj / mlp.router / mlp.experts.{gate,up,down}_proj (attention_bias=true reaches beyond
// q/k/v), neither of which engine/metal's quantised decode path consumes yet (Q/K/V bias DOES already flow
// through the existing BQ/BK/BV mechanism — see weights.go). Landing either is real SDPA/MoE-kernel surgery
// this pass did not attempt without a GPU to verify it on. Recognised + configured + geometry-verified +
// weight-mapped + the two requested engine primitives landed; full serving remains the named follow-up.
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
		AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1,
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
// schedule, YaRN-corrected rope) and, once that succeeds, still refuses: this pass wired the parse, the
// weight-name mapping (weights.go), and gpt_oss's two requested engine primitives (the clamped-SwiGLU MoE
// activation and the YaRN frequency table — both in this package/engine/metal, byte-gated), but THREE
// checkpoint-real primitives remain unconsumed by engine/metal — attention sinks, the o_proj/router/expert
// additive biases, and the YaRN attention_factor/mscale cos-sin postscale (arch.RopeScale multiplies the
// rope ANGLE pre-cos/sin — see rope_freqs.go's lthn_qknorm_rope_bf16.metal "theta = scale·offset·inv_freq"
// — not cos/sin post-hoc, so reusing it for mscale would be confidently WRONG, not merely absent). A config
// that fails buildArch's structural validation surfaces THAT error instead — the boundary below is reached
// only once the geometry is genuinely sound. See config.go's package doc for the full boundary and the
// commit history for the formula sources.
func (c *Config) Arch() (model.Arch, error) {
	a, err := c.buildArch()
	if err != nil {
		return model.Arch{}, err
	}
	mscale := yarnAttentionFactor(c.RopeScaling.Factor)
	return model.Arch{}, core.NewError(core.Sprintf(
		"gptoss.Config.Arch: gpt_oss geometry resolves cleanly (%d layers alternating sliding/full attention, "+
			"%d local experts / %d per token, hidden %d, vocab %d) and the weight-name map + the clamped-SwiGLU "+
			"activation + YaRN rope table are wired — but engine/metal's SDPA has no consumer for the per-head "+
			"self_attn.sinks softmax bias every layer carries, and its quantised decode path has no consumer for "+
			"the additive biases on o_proj/router/expert projections (attention_bias=true reaches beyond q/k/v, "+
			"which already flow through the existing BQ/BK/BV mechanism), and the YaRN attention_factor postscale "+
			"(~%.4f for this config's factor=%.1f) has no cos/sin-postscale hook either; recognised + configured "+
			"+ geometry-verified + weight-mapped + activation/rope landed, full serving remains a named follow-up",
		len(a.Layer), a.Experts, a.TopK, a.Hidden, a.Vocab, mscale, c.RopeScaling.Factor))
}
