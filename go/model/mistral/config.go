// SPDX-Licence-Identifier: EUPL-1.2

// Package mistral is the backend-agnostic config for the Mistral / Ministral text architecture
// (model_type "mistral3" / "ministral3"). Ministral-3 is a standard Mistral transformer — GQA,
// RoPE, SwiGLU, RMSNorm, full attention — which is architecturally a SUBSET of gemma4 (no
// QK-norm, post-FF norm, soft-cap, sliding window, partial rotary, per-layer-input or MoE). So
// it reuses model.Arch (the de-facto generic decode declaration; a shared model.Arch is a
// later cleanup) with every gemma4-specific feature off, and the native executor / session /
// ops run it unchanged.
package mistral

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture-relevant subset of a Ministral-3 config.json. Real packs are the
// multimodal wrapper (Mistral3ForConditionalGeneration): the text arch nests under text_config,
// with vision_config a sibling — Arch() resolves it.
type Config struct {
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	IntermediateSize  int     `json:"intermediate_size"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	SlidingWindow     int     `json:"sliding_window"` // null in Ministral-3 → 0 → full attention

	RopeParameters *RopeParams `json:"rope_parameters"`
	TextConfig     *Config     `json:"text_config"`
}

// RopeParams is Ministral's RoPE config. Ministral-3 uses YaRN (rope_type "yarn") for long
// context; only RopeTheta is consumed today — the YaRN per-dim frequency remapping (Factor,
// BetaFast/Slow, OriginalMaxPositionEmbeddings) is a long-context faithfulness refinement. At
// short context YaRN preserves the high-frequency dims and only interpolates the long-range
// low-frequency ones, so the base theta is a good first approximation.
type RopeParams struct {
	RopeTheta                     float32 `json:"rope_theta"`
	RopeType                      string  `json:"rope_type"` // "yarn" for Ministral-3
	Factor                        float32 `json:"factor"`
	BetaFast                      float32 `json:"beta_fast"`
	BetaSlow                      float32 `json:"beta_slow"`
	OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
}

const defaultRopeTheta float32 = 1_000_000 // Ministral-3 rope_theta

// InferFromWeights satisfies model.ArchConfig: Mistral declares every dim (Arch() falls back to
// hidden/heads when head_dim is absent), so it reads nothing from the weight shapes.
func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch builds a backend-agnostic model.Arch from a Ministral config: the neutral transformer
// dims, full attention on every layer (no sliding, no KV-share), full rotary (no partial), and
// every gemma4-specific extra off (no soft-cap / QK-norm / per-layer-input / MoE). HeadDim
// defaults to hidden/heads, KVHeads to heads, eps to 1e-5, rope to rope_theta or 1e6.
func (c Config) Arch() (model.Arch, error) {
	if c.TextConfig != nil { // multimodal wrapper: the text arch is nested
		return c.TextConfig.Arch()
	}
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 {
		return model.Arch{}, core.NewError("mistral.Config.Arch: hidden_size, num_hidden_layers, num_attention_heads must be > 0")
	}
	headDim := c.HeadDim
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("mistral.Config.Arch: head_dim absent and hidden_size not divisible by num_attention_heads")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("mistral.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	eps := c.RMSNormEps
	if eps == 0 {
		eps = 1e-5
	}
	ropeBase := defaultRopeTheta
	if c.RopeParameters != nil && c.RopeParameters.RopeTheta != 0 {
		ropeBase = c.RopeParameters.RopeTheta
	}
	// every layer full attention, no KV-share — DeriveLayers over all-global layer types.
	layerTypes := make([]string, c.NumHiddenLayers)
	for i := range layerTypes {
		layerTypes[i] = "full_attention"
	}
	// Mistral is a standard transformer: ONE head_dim across layers (no per-type
	// distinction) and the standard SDPA scale 1/√headDim (no QK-norm to do the
	// scaling, unlike gemma4). The model declares it; the engine applies it.
	layers := model.DeriveLayers(layerTypes, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
	}
	arch := model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads, HeadDim: headDim,
		GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		FF: c.IntermediateSize, Vocab: c.VocabSize, Eps: eps,
		AttnScale: float32(1.0 / math.Sqrt(float64(headDim))),
		RopeBase:  ropeBase, RopeLocalBase: ropeBase, RotaryDim: headDim, RotaryDimLocal: headDim, RopeScale: 1,
		SlidingWindow: c.SlidingWindow,
		Layer:         layers,
	}
	// YaRN long-context: when rope_type is "yarn" with an extension factor, the
	// rotary frequencies are the NTK-by-parts remap rather than the uniform base.
	// Resolve them onto the arch so the backend's RoPE uses them; beta_fast/slow
	// default to the YaRN paper's 32/1 when a config declares yarn but omits them.
	if rp := c.RopeParameters; rp != nil && rp.RopeType == "yarn" && rp.Factor > 1 && rp.OriginalMaxPositionEmbeddings > 0 {
		betaFast, betaSlow := rp.BetaFast, rp.BetaSlow
		if betaFast == 0 {
			betaFast = 32
		}
		if betaSlow == 0 {
			betaSlow = 1
		}
		arch.RopeFreqs = YaRNInvFreqs(float64(ropeBase), float64(rp.Factor), float64(betaFast), float64(betaSlow), rp.OriginalMaxPositionEmbeddings, arch.RotaryDim)
	}
	return arch, nil
}
