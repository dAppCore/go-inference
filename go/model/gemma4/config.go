// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// Config is the backend-agnostic gemma4 model configuration: the architecture-
// relevant subset of the HF config.json. The json tags match config.json so a raw
// config unmarshals straight into it (core.JSONUnmarshal), and Arch() fills a complete
// backend-agnostic Arch — the dims-from-config step a loader needs so it never
// hand-assembles transformer dims. pkg/metal's Gemma4TextConfig carries the same
// fields (plus backend/runtime extras); this is the neutral, all-platforms mirror.
type Config struct {
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`        // sliding-attention head_dim (the default for every layer when global_head_dim is absent)
	GlobalHeadDim     int     `json:"global_head_dim"` // full_attention head_dim — gemma4 uses a larger one (E2B/E4B/12B/31B/26B: 512 vs sliding 256); 0 ⇒ same as HeadDim
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`

	// NumGlobalKeyValueHeads is the full_attention KV-head count when it differs from
	// the sliding num_key_value_heads (gemma4 may carry it); 0 ⇒ same as NumKeyValueHeads.
	NumGlobalKeyValueHeads int `json:"num_global_key_value_heads"`

	FinalLogitSoftcapping float32              `json:"final_logit_softcapping"`
	SlidingWindow         int                  `json:"sliding_window"`
	MaxPositionEmbeddings int                  `json:"max_position_embeddings"` // required by the copied parser (don't-guess); not consumed by Arch (maxLen is a session arg)
	NumKVSharedLayers     int                  `json:"num_kv_shared_layers"`
	LayerTypes            []string             `json:"layer_types"`
	AttentionKEqV         bool                 `json:"attention_k_eq_v"`
	RopeParameters        map[string]RopeParam `json:"rope_parameters"` // per-attention-type RoPE (full_attention / sliding_attention)

	VocabSizePerLayerInput  int `json:"vocab_size_per_layer_input"`
	HiddenSizePerLayerInput int `json:"hidden_size_per_layer_input"`

	EnableMoEBlock      bool `json:"enable_moe_block"`
	NumExperts          int  `json:"num_experts"`
	TopKExperts         int  `json:"top_k_experts"`
	MoEIntermediateSize int  `json:"moe_intermediate_size"`

	Quantization *model.QuantConfig `json:"quantization"` // present in 4-bit checkpoints (mlx group-affine)

	// TextConfig holds the text-model arch when the checkpoint is the multimodal wrapper
	// (gemma4_text / gemma4_unified_text): real packs nest hidden_size/layers/rope_parameters/…
	// under "text_config", with quantization left at the top level. nil for a flat (text-only or
	// synthetic) config. Arch() / ResolvedQuant() resolve it.
	TextConfig *Config `json:"text_config"`
}

// ResolvedQuant returns the checkpoint's quantization block, preferring the top-level one (where
// the multimodal wrapper puts it) and falling back to the nested text_config. nil = bf16.
func (c Config) ResolvedQuant() *model.QuantConfig {
	if c.Quantization != nil {
		return c.Quantization
	}
	if c.TextConfig != nil {
		return c.TextConfig.Quantization
	}
	return nil
}

// RopeParam is one attention type's RoPE configuration: the theta and the partial-rotary
// factor (gemma4 full_attention uses 0.25 — only a quarter of each head's dims are rotated).
// rope_type / factor scaling is a later refinement.
type RopeParam struct {
	RopeTheta           float32 `json:"rope_theta"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"` // fraction of head dims rotated (default 1.0 = full)
	RopeType            string  `json:"rope_type"`             // "proportional" (gemma4 full_attention) or "default"
	Factor              float32 `json:"factor"`                // proportional scaling factor (absent → 1; folding it is a later refinement)
}

// gemma4 defaults applied when a config omits the field.
const (
	defaultRopeTheta      float32 = 1_000_000 // gemma4 global (full_attention) RoPE base
	defaultRopeLocalTheta float32 = 10_000    // gemma4 sliding_attention RoPE base
	defaultRMSNormEps     float32 = 1e-6
)

// Arch builds the backend-agnostic Arch from the config: it fills the neutral
// transformer dims + gemma4-specifics, derives the per-layer attention/KV-share specs
// (model.DeriveLayers over layer_types + num_kv_shared_layers), and marks every layer MoE
// when enable_moe_block is set — gemma4 applies MoE uniformly across layers, not
// interleaved (matching pkg/metal's per-layer EnableMoE = the model-wide flag).
// HeadDim defaults to hidden_size / num_attention_heads, NumKeyValueHeads to
// NumAttentionHeads (MHA), eps/rope to the gemma4 defaults, when the config omits
// them. Validates the load-bearing invariants. RoPE is per-attention-type: RopeBase is the
// global (full_attention) theta, RopeLocalBase the sliding_attention theta (gemma4 defaults
// 1e6 / 1e4, overridden by rope_parameters). RopeScale (the rope_type/factor scaling) is the
// single global 1.0 today — proportional/yarn scaling is a later refinement.
func (c Config) Arch() (model.Arch, error) {
	// multimodal wrapper: the text arch lives under text_config (the top level carries only
	// modality configs + quantization). Derive from it — the arch is representation-agnostic,
	// so the top-level quantization is irrelevant here (ResolvedQuant handles it for the loader).
	if c.TextConfig != nil {
		return c.TextConfig.Arch()
	}
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 {
		return model.Arch{}, core.NewError("gemma4.Config.Arch: hidden_size, num_hidden_layers, num_attention_heads must be > 0")
	}

	headDim := c.HeadDim
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("gemma4.Config.Arch: head_dim absent and hidden_size not divisible by num_attention_heads")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("gemma4.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	// per-attention-type attention geometry: gemma4 full_attention layers use a larger
	// head_dim (global_head_dim) than sliding (head_dim), and may carry a distinct KV
	// head count (num_global_key_value_heads). Absent ⇒ no distinction (the global
	// values mirror the sliding/default), so uniform packs are unaffected.
	globalHeadDim := c.GlobalHeadDim
	if globalHeadDim == 0 {
		globalHeadDim = headDim
	}
	globalKVHeads := c.NumGlobalKeyValueHeads
	if globalKVHeads == 0 {
		globalKVHeads = kvHeads
	}
	if c.NumAttentionHeads%globalKVHeads != 0 {
		return model.Arch{}, core.NewError("gemma4.Config.Arch: num_attention_heads must be a multiple of num_global_key_value_heads")
	}

	layerTypes := c.LayerTypes
	if len(layerTypes) == 0 {
		// no per-layer types declared → all global attention.
		layerTypes = make([]string, c.NumHiddenLayers)
		for i := range layerTypes {
			layerTypes[i] = "full_attention"
		}
	}
	if len(layerTypes) != c.NumHiddenLayers {
		return model.Arch{}, core.NewError("gemma4.Config.Arch: layer_types length must equal num_hidden_layers")
	}

	experts, topK, expertFF := 0, 0, 0
	if c.EnableMoEBlock {
		if c.NumExperts <= 0 || c.TopKExperts <= 0 {
			return model.Arch{}, core.NewError("gemma4.Config.Arch: enable_moe_block set but num_experts / top_k_experts not declared")
		}
		if c.TopKExperts > c.NumExperts {
			return model.Arch{}, core.NewError("gemma4.Config.Arch: top_k_experts must not exceed num_experts")
		}
		experts, topK = c.NumExperts, c.TopKExperts
		expertFF = c.MoEIntermediateSize
		if expertFF == 0 {
			expertFF = c.IntermediateSize // fall back to the dense FF when unspecified
		}
	}

	eps := c.RMSNormEps
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	// per-attention-type RoPE theta: global (full_attention) defaults to rope_theta or 1e6;
	// sliding_attention to 1e4 — overridden by rope_parameters when present.
	ropeBase := c.RopeTheta
	if ropeBase == 0 {
		ropeBase = defaultRopeTheta
	}
	if rp, ok := c.RopeParameters["full_attention"]; ok && rp.RopeTheta != 0 {
		ropeBase = rp.RopeTheta
	}
	ropeLocalBase := defaultRopeLocalTheta
	if rp, ok := c.RopeParameters["sliding_attention"]; ok && rp.RopeTheta != 0 {
		ropeLocalBase = rp.RopeTheta
	}
	// partial rotary: the fraction of each head's dims that RoPE rotates (gemma4
	// full_attention = 0.25, sliding = full). rotaryDim = floor(headDim · factor),
	// defaulting to the full headDim when no factor is declared (mirrors mlx).
	// rotaryDim is per-attention-type AND per-head-dim: full_attention rotates a
	// fraction of GlobalHeadDim, sliding a fraction of HeadDim.
	rotaryDim, rotaryDimLocal := globalHeadDim, headDim
	if rp, ok := c.RopeParameters["full_attention"]; ok && rp.PartialRotaryFactor > 0 {
		rotaryDim = int(float32(globalHeadDim) * rp.PartialRotaryFactor)
	}
	if rp, ok := c.RopeParameters["sliding_attention"]; ok && rp.PartialRotaryFactor > 0 {
		rotaryDimLocal = int(float32(headDim) * rp.PartialRotaryFactor)
	}
	// proportional RoPE (gemma4 full_attention): the partial-rotary frequencies are normalised
	// over the FULL headDim, not the rotated subset — exactly equivalent to default RoPE with an
	// effective base of base^(rotaryDim/headDim), since (base^(rd/hd))^(-2i/rd) = base^(-2i/hd).
	// Folding it into the base means the decode needs no proportional-specific path (full rotary
	// → base^1 unchanged; "default" type → unchanged). A non-unit `factor` (absent in current
	// packs) would additionally scale the angle — a later refinement.
	ropeBase = proportionalBase(ropeBase, rotaryDim, globalHeadDim, c.RopeParameters["full_attention"].RopeType)
	ropeLocalBase = proportionalBase(ropeLocalBase, rotaryDimLocal, headDim, c.RopeParameters["sliding_attention"].RopeType)

	layers := model.DeriveLayers(layerTypes, c.NumKVSharedLayers)
	// resolve each layer's attention geometry from its type (full → global dims,
	// sliding → the default dims), and apply MoE uniformly when enabled.
	for i := range layers {
		if layers[i].Attention == model.GlobalAttention {
			layers[i].HeadDim, layers[i].KVHeads = globalHeadDim, globalKVHeads
		} else {
			layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
		}
		if c.EnableMoEBlock {
			layers[i].MoE = true
		}
	}

	return model.Arch{
		Hidden:              c.HiddenSize,
		Heads:               c.NumAttentionHeads,
		KVHeads:             kvHeads,
		HeadDim:             headDim,
		GlobalHeadDim:       globalHeadDim,
		GlobalKVHeads:       globalKVHeads,
		FF:                  c.IntermediateSize,
		Vocab:               c.VocabSize,
		Experts:             experts,
		TopK:                topK,
		ExpertFF:            expertFF,
		Eps:                 eps,
		AttnScale:           1,                                         // gemma4: the per-head QK-norm does the scaling, so SDPA scale is 1.0 (metal's gemma4AttentionScale), NOT 1/√headDim
		EmbedScale:          float32(math.Sqrt(float64(c.HiddenSize))), // gemma-family √hidden token-embedding multiplier, declared so backends never assume it
		RopeBase:            ropeBase,
		RopeLocalBase:       ropeLocalBase,
		RotaryDim:           rotaryDim,
		RotaryDimLocal:      rotaryDimLocal,
		RopeScale:           1,
		SoftCap:             c.FinalLogitSoftcapping,
		SlidingWindow:       c.SlidingWindow,
		PerLayerInputVocab:  c.VocabSizePerLayerInput,
		PerLayerInputHidden: c.HiddenSizePerLayerInput,
		AttentionKEqV:       c.AttentionKEqV,
		ValueNorm:           true, // gemma4 value-norms V (no-scale per-head RMSNorm) in every attention layer
		Layer:               layers,
	}, nil
}

// proportionalBase returns the effective RoPE base for the "proportional" rope_type, which
// normalises the partial-rotary frequencies over the full headDim: base^(rotaryDim/headDim),
// so the default-rope kernel (which normalises over rotaryDim) reproduces it exactly. Full
// rotary (rotaryDim == headDim) or any non-proportional type returns base unchanged.
func proportionalBase(base float32, rotaryDim, headDim int, ropeType string) float32 {
	if ropeType != "proportional" || rotaryDim <= 0 || rotaryDim >= headDim {
		return base
	}
	return float32(math.Pow(float64(base), float64(rotaryDim)/float64(headDim)))
}
