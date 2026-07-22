// SPDX-Licence-Identifier: EUPL-1.2

// Package qwen35 declares the Qwen 3.6 hybrid architecture — one architecture released under three
// model_type families: qwen3_5 / qwen3_5_moe (nested text_config qwen3_5_text / qwen3_5_moe_text),
// qwen3_6 / qwen3_6_moe (the same hybrid under its other released name), and qwen3_next (Qwen 3.6's
// predecessor) — to the engine's reactive loader. It is a 3:1
// GatedDeltaNet/full-attention hybrid: linear_attention layers carry the gated-delta recurrence
// (MixerGatedDelta — a recurrent state, no KV cache) and every full_attention_interval-th layer is standard
// GQA with GATED output (attn_output_gate: q_proj emits [q ; gate] and the attention output is sigmoid-gated
// before o_proj). Rotary is PARTIAL (partial_rotary_factor 0.25) and RMSNorm is plain (qwen is not gemma —
// no +1 fold). The FFN is dense SwiGLU (qwen3_5) or a sparse SiLU MoE with one always-on shared expert
// (qwen3_5_moe, 256 experts top-8).
//
// This is the factory-native declaration the composed loader is being retired in favour of (#18): the
// per-layer schedule (DeriveLayers maps linear_attention → MixerGatedDelta) plus the neutral Arch
// declarations here feed model.Assemble (weights) and arch_session (decode) — the same reactive path
// gemma4 rides — rather than the retired composed engine's parallel loader (#50). It carries the schedule + declarations only;
// the gated-delta geometry (key/value heads, conv kernel) is DERIVED from the weight shapes at assemble
// time (assembleGatedDelta), not trusted from config field names a checkpoint might spell differently.
package qwen35

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const (
	// defaultRopeTheta matches the retired composed engine's ropeTheta() fallback (1e6): the real Qwen 3.6 checkpoints
	// carry rope_theta 1e7 under rope_parameters, so this fires only for a config that omits it entirely.
	defaultRopeTheta  float32 = 1_000_000
	defaultRMSNormEps float32 = 1e-6
)

// ropeParameters is the rope_parameters block (Qwen 3.6 nests rope config here, not under rope_scaling).
// For pure-text decode the mRoPE collapses to standard partial rotary, so only the theta + partial factor
// are read here (mrope_section is the multimodal position split, out of scope for the text schedule).
type ropeParameters struct {
	RopeTheta           float32 `json:"rope_theta"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
}

// TokenID parses a HF token-id config field that ships as EITHER a scalar or a list (a multimodal
// wrapper's eos_token_id is [id, …] while the nested text_config's is a plain int) — keeping the first
// id. An absent/null/unreadable field stays 0 rather than failing the whole parse (the composed
// loader's tokenID contract, ported with the vision wiring).
type TokenID int32

// UnmarshalJSON accepts a JSON number or the first element of a JSON array.
func (t *TokenID) UnmarshalJSON(b []byte) error {
	var n float64
	if r := core.JSONUnmarshal(b, &n); r.OK {
		*t = TokenID(n)
		return nil
	}
	var arr []float64
	if r := core.JSONUnmarshal(b, &arr); r.OK && len(arr) > 0 {
		*t = TokenID(arr[0])
	}
	return nil
}

// VisionArchConfig is the multimodal wrapper's sibling vision_config. Only PatchSize is load-bearing
// (pixel geometry has no tensor to derive it from); Hidden/Depth/OutHiddenSize are carried for
// documentation — LoadVisionTower (vision_loader.go) DERIVES the tower geometry from the checkpoint's
// own tensor shapes and cross-validates the config fields that overlap (SpatialMergeSize, NumHeads as
// the head-dim fallback), the same weight-derived stance the text schedule takes on the gated-delta
// geometry.
type VisionArchConfig struct {
	ModelType        string  `json:"model_type"`
	Depth            int     `json:"depth"`
	HiddenSize       int     `json:"hidden_size"`
	OutHiddenSize    int     `json:"out_hidden_size"`
	PatchSize        int     `json:"patch_size"`
	InChannels       int     `json:"in_channels"`
	TemporalPatch    int     `json:"temporal_patch_size"`
	NumHeads         int     `json:"num_heads"`
	NumKeyValueHeads int     `json:"num_key_value_heads"`
	SpatialMergeSize int     `json:"spatial_merge_size"`
	RMSNormEps       float32 `json:"rms_norm_eps"`
	RopeTheta        float32 `json:"rope_theta"`
}

// Config is the architecture-relevant subset of a Qwen 3.6 config.json. The multimodal wrapper nests the
// text fields under text_config (effective() resolves the nesting); rope lives under rope_parameters.
type Config struct {
	HiddenSize        int `json:"hidden_size"`
	NumHiddenLayers   int `json:"num_hidden_layers"`
	IntermediateSize  int `json:"intermediate_size"`
	NumAttentionHeads int `json:"num_attention_heads"`
	NumKeyValueHeads  int `json:"num_key_value_heads"`
	HeadDim           int `json:"head_dim"`
	VocabSize         int `json:"vocab_size"`

	RMSNormEps          float32 `json:"rms_norm_eps"`
	RopeTheta           float32 `json:"rope_theta"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`

	LayerTypes            []string `json:"layer_types"`
	FullAttentionInterval int      `json:"full_attention_interval"`

	// AttnOutputGate declares the gated attention output (q_proj emits [q ; gate], sigmoid gate before
	// o_proj) — the Qwen 3.6 full-attention layers set it; carried through to Arch.AttnOutputGate.
	AttnOutputGate bool `json:"attn_output_gate"`

	// MoE (qwen3_5_moe). Zero on the dense qwen3_5. NormTopKProb is a pointer so an absent key defaults to
	// true (the reference renormalises the top-k router weights) while an explicit false is honoured.
	NumExperts                   int   `json:"num_experts"`
	NumExpertsPerTok             int   `json:"num_experts_per_tok"`
	MoEIntermediateSize          int   `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize int   `json:"shared_expert_intermediate_size"`
	NormTopKProb                 *bool `json:"norm_topk_prob"`

	RopeParameters    *ropeParameters    `json:"rope_parameters"`
	TieWordEmbeddings *bool              `json:"tie_word_embeddings"`
	Quantization      *model.QuantConfig `json:"quantization"`
	TextConfig        *Config            `json:"text_config"`

	// Multimodal wrapper fields (the vision-towered releases). ImageTokenID is the id one image
	// soft-token occupies in the prompt (`<|image_pad|>`); the vision start/end ids bracket the run.
	// All zero on a text-only config — the tower-presence probe (LoadVisionTower) is the real gate,
	// these are carried facts. TokenID because HF ships the wrapper-root ids polymorphically
	// (scalar or list).
	ImageTokenID       TokenID           `json:"image_token_id"`
	VideoTokenID       TokenID           `json:"video_token_id"`
	VisionStartTokenID TokenID           `json:"vision_start_token_id"`
	VisionEndTokenID   TokenID           `json:"vision_end_token_id"`
	VisionConfig       *VisionArchConfig `json:"vision_config"`

	// MTP drafter fields (qwen3_5_mtp / qwen3_6_mtp — see mtp_drafter.go). Carried directly on THIS
	// type, not a separate one, because a real drafter checkpoint nests mtp_num_hidden_layers INSIDE
	// its own text_config (the base's text_config, reused verbatim) — only a field declared on Config
	// itself survives effective()'s recursive resolution down to TextConfig. MTPNumHiddenLayers is the
	// head's OWN transformer depth (distinct from NumHiddenLayers, which a drafter config still
	// carries as an artefact of reusing the base's text_config shape, but never as the head's actual
	// depth). BlockSize is the checkpoint's declared draft length (top-level only on the real
	// checkpoint, so read via c.BlockSize directly, never c.effective().BlockSize); a future pair
	// loader's default draft block when the caller pins none, falling back to its own constant when
	// absent (0). The base's own Arch()/InferFromWeights never read either field.
	MTPNumHiddenLayers int `json:"mtp_num_hidden_layers"`
	BlockSize          int `json:"block_size"`
}

// effective returns the text config (self, or the nested text_config for the multimodal wrapper).
func (c *Config) effective() *Config {
	if c.TextConfig != nil {
		return c.TextConfig
	}
	return c
}

// ResolvedQuant returns the checkpoint's quantization block (top-level or nested), nil = bf16.
func (c *Config) ResolvedQuant() *model.QuantConfig {
	if c.Quantization != nil {
		return c.Quantization
	}
	if c.TextConfig != nil {
		return c.TextConfig.Quantization
	}
	return nil
}

func (c *Config) ropeTheta() float32 {
	if c.RopeTheta > 0 {
		return c.RopeTheta
	}
	if c.RopeParameters != nil && c.RopeParameters.RopeTheta > 0 {
		return c.RopeParameters.RopeTheta
	}
	return defaultRopeTheta
}

func (c *Config) partialRotary() float32 {
	if c.PartialRotaryFactor > 0 {
		return c.PartialRotaryFactor
	}
	if c.RopeParameters != nil && c.RopeParameters.PartialRotaryFactor > 0 {
		return c.RopeParameters.PartialRotaryFactor
	}
	return 1
}

// normTopKProb resolves the top-k renormalisation flag: absent ⇒ true (the reference default), an explicit
// value honoured — threaded into Arch.NormaliseMoETopK so routing is config-driven, not assumed.
func (c *Config) normTopKProb() bool {
	if c.NormTopKProb == nil {
		return true
	}
	return *c.NormTopKProb
}

// layerTypes returns the per-layer attention schedule: the config's explicit layer_types when present
// (validated against num_hidden_layers), else synthesised from full_attention_interval (every interval-th
// layer full_attention, the rest linear_attention — the Qwen 3.6 3:1 schedule).
func (c *Config) layerTypes() ([]string, error) {
	if len(c.LayerTypes) > 0 {
		if len(c.LayerTypes) != c.NumHiddenLayers {
			return nil, core.NewError(core.Sprintf("qwen35.Config: layer_types length %d != num_hidden_layers %d", len(c.LayerTypes), c.NumHiddenLayers))
		}
		return c.LayerTypes, nil
	}
	if c.FullAttentionInterval <= 0 {
		return nil, core.NewError("qwen35.Config: neither layer_types nor full_attention_interval present")
	}
	types := make([]string, c.NumHiddenLayers)
	for i := range types {
		if (i+1)%c.FullAttentionInterval == 0 {
			types[i] = "full_attention"
		} else {
			types[i] = "linear_attention"
		}
	}
	return types, nil
}

// InferFromWeights resolves vocab (and, defensively, head_dim) from the checkpoint when the config omits
// them. The real Qwen 3.6 configs carry both, so this is a fallback; note the gated q_proj emits [q ; gate],
// so its rows are 2× the plain q size — head_dim inference from q_proj would double it, but a present
// head_dim (always the case here) short-circuits before that path. Satisfies model.ArchConfig.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) {
	if c.TextConfig != nil {
		c.TextConfig.InferFromWeights(weights)
		return
	}
	if c.HeadDim == 0 {
		for i := 0; i < c.NumHiddenLayers; i++ {
			if hd := model.InferHeadDim(weights, core.Sprintf("model.layers.%d.self_attn.q_proj.weight", i), c.NumAttentionHeads); hd > 0 {
				c.HeadDim = hd
				break
			}
		}
	}
	if c.VocabSize == 0 {
		if w, ok := model.WeightAny(weights, "model.embed_tokens.weight", "model.embed_tokens"); ok && len(w.Shape) > 0 && w.Shape[0] > 0 {
			c.VocabSize = int(w.Shape[0])
		}
	}
}

// Arch builds the neutral model.Arch for the Qwen 3.6 hybrid: the 3:1 gated-delta/full-attention schedule
// (DeriveLayers maps linear_attention → MixerGatedDelta), partial rotary (partial_rotary_factor·head_dim),
// plain RMSNorm, SiLU feed-forward, and gated attention. Dense (qwen3_5) leaves the MoE dims zero; the MoE
// variant (qwen3_5_moe) sets Experts/TopK/ExpertFF + one shared expert. Satisfies model.ArchConfig.
func (c *Config) Arch() (model.Arch, error) {
	eff := c.effective()
	if eff.HiddenSize <= 0 || eff.NumHiddenLayers <= 0 || eff.NumAttentionHeads <= 0 {
		return model.Arch{}, core.NewError("qwen35.Config.Arch: hidden_size, num_hidden_layers, num_attention_heads must be > 0")
	}
	headDim := eff.HeadDim
	if headDim == 0 {
		if eff.HiddenSize%eff.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("qwen35.Config.Arch: head_dim absent and hidden_size not divisible by num_attention_heads")
		}
		headDim = eff.HiddenSize / eff.NumAttentionHeads
	}
	kvHeads := eff.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = eff.NumAttentionHeads
	}
	if kvHeads <= 0 || eff.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("qwen35.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	eps := eff.RMSNormEps
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	ropeBase := eff.ropeTheta()
	rotaryDim := int(float32(headDim) * eff.partialRotary())
	if rotaryDim <= 0 {
		rotaryDim = headDim
	}

	types, err := eff.layerTypes()
	if err != nil {
		return model.Arch{}, err
	}

	moe := eff.NumExperts > 0
	if moe && eff.NumExpertsPerTok > eff.NumExperts {
		return model.Arch{}, core.NewError("qwen35.Config.Arch: num_experts_per_tok exceeds num_experts")
	}

	// DeriveLayers assigns MixerGatedDelta (no KV cache) to linear_attention layers and the attention
	// mixer + a KV-cache slot to full_attention layers; the attention geometry is filled per attention
	// layer, and the FFN is MoE on every layer for the sparse variant (dense leaves MoE false).
	layers := model.DeriveLayers(types, 0)
	for i := range layers {
		if layers[i].Mixer == model.MixerAttention {
			layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
		}
		layers[i].MoE = moe
	}

	a := model.Arch{
		Hidden: eff.HiddenSize, Heads: eff.NumAttentionHeads, KVHeads: kvHeads,
		HeadDim: headDim, GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		FF: eff.IntermediateSize, Vocab: eff.VocabSize, Eps: eps,
		AttnScale: float32(1 / math.Sqrt(float64(headDim))), EmbedScale: 1,
		RopeBase: ropeBase, RopeLocalBase: ropeBase, RopeScale: 1,
		RotaryDim: rotaryDim, RotaryDimLocal: rotaryDim,
		Activation:        "silu",
		AttnOutputGate:    eff.AttnOutputGate,
		TieWordEmbeddings: eff.TieWordEmbeddings,
		Layer:             layers,
	}
	if moe {
		a.Experts = eff.NumExperts
		a.TopK = eff.NumExpertsPerTok
		a.ExpertFF = eff.MoEIntermediateSize
		a.MoEGating = model.MoEGatingSoftmax
		a.NormaliseMoETopK = eff.normTopKProb()
		if eff.SharedExpertIntermediateSize > 0 {
			a.SharedExperts = 1
			// SharedExpertFF (#61): the shared expert's OWN intermediate size — real Qwen 3.6 MoE
			// fixtures ship it equal to ExpertFF today, but the field is a direct passthrough (the
			// same idiom qwenmoe.Config.Arch uses, #57) so a same-family checkpoint that DOES diverge
			// sizes correctly. Feeds engine/metal/arch_qwen_moe.go's encQwenMoEHalf (#61) via
			// model.assembleMoE's SharedDown InDim and engine/metal's MoEQuantLayerWeights.SharedDFF.
			a.SharedExpertFF = eff.SharedExpertIntermediateSize
		}
	}
	return a, nil
}
