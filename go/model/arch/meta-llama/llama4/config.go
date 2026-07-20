// SPDX-Licence-Identifier: EUPL-1.2

// Package llama4 declares Meta's Llama 4 sparse text transformer architecture.
// The multimodal vision tower is deliberately outside this package's text lane.
package llama4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// RopeScaling is Llama 4's Llama 3 long-context RoPE declaration.
type RopeScaling struct {
	Factor                        float32 `json:"factor"`
	OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
	RopeType                      string  `json:"rope_type"`
}

// Config is the architecture-relevant subset of a Llama4TextConfig.
type Config struct {
	HiddenSize             int         `json:"hidden_size"`
	IntermediateSize       int         `json:"intermediate_size"`
	IntermediateSizeMLP    int         `json:"intermediate_size_mlp"`
	NumHiddenLayers        int         `json:"num_hidden_layers"`
	NumAttentionHeads      int         `json:"num_attention_heads"`
	NumKeyValueHeads       int         `json:"num_key_value_heads"`
	HeadDim                int         `json:"head_dim"`
	NumLocalExperts        int         `json:"num_local_experts"`
	NumExpertsPerTok       int         `json:"num_experts_per_tok"`
	InterleaveMoELayerStep int         `json:"interleave_moe_layer_step"`
	MoELayers              []int       `json:"moe_layers"`
	NoRopeLayers           []int       `json:"no_rope_layers"`
	NoRopeLayerInterval    int         `json:"no_rope_layer_interval"`
	VocabSize              int         `json:"vocab_size"`
	RMSNormEps             float32     `json:"rms_norm_eps"`
	RopeTheta              float32     `json:"rope_theta"`
	RopeScaling            RopeScaling `json:"rope_scaling"`
	UseQKNorm              bool        `json:"use_qk_norm"`
	TieWordEmbeddings      *bool       `json:"tie_word_embeddings"`
}

// InferFromWeights satisfies model.ArchConfig. Llama 4 declares its geometry.
func (c *Config) InferFromWeights(weights map[string]safetensors.Tensor) { weights = nil }

// Arch resolves the sparse/dense interleave, top-k sigmoid routing, and per-layer NoPE pattern.
func (c Config) Arch() (model.Arch, error) {
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.IntermediateSizeMLP <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 || c.NumLocalExperts <= 0 || c.NumExpertsPerTok <= 0 {
		return model.Arch{}, core.NewError("llama4.Config.Arch: all architecture dimensions must be > 0")
	}
	headDim := c.HeadDim
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("llama4.Config.Arch: hidden_size must be divisible by num_attention_heads when head_dim is absent")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("llama4.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	if c.NumExpertsPerTok > c.NumLocalExperts {
		return model.Arch{}, core.NewError("llama4.Config.Arch: num_experts_per_tok exceeds num_local_experts")
	}
	noRope := c.NoRopeLayers
	if len(noRope) == 0 {
		interval := c.NoRopeLayerInterval
		if interval <= 0 {
			interval = 4
		}
		noRope = make([]int, c.NumHiddenLayers)
		for i := range noRope {
			if (i+1)%interval != 0 {
				noRope[i] = 1
			}
		}
	}
	if len(noRope) != c.NumHiddenLayers {
		return model.Arch{}, core.NewError("llama4.Config.Arch: no_rope_layers length must equal num_hidden_layers")
	}
	moe := make(map[int]bool, len(c.MoELayers))
	if len(c.MoELayers) > 0 {
		for _, layer := range c.MoELayers {
			if layer < 0 || layer >= c.NumHiddenLayers {
				return model.Arch{}, core.NewError("llama4.Config.Arch: moe_layers index out of range")
			}
			moe[layer] = true
		}
	} else {
		step := c.InterleaveMoELayerStep
		if step <= 0 {
			step = 1
		}
		for i := step - 1; i < c.NumHiddenLayers; i += step {
			moe[i] = true
		}
	}
	layerTypes := make([]string, c.NumHiddenLayers)
	for i := range layerTypes {
		layerTypes[i] = "full_attention"
	}
	layers := model.DeriveLayers(layerTypes, 0)
	for i := range layers {
		layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
		layers[i].MoE, layers[i].DisableRotary = moe[i], noRope[i] == 0
	}
	eps := c.RMSNormEps
	if eps == 0 {
		eps = 1e-5
	}
	rope := c.RopeTheta
	if rope == 0 {
		rope = 500_000
	}
	ropeScale := c.RopeScaling.Factor
	if ropeScale == 0 {
		ropeScale = 1
	}
	qk := model.QKNone
	if c.UseQKNorm {
		qk = model.QKL2Norm
	}
	return model.Arch{
		Hidden: c.HiddenSize, Heads: c.NumAttentionHeads, KVHeads: kvHeads, HeadDim: headDim,
		GlobalHeadDim: headDim, GlobalKVHeads: kvHeads, FF: c.IntermediateSizeMLP, Vocab: c.VocabSize,
		Experts: c.NumLocalExperts, TopK: c.NumExpertsPerTok, ExpertFF: c.IntermediateSize,
		// SharedExpertFF (#61): Llama 4's always-on shared expert is structurally the SAME dense
		// SwiGLU block as a non-MoE layer's MLP, so it shares intermediate_size_mlp — genuinely
		// DISTINCT from ExpertFF (intermediate_size) above: real Scout ships 16384 (shared,
		// IntermediateSizeMLP, == FF above) vs 8192 (routed, IntermediateSize/ExpertFF), a live 2x
		// mismatch reaching engine/metal/arch_qwen_moe.go's encQwenMoEHalf shared-expert dispatch.
		SharedExpertFF: c.IntermediateSizeMLP,
		MoEGating:      model.MoEGatingSigmoid, NormaliseMoETopK: false, SharedExperts: 1,
		Eps: eps, AttnScale: float32(1 / core.Pow(float64(headDim), 0.5)), EmbedScale: 1,
		RopeBase: rope, RopeLocalBase: rope, RopeScale: ropeScale, RotaryDim: headDim, RotaryDimLocal: headDim,
		RopeOriginalContext: c.RopeScaling.OriginalMaxPositionEmbeddings,
		TieWordEmbeddings:   c.TieWordEmbeddings, QKNormalization: qk, Layer: layers,
	}, nil
}
