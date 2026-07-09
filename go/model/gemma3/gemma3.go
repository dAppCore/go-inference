// SPDX-Licence-Identifier: EUPL-1.2

// Package gemma3 declares the gemma3 architecture to the engine's reactive loader (model.RegisterArch),
// so model.Load parses + assembles a gemma3 checkpoint into the neutral decode path with no central
// switch. gemma3 is a gemma-family transformer that REUSES the shared decode: 4 RMSNorms per block
// (input / post-attention / pre-feedforward / post-feedforward), per-head QK-norm, GQA, and a
// sliding/global attention pattern (every Nth layer is global). It differs from gemma4 in the details
// verified against the working metal gemma3: the SDPA scale is 1/sqrt(head_dim) (gemma4 uses 1.0 because
// its QK-norm carries the scaling), there is NO value-norm, NO PLE / MoE / MatFormer, full (not partial)
// rotary, and NO logit softcapping. The gemma "(1 + weight)" RMSNorm convention is folded at load via
// the ArchSpec's NormBiasOne (see model/norm_bias.go), exactly as metal precomputes NormScaled.
package gemma3

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

const (
	defaultRopeTheta            float32 = 1_000_000 // gemma3 global (full_attention) RoPE base
	defaultRopeLocalTheta       float32 = 10_000    // gemma3 sliding_attention RoPE base
	defaultRMSNormEps           float32 = 1e-6
	defaultSlidingWindowPattern int     = 6 // every 6th layer is global, the rest sliding (metal isLayerSliding)
)

// Config is the arch-relevant subset of a gemma3 config.json (json tags match so it unmarshals directly).
type Config struct {
	HiddenSize           int     `json:"hidden_size"`
	NumHiddenLayers      int     `json:"num_hidden_layers"`
	IntermediateSize     int     `json:"intermediate_size"`
	NumAttentionHeads    int     `json:"num_attention_heads"`
	NumKeyValueHeads     int     `json:"num_key_value_heads"`
	HeadDim              int     `json:"head_dim"`
	VocabSize            int     `json:"vocab_size"`
	RMSNormEps           float32 `json:"rms_norm_eps"`
	RopeTheta            float32 `json:"rope_theta"`
	RopeLocalBaseFreq    float32 `json:"rope_local_base_freq"`
	SlidingWindow        int     `json:"sliding_window"`
	SlidingWindowPattern int     `json:"sliding_window_pattern"`

	// TextConfig holds the text arch when the checkpoint is the gemma3 multimodal wrapper (the text
	// fields nest under "text_config"); nil for a flat text-only config.
	TextConfig   *Config            `json:"text_config"`
	Quantization *model.QuantConfig `json:"quantization"`
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

// InferFromWeights resolves the dims gemma3 reads from the weight SHAPES (the don't-guess rule):
// head_dim from a q_proj's rows (gemma3 uses head_dim=256, which differs from hidden/heads), and vocab
// from the embedding rows. Satisfies model.ArchConfig.
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
		if c.HeadDim == 0 && c.HiddenSize > 0 && c.NumAttentionHeads > 0 {
			c.HeadDim = c.HiddenSize / c.NumAttentionHeads
		}
	}
	if c.VocabSize == 0 {
		if w, ok := model.WeightAny(weights, "model.embed_tokens.weight", "model.embed_tokens"); ok && len(w.Shape) > 0 && w.Shape[0] > 0 {
			c.VocabSize = int(w.Shape[0])
		}
	}
}

// Arch builds the neutral model.Arch from the gemma3 config: the transformer dims, the per-layer
// sliding/global pattern, and gemma3's specifics (scale 1/sqrt(head_dim), full rotary, no value-norm,
// no softcap). Satisfies model.ArchConfig.
func (c *Config) Arch() (model.Arch, error) {
	if c.TextConfig != nil {
		return c.TextConfig.Arch()
	}
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 {
		return model.Arch{}, core.NewError("gemma3.Config.Arch: hidden_size, num_hidden_layers, num_attention_heads must be > 0")
	}
	headDim := c.HeadDim
	if headDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return model.Arch{}, core.NewError("gemma3.Config.Arch: head_dim absent and hidden_size not divisible by num_attention_heads")
		}
		headDim = c.HiddenSize / c.NumAttentionHeads
	}
	kvHeads := c.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%kvHeads != 0 {
		return model.Arch{}, core.NewError("gemma3.Config.Arch: num_attention_heads must be a multiple of num_key_value_heads")
	}
	pattern := c.SlidingWindowPattern
	if pattern <= 0 {
		pattern = defaultSlidingWindowPattern
	}
	layerTypes := make([]string, c.NumHiddenLayers)
	for i := range layerTypes {
		if (i+1)%pattern == 0 { // metal gemma3 isLayerSliding: global when (i+1)%pattern == 0
			layerTypes[i] = "full_attention"
		} else {
			layerTypes[i] = "sliding_attention"
		}
	}
	eps := c.RMSNormEps
	if eps == 0 {
		eps = defaultRMSNormEps
	}
	ropeBase := c.RopeTheta
	if ropeBase == 0 {
		ropeBase = defaultRopeTheta
	}
	ropeLocalBase := c.RopeLocalBaseFreq
	if ropeLocalBase == 0 {
		ropeLocalBase = defaultRopeLocalTheta
	}

	layers := model.DeriveLayers(layerTypes, 0)
	for i := range layers { // gemma3 is uniform-geometry: every layer the same head_dim / kv-heads
		layers[i].HeadDim, layers[i].KVHeads = headDim, kvHeads
	}

	return model.Arch{
		Hidden:         c.HiddenSize,
		Heads:          c.NumAttentionHeads,
		KVHeads:        kvHeads,
		HeadDim:        headDim,
		GlobalHeadDim:  headDim,
		GlobalKVHeads:  kvHeads,
		FF:             c.IntermediateSize,
		Vocab:          c.VocabSize,
		Eps:            eps,
		AttnScale:      float32(1.0 / math.Sqrt(float64(headDim))), // gemma3: 1/sqrt(head_dim) (gemma4 uses 1.0; verified vs metal gemma3 Scale)
		RopeBase:       ropeBase,
		RopeLocalBase:  ropeLocalBase,
		RotaryDim:      headDim, // gemma3 full rotary
		RotaryDimLocal: headDim,
		RopeScale:      1,
		SoftCap:        0, // gemma3 has no logit softcapping
		SlidingWindow:  c.SlidingWindow,
		ValueNorm:      false, // gemma3 does not value-norm V (gemma4 does)
		Layer:          layers,
	}, nil
}
