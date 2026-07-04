// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "dappco.re/go/inference/model"

// vision_config.go — copied verbatim from pkg/metal/model/gemma4/vision.go (the vision_config structs
// + normalizer the literal config port parses). The neutral TransformerConfig core is pkg/model's
// (model.TransformerConfig); the vision DECODE towers stay in metal (a later, separate feature port).

type Gemma4VisionRopeParameters struct {
	RopeType  string  `json:"rope_type"`
	RopeTheta float32 `json:"rope_theta"`
}

// Gemma4VisionConfig holds the Gemma 4 SigLIP-derived vision tower configuration.
type Gemma4VisionConfig struct {
	// Embedded neutral core — promotes ModelType/HiddenSize/IntermediateSize/
	// NumHiddenLayers/NumAttentionHeads/NumKeyValueHeads/HeadDim/RMSNormEps/
	// MaxPositionEmbeddings (the vision tower is a transformer; VocabSize is
	// carried by the core but unused here).
	model.TransformerConfig

	ImageSize             int32                      `json:"image_size"`
	PatchSize             int32                      `json:"patch_size"`
	NumChannels           int32                      `json:"num_channels"`
	HiddenActivation      string                     `json:"hidden_activation"`
	LayerNormEps          float32                    `json:"layer_norm_eps"`
	MMEmbedDim            int32                      `json:"mm_embed_dim"`
	MMPosembSize          int32                      `json:"mm_posemb_size"`
	ModelPatchSize        int32                      `json:"model_patch_size"`
	NumSoftTokens         int32                      `json:"num_soft_tokens"`
	OutputProjDims        int32                      `json:"output_proj_dims"`
	AttentionBias         bool                       `json:"attention_bias"`
	AttentionDropout      float32                    `json:"attention_dropout"`
	RopeParameters        Gemma4VisionRopeParameters `json:"rope_parameters"`
	PoolingKernelSize     int32                      `json:"pooling_kernel_size"`
	PositionEmbeddingSize int32                      `json:"position_embedding_size"`
	UseClippedLinears     bool                       `json:"use_clipped_linears"`
	Standardize           bool                       `json:"standardize"`
	InitializerRange      float32                    `json:"initializer_range"`
}

func normalizeGemma4VisionConfig(cfg *Gemma4VisionConfig) *Gemma4VisionConfig {
	if cfg == nil {
		return nil
	}
	if cfg.ModelType == "" {
		cfg.ModelType = "gemma4_vision"
	}
	if cfg.NumChannels == 0 {
		cfg.NumChannels = 3 // RGB — physical, not a tuned guess
	}
	if cfg.HiddenActivation == "" {
		cfg.HiddenActivation = "gelu_pytorch_tanh"
	}
	// RMS/Layer-norm epsilon: cross-fill the two names, then the Gemma constant.
	if cfg.LayerNormEps == 0 && cfg.RMSNormEps != 0 {
		cfg.LayerNormEps = cfg.RMSNormEps
	}
	if cfg.RMSNormEps == 0 && cfg.LayerNormEps != 0 {
		cfg.RMSNormEps = cfg.LayerNormEps
	}
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-6
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.RopeParameters.RopeType == "" {
		cfg.RopeParameters.RopeType = "default"
	}
	if cfg.RopeParameters.RopeTheta == 0 {
		cfg.RopeParameters.RopeTheta = 100
	}
	if cfg.PoolingKernelSize == 0 {
		cfg.PoolingKernelSize = 3
	}
	// Derivations from the model's own declared dims — not cross-model guesses.
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HeadDim == 0 && cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	return cfg
}
