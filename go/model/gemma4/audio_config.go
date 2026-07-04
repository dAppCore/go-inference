// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

// audio_config.go — copied verbatim from pkg/metal/model/gemma4/audio.go (the audio_config struct
// + normalizer the literal config port parses). The audio DECODE encoder stays in metal.

type Gemma4AudioConfig struct {
	ModelType               string  `json:"model_type"`
	HiddenSize              int32   `json:"hidden_size"`
	NumHiddenLayers         int32   `json:"num_hidden_layers"`
	NumAttentionHeads       int32   `json:"num_attention_heads"`
	AttentionChunkSize      int32   `json:"attention_chunk_size"`
	AttentionContextLeft    int32   `json:"attention_context_left"`
	AttentionContextRight   int32   `json:"attention_context_right"`
	AttentionLogitCap       float32 `json:"attention_logit_cap"`
	ConvKernelSize          int32   `json:"conv_kernel_size"`
	SubsamplingConvChannels []int32 `json:"subsampling_conv_channels"`
	ResidualWeight          float32 `json:"residual_weight"`
	HiddenAct               string  `json:"hidden_act"`
	UseClippedLinears       bool    `json:"use_clipped_linears"`
	OutputProjDims          int32   `json:"output_proj_dims"`
	RMSNormEps              float32 `json:"rms_norm_eps"`
	// GradientClipping clamps activations between Conformer sub-blocks
	// (training-stability carry-over the reference applies at inference too).
	GradientClipping float32 `json:"gradient_clipping"`
	// AttentionInvalidLogitsValue replaces masked attention logits.
	AttentionInvalidLogitsValue float32 `json:"attention_invalid_logits_value"`
}

func normalizeGemma4AudioConfig(cfg *Gemma4AudioConfig) *Gemma4AudioConfig {
	if cfg == nil {
		return nil
	}
	if cfg.ModelType == "" {
		cfg.ModelType = "gemma4_unified_audio"
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	// Non-dimensional knobs absent from a checkpoint config take the HF
	// Gemma4AudioConfig defaults (configuration_gemma4.py) — published spec,
	// not invention. Dimensions stay zero and fail loud at encoder build.
	if cfg.GradientClipping == 0 {
		cfg.GradientClipping = 1e10
	}
	if cfg.AttentionInvalidLogitsValue == 0 {
		cfg.AttentionInvalidLogitsValue = -1.0e9
	}
	if cfg.HiddenAct == "" {
		cfg.HiddenAct = "silu"
	}
	if cfg.ResidualWeight == 0 {
		cfg.ResidualWeight = 0.5
	}
	return cfg
}
