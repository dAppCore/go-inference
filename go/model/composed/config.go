// SPDX-Licence-Identifier: EUPL-1.2

package composed

import core "dappco.re/go"

// config.go is the arch-relevant config for the Qwen 3.6 hybrid (model_type qwen3_5 / qwen3_5_moe,
// nested text_config model_types qwen3_5_text / qwen3_5_moe_text). It parses BOTH the flat text config
// and the multimodal wrapper (nested text_config + a sibling vision_config), carrying every field the
// composed loader consumes or validates. Vision is parsed-and-carried only — the tower assembly is out of
// scope (the composed model is text-only today). The linear-attention geometry is DERIVED from the weight
// shapes by the loader; when the config declares the linear_* fields they are validated against that
// derivation (a mismatched config/checkpoint pairing fails loudly rather than mis-loading).

// ropeParams is the rope_parameters block (Qwen 3.6 nests rope config here, not under rope_scaling). For
// pure-text decode the mRoPE collapses to standard partial rotary (all three position dims share the text
// position), so mrope_interleaved / mrope_section are carried for completeness + the rotary reduction
// property, not consumed by the text attention forward — see rotary.go.
type ropeParams struct {
	RopeTheta           float32 `json:"rope_theta"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	MRopeInterleaved    bool    `json:"mrope_interleaved"`
	MRopeSection        []int   `json:"mrope_section"`
}

// visionConfig is the multimodal wrapper's sibling vision tower config — carried (parse-and-carry) so a
// wrapper checkpoint parses cleanly; the encoder assembly itself is out of scope (a later cut).
type visionConfig struct {
	ModelType     string `json:"model_type"`
	Depth         int    `json:"depth"`
	HiddenSize    int    `json:"hidden_size"`
	OutHiddenSize int    `json:"out_hidden_size"`
	PatchSize     int    `json:"patch_size"`
}

// loaderConfig is the arch-relevant subset of a Qwen 3.6 config.json. Text fields nest under text_config in
// the multimodal wrapper (effective() resolves the nesting); rope lives under rope_parameters.
type loaderConfig struct {
	ModelType             string   `json:"model_type"`
	HiddenSize            int      `json:"hidden_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	IntermediateSize      int      `json:"intermediate_size"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	HeadDim               int      `json:"head_dim"`
	VocabSize             int      `json:"vocab_size"`
	RMSNormEps            float32  `json:"rms_norm_eps"`
	LayerNormEps          float32  `json:"layer_norm_eps"`
	LogitScale            float32  `json:"logit_scale"`
	UseQKNorm             *bool    `json:"use_qk_norm"`
	UseLayerNorm          bool     `json:"use_layer_norm"`
	QKVClip               float32  `json:"qkv_clip"`
	SlidingWindow         int      `json:"sliding_window"`
	SlidingWindowPattern  int      `json:"sliding_window_pattern"`
	RopeTheta             float32  `json:"rope_theta"`
	PartialRotaryFactor   float32  `json:"partial_rotary_factor"`
	LayerTypes            []string `json:"layer_types"`
	FullAttentionInterval int      `json:"full_attention_interval"`

	// Gated attention: q_proj emits [q ; gate] and the attention output is gated by the gate before
	// o_proj (attn_output_gate). OutputGateType is carried for completeness — the transformers qwen3_5
	// reference hardcodes sigmoid for the attention output gate and does not consume output_gate_type
	// (see attention.go).
	AttnOutputGate bool   `json:"attn_output_gate"`
	OutputGateType string `json:"output_gate_type"`

	// Linear-attention (gated-delta) geometry. The loader DERIVES these from the weight shapes; when the
	// config declares them they are validated against the derivation (validateLinearGeometry).
	LinearNumKeyHeads   int `json:"linear_num_key_heads"`
	LinearNumValueHeads int `json:"linear_num_value_heads"`
	LinearKeyHeadDim    int `json:"linear_key_head_dim"`
	LinearValueHeadDim  int `json:"linear_value_head_dim"`
	LinearConvKernelDim int `json:"linear_conv_kernel_dim"`

	// MoE (qwen3_5_moe). NormTopKProb is a pointer so an absent key defaults to true (the reference
	// renormalises the top-k router weights) while an explicit false is honoured — normTopKProb().
	NumExperts                   int   `json:"num_experts"`
	NumExpertsPerTok             int   `json:"num_experts_per_tok"`
	MoEIntermediateSize          int   `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize int   `json:"shared_expert_intermediate_size"`
	NormTopKProb                 *bool `json:"norm_topk_prob"`

	// The in-checkpoint multi-token-prediction head depth. Carried; the speculative wiring is out of scope.
	MTPNumHiddenLayers int `json:"mtp_num_hidden_layers"`

	// Token ids (Qwen 3.6: bos == eos == 248044).
	BosTokenID int `json:"bos_token_id"`
	EosTokenID int `json:"eos_token_id"`

	RopeParameters *ropeParams   `json:"rope_parameters"`
	VisionConfig   *visionConfig `json:"vision_config"`
	TextConfig     *loaderConfig `json:"text_config"`
}

// effective returns the text config (self, or the nested text_config for the multimodal wrapper).
func (c *loaderConfig) effective() *loaderConfig {
	if c.TextConfig != nil {
		return c.TextConfig
	}
	return c
}

func (c *loaderConfig) ropeTheta() float32 {
	if c.RopeTheta > 0 {
		return c.RopeTheta
	}
	if c.RopeParameters != nil && c.RopeParameters.RopeTheta > 0 {
		return c.RopeParameters.RopeTheta
	}
	return 1e6
}

func (c *loaderConfig) partialRotary() float32 {
	if c.PartialRotaryFactor > 0 {
		return c.PartialRotaryFactor
	}
	if c.RopeParameters != nil && c.RopeParameters.PartialRotaryFactor > 0 {
		return c.RopeParameters.PartialRotaryFactor
	}
	return 1
}

// normTopKProb resolves the top-k renormalisation flag: absent ⇒ true (the reference default), an explicit
// value honoured. It is threaded into MoEMLP so the routing behaviour is config-driven, not assumed.
func (c *loaderConfig) normTopKProb() bool {
	if c.NormTopKProb == nil {
		return true
	}
	return *c.NormTopKProb
}

// validateLinearGeometry checks the gated-delta geometry the loader DERIVED from the weight shapes against
// the config's declared linear_* fields, when present. The loader's delta state is square
// (KeyHeadDim == ValueHeadDim == HeadDim), so both declared head dims are validated against the single
// derived headDim. A mismatch means the config and the checkpoint disagree — fail rather than mis-load.
func (c *loaderConfig) validateLinearGeometry(keyHeads, valueHeads, headDim, convKernel int) error {
	mism := func(name string, got, want int) error {
		return core.NewError(core.Sprintf("composed: linear_attention %s derived %d != config %d (config/checkpoint mismatch)", name, got, want))
	}
	if c.LinearNumKeyHeads > 0 && c.LinearNumKeyHeads != keyHeads {
		return mism("num_key_heads", keyHeads, c.LinearNumKeyHeads)
	}
	if c.LinearNumValueHeads > 0 && c.LinearNumValueHeads != valueHeads {
		return mism("num_value_heads", valueHeads, c.LinearNumValueHeads)
	}
	if c.LinearKeyHeadDim > 0 && c.LinearKeyHeadDim != headDim {
		return mism("key_head_dim", headDim, c.LinearKeyHeadDim)
	}
	if c.LinearValueHeadDim > 0 && c.LinearValueHeadDim != headDim {
		return mism("value_head_dim", headDim, c.LinearValueHeadDim)
	}
	if c.LinearConvKernelDim > 0 && c.LinearConvKernelDim != convKernel {
		return mism("conv_kernel_dim", convKernel, c.LinearConvKernelDim)
	}
	return nil
}
