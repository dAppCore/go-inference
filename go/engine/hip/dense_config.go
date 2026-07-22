// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"encoding/json"
	"math"

	core "dappco.re/go"
)

// DenseConfig is the shared dense-transformer config used by Qwen, Llama,
// Mistral, Hermes, Granite, Phi, GLM and related MoE families.
type DenseConfig struct {
	ModelType             string                   `json:"model_type"`
	Architectures         []string                 `json:"architectures"`
	VocabSize             int                      `json:"vocab_size"`
	HiddenSize            int                      `json:"hidden_size"`
	NumHiddenLayers       int                      `json:"num_hidden_layers"`
	IntermediateSize      int                      `json:"intermediate_size"`
	MoEIntermediateSize   int                      `json:"moe_intermediate_size"`
	NumAttentionHeads     int                      `json:"num_attention_heads"`
	NumKeyValueHeads      int                      `json:"num_key_value_heads"`
	NumExperts            int                      `json:"num_experts"`
	NumExpertsPerTok      int                      `json:"num_experts_per_tok"`
	TopKExperts           int                      `json:"top_k_experts"`
	DecoderSparseStep     int                      `json:"decoder_sparse_step"`
	HeadDim               int                      `json:"head_dim"`
	Scale                 float32                  `json:"-"`
	RMSNormEps            float64                  `json:"rms_norm_eps"`
	RopeTheta             float64                  `json:"rope_theta"`
	// NormTopKProb is qwen3_moe's router combine-weight order (norm_topk_prob):
	// true softmaxes over all experts then renormalises the gathered top-k to sum
	// to one; false gathers the top-k without renormalising. See
	// hipQwen3MoERouterSelect (qwen3_moe's only reader today) and
	// engine/metal/router.go's identical #65 distinction. A plain bool, not a
	// pointer, mirroring model/arch/Qwen/qwenmoe.Config's own NormTopKProb field —
	// the checkpoint's config.json is trusted to state this explicitly.
	NormTopKProb bool `json:"norm_topk_prob"`
	PartialRotaryFactor   float64                  `json:"partial_rotary_factor"`
	MaxPositionEmbeddings int                      `json:"max_position_embeddings"`
	LayerTypes            []string                 `json:"layer_types"`
	Quantization          *DenseQuantizationConfig `json:"quantization_config,omitempty"`
}

// DenseQuantizationConfig captures the common quantization config identifiers
// needed by loader-neutral dense-family routing.
type DenseQuantizationConfig struct {
	QuantMethod  string `json:"quant_method,omitempty"`
	Algorithm    string `json:"algorithm,omitempty"`
	Format       string `json:"format,omitempty"`
	WeightFormat string `json:"weight_format,omitempty"`
	Scheme       string `json:"scheme,omitempty"`
	Type         string `json:"type,omitempty"`
	Bits         int    `json:"bits,omitempty"`
	GroupSize    int    `json:"group_size,omitempty"`
	Iters        int    `json:"iters,omitempty"`
	NSamples     int    `json:"nsamples,omitempty"`
	SeqLen       int    `json:"seqlen,omitempty"`
	Sym          *bool  `json:"sym,omitempty"`
	Asym         *bool  `json:"asym,omitempty"`
}

// ParseDenseConfig reads the shared dense-transformer config surface used by
// the Qwen-derived dense and sparse families.
func ParseDenseConfig(data []byte) (*DenseConfig, error) {
	var cfg DenseConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, core.E("rocm.dense.ParseConfig", "parse config", err)
	}
	var wrapper struct {
		TextConfig         *DenseConfig             `json:"text_config"`
		Quantization       *DenseQuantizationConfig `json:"quantization"`
		QuantizationConfig *DenseQuantizationConfig `json:"quantization_config"`
	}
	if err := json.Unmarshal(data, &wrapper); err != nil {
		return nil, core.E("rocm.dense.ParseConfig", "parse nested config", err)
	}
	if wrapper.TextConfig != nil {
		cfg = mergeDenseTextConfig(cfg, *wrapper.TextConfig)
	}
	if cfg.ModelType == "" {
		cfg.ModelType = firstDenseArchitecture(cfg.Architectures)
	}
	cfg.ModelType = normalizeROCmArchitecture(cfg.ModelType)
	cfg.Quantization = FirstDenseQuantization(wrapper.Quantization, wrapper.QuantizationConfig, cfg.Quantization)
	if cfg.NumExpertsPerTok == 0 {
		cfg.NumExpertsPerTok = cfg.TopKExperts
	}
	if cfg.HeadDim == 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.HeadDim > 0 {
		cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 151936
	}
	return &cfg, nil
}

// FirstDenseQuantization returns the first non-nil DenseQuantizationConfig.
func FirstDenseQuantization(configs ...*DenseQuantizationConfig) *DenseQuantizationConfig {
	for _, cfg := range configs {
		if cfg != nil {
			return cfg
		}
	}
	return nil
}

func (cfg *DenseQuantizationConfig) Method() string {
	if cfg == nil {
		return ""
	}
	return normalizeROCmQuantizationAlias(firstNonEmptyString(cfg.Algorithm, cfg.QuantMethod, cfg.WeightFormat, cfg.Format, cfg.Type))
}

func (cfg *DenseQuantizationConfig) IsAutoRound() bool {
	if cfg == nil {
		return false
	}
	return rocmQuantizationAliasIsAutoRound(cfg.Algorithm, cfg.QuantMethod, cfg.WeightFormat, cfg.Format, cfg.Type)
}

func (cfg *DenseQuantizationConfig) AutoRoundProfile() (ProductionAutoRoundQuantizationProfile, bool) {
	if cfg == nil || !cfg.IsAutoRound() {
		return ProductionAutoRoundQuantizationProfile{}, false
	}
	return productionAutoRoundQuantizationProfileForFields(cfg.Scheme, firstNonEmptyString(cfg.WeightFormat, cfg.Format), cfg.GroupSize)
}

func (cfg *DenseQuantizationConfig) AutoRoundCalibrationPlan() (ProductionAutoRoundCalibrationPlan, bool) {
	profile, ok := cfg.AutoRoundProfile()
	if !ok {
		return ProductionAutoRoundCalibrationPlan{}, false
	}
	return productionAutoRoundCalibrationPlan(profile, cfg.NSamples, cfg.SeqLen, cfg.Iters), true
}

// IsMoE reports whether cfg describes a sparse expert model.
func (cfg *DenseConfig) IsMoE() bool {
	return cfg != nil && (cfg.ModelType == "qwen3_moe" || cfg.ModelType == "qwen3_6_moe" || cfg.NumExperts > 0 || cfg.NumExpertsPerTok > 0 || cfg.MoEIntermediateSize > 0)
}

// IsQwen36Hybrid reports whether cfg uses Qwen3.6 hybrid linear/full attention.
func (cfg *DenseConfig) IsQwen36Hybrid() bool {
	if cfg == nil {
		return false
	}
	switch normalizeROCmArchitecture(cfg.ModelType) {
	case "qwen3_6", "qwen3_6_moe":
		return true
	}
	for _, layerType := range cfg.LayerTypes {
		if NormalizeDenseLayerType(layerType) == HybridAttentionLinear {
			return true
		}
	}
	return cfg.PartialRotaryFactor > 0 && cfg.PartialRotaryFactor < 1
}

// NormalizeDenseLayerType canonicalises layer type identifiers from dense
// family configs.
func NormalizeDenseLayerType(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	value = core.Replace(value, ".", "_")
	return core.Replace(value, " ", "_")
}

// Qwen36NativeGuardMessage keeps staged Qwen3.6 diagnostics consistent across
// dense and MoE loaders.
func Qwen36NativeGuardMessage(modelType string) string {
	if normalizeROCmArchitecture(modelType) == "qwen3_6_moe" {
		return "qwen3_6_moe hybrid linear attention and sparse expert routing are not implemented in the native ROCm loader yet"
	}
	return "qwen3_6 hybrid linear attention is not implemented in the native ROCm loader yet"
}

// DenseWeightNameCandidates returns the standard model/language_model aliases
// for a checkpoint tensor name.
func DenseWeightNameCandidates(name string) []string {
	candidates := []string{name}
	if core.HasPrefix(name, "model.") {
		suffix := core.TrimPrefix(name, "model.")
		return append(candidates,
			"language_model."+name,
			"language_model.model."+suffix,
			"model.language_model."+suffix,
			"model.language_model.model."+suffix,
		)
	}
	return append(candidates,
		"model."+name,
		"language_model."+name,
		"language_model.model."+name,
		"model.language_model."+name,
		"model.language_model.model."+name,
	)
}

// HasResolvedDenseWeightName reports whether a tensor exists under the standard
// model and language_model aliases.
func HasResolvedDenseWeightName(names map[string]bool, name string) bool {
	for _, candidate := range DenseWeightNameCandidates(name) {
		if names[candidate] {
			return true
		}
	}
	return false
}

// DetectDenseModelType selects the concrete dense-family architecture from
// config metadata or Qwen3 Q/K norm tensor names.
func DetectDenseModelType(configData []byte, names map[string]bool) string {
	if cfg, err := ParseDenseConfig(configData); err == nil {
		switch cfg.ModelType {
		case "llama", "mistral", "hermes", "granite", "phi", "glm", "glm4", "qwen2", "qwen3", "qwen3_next", "qwen3_6", "qwen3_6_moe", "qwen3_moe":
			return cfg.ModelType
		}
	}
	if HasResolvedDenseWeightName(names, "model.layers.0.self_attn.q_norm.weight") {
		return "qwen3"
	}
	return "qwen2"
}

func mergeDenseTextConfig(top, text DenseConfig) DenseConfig {
	if text.ModelType == "" {
		text.ModelType = top.ModelType
	}
	if len(text.Architectures) == 0 && len(top.Architectures) > 0 {
		text.Architectures = append([]string(nil), top.Architectures...)
	}
	text.Quantization = FirstDenseQuantization(text.Quantization, top.Quantization)
	if text.VocabSize == 0 {
		text.VocabSize = top.VocabSize
	}
	if text.HiddenSize == 0 {
		text.HiddenSize = top.HiddenSize
	}
	if text.NumHiddenLayers == 0 {
		text.NumHiddenLayers = top.NumHiddenLayers
	}
	if text.IntermediateSize == 0 {
		text.IntermediateSize = top.IntermediateSize
	}
	if text.MoEIntermediateSize == 0 {
		text.MoEIntermediateSize = top.MoEIntermediateSize
	}
	if text.NumAttentionHeads == 0 {
		text.NumAttentionHeads = top.NumAttentionHeads
	}
	if text.NumKeyValueHeads == 0 {
		text.NumKeyValueHeads = top.NumKeyValueHeads
	}
	if text.NumExperts == 0 {
		text.NumExperts = top.NumExperts
	}
	if text.NumExpertsPerTok == 0 {
		text.NumExpertsPerTok = firstPositiveInt(top.NumExpertsPerTok, top.TopKExperts)
	}
	if text.TopKExperts == 0 {
		text.TopKExperts = top.TopKExperts
	}
	if text.DecoderSparseStep == 0 {
		text.DecoderSparseStep = top.DecoderSparseStep
	}
	if text.HeadDim == 0 {
		text.HeadDim = top.HeadDim
	}
	if text.RMSNormEps == 0 {
		text.RMSNormEps = top.RMSNormEps
	}
	if text.RopeTheta == 0 {
		text.RopeTheta = top.RopeTheta
	}
	if text.PartialRotaryFactor == 0 {
		text.PartialRotaryFactor = top.PartialRotaryFactor
	}
	if text.MaxPositionEmbeddings == 0 {
		text.MaxPositionEmbeddings = top.MaxPositionEmbeddings
	}
	if len(text.LayerTypes) == 0 && len(top.LayerTypes) > 0 {
		text.LayerTypes = append([]string(nil), top.LayerTypes...)
	}
	return text
}

func firstDenseArchitecture(architectures []string) string {
	for _, architecture := range architectures {
		if normalized := normalizeROCmArchitecture(architecture); normalized != "" {
			return normalized
		}
	}
	return ""
}
