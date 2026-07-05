// SPDX-Licence-Identifier: EUPL-1.2

package modelpack

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/quant/autoround"
	"dappco.re/go/inference/model/quant/codebook"
	"dappco.re/go/inference/model/quant/jang"
	"dappco.re/go/inference/profile"
)

// ModelPackFormat names the model weight container found in a pack.
type ModelPackFormat string

const (
	ModelPackFormatMissing     ModelPackFormat = "missing"
	ModelPackFormatSafetensors ModelPackFormat = "safetensors"
	ModelPackFormatGGUF        ModelPackFormat = "gguf"
	ModelPackFormatMixed       ModelPackFormat = "mixed"
)

// ModelPackChatTemplateSource records where chat formatting came from.
type ModelPackChatTemplateSource string

const (
	ModelPackChatTemplateNone   ModelPackChatTemplateSource = ""
	ModelPackChatTemplateFile   ModelPackChatTemplateSource = "tokenizer_config.json"
	ModelPackChatTemplateJinja  ModelPackChatTemplateSource = "chat_template.jinja"
	ModelPackChatTemplateNative ModelPackChatTemplateSource = "native"
)

// ModelPackIssueSeverity classifies a validation issue.
type ModelPackIssueSeverity string

const (
	ModelPackIssueError   ModelPackIssueSeverity = "error"
	ModelPackIssueWarning ModelPackIssueSeverity = "warning"
)

// ModelPackIssueCode is a stable machine-readable pack validation code.
type ModelPackIssueCode string

const (
	ModelPackIssueMissingConfig           ModelPackIssueCode = "missing_config"
	ModelPackIssueInvalidConfig           ModelPackIssueCode = "invalid_config"
	ModelPackIssueMissingWeights          ModelPackIssueCode = "missing_weights"
	ModelPackIssueMultipleGGUF            ModelPackIssueCode = "multiple_gguf"
	ModelPackIssueMixedWeightFormats      ModelPackIssueCode = "mixed_weight_formats"
	ModelPackIssueInvalidGGUF             ModelPackIssueCode = "invalid_gguf"
	ModelPackIssueMissingTokenizer        ModelPackIssueCode = "missing_tokenizer"
	ModelPackIssueInvalidTokenizer        ModelPackIssueCode = "invalid_tokenizer"
	ModelPackIssueUnsupportedArchitecture ModelPackIssueCode = "unsupported_architecture"
	ModelPackIssueUnsupportedRuntime      ModelPackIssueCode = "unsupported_runtime"
	ModelPackIssueMissingArchitecture     ModelPackIssueCode = "missing_architecture"
	ModelPackIssueMissingChatTemplate     ModelPackIssueCode = "missing_chat_template"
	ModelPackIssueQuantizationMismatch    ModelPackIssueCode = "quantization_mismatch"
	ModelPackIssueContextTooLarge         ModelPackIssueCode = "context_too_large"
	ModelPackIssueMiniMaxM2LayerSkeleton  ModelPackIssueCode = "minimax_m2_layer_skeleton"
	ModelPackIssueUnsupportedCodebook     ModelPackIssueCode = "unsupported_codebook"
	ModelPackIssueUnsupportedAutoRound    ModelPackIssueCode = "unsupported_auto_round"
)

// ModelPackIssue describes one pack validation finding.
type ModelPackIssue struct {
	Severity ModelPackIssueSeverity `json:"severity"`
	Code     ModelPackIssueCode     `json:"code"`
	Message  string                 `json:"message"`
	Path     string                 `json:"path,omitempty"`
}

// ModelEmbeddingProfile records metadata for encoder-style embedding packs.
type ModelEmbeddingProfile struct {
	Dimension         int    `json:"dimension,omitempty"`
	Pooling           string `json:"pooling,omitempty"`
	Normalize         bool   `json:"normalize,omitempty"`
	MaxSequenceLength int    `json:"max_sequence_length,omitempty"`
	Source            string `json:"source,omitempty"`
}

// ModelRerankProfile records metadata for cross-encoder rerank packs.
type ModelRerankProfile struct {
	Method            string `json:"method,omitempty"`
	MaxSequenceLength int    `json:"max_sequence_length,omitempty"`
	Source            string `json:"source,omitempty"`
}

// ModelPack summarises whether a local model directory is natively loadable.
//
// Fields Quantization, GGUF, MiniMaxM2, MiniMaxM2LayerSkeleton are typed as
// `any` to break the import cycle with mlx-root concrete types
// (GGUFInfo, GGUFQuantizationInfo, MiniMaxM2TensorPlan, etc.). Mlx-root
// inspectors populate these with concrete pointer values; consumers that
// need the typed value perform the type assertion.
type ModelPack struct {
	Path                   string                            `json:"path"`
	Root                   string                            `json:"root"`
	Format                 ModelPackFormat                   `json:"format"`
	ConfigPath             string                            `json:"config_path,omitempty"`
	WeightFiles            []string                          `json:"weight_files,omitempty"`
	TokenizerPath          string                            `json:"tokenizer_path,omitempty"`
	TokenizerConfigPath    string                            `json:"tokenizer_config_path,omitempty"`
	Architecture           string                            `json:"architecture,omitempty"`
	SupportedArchitecture  bool                              `json:"supported_architecture"`
	NativeLoadable         bool                              `json:"native_loadable"`
	HasTokenizer           bool                              `json:"has_tokenizer"`
	HasChatTemplate        bool                              `json:"has_chat_template"`
	ChatTemplateSource     ModelPackChatTemplateSource       `json:"chat_template_source,omitempty"`
	ChatTemplate           string                            `json:"chat_template,omitempty"`
	QuantBits              int                               `json:"quant_bits,omitempty"`
	QuantGroup             int                               `json:"quant_group,omitempty"`
	QuantType              string                            `json:"quant_type,omitempty"`
	QuantFamily            string                            `json:"quant_family,omitempty"`
	Quantization           any                               `json:"quantization,omitempty"`
	JANG                   *jang.Info                        `json:"jang,omitempty"`
	PackedQuantization     *jang.PackedProfile               `json:"packed_quantization,omitempty"`
	AutoRound              *autoround.PackInfo               `json:"auto_round,omitempty"`
	Codebook               *codebook.Profile                 `json:"codebook,omitempty"`
	MiniMaxM2              any                               `json:"minimax_m2,omitempty"`
	MiniMaxM2LayerSkeleton any                               `json:"minimax_m2_layer_skeleton,omitempty"`
	ArchitectureProfile    *profile.ModelArchitectureProfile `json:"architecture_profile,omitempty"`
	Embedding              *ModelEmbeddingProfile            `json:"embedding,omitempty"`
	Rerank                 *ModelRerankProfile               `json:"rerank,omitempty"`
	Capabilities           []inference.Capability            `json:"capabilities,omitempty"`
	WeightBytes            uint64                            `json:"weight_bytes,omitempty"`
	ContextLength          int                               `json:"context_length,omitempty"`
	NumLayers              int                               `json:"num_layers,omitempty"`
	HiddenSize             int                               `json:"hidden_size,omitempty"`
	NumKVHeads             int                               `json:"num_kv_heads,omitempty"`
	HeadDim                int                               `json:"head_dim,omitempty"`
	VocabSize              int                               `json:"vocab_size,omitempty"`
	GGUF                   any                               `json:"gguf,omitempty"`
	Issues                 []ModelPackIssue                  `json:"issues,omitempty"`
	OK                     bool                              `json:"valid"`
}

// Valid reports whether the pack has no error-severity validation issues.
func (p ModelPack) Valid() bool { return p.OK }

// HasIssue reports whether a validation issue code is present.
func (p ModelPack) HasIssue(code ModelPackIssueCode) bool {
	for _, issue := range p.Issues {
		if issue.Code == code {
			return true
		}
	}
	return false
}

// ModelPackConfig configures pack validation.
type ModelPackConfig struct {
	ExpectedQuantBits   int
	MaxContextLength    int
	RequireChatTemplate bool
}

// ModelPackOption configures model-pack inspection.
type ModelPackOption func(*ModelPackConfig)

// WithPackQuantization requires a specific quantization width when metadata exposes one.
func WithPackQuantization(bits int) ModelPackOption {
	return func(cfg *ModelPackConfig) { cfg.ExpectedQuantBits = bits }
}

// WithPackMaxContextLength rejects packs whose declared context exceeds n.
func WithPackMaxContextLength(n int) ModelPackOption {
	return func(cfg *ModelPackConfig) { cfg.MaxContextLength = n }
}

// WithPackRequireChatTemplate controls whether a chat template is mandatory.
func WithPackRequireChatTemplate(required bool) ModelPackOption {
	return func(cfg *ModelPackConfig) { cfg.RequireChatTemplate = required }
}

// ApplyOptions reduces a list of options into a ModelPackConfig with defaults.
//
//	cfg := pack.ApplyOptions(opts)
func ApplyOptions(opts []ModelPackOption) ModelPackConfig {
	// Fast-path the zero-opts case so cfg stays on the caller's stack
	// frame. The for-loop body takes &cfg, which would otherwise force
	// the compiler to heap-allocate cfg even when opts is empty.
	if len(opts) == 0 {
		return ModelPackConfig{RequireChatTemplate: true}
	}
	cfg := ModelPackConfig{RequireChatTemplate: true}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// AddIssue appends a validation issue to the pack.
//
//	p.AddIssue(pack.ModelPackIssueError, pack.ModelPackIssueMissingConfig, "...", path)
func (p *ModelPack) AddIssue(severity ModelPackIssueSeverity, code ModelPackIssueCode, message, path string) {
	p.Issues = append(p.Issues, ModelPackIssue{
		Severity: severity,
		Code:     code,
		Message:  message,
		Path:     path,
	})
}

// HasErrorIssue reports whether any issue has error severity.
func (p ModelPack) HasErrorIssue() bool {
	for _, issue := range p.Issues {
		if issue.Severity == ModelPackIssueError {
			return true
		}
	}
	return false
}

// IssueSummary returns a comma-separated list of error-severity issue codes.
func (p ModelPack) IssueSummary() string {
	if len(p.Issues) == 0 {
		return "unknown"
	}
	// Single-pass build — skip the intermediate codes slice. Pre-size
	// the Builder against the total error-code byte count so its
	// internal buffer never grows. The earlier "collect into []string,
	// then core.Join" path took two allocs (slice header + Builder);
	// streaming directly into the Builder drops it to one.
	total := 0
	count := 0
	for _, issue := range p.Issues {
		if issue.Severity == ModelPackIssueError {
			total += len(issue.Code)
			count++
		}
	}
	if count == 0 {
		return "unknown"
	}
	total += 2 * (count - 1) // ", " separators
	// Build directly into a pre-sized byte slice and AsString the
	// result — Builder's WriteString carries non-trivial dispatch per
	// call and a strings.Builder still ends up doing the same
	// unsafe-cast in String(). One make([]byte, 0, total) + AsString
	// keeps the alloc count at one (the buffer itself) and avoids the
	// per-WriteString interface overhead.
	buf := make([]byte, 0, total)
	first := true
	for _, issue := range p.Issues {
		if issue.Severity != ModelPackIssueError {
			continue
		}
		if !first {
			buf = append(buf, ", "...)
		}
		first = false
		buf = append(buf, issue.Code...)
	}
	return core.AsString(buf)
}
