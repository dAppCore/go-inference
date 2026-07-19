// SPDX-Licence-Identifier: EUPL-1.2

// Package hf is the engine-agnostic HuggingFace Hub primitive shared by
// every LEM Engine (mlx, rocm, cpu). It covers two concerns:
//
//   - Hub metadata: RemoteSource queries the public HF Hub API for model
//     search results and per-model metadata (config.json shape, file
//     listing).
//   - Local cache resolution: InspectLocalMetadata and its helpers navigate
//     the standard `~/.cache/huggingface/hub/models--<org>--<name>/
//     snapshots/<rev>/` cache layout that huggingface_hub (and every
//     HF-compatible tool) writes, so a caller resolves a local repo root to
//     its metadata without re-deriving the cache convention per engine.
//
// It carries no engine-specific types — no MLX/CUDA/ROCm, no cgo. Per-engine
// concerns (Apple unified-memory fit planning, architecture-profile
// registries, native-runtime support tables) stay in each engine's own
// package, built on top of the ModelMetadata this package resolves.
//
//	source := hf.NewRemoteSource(hf.RemoteConfig{Token: hfToken})
//	meta, err := source.ModelMetadata(ctx, "Qwen/Qwen3-0.6B")
//
//	meta, root, err := hf.InspectLocalMetadata("/models/cache/models--org--name")
package hf

import (
	"context"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/jang"
)

const (
	// SourceRemote tags metadata resolved from the HF Hub API.
	SourceRemote = "huggingface"
	// SourceLocal tags metadata resolved from an on-disk cache/snapshot.
	SourceLocal = "local"

	defaultBaseURL = "https://huggingface.co"
)

// ModelSource provides optional Hugging Face metadata lookup/search. Every
// engine's own fit-planner accepts a ModelSource so tests can inject a
// fixture instead of hitting the network — RemoteSource is the production
// implementation.
type ModelSource interface {
	SearchModels(context.Context, string, int) ([]ModelMetadata, error)
	ModelMetadata(context.Context, string) (ModelMetadata, error)
}

// RemoteConfig configures the optional HF Hub metadata source.
type RemoteConfig struct {
	BaseURL   string
	Token     string
	UserAgent string
	Client    *core.HTTPClient
}

// RemoteSource reads model metadata from the Hugging Face Hub API.
type RemoteSource struct {
	baseURL   string
	token     string
	userAgent string
	authValue string // pre-built "Bearer <token>"; empty when no token
	client    *core.HTTPClient
}

// NewRemoteSource creates a network-backed HF metadata source.
func NewRemoteSource(cfg RemoteConfig) *RemoteSource {
	baseURL := core.TrimSuffix(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	client := cfg.Client
	if client == nil {
		client = &core.HTTPClient{}
	}
	// Pre-build the Authorization header value once at constructor time —
	// the token is immutable after construction, so the formatted value is
	// too. Avoids a core.Concat("Bearer ", token) allocation per request.
	var authValue string
	if cfg.Token != "" {
		authValue = core.Concat("Bearer ", cfg.Token)
	}
	return &RemoteSource{
		baseURL:   baseURL,
		token:     cfg.Token,
		userAgent: core.FirstNonBlank(cfg.UserAgent, "go-inference"),
		authValue: authValue,
		client:    client,
	}
}

// SearchModels queries HF model metadata. Network use is explicit via this source.
func (s *RemoteSource) SearchModels(ctx context.Context, query string, limit int) ([]ModelMetadata, error) {
	if s == nil {
		return nil, core.NewError("hf: nil RemoteSource")
	}
	if limit <= 0 {
		limit = 10
	}
	// Build the query string directly via Concat rather than url.Values —
	// the HF /api/models endpoint doesn't care about parameter order, so a
	// direct Concat is equivalent on the wire and saves the map + Encode
	// allocations.
	var models []ModelMetadata
	target := core.Concat(
		s.baseURL,
		"/api/models?full=true&limit=",
		strconv.Itoa(limit),
		"&search=",
		core.URLEncode(query),
	)
	if err := s.getJSON(ctx, target, &models); err != nil {
		return nil, err
	}
	return models, nil
}

// ModelMetadata returns detailed HF metadata for one model id.
func (s *RemoteSource) ModelMetadata(ctx context.Context, modelID string) (ModelMetadata, error) {
	if s == nil {
		return ModelMetadata{}, core.NewError("hf: nil RemoteSource")
	}
	target := core.Concat(s.baseURL, "/api/models/", core.URLPathEscape(modelID))
	var meta ModelMetadata
	if err := s.getJSON(ctx, target, &meta); err != nil {
		return ModelMetadata{}, err
	}
	if meta.ID == "" && meta.ModelID == "" {
		meta.ID = modelID
	}
	return meta, nil
}

func (s *RemoteSource) getJSON(ctx context.Context, target string, out any) error {
	reqResult := core.NewHTTPRequestContext(ctx, "GET", target, nil)
	if !reqResult.OK {
		return core.E("RemoteSource", "build request", reqResult.Err())
	}
	req := reqResult.Value.(*core.Request)
	req.Header.Set("Accept", "application/json")
	if s.userAgent != "" {
		req.Header.Set("User-Agent", s.userAgent)
	}
	if s.authValue != "" {
		// authValue is pre-built at constructor time; skips the per-call
		// core.Concat("Bearer ", s.token) allocation.
		req.Header.Set("Authorization", s.authValue)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return core.E("RemoteSource", "GET metadata", err)
	}
	read := core.ReadAll(resp.Body)
	if !read.OK {
		return core.E("RemoteSource", "read response", read.Err())
	}
	body := read.String()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Avoid core.Sprintf — its fmt machinery is hot-path heavy for what
		// is just an int + string assembly. strconv.Itoa+Concat is cheaper
		// for this error message shape.
		return core.NewError(core.Concat(
			"hf: HF metadata request failed: ",
			strconv.Itoa(resp.StatusCode),
			" ",
			core.Trim(body),
		))
	}
	// JSONUnmarshalString takes a string and zero-copies it to []byte via
	// AsBytes — json.Unmarshal treats the buffer as read-only and copies
	// strings into the target via SetString. Saves the []byte(body) copy
	// that would otherwise duplicate the whole response body on every call.
	if result := core.JSONUnmarshalString(body, out); !result.OK {
		return core.E("RemoteSource", "parse response", result.Err())
	}
	return nil
}

// ModelMetadata is the subset of Hugging Face/local metadata every engine
// needs to identify, plan for, and load a model.
type ModelMetadata struct {
	ID          string      `json:"id,omitempty"`
	ModelID     string      `json:"modelId,omitempty"`
	Tags        []string    `json:"tags,omitempty"`
	PipelineTag string      `json:"pipeline_tag,omitempty"`
	Config      ModelConfig `json:"config"`
	Files       []ModelFile `json:"siblings,omitempty"`
	JANG        *jang.Info  `json:"jang,omitempty"`
}

// ModelFile describes one model repository file.
type ModelFile struct {
	Name      string `json:"name,omitempty"`
	RFilename string `json:"rfilename,omitempty"`
	Size      uint64 `json:"size,omitempty"`
	SizeBytes uint64 `json:"sizeBytes,omitempty"`
}

// filename returns Name, falling back to RFilename.
func (file ModelFile) filename() string {
	return core.FirstNonBlank(file.Name, file.RFilename)
}

// byteSize returns Size, falling back to SizeBytes.
func (file ModelFile) byteSize() uint64 {
	if file.Size > 0 {
		return file.Size
	}
	return file.SizeBytes
}

// ModelConfig mirrors common transformer config fields exposed by HF. It is
// a plain data projection of config.json — deliberately free of any
// architecture-support or quantisation-normalisation logic. Interpreting
// these fields against "what can this engine actually run" is each engine's
// own concern, built on top of this data.
type ModelConfig struct {
	ModelType             string              `json:"model_type,omitempty"`
	Architectures         []string            `json:"architectures,omitempty"`
	VocabSize             int                 `json:"vocab_size,omitempty"`
	HiddenSize            int                 `json:"hidden_size,omitempty"`
	IntermediateSize      int                 `json:"intermediate_size,omitempty"`
	NumHiddenLayers       int                 `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int                 `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int                 `json:"num_key_value_heads,omitempty"`
	HeadDim               int                 `json:"head_dim,omitempty"`
	MaxPositionEmbeddings int                 `json:"max_position_embeddings,omitempty"`
	ContextLength         int                 `json:"context_length,omitempty"`
	Quantization          *QuantizationConfig `json:"quantization,omitempty"`
	QuantizationConfig    *QuantizationConfig `json:"quantization_config,omitempty"`
	TextConfig            *ModelConfig        `json:"text_config,omitempty"`
}

// QuantizationConfig captures quantization metadata when present.
type QuantizationConfig struct {
	Bits      int    `json:"bits,omitempty"`
	GroupSize int    `json:"group_size,omitempty"`
	Type      string `json:"type,omitempty"`
}
