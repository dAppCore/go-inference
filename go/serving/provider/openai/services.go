// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"io"
	"net/http"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	DefaultEmbeddingsPath   = "/v1/embeddings"
	DefaultRerankPath       = "/v1/rerank"
	DefaultCapabilitiesPath = "/v1/models/capabilities"
	DefaultCacheStatsPath   = "/v1/cache/stats"
	DefaultCacheWarmPath    = "/v1/cache/warm"
	DefaultCacheClearPath   = "/v1/cache/clear"
	DefaultCancelPath       = "/v1/cancel"
)

// EmbeddingRequest is the OpenAI-compatible embedding request body.
type EmbeddingRequest struct {
	Model          string         `json:"model"`
	Input          EmbeddingInput `json:"input"`
	EncodingFormat string         `json:"encoding_format,omitempty"`
	Dimensions     *int           `json:"dimensions,omitempty"`
	User           string         `json:"user,omitempty"`
	Normalize      bool           `json:"normalize,omitempty"`
}

// EmbeddingInput accepts either a string or an array of strings.
type EmbeddingInput []string

func (input *EmbeddingInput) UnmarshalJSON(data []byte) error {
	// Hot path — fires per embeddings request. parseJSONStringList
	// walks the variant string-or-array shape in a single pass —
	// drops the recursive core.JSONUnmarshal allocs (encoder state
	// + per-element string).
	values, err := parseJSONStringList(data)
	if err != nil {
		return err
	}
	*input = values
	return nil
}

type EmbeddingResponse struct {
	Object string                   `json:"object"`
	Data   []EmbeddingResponseDatum `json:"data"`
	Model  string                   `json:"model"`
	Usage  inference.EmbeddingUsage `json:"usage"`
}

type EmbeddingResponseDatum struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type RerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

type RerankResponse struct {
	Object  string                  `json:"object"`
	Model   string                  `json:"model"`
	Results []inference.RerankScore `json:"results"`
}

type CacheWarmRequest struct {
	Model  string            `json:"model"`
	Prompt string            `json:"prompt,omitempty"`
	Tokens []int32           `json:"tokens,omitempty"`
	Mode   string            `json:"mode,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`
}

type CacheClearRequest struct {
	Model  string            `json:"model"`
	Labels map[string]string `json:"labels,omitempty"`
}

type CancelRequest struct {
	Model string `json:"model"`
	ID    string `json:"id"`
}

type serviceHandler struct {
	resolver Resolver
}

type EmbeddingsHandler struct{ serviceHandler }
type RerankHandler struct{ serviceHandler }
type CapabilityHandler struct{ serviceHandler }
type CacheStatsHandler struct{ serviceHandler }
type CacheWarmHandler struct{ serviceHandler }
type CacheClearHandler struct{ serviceHandler }
type CancelHandler struct{ serviceHandler }

func NewEmbeddingsHandler(resolver Resolver) *EmbeddingsHandler {
	return &EmbeddingsHandler{serviceHandler{resolver: resolver}}
}

func NewRerankHandler(resolver Resolver) *RerankHandler {
	return &RerankHandler{serviceHandler{resolver: resolver}}
}

func NewCapabilityHandler(resolver Resolver) *CapabilityHandler {
	return &CapabilityHandler{serviceHandler{resolver: resolver}}
}

func NewCacheStatsHandler(resolver Resolver) *CacheStatsHandler {
	return &CacheStatsHandler{serviceHandler{resolver: resolver}}
}

func NewCacheWarmHandler(resolver Resolver) *CacheWarmHandler {
	return &CacheWarmHandler{serviceHandler{resolver: resolver}}
}

func NewCacheClearHandler(resolver Resolver) *CacheClearHandler {
	return &CacheClearHandler{serviceHandler{resolver: resolver}}
}

func NewCancelHandler(resolver Resolver) *CancelHandler {
	return &CancelHandler{serviceHandler{resolver: resolver}}
}

func (h *EmbeddingsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireServiceMethod(w, r, http.MethodPost) {
		return
	}
	var req EmbeddingRequest
	if !decodeServiceRequest(w, r, &req, "openai.EmbeddingsHandler") {
		return
	}
	if core.Trim(req.Model) == "" {
		writeError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	if len(req.Input) == 0 {
		writeError(w, http.StatusBadRequest, "input must not be empty", "input")
		return
	}
	model, ok := h.resolve(w, r.Context(), req.Model)
	if !ok {
		return
	}
	embeddingModel, ok := model.(inference.EmbeddingModel)
	if !ok {
		writeError(w, http.StatusNotImplemented, "model does not support embeddings", "model")
		return
	}
	result, err := embeddingModel.Embed(r.Context(), inference.EmbeddingRequest{
		Model:     req.Model,
		Input:     []string(req.Input),
		Normalize: req.Normalize,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	data := make([]EmbeddingResponseDatum, 0, len(result.Vectors))
	for i, vector := range result.Vectors {
		data = append(data, EmbeddingResponseDatum{Object: "embedding", Index: i, Embedding: vector})
	}
	writeJSON(w, http.StatusOK, EmbeddingResponse{Object: "list", Data: data, Model: req.Model, Usage: result.Usage})
}

func (h *RerankHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireServiceMethod(w, r, http.MethodPost) {
		return
	}
	var req RerankRequest
	if !decodeServiceRequest(w, r, &req, "openai.RerankHandler") {
		return
	}
	if core.Trim(req.Model) == "" {
		writeError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	if core.Trim(req.Query) == "" {
		writeError(w, http.StatusBadRequest, "query is required", "query")
		return
	}
	if len(req.Documents) == 0 {
		writeError(w, http.StatusBadRequest, "documents must not be empty", "documents")
		return
	}
	model, ok := h.resolve(w, r.Context(), req.Model)
	if !ok {
		return
	}
	rerankModel, ok := model.(inference.RerankModel)
	if !ok {
		writeError(w, http.StatusNotImplemented, "model does not support rerank", "model")
		return
	}
	result, err := rerankModel.Rerank(r.Context(), inference.RerankRequest{
		Model:     req.Model,
		Query:     req.Query,
		Documents: req.Documents,
		TopN:      req.TopN,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	writeJSON(w, http.StatusOK, RerankResponse{Object: "list", Model: req.Model, Results: result.Results})
}

func (h *CapabilityHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireServiceMethod(w, r, http.MethodGet) {
		return
	}
	modelName := queryModel(r)
	if modelName == "" {
		writeError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	model, ok := h.resolve(w, r.Context(), modelName)
	if !ok {
		return
	}
	if reporter, ok := model.(inference.CapabilityReporter); ok {
		writeJSON(w, http.StatusOK, reporter.Capabilities())
		return
	}
	writeJSON(w, http.StatusOK, inference.TextModelCapabilities(inference.RuntimeIdentity{}, model))
}

func (h *CacheStatsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireServiceMethod(w, r, http.MethodGet) {
		return
	}
	model, ok := h.resolveCacheService(w, r.Context(), queryModel(r))
	if !ok {
		return
	}
	stats, err := model.CacheStats(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "cache")
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (h *CacheWarmHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireServiceMethod(w, r, http.MethodPost) {
		return
	}
	var req CacheWarmRequest
	if !decodeServiceRequest(w, r, &req, "openai.CacheWarmHandler") {
		return
	}
	model, ok := h.resolveCacheService(w, r.Context(), req.Model)
	if !ok {
		return
	}
	result, err := model.WarmCache(r.Context(), inference.CacheWarmRequest{
		Model:  inference.ModelIdentity{ID: req.Model},
		Prompt: req.Prompt,
		Tokens: req.Tokens,
		Mode:   req.Mode,
		Labels: req.Labels,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "cache")
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (h *CacheClearHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireServiceMethod(w, r, http.MethodPost) {
		return
	}
	var req CacheClearRequest
	if !decodeServiceRequest(w, r, &req, "openai.CacheClearHandler") {
		return
	}
	model, ok := h.resolveCacheService(w, r.Context(), req.Model)
	if !ok {
		return
	}
	stats, err := model.ClearCache(r.Context(), req.Labels)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "cache")
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (h *CancelHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireServiceMethod(w, r, http.MethodPost) {
		return
	}
	var req CancelRequest
	if !decodeServiceRequest(w, r, &req, "openai.CancelHandler") {
		return
	}
	if core.Trim(req.ID) == "" {
		writeError(w, http.StatusBadRequest, "id is required", "id")
		return
	}
	model, ok := h.resolve(w, r.Context(), req.Model)
	if !ok {
		return
	}
	cancellable, ok := model.(inference.CancellableModel)
	if !ok {
		writeError(w, http.StatusNotImplemented, "model does not support request cancellation", "model")
		return
	}
	result, err := cancellable.CancelRequest(r.Context(), req.ID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (h *serviceHandler) resolve(w http.ResponseWriter, ctx context.Context, modelName string) (inference.TextModel, bool) {
	if h == nil || h.resolver == nil {
		writeError(w, http.StatusServiceUnavailable, "handler is not configured", "model")
		return nil, false
	}
	modelName = core.Trim(modelName)
	if modelName == "" {
		writeError(w, http.StatusBadRequest, "model is required", "model")
		return nil, false
	}
	model, err := h.resolver.ResolveModel(ctx, modelName)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error(), "model")
		return nil, false
	}
	return model, true
}

func (h *serviceHandler) resolveCacheService(w http.ResponseWriter, ctx context.Context, modelName string) (inference.CacheService, bool) {
	model, ok := h.resolve(w, ctx, modelName)
	if !ok {
		return nil, false
	}
	cache, ok := model.(inference.CacheService)
	if !ok {
		writeError(w, http.StatusNotImplemented, "model does not support cache service operations", "model")
		return nil, false
	}
	return cache, true
}

func decodeServiceRequest(w http.ResponseWriter, r *http.Request, into any, scope string) bool {
	if r == nil || r.Body == nil {
		writeError(w, http.StatusBadRequest, "request body is nil", "body")
		return false
	}
	data, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "read request body failed", "body")
		return false
	}
	result := core.JSONUnmarshal(data, into)
	if !result.OK {
		err := result.Err()
		message := "invalid request body"
		if err != nil && core.Trim(err.Error()) != "" {
			message = core.Concat(scope, ": ", err.Error())
		}
		writeError(w, http.StatusBadRequest, message, "body")
		return false
	}
	return true
}

func requireServiceMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r == nil {
		writeError(w, http.StatusBadRequest, "request is nil", "request")
		return false
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return false
	}
	return true
}

func queryModel(r *http.Request) string {
	if r == nil || r.URL == nil {
		return ""
	}
	return core.Trim(r.URL.Query().Get("model"))
}
