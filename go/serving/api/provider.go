// SPDX-License-Identifier: EUPL-1.2

// Package api exposes the inference stack provider routes for core/api.
package api

import (
	"net/http"

	coreapi "dappco.re/go/api"
	coreprovider "dappco.re/go/api/pkg/provider"
	"dappco.re/go/inference"
	"github.com/gin-gonic/gin"
)

// AIProvider exposes the inference stack embedding and scoring surfaces as a core/api
// provider.
//
// The scoring and behavioural-imprint surfaces are pure — they run the
// in-process lem-scorer (eval/score/lek, built on go-i18n's grammar imprint)
// and need no model. The neural text-embedding surface needs an
// [inference.EmbeddingModel]; inject one with [WithEmbedder] or /embeddings/text
// reports 503.
type AIProvider struct {
	embedder inference.EmbeddingModel
}

var (
	_ coreprovider.Provider    = (*AIProvider)(nil)
	_ coreprovider.Describable = (*AIProvider)(nil)
)

// Option configures an AIProvider at construction.
type Option func(*AIProvider)

// WithEmbedder injects the text-embedding model backing POST /embeddings/text.
// Without it that one endpoint reports 503; the scoring and behavioural-imprint
// endpoints work regardless because they need no model.
//
//	p := api.NewProvider(api.WithEmbedder(model))
func WithEmbedder(m inference.EmbeddingModel) Option {
	return func(p *AIProvider) { p.embedder = m }
}

// NewProvider creates the the inference stack HTTP provider. Pass [WithEmbedder]
// to wire the neural text-embedding endpoint.
func NewProvider(opts ...Option) *AIProvider {
	p := &AIProvider{}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// New creates the the inference stack HTTP provider for core/api registration call sites that
// alias this package as provider.
func New(opts ...Option) *AIProvider {
	return NewProvider(opts...)
}

// Name implements api.RouteGroup.
func (p *AIProvider) Name() string { return "ai" }

// BasePath implements api.RouteGroup.
func (p *AIProvider) BasePath() string { return "/v1" }

// RegisterRoutes implements api.RouteGroup.
func (p *AIProvider) RegisterRoutes(rg *gin.RouterGroup) {
	if p == nil || rg == nil {
		return
	}

	rg.POST("/embeddings/text", p.embedText)
	rg.POST("/embeddings/behavioural", p.embedBehavioural)
	rg.POST("/score/content", p.scoreContent)
	rg.POST("/score/imprint", p.scoreImprint)
	rg.GET("/score/:id", p.getScore)
	rg.GET("/health", p.health)
}

// Describe implements api.DescribableGroup for OpenAPI generation when core/api
// mounts the provider.
func (p *AIProvider) Describe() []coreapi.RouteDescription {
	return []coreapi.RouteDescription{
		{
			Method:      http.MethodPost,
			Path:        "/embeddings/text",
			Summary:     "Create a text embedding",
			Description: "Accepts text and returns a neural embedding vector from the injected embedding model. Reports 503 when no embedding model is configured.",
			Tags:        []string{"ai", "embeddings"},
			RequestBody: map[string]any{
				"type":     "object",
				"required": []string{"text"},
				"properties": map[string]any{
					"text":  map[string]any{"type": "string"},
					"model": map[string]any{"type": "string"},
				},
			},
			Response: embeddingSchema(),
		},
		{
			Method:      http.MethodPost,
			Path:        "/embeddings/behavioural",
			Summary:     "Create a behavioural embedding",
			Description: "Accepts text and returns the grammar imprint as a fixed-order behavioural fingerprint vector (the lem-scorer imprint dimensions). No model required.",
			Tags:        []string{"ai", "embeddings"},
			RequestBody: map[string]any{
				"type":     "object",
				"required": []string{"text"},
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
				},
			},
			Response: embeddingSchema(),
		},
		{
			Method:      http.MethodPost,
			Path:        "/score/content",
			Summary:     "Score content",
			Description: "Runs the in-process lem-scorer (sycophancy, LEK, hostility, grammar imprint). Send text for a single-text ScoreResult, or prompt+response for a DiffResult with cross-text differential and authority signal. No model required.",
			Tags:        []string{"ai", "scoring"},
			RequestBody: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text":     map[string]any{"type": "string"},
					"prompt":   map[string]any{"type": "string"},
					"response": map[string]any{"type": "string"},
				},
			},
			Response: map[string]any{"type": "object"},
		},
		{
			Method:      http.MethodPost,
			Path:        "/score/imprint",
			Summary:     "Score imprint",
			Description: "Accepts text and returns the 14-dimensional grammar+phonetic imprint (imprint is null when the text produces no tokens). No model required.",
			Tags:        []string{"ai", "scoring"},
			RequestBody: map[string]any{
				"type":     "object",
				"required": []string{"text"},
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
				},
			},
			Response: map[string]any{
				"type":       "object",
				"properties": map[string]any{"imprint": map[string]any{"type": "object"}},
			},
		},
		{
			Method:      http.MethodGet,
			Path:        "/score/:id",
			Summary:     "Get score result",
			Description: "Retrieves a stored score result once persistence for the the inference stack provider surface is selected.",
			Tags:        []string{"ai", "scoring"},
			Response:    notImplementedSchema(),
		},
		{
			Method:      http.MethodGet,
			Path:        "/health",
			Summary:     "Health check",
			Description: "Returns basic the inference stack provider health.",
			Tags:        []string{"ai"},
			Response: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"ok":       map[string]any{"type": "boolean"},
					"provider": map[string]any{"type": "string"},
					"status":   map[string]any{"type": "string"},
				},
			},
		},
	}
}

func notImplementedSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"error":   map[string]any{"type": "string"},
			"message": map[string]any{"type": "string"},
			"todo":    map[string]any{"type": "string"},
		},
	}
}

// Registration note: core/api should import this package, commonly aliased as
// provider, and mount it with Engine.Register(provider.New()).
