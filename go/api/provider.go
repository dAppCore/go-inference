// SPDX-License-Identifier: EUPL-1.2

// Package api exposes the inference stack provider routes for core/api.
package api

import (
	"net/http"

	coreapi "dappco.re/go/api"
	coreprovider "dappco.re/go/api/pkg/provider"
	"github.com/gin-gonic/gin"
)

// AIProvider exposes the inference stack embedding and scoring surfaces as a core/api
// provider.
type AIProvider struct{}

var (
	_ coreprovider.Provider    = (*AIProvider)(nil)
	_ coreprovider.Describable = (*AIProvider)(nil)
)

// NewProvider creates the the inference stack HTTP provider.
func NewProvider() *AIProvider {
	return &AIProvider{}
}

// New creates the the inference stack HTTP provider for core/api registration call sites that
// alias this package as provider.
func New() *AIProvider {
	return NewProvider()
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
			Description: "Accepts text and returns an embedding vector once the the inference stack provider architecture is selected.",
			Tags:        []string{"ai", "embeddings"},
			RequestBody: map[string]any{
				"type":     "object",
				"required": []string{"text"},
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
				},
			},
			Response: notImplementedSchema(),
		},
		{
			Method:      http.MethodPost,
			Path:        "/embeddings/behavioural",
			Summary:     "Create a behavioural embedding",
			Description: "Accepts a behavioural sequence and returns an OFM B1 fingerprint once the the inference stack provider architecture is selected.",
			Tags:        []string{"ai", "embeddings"},
			RequestBody: map[string]any{
				"type":     "object",
				"required": []string{"sequence"},
				"properties": map[string]any{
					"sequence": map[string]any{"type": "array", "items": map[string]any{"type": "object"}},
				},
			},
			Response: notImplementedSchema(),
		},
		{
			Method:      http.MethodPost,
			Path:        "/score/content",
			Summary:     "Score content",
			Description: "Accepts text and returns ethical and sycophancy scoring once the the inference stack provider architecture is selected.",
			Tags:        []string{"ai", "scoring"},
			RequestBody: map[string]any{
				"type":     "object",
				"required": []string{"text"},
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
				},
			},
			Response: notImplementedSchema(),
		},
		{
			Method:      http.MethodPost,
			Path:        "/score/imprint",
			Summary:     "Score imprint",
			Description: "Accepts imprint material and returns ScoreImprint output once the the inference stack provider architecture is selected.",
			Tags:        []string{"ai", "scoring"},
			RequestBody: map[string]any{
				"type": "object",
			},
			Response: notImplementedSchema(),
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
