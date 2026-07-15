// SPDX-License-Identifier: EUPL-1.2

// Package api exposes the inference stack provider routes for core/api.
package api

import (
	"net/http"

	coreapi "dappco.re/go/api"
	coreprovider "dappco.re/go/api/pkg/provider"
	"github.com/gin-gonic/gin"
)

// AIProvider exposes the inference stack scoring surfaces as a core/api provider.
//
// Every surface is pure — it runs the in-process lem-scorer (eval/score/lek,
// built on go-i18n's grammar imprint) over text supplied in the request and
// needs no model. The caller (e.g. lthn/desktop) presents the text and uses the
// returned score as request metadata; nothing is tracked or persisted here.
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

	rg.POST("/embeddings/behavioural", p.embedBehavioural)
	rg.POST("/score/content", p.scoreContent)
	rg.POST("/score/imprint", p.scoreImprint)
	rg.POST("/score/session", p.scoreSession)
	rg.GET("/score/:id", p.getScore)
	rg.GET("/health", p.health)
}

// Describe implements api.DescribableGroup for OpenAPI generation when core/api
// mounts the provider.
func (p *AIProvider) Describe() []coreapi.RouteDescription {
	return []coreapi.RouteDescription{
		{
			Method:      http.MethodPost,
			Path:        "/embeddings/behavioural",
			Summary:     "Create a behavioural embedding",
			Description: "Accepts text and returns the grammar imprint as a fixed-order behavioural fingerprint vector (the lem-scorer imprint dimensions). No model required.",
			Tags:        []string{"embeddings"},
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
			Tags:        []string{"scoring"},
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
			Tags:        []string{"scoring"},
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
			Method:      http.MethodPost,
			Path:        "/score/session",
			Summary:     "Score session history",
			Description: "Runs the lem-scorer after the fact over a conversation's turns: each assistant turn is scored against the preceding user turn (the same pairing the live pipeline applies). Stateless — the caller supplies the turns loaded from session history. No model required.",
			Tags:        []string{"scoring"},
			RequestBody: map[string]any{
				"type":     "object",
				"required": []string{"turns"},
				"properties": map[string]any{
					"turns": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"role":    map[string]any{"type": "string"},
								"content": map[string]any{"type": "string"},
							},
						},
					},
				},
			},
			Response: map[string]any{
				"type":       "object",
				"properties": map[string]any{"scores": map[string]any{"type": "array", "items": map[string]any{"type": "object"}}},
			},
		},
		{
			Method:      http.MethodGet,
			Path:        "/score/:id",
			Summary:     "Get score result",
			Description: "Retrieves a stored score result once persistence for the the inference stack provider surface is selected.",
			Tags:        []string{"scoring"},
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
