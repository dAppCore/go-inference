// SPDX-License-Identifier: EUPL-1.2

package api

import (
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func TestHandlers_Good(t *testing.T) {
	router := setupTestRouter()

	tests := []struct {
		name       string
		method     string
		path       string
		wantStatus int
		wantBody   string
	}{
		{
			name:       "text embeddings",
			method:     http.MethodPost,
			path:       "/v1/embeddings/text",
			wantStatus: http.StatusNotImplemented,
			wantBody:   "text embedding generation",
		},
		{
			name:       "behavioural embeddings",
			method:     http.MethodPost,
			path:       "/v1/embeddings/behavioural",
			wantStatus: http.StatusNotImplemented,
			wantBody:   "behavioural embedding generation",
		},
		{
			name:       "score content",
			method:     http.MethodPost,
			path:       "/v1/score/content",
			wantStatus: http.StatusNotImplemented,
			wantBody:   "content scoring",
		},
		{
			name:       "score imprint",
			method:     http.MethodPost,
			path:       "/v1/score/imprint",
			wantStatus: http.StatusNotImplemented,
			wantBody:   "imprint scoring",
		},
		{
			name:       "score retrieval",
			method:     http.MethodGet,
			path:       "/v1/score/example-score",
			wantStatus: http.StatusNotImplemented,
			wantBody:   "score retrieval",
		},
		{
			name:       "health",
			method:     http.MethodGet,
			path:       "/v1/health",
			wantStatus: http.StatusOK,
			wantBody:   "healthy",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := performRequest(router, tt.method, tt.path)
			if rec.Code != tt.wantStatus {
				t.Fatalf("expected status %d, got %d with body %s", tt.wantStatus, rec.Code, rec.Body.String())
			}
			if !core.Contains(rec.Body.String(), tt.wantBody) {
				t.Fatalf("expected body to contain %q, got %s", tt.wantBody, rec.Body.String())
			}
		})
	}
}

func TestHandlers_Bad(t *testing.T) {
	router := setupTestRouter()

	tests := []struct {
		name   string
		method string
		path   string
	}{
		{name: "text embeddings rejects GET", method: http.MethodGet, path: "/v1/embeddings/text"},
		{name: "behavioural embeddings rejects GET", method: http.MethodGet, path: "/v1/embeddings/behavioural"},
		{name: "score content rejects PUT", method: http.MethodPut, path: "/v1/score/content"},
		{name: "score imprint rejects PUT", method: http.MethodPut, path: "/v1/score/imprint"},
		{name: "score retrieval rejects POST", method: http.MethodPost, path: "/v1/score/example-score"},
		{name: "health rejects POST", method: http.MethodPost, path: "/v1/health"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := performRequest(router, tt.method, tt.path)
			if rec.Code != http.StatusNotFound {
				t.Fatalf("expected status %d, got %d with body %s", http.StatusNotFound, rec.Code, rec.Body.String())
			}
		})
	}
}

func TestHandlers_Ugly(t *testing.T) {
	for _, handler := range []func(*AIProvider, *gin.Context){
		func(p *AIProvider, c *gin.Context) { p.embedText(c) },
		func(p *AIProvider, c *gin.Context) { p.embedBehavioural(c) },
		func(p *AIProvider, c *gin.Context) { p.scoreContent(c) },
		func(p *AIProvider, c *gin.Context) { p.scoreImprint(c) },
		func(p *AIProvider, c *gin.Context) { p.getScore(c) },
		func(p *AIProvider, c *gin.Context) { p.health(c) },
	} {
		assertDoesNotPanic(t, func() {
			handler(NewProvider(), nil)
		})
	}
}

func TestHandlersNotImplementedBody(t *testing.T) {
	router := setupTestRouter()
	rec := performRequest(router, http.MethodPost, "/v1/score/content")

	var body map[string]string
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &body); !r.OK {
		t.Fatalf("decode response: %v", r.Error())
	}
	if body["error"] != "not_implemented" {
		t.Fatalf("expected not_implemented error, got %q", body["error"])
	}
	if !core.Contains(body["todo"], "architectural-decision-needed") {
		t.Fatalf("expected architecture TODO, got %q", body["todo"])
	}
}

func setupTestRouter() *gin.Engine {
	provider := NewProvider()
	router := gin.New()
	provider.RegisterRoutes(router.Group(provider.BasePath()))
	return router
}

func performRequest(router *gin.Engine, method, path string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}
