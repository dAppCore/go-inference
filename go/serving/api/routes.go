// SPDX-License-Identifier: EUPL-1.2

// Package api provides REST endpoints for the ML inference service,
// implementing go-api's RouteGroup and StreamGroup interfaces.
package api

import (
	"net/http"

	goapi "dappco.re/go/api"
	"dappco.re/go/inference/serving"
	"dappco.re/go/inference/serving/mlservice"
	"github.com/gin-gonic/gin"
)

// Routes implements api.RouteGroup and api.StreamGroup for ML inference endpoints.
// Usage example: engine.Register(api.NewRoutes(service))
type Routes struct {
	service *mlservice.Service
}

// NewRoutes creates an ML route group wrapping the given service.
// Usage example: routes := api.NewRoutes(service)
func NewRoutes(svc *mlservice.Service) *Routes {
	return &Routes{service: svc}
}

// Name returns the route-group name used by the API engine.
func (r *Routes) Name() string { return "ml" }

// BasePath returns the URL prefix for the ML HTTP routes.
func (r *Routes) BasePath() string { return "/v1/ml" }

// RegisterRoutes mounts the ML HTTP handlers on the provided router group.
func (r *Routes) RegisterRoutes(rg *gin.RouterGroup) {
	rg.GET("/backends", r.ListBackends)
	rg.GET("/status", r.Status)
	rg.POST("/generate", r.Generate)
}

// Channels declares WebSocket channels for ML events.
func (r *Routes) Channels() []string {
	return []string{"ml.generate", "ml.status"}
}

// backendInfo describes a registered inference backend.
type backendInfo struct {
	Name      string `json:"name"`
	Available bool   `json:"available"`
}

// statusResponse describes the service status.
type statusResponse struct {
	Ready    bool     `json:"ready"`
	Backends []string `json:"backends"`
	HasJudge bool     `json:"has_judge"`
}

// generateRequest is the payload for POST /generate.
type generateRequest struct {
	Prompt      string  `json:"prompt" binding:"required"`
	Backend     string  `json:"backend,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
}

// generateResponse wraps the generation result.
type generateResponse struct {
	Text string `json:"text"`
}

// ListBackends returns all registered inference backends with their availability.
func (r *Routes) ListBackends(c *gin.Context) {
	if r.service == nil {
		c.JSON(http.StatusServiceUnavailable, goapi.Fail("SERVICE_UNAVAILABLE", "ML service not initialised"))
		return
	}

	names := r.service.Backends()
	backends := make([]backendInfo, 0, len(names))
	for _, name := range names {
		b := r.service.Backend(name)
		available := false
		if b != nil {
			available = b.Available()
		}
		backends = append(backends, backendInfo{
			Name:      name,
			Available: available,
		})
	}

	c.JSON(http.StatusOK, goapi.OK(backends))
}

// Status returns the overall service status.
func (r *Routes) Status(c *gin.Context) {
	if r.service == nil {
		c.JSON(http.StatusServiceUnavailable, goapi.Fail("SERVICE_UNAVAILABLE", "ML service not initialised"))
		return
	}

	names := r.service.Backends()
	c.JSON(http.StatusOK, goapi.OK(statusResponse{
		Ready:    len(names) > 0,
		Backends: names,
		HasJudge: r.service.Judge() != nil,
	}))
}

// Generate runs text generation against a backend.
func (r *Routes) Generate(c *gin.Context) {
	if r.service == nil {
		c.JSON(http.StatusServiceUnavailable, goapi.Fail("SERVICE_UNAVAILABLE", "ML service not initialised"))
		return
	}

	var req generateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, goapi.Fail("INVALID_REQUEST", err.Error()))
		return
	}

	opts := serving.DefaultGenOpts()
	if req.Temperature > 0 {
		opts.Temperature = req.Temperature
	}
	if req.MaxTokens > 0 {
		opts.MaxTokens = req.MaxTokens
	}

	result := r.service.Generate(c.Request.Context(), req.Backend, req.Prompt, opts)
	if !result.OK {
		c.JSON(http.StatusInternalServerError, goapi.Fail("GENERATION_FAILED", result.Error()))
		return
	}
	res := result.Value.(serving.Result)

	c.JSON(http.StatusOK, goapi.OK(generateResponse{Text: res.Text}))
}
