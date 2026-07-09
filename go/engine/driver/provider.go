// SPDX-License-Identifier: EUPL-1.2

package driver

import (
	"net/http"

	core "dappco.re/go"
	coreapi "dappco.re/go/api"
	coreprovider "dappco.re/go/api/pkg/provider"
	"github.com/gin-gonic/gin"
)

// Provider exposes the driver-orchestration surface as a core/api RouteGroup at
// /v1/driver: serve a model on a runtime, list the catalogue, read status, stop
// a driver. Generic process health/list comes from the go-process provider at
// /api/process; this group is the model-semantic view over it.
//
// Usage example:
//
//	engine.Register(driver.NewProvider(driver.NewService(procSvc)))
type Provider struct {
	svc *Service
}

var (
	_ coreapi.RouteGroup       = (*Provider)(nil)
	_ coreprovider.Describable = (*Provider)(nil)
)

// NewProvider wraps a driver Service as a mountable RouteGroup.
func NewProvider(svc *Service) *Provider { return &Provider{svc: svc} }

// Name implements api.RouteGroup.
func (p *Provider) Name() string { return "driver" }

// BasePath implements api.RouteGroup.
func (p *Provider) BasePath() string { return "/v1/driver" }

// RegisterRoutes implements api.RouteGroup.
func (p *Provider) RegisterRoutes(rg *gin.RouterGroup) {
	if p == nil || rg == nil {
		return
	}
	rg.GET("/models", p.models)
	rg.POST("/serve", p.serve)
	rg.GET("/status", p.status)
	rg.POST("/stop", p.stop)
}

// Describe implements coreprovider.Describable so the driver-orchestration
// routes appear in the OpenAPI document when core/api mounts the provider —
// which is what lets the SDK generators emit a typed client for them.
func (p *Provider) Describe() []coreapi.RouteDescription {
	serveBody := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"model":         map[string]any{"type": "string"},
			"profile":       map[string]any{"type": "string"},
			"runtime":       map[string]any{"type": "string"},
			"addr":          map[string]any{"type": "string"},
			"context":       map[string]any{"type": "integer"},
			"noAutoProfile": map[string]any{"type": "boolean"},
		},
	}
	return []coreapi.RouteDescription{
		{
			Method:      http.MethodGet,
			Path:        "/models",
			Summary:     "List loadable models and serve profiles",
			Description: "Lists the weights the host can load and the serve profiles available for them.",
			Tags:        []string{"driver"},
		},
		{
			Method:      http.MethodPost,
			Path:        "/serve",
			Summary:     "Serve a model on a runtime",
			Description: "Cold-starts a driver for the (model, profile) on the requested runtime (mlx | cuda | amd; empty defaults to mlx). An empty model starts the driver model-less to load later.",
			Tags:        []string{"driver"},
			RequestBody: serveBody,
		},
		{
			Method:      http.MethodGet,
			Path:        "/status",
			Summary:     "List supervised drivers",
			Description: "Snapshot of every driver the host is currently supervising.",
			Tags:        []string{"driver"},
		},
		{
			Method:      http.MethodPost,
			Path:        "/stop",
			Summary:     "Stop a driver",
			Description: "Drains and terminates the driver for the given runtime. An empty body defaults to mlx.",
			Tags:        []string{"driver"},
			RequestBody: map[string]any{
				"type":       "object",
				"properties": map[string]any{"runtime": map[string]any{"type": "string"}},
			},
		},
	}
}

// models — GET /v1/driver/models. Lists loadable weights + serve profiles.
func (p *Provider) models(c *gin.Context) {
	r := p.svc.Models()
	if !r.OK {
		c.JSON(http.StatusInternalServerError, fail(r.Error()))
		return
	}
	c.JSON(http.StatusOK, r)
}

// serve — POST /v1/driver/serve. Cold-starts a driver for the (model, profile)
// on the requested runtime.
func (p *Provider) serve(c *gin.Context) {
	var req ServeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, fail("invalid request body: "+err.Error()))
		return
	}
	r := p.svc.Serve(req)
	if !r.OK {
		c.JSON(http.StatusInternalServerError, fail(r.Error()))
		return
	}
	c.JSON(http.StatusOK, r)
}

// status — GET /v1/driver/status. Snapshot of every supervised driver.
func (p *Provider) status(c *gin.Context) {
	c.JSON(http.StatusOK, core.Ok(p.svc.Status()))
}

// stopRequest selects which driver to stop. An empty body defaults to mlx.
type stopRequest struct {
	Runtime string `json:"runtime"`
}

// stop — POST /v1/driver/stop. Drains + terminates a driver.
func (p *Provider) stop(c *gin.Context) {
	var req stopRequest
	_ = c.ShouldBindJSON(&req) // empty body is valid — defaults to mlx
	r := p.svc.Stop(req.Runtime)
	if !r.OK {
		c.JSON(http.StatusNotFound, fail(r.Error()))
		return
	}
	c.JSON(http.StatusOK, r)
}

// fail renders a uniform error envelope so clients branch on OK like every
// other core/api response.
func fail(msg string) gin.H {
	return gin.H{"OK": false, "error": msg}
}
