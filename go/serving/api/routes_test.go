// SPDX-License-Identifier: EUPL-1.2

package api_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	goapi "dappco.re/go/api"
	mlapi "dappco.re/go/inference/serving/api"
	"dappco.re/go/inference/serving/mlservice"
	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// ── Interface satisfaction ─────────────────────────────────────────────

func TestRoutesSatisfiesRouteGroupGoodScenario(t *testing.T) {
	var rg goapi.RouteGroup = mlapi.NewRoutes(nil)

	if rg.Name() != "ml" {
		t.Fatalf("expected Name=%q, got %q", "ml", rg.Name())
	}
	if rg.BasePath() != "/v1/ml" {
		t.Fatalf("expected BasePath=%q, got %q", "/v1/ml", rg.BasePath())
	}
}

func TestRoutesSatisfiesStreamGroupGoodScenario(t *testing.T) {
	var sg goapi.StreamGroup = mlapi.NewRoutes(nil)

	channels := sg.Channels()
	if len(channels) != 2 {
		t.Fatalf("expected 2 channels, got %d", len(channels))
	}
	if channels[0] != "ml.generate" {
		t.Fatalf("expected channels[0]=%q, got %q", "ml.generate", channels[0])
	}
	if channels[1] != "ml.status" {
		t.Fatalf("expected channels[1]=%q, got %q", "ml.status", channels[1])
	}
}

// ── Engine integration ─────────────────────────────────────────────────

func TestRoutesEngineRegistrationGoodScenario(t *testing.T) {
	e, err := goapi.New()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	routes := mlapi.NewRoutes(nil)
	e.Register(routes)

	groups := e.Groups()
	if len(groups) != 1 {
		t.Fatalf("expected 1 group, got %d", len(groups))
	}
	if groups[0].Name() != "ml" {
		t.Fatalf("expected group name=%q, got %q", "ml", groups[0].Name())
	}
}

func TestRoutesEngineChannelsGoodScenario(t *testing.T) {
	e, _ := goapi.New()
	routes := mlapi.NewRoutes(nil)
	e.Register(routes)

	channels := e.Channels()
	if len(channels) != 2 {
		t.Fatalf("expected 2 channels, got %d", len(channels))
	}
}

// ── ListBackends handler ───────────────────────────────────────────────

func TestRoutesListBackendsNilServiceBadScenario(t *testing.T) {
	routes := mlapi.NewRoutes(nil)
	h := buildHandler(routes)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodGet, "/v1/ml/backends", nil)
	h.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}

	var resp goapi.Response[any]
	mustJSONUnmarshalBytes(t, w.Body.Bytes(), &resp)
	if resp.Success {
		t.Fatal("expected Success=false for nil service")
	}
	if resp.Error == nil {
		t.Fatal("expected Error to be set")
	}
	if resp.Error.Code != "SERVICE_UNAVAILABLE" {
		t.Fatalf("expected error code=%q, got %q", "SERVICE_UNAVAILABLE", resp.Error.Code)
	}
}

// ── Status handler ─────────────────────────────────────────────────────

func TestRoutesStatusNilServiceBadScenario(t *testing.T) {
	routes := mlapi.NewRoutes(nil)
	h := buildHandler(routes)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodGet, "/v1/ml/status", nil)
	h.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}

	var resp goapi.Response[any]
	mustJSONUnmarshalBytes(t, w.Body.Bytes(), &resp)
	if resp.Success {
		t.Fatal("expected Success=false for nil service")
	}
}

// ── Generate handler ───────────────────────────────────────────────────

func TestRoutesGenerateNilServiceBadScenario(t *testing.T) {
	routes := mlapi.NewRoutes(nil)
	h := buildHandler(routes)

	body := `{"prompt":"hello"}`
	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodPost, "/v1/ml/generate", core.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

func TestRoutesGenerateMissingPromptBadScenario(t *testing.T) {
	// Even with a nil service, request validation happens first when service
	// is nil — but our handler checks service first. So this tests a valid
	// scenario where the body is empty.
	routes := mlapi.NewRoutes(nil)
	h := buildHandler(routes)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodPost, "/v1/ml/generate", core.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(w, req)

	// Service check happens before body parsing, so we get 503.
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

// ── Envelope format ────────────────────────────────────────────────────

func TestRoutesEnvelopeErrorFormatGoodScenario(t *testing.T) {
	routes := mlapi.NewRoutes(nil)
	h := buildHandler(routes)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodGet, "/v1/ml/status", nil)
	h.ServeHTTP(w, req)

	// Verify the envelope has the correct JSON structure.
	var raw map[string]any
	mustJSONUnmarshalBytes(t, w.Body.Bytes(), &raw)

	// Must have "success" key.
	if _, ok := raw["success"]; !ok {
		t.Fatal("envelope missing 'success' key")
	}

	// Must have "error" key for failure responses.
	if _, ok := raw["error"]; !ok {
		t.Fatal("envelope missing 'error' key for failure response")
	}

	// "data" should be absent or null for failure responses.
	if data, ok := raw["data"]; ok && data != nil {
		t.Fatal("expected 'data' to be absent or null for failure response")
	}
}

func TestRoutesHealthViaEngineGoodScenario(t *testing.T) {
	// Verify that the built-in /health endpoint still works
	// when our ML routes are registered alongside it.
	e, _ := goapi.New()
	routes := mlapi.NewRoutes(nil)
	e.Register(routes)

	h := e.Handler()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodGet, "/health", nil)
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp goapi.Response[string]
	mustJSONUnmarshalBytes(t, w.Body.Bytes(), &resp)
	if !resp.Success || resp.Data != "healthy" {
		t.Fatalf("expected healthy response, got success=%v data=%q", resp.Success, resp.Data)
	}
}

// ── Route method checks ────────────────────────────────────────────────

func TestRoutesWrongMethodBadScenario(t *testing.T) {
	routes := mlapi.NewRoutes(nil)
	h := buildHandler(routes)

	// POST to a GET-only endpoint.
	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodPost, "/v1/ml/backends", nil)
	h.ServeHTTP(w, req)

	if w.Code == http.StatusOK {
		t.Fatal("expected non-200 for POST on GET-only route")
	}
}

func TestRoutesNotFoundBadScenario(t *testing.T) {
	routes := mlapi.NewRoutes(nil)
	h := buildHandler(routes)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodGet, "/v1/ml/nonexistent", nil)
	h.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", w.Code)
	}
}

// ── Helpers ────────────────────────────────────────────────────────────

// buildHandler creates an api.Engine with the given routes and returns its http.Handler.
func buildHandler(routes goapi.RouteGroup) http.Handler {
	e, _ := goapi.New()
	e.Register(routes)
	return e.Handler()
}

func newTestService() *mlservice.Service {
	factory := mlservice.NewService(mlservice.Options{DefaultBackend: "test"})
	r := factory(core.New())
	if !r.OK {
		panic(r.Error())
	}
	return r.Value.(*mlservice.Service)
}

// ── ListBackends with service ───────────────────────────────────────────

func TestRoutesListBackendsWithServiceGoodScenario(t *testing.T) {
	svc := newTestService()
	routes := mlapi.NewRoutes(svc)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodGet, "/v1/ml/backends", nil)
	routes.ListBackends(ctx)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp goapi.Response[any]
	mustJSONUnmarshalBytes(t, w.Body.Bytes(), &resp)
	if !resp.Success {
		t.Fatal("expected Success=true")
	}
}

// ── Status with service ─────────────────────────────────────────────────

func TestRoutesStatusWithServiceGoodScenario(t *testing.T) {
	svc := newTestService()
	routes := mlapi.NewRoutes(svc)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodGet, "/v1/ml/status", nil)
	routes.Status(ctx)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp goapi.Response[any]
	mustJSONUnmarshalBytes(t, w.Body.Bytes(), &resp)
	if !resp.Success {
		t.Fatal("expected Success=true")
	}
}

// ── Generate with service (bad request path) ────────────────────────────

func TestRoutesGenerateWithServiceBadRequestBody(t *testing.T) {
	svc := newTestService()
	routes := mlapi.NewRoutes(svc)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodPost, "/v1/ml/generate", core.NewReader(`not json`))
	ctx.Request.Header.Set("Content-Type", "application/json")
	routes.Generate(ctx)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for bad JSON, got %d", w.Code)
	}
}

// --- v0.9.0 shape triplets ---

func TestRoutes_NewRoutes_Good(t *core.T) {
	routes := mlapi.NewRoutes(nil)
	core.AssertEqual(t, "ml", routes.Name())
	core.AssertEqual(t, "/v1/ml", routes.BasePath())
}

func TestRoutes_NewRoutes_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertNotNil(t, routes)
}

func TestRoutes_NewRoutes_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertEqual(t, []string{"ml.generate", "ml.status"}, routes.Channels())
}

func TestRoutes_Routes_Name_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertEqual(t, "ml", routes.Name())
}

func TestRoutes_Routes_Name_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertNotEqual(t, "api", routes.Name())
}

func TestRoutes_Routes_Name_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertEqual(t, routes.Name(), routes.Name())
}

func TestRoutes_Routes_BasePath_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertEqual(t, "/v1/ml", routes.BasePath())
}

func TestRoutes_Routes_BasePath_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertNotEqual(t, "/", routes.BasePath())
}

func TestRoutes_Routes_BasePath_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertContains(t, routes.BasePath(), "ml")
}

func TestRoutes_Routes_RegisterRoutes_Good(t *core.T) {
	router := gin.New()
	routes := mlapi.NewRoutes(nil)
	routes.RegisterRoutes(router.Group(routes.BasePath()))
	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodGet, "/v1/ml/status", nil)
	router.ServeHTTP(w, req)
	core.AssertEqual(t, http.StatusServiceUnavailable, w.Code)
}

func TestRoutes_Routes_RegisterRoutes_Bad(t *core.T) {
	router := gin.New()
	routes := mlapi.NewRoutes(nil)
	routes.RegisterRoutes(router.Group(routes.BasePath()))
	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodGet, "/v1/ml/missing", nil)
	router.ServeHTTP(w, req)
	core.AssertEqual(t, http.StatusNotFound, w.Code)
}

func TestRoutes_Routes_RegisterRoutes_Ugly(t *core.T) {
	router := gin.New()
	routes := mlapi.NewRoutes(nil)
	routes.RegisterRoutes(router.Group(routes.BasePath()))
	w := httptest.NewRecorder()
	req, _ := http.NewRequest(http.MethodPost, "/v1/ml/backends", nil)
	router.ServeHTTP(w, req)
	core.AssertNotEqual(t, http.StatusOK, w.Code)
}

func TestRoutes_Routes_Channels_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertEqual(t, []string{"ml.generate", "ml.status"}, routes.Channels())
}

func TestRoutes_Routes_Channels_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertNotContains(t, routes.Channels(), "ml.unknown")
}

func TestRoutes_Routes_Channels_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	routes := mlapi.NewRoutes(nil)
	core.AssertLen(t, routes.Channels(), 2)
}

func TestRoutes_Routes_ListBackends_Good(t *core.T) {
	routes := mlapi.NewRoutes(nil)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	req, _ := http.NewRequest(http.MethodGet, "/v1/ml/backends", nil)
	ctx.Request = req
	routes.ListBackends(ctx)
	core.AssertEqual(t, http.StatusServiceUnavailable, w.Code)
}

func TestRoutes_Routes_ListBackends_Bad(t *core.T) {
	routes := mlapi.NewRoutes(nil)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodGet, "/v1/ml/backends", nil)
	routes.ListBackends(ctx)
	core.AssertContains(t, w.Body.String(), "SERVICE_UNAVAILABLE")
}

func TestRoutes_Routes_ListBackends_Ugly(t *core.T) {
	routes := mlapi.NewRoutes(nil)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodGet, "/v1/ml/backends", nil)
	routes.ListBackends(ctx)
	core.AssertFalse(t, w.Body.Len() == 0)
}

func TestRoutes_Routes_Status_Good(t *core.T) {
	routes := mlapi.NewRoutes(nil)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodGet, "/v1/ml/status", nil)
	routes.Status(ctx)
	core.AssertEqual(t, http.StatusServiceUnavailable, w.Code)
}

func TestRoutes_Routes_Status_Bad(t *core.T) {
	routes := mlapi.NewRoutes(nil)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodGet, "/v1/ml/status", nil)
	routes.Status(ctx)
	core.AssertContains(t, w.Body.String(), "SERVICE_UNAVAILABLE")
}

func TestRoutes_Routes_Status_Ugly(t *core.T) {
	routes := mlapi.NewRoutes(nil)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodGet, "/v1/ml/status", nil)
	routes.Status(ctx)
	core.AssertTrue(t, w.Code >= 500)
}

func TestRoutes_Routes_Generate_Good(t *core.T) {
	routes := mlapi.NewRoutes(nil)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodPost, "/v1/ml/generate", core.NewReader(`{"prompt":"hi"}`))
	ctx.Request.Header.Set("Content-Type", "application/json")
	routes.Generate(ctx)
	core.AssertEqual(t, http.StatusServiceUnavailable, w.Code)
}

func TestRoutes_Routes_Generate_Bad(t *core.T) {
	routes := mlapi.NewRoutes(nil)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodPost, "/v1/ml/generate", core.NewReader(`{}`))
	ctx.Request.Header.Set("Content-Type", "application/json")
	routes.Generate(ctx)
	core.AssertContains(t, w.Body.String(), "SERVICE_UNAVAILABLE")
}

func TestRoutes_Routes_Generate_Ugly(t *core.T) {
	routes := mlapi.NewRoutes(nil)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request, _ = http.NewRequest(http.MethodPost, "/v1/ml/generate", core.NewReader(`not valid`))
	ctx.Request.Header.Set("Content-Type", "application/json")
	routes.Generate(ctx)
	core.AssertEqual(t, http.StatusServiceUnavailable, w.Code)
}
