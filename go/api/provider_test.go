// SPDX-License-Identifier: EUPL-1.2

package api

import (
	core "dappco.re/go"
	"net/http"
	"net/http/httptest"
	"testing"

	coreprovider "dappco.re/go/api/pkg/provider"
	"github.com/gin-gonic/gin"
)

func TestNewProvider_Good(t *testing.T) {
	p := NewProvider()
	if p == nil {
		t.Fatal("expected provider")
	}
	if New() == nil {
		t.Fatal("expected New alias to return provider")
	}

	var provider coreprovider.Provider = p
	if provider.Name() != "ai" {
		t.Fatalf("expected name %q, got %q", "ai", provider.Name())
	}
	if provider.BasePath() != "/v1" {
		t.Fatalf("expected base path %q, got %q", "/v1", provider.BasePath())
	}

	want := map[string]bool{
		http.MethodPost + " /embeddings/text":        false,
		http.MethodPost + " /embeddings/behavioural": false,
		http.MethodPost + " /score/content":          false,
		http.MethodPost + " /score/imprint":          false,
		http.MethodGet + " /score/:id":               false,
		http.MethodGet + " /health":                  false,
	}
	for _, desc := range p.Describe() {
		key := desc.Method + " " + desc.Path
		if _, ok := want[key]; ok {
			want[key] = true
		}
	}
	for key, seen := range want {
		if !seen {
			t.Fatalf("expected route description for %s", key)
		}
	}
}

func TestNewProvider_Bad(t *testing.T) {
	p := NewProvider()

	assertDoesNotPanic(t, func() {
		p.RegisterRoutes(nil)
	})
}

func TestNewProvider_Ugly(t *testing.T) {
	var p *AIProvider
	router := gin.New()

	assertDoesNotPanic(t, func() {
		p.RegisterRoutes(router.Group("/v1"))
	})
}

func assertDoesNotPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("expected no panic, got %v", r)
		}
	}()
	fn()
}

// --- AX-7 canonical triplets ---

func TestProvider_New_Good(t *core.T) {
	provider := New()
	name := provider.Name()
	basePath := provider.BasePath()

	core.AssertNotNil(t, provider)
	core.AssertEqual(t, "ai", name)
	core.AssertEqual(t, "/v1", basePath)
}

func TestProvider_New_Bad(t *core.T) {
	first := New()
	second := New()
	same := first == second

	core.AssertNotNil(t, first)
	core.AssertNotNil(t, second)
	core.AssertFalse(t, same)
}

func TestProvider_New_Ugly(t *core.T) {
	provider := New()
	descriptions := provider.Describe()
	got := len(descriptions)

	core.AssertTrue(t, got > 0)
	core.AssertEqual(t, "ai", provider.Name())
}

func TestProvider_NewProvider_Good(t *core.T) {
	provider := NewProvider()
	name := provider.Name()
	basePath := provider.BasePath()

	core.AssertNotNil(t, provider)
	core.AssertEqual(t, "ai", name)
	core.AssertEqual(t, "/v1", basePath)
}

func TestProvider_NewProvider_Bad(t *core.T) {
	first := NewProvider()
	second := NewProvider()
	same := first == second

	core.AssertNotNil(t, first)
	core.AssertNotNil(t, second)
	core.AssertFalse(t, same)
}

func TestProvider_NewProvider_Ugly(t *core.T) {
	provider := NewProvider()
	descriptions := provider.Describe()
	got := len(descriptions)

	core.AssertEqual(t, 6, got)
	core.AssertEqual(t, "ai", provider.Name())
}

func TestProvider_AIProvider_Name_Good(t *core.T) {
	provider := &AIProvider{}
	got := provider.Name()
	want := "ai"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestProvider_AIProvider_Name_Bad(t *core.T) {
	var provider *AIProvider
	got := provider.Name()
	want := "ai"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestProvider_AIProvider_Name_Ugly(t *core.T) {
	provider := NewProvider()
	got := provider.Name()
	again := provider.Name()

	core.AssertEqual(t, got, again)
	core.AssertEqual(t, "ai", got)
}

func TestProvider_AIProvider_BasePath_Good(t *core.T) {
	provider := &AIProvider{}
	got := provider.BasePath()
	want := "/v1"

	core.AssertEqual(t, want, got)
	core.AssertTrue(t, core.HasPrefix(got, "/"))
}

func TestProvider_AIProvider_BasePath_Bad(t *core.T) {
	var provider *AIProvider
	got := provider.BasePath()
	want := "/v1"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestProvider_AIProvider_BasePath_Ugly(t *core.T) {
	provider := NewProvider()
	got := provider.BasePath()
	again := provider.BasePath()

	core.AssertEqual(t, got, again)
	core.AssertEqual(t, "/v1", got)
}

func TestProvider_AIProvider_RegisterRoutes_Good(t *core.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	NewProvider().RegisterRoutes(router.Group("/v1"))

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	router.ServeHTTP(rec, req)
	core.AssertEqual(t, http.StatusOK, rec.Code)
}

func TestProvider_AIProvider_RegisterRoutes_Bad(t *core.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	var provider *AIProvider

	provider.RegisterRoutes(router.Group("/v1"))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	router.ServeHTTP(rec, req)
	core.AssertEqual(t, http.StatusNotFound, rec.Code)
}

func TestProvider_AIProvider_RegisterRoutes_Ugly(t *core.T) {
	provider := NewProvider()
	core.AssertNotPanics(t, func() {
		provider.RegisterRoutes(nil)
	})
	core.AssertEqual(t, "ai", provider.Name())
}

func TestProvider_AIProvider_Describe_Good(t *core.T) {
	provider := NewProvider()
	descriptions := provider.Describe()
	first := descriptions[0]

	core.AssertLen(t, descriptions, 6)
	core.AssertEqual(t, http.MethodPost, first.Method)
}

func TestProvider_AIProvider_Describe_Bad(t *core.T) {
	var provider *AIProvider
	descriptions := provider.Describe()
	got := len(descriptions)

	core.AssertEqual(t, 6, got)
	core.AssertEqual(t, "/health", descriptions[5].Path)
}

func TestProvider_AIProvider_Describe_Ugly(t *core.T) {
	provider := NewProvider()
	descriptions := provider.Describe()
	health := descriptions[5]

	core.AssertEqual(t, http.MethodGet, health.Method)
	core.AssertEqual(t, "/health", health.Path)
}
