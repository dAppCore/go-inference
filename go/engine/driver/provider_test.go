// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	core "dappco.re/go"
	coreprovider "dappco.re/go/api/pkg/provider"
	"github.com/gin-gonic/gin"
)

// --- NewProvider / Name / BasePath ---

func TestProvider_NewProvider_Good(t *testing.T) {
	svc := &Service{}
	p := NewProvider(svc)
	if p == nil || p.svc != svc {
		t.Fatalf("NewProvider = %+v, want it wrapping the given Service", p)
	}
}

// TestProvider_NewProvider_Bad proves NewProvider always allocates a fresh
// wrapper — two calls never alias the same *Provider.
func TestProvider_NewProvider_Bad(t *testing.T) {
	first := NewProvider(&Service{})
	second := NewProvider(&Service{})
	if first == second {
		t.Fatal("NewProvider returned the same *Provider instance across two calls")
	}
}

// TestProvider_NewProvider_Ugly covers the intended aliasing edge: two
// providers wrapping the SAME underlying Service must share it (the wrapper
// is thin — it never copies the Service it's given).
func TestProvider_NewProvider_Ugly(t *testing.T) {
	svc := &Service{}
	first := NewProvider(svc)
	second := NewProvider(svc)
	if first.svc != second.svc {
		t.Fatal("NewProvider wrapping the same Service produced providers pointing at different services")
	}
}

func TestProvider_Name_Good(t *testing.T) {
	p := NewProvider(nil)
	if got := p.Name(); got != "driver" {
		t.Fatalf("Name() = %q, want %q", got, "driver")
	}
}

// TestProvider_Name_Bad covers a nil receiver — Name() never touches p, so it
// must still answer the constant rather than panic.
func TestProvider_Name_Bad(t *testing.T) {
	var p *Provider
	if got := p.Name(); got != "driver" {
		t.Fatalf("Name() on a nil *Provider = %q, want %q", got, "driver")
	}
}

// TestProvider_Name_Ugly proves Name() is idempotent regardless of the
// wrapped Service's state.
func TestProvider_Name_Ugly(t *testing.T) {
	p := NewProvider(&Service{served: map[string]*Served{}})
	first, second := p.Name(), p.Name()
	if first != second || first != "driver" {
		t.Fatalf("Name() = (%q, %q), want both calls to answer %q", first, second, "driver")
	}
}

func TestProvider_BasePath_Good(t *testing.T) {
	p := NewProvider(nil)
	if got := p.BasePath(); got != "/v1/driver" {
		t.Fatalf("BasePath() = %q, want %q", got, "/v1/driver")
	}
}

// TestProvider_BasePath_Bad covers a nil receiver — BasePath() never touches
// p, so it must still answer the constant rather than panic.
func TestProvider_BasePath_Bad(t *testing.T) {
	var p *Provider
	if got := p.BasePath(); got != "/v1/driver" {
		t.Fatalf("BasePath() on a nil *Provider = %q, want %q", got, "/v1/driver")
	}
}

// TestProvider_BasePath_Ugly proves BasePath() is idempotent regardless of
// the wrapped Service's state.
func TestProvider_BasePath_Ugly(t *testing.T) {
	p := NewProvider(&Service{served: map[string]*Served{}})
	first, second := p.BasePath(), p.BasePath()
	if first != second || first != "/v1/driver" {
		t.Fatalf("BasePath() = (%q, %q), want both calls to answer %q", first, second, "/v1/driver")
	}
}

// --- RegisterRoutes ---

func TestProvider_RegisterRoutes_Good(t *testing.T) {
	p := NewProvider(&Service{served: map[string]*Served{}})
	engine := gin.New()
	grp := engine.Group(p.BasePath())
	p.RegisterRoutes(grp)

	want := map[string]bool{
		http.MethodGet + " /v1/driver/models": false,
		http.MethodPost + " /v1/driver/serve": false,
		http.MethodGet + " /v1/driver/status": false,
		http.MethodPost + " /v1/driver/stop":  false,
	}
	for _, route := range engine.Routes() {
		key := route.Method + " " + route.Path
		if _, ok := want[key]; ok {
			want[key] = true
		}
	}
	for key, seen := range want {
		if !seen {
			t.Fatalf("RegisterRoutes did not register %s", key)
		}
	}
}

// TestProvider_RegisterRoutes_Bad covers the nil-receiver guard — a nil
// Provider must be a safe no-op, not a panic.
func TestProvider_RegisterRoutes_Bad(t *testing.T) {
	engine := gin.New()
	grp := engine.Group("/v1/driver")
	var p *Provider
	p.RegisterRoutes(grp)
	if len(engine.Routes()) != 0 {
		t.Fatalf("RegisterRoutes on a nil provider registered %d routes, want 0", len(engine.Routes()))
	}
}

// TestProvider_RegisterRoutes_Ugly covers the nil-group guard on an
// otherwise-valid provider — a nil *gin.RouterGroup must be a safe no-op too.
func TestProvider_RegisterRoutes_Ugly(t *testing.T) {
	p := NewProvider(&Service{served: map[string]*Served{}})
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("RegisterRoutes(nil) panicked: %v", r)
			}
		}()
		p.RegisterRoutes(nil)
	}()
	if got := p.Name(); got != "driver" {
		t.Fatalf("Name() after RegisterRoutes(nil) = %q, want %q", got, "driver")
	}
}

// --- models ---

func TestProvider_Models_Good(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	modelsDir := core.PathJoin(home, "Lethean", "lem", "models")
	if r := core.MkdirAll(modelsDir, 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(modelsDir, "gemma"), []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed: %v", r.Value)
	}
	p := NewProvider(&Service{})
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/driver/models", nil)

	p.models(c)

	if w.Code != http.StatusOK {
		t.Fatalf("models status = %d, want 200; body=%s", w.Code, w.Body.String())
	}
	if !core.Contains(w.Body.String(), "gemma") {
		t.Fatalf("models body = %s, want it listing the seeded model", w.Body.String())
	}
}

func TestProvider_Models_Bad(t *testing.T) {
	t.Setenv("HOME", "")
	p := NewProvider(&Service{})
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/driver/models", nil)

	p.models(c)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("models status = %d, want 500 when the home dir can't resolve", w.Code)
	}
}

// --- serve ---

func TestProvider_Serve_Good(t *testing.T) {
	newHealthyDriver(t, RuntimeMLX)
	addr := newHealthServer(t, true)
	proc := benchProcSvc(t)
	svc := NewService(proc, nil)
	t.Cleanup(func() { svc.Stop(RuntimeMLX) })
	p := NewProvider(svc)

	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	body := core.Sprintf(`{"runtime":"mlx","addr":%q,"model":"org/model"}`, addr)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/driver/serve", bytes.NewReader([]byte(body)))
	c.Request.Header.Set("Content-Type", "application/json")

	p.serve(c)

	if w.Code != http.StatusOK {
		t.Fatalf("serve status = %d, want 200; body=%s", w.Code, w.Body.String())
	}
	if !core.Contains(w.Body.String(), "org/model") {
		t.Fatalf("serve body = %s, want the served model reflected", w.Body.String())
	}
}

func TestProvider_Serve_Bad(t *testing.T) {
	p := NewProvider(&Service{served: map[string]*Served{}})
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/driver/serve", bytes.NewReader([]byte("not json")))
	c.Request.Header.Set("Content-Type", "application/json")

	p.serve(c)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("serve status = %d, want 400 over an invalid body", w.Code)
	}
}

// TestProvider_Serve_Ugly covers a well-formed request that Serve itself
// refuses (an unknown runtime) — the handler must translate that into 500,
// not treat it as a binding failure.
func TestProvider_Serve_Ugly(t *testing.T) {
	p := NewProvider(&Service{served: map[string]*Served{}, everReady: map[string]bool{}, restartLog: map[string][]time.Time{}})
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/driver/serve", bytes.NewReader([]byte(`{"runtime":"bogus"}`)))
	c.Request.Header.Set("Content-Type", "application/json")

	p.serve(c)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("serve status = %d, want 500 when Serve itself refuses (unknown runtime)", w.Code)
	}
}

// --- status ---

func TestProvider_Status_Good(t *testing.T) {
	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	svc := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: pid, Running: true, Addr: "127.0.0.1:1"},
	}}
	p := NewProvider(svc)
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/driver/status", nil)

	p.status(c)

	if w.Code != http.StatusOK {
		t.Fatalf("status code = %d, want 200", w.Code)
	}
	if !core.Contains(w.Body.String(), RuntimeMLX) {
		t.Fatalf("status body = %s, want the tracked runtime reflected", w.Body.String())
	}
}

// --- stop ---

func TestProvider_Stop_Good(t *testing.T) {
	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	svc := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: pid, Running: true},
	}, everReady: map[string]bool{}, restartLog: map[string][]time.Time{}}
	p := NewProvider(svc)
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/driver/stop", bytes.NewReader([]byte(`{}`))) // empty body defaults to mlx

	p.stop(c)

	if w.Code != http.StatusOK {
		t.Fatalf("stop status = %d, want 200; body=%s", w.Code, w.Body.String())
	}
}

func TestProvider_Stop_Bad(t *testing.T) {
	p := NewProvider(&Service{served: map[string]*Served{}})
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/driver/stop", bytes.NewReader([]byte(`{}`)))

	p.stop(c)

	if w.Code != http.StatusNotFound {
		t.Fatalf("stop status = %d, want 404 with nothing served", w.Code)
	}
}

// --- fail ---

func TestProvider_Fail_Good(t *testing.T) {
	got := fail("boom")
	if got["OK"] != false || got["error"] != "boom" {
		t.Fatalf("fail(\"boom\") = %+v, want {OK:false, error:\"boom\"}", got)
	}
}

// --- Describe ---

// TestProvider_Describe_Good verifies the driver-orchestration route group is
// OpenAPI-describable and surfaces every route it registers, so the core/api
// engine mounts it into the generated spec (and the SDK generators emit a typed
// client for it).
func TestProvider_Describe_Good(t *testing.T) {
	var _ coreprovider.Describable = (*Provider)(nil)

	p := NewProvider(nil)
	want := map[string]bool{
		http.MethodGet + " /models": false,
		http.MethodPost + " /serve": false,
		http.MethodGet + " /status": false,
		http.MethodPost + " /stop":  false,
	}
	descriptions := p.Describe()
	if len(descriptions) == 0 {
		t.Fatal("Describe returned no route descriptions")
	}
	for _, desc := range descriptions {
		if _, ok := want[desc.Method+" "+desc.Path]; ok {
			want[desc.Method+" "+desc.Path] = true
		}
	}
	for key, seen := range want {
		if !seen {
			t.Fatalf("expected route description for %s", key)
		}
	}
}

// TestProvider_Describe_Bad covers a nil receiver — Describe() builds its
// descriptions from static data only, never touching p, so it must still
// answer the full route list rather than panic.
func TestProvider_Describe_Bad(t *testing.T) {
	var p *Provider
	descriptions := p.Describe()
	if len(descriptions) != 4 {
		t.Fatalf("Describe() on a nil *Provider returned %d descriptions, want 4", len(descriptions))
	}
}

// TestProvider_Describe_Ugly checks the /serve route's request-body schema
// carries the full ServeRequest field set — the detail the SDK generators
// rely on to emit a typed client, not just the route list.
func TestProvider_Describe_Ugly(t *testing.T) {
	p := NewProvider(nil)
	for _, desc := range p.Describe() {
		if desc.Method != http.MethodPost || desc.Path != "/serve" {
			continue
		}
		body := desc.RequestBody
		if body == nil {
			t.Fatal("/serve RequestBody is nil, want a map schema")
		}
		props, ok := body["properties"].(map[string]any)
		if !ok {
			t.Fatalf("/serve RequestBody properties = %T, want a map", body["properties"])
		}
		for _, field := range []string{"model", "profile", "runtime", "addr", "context", "noAutoProfile"} {
			if _, ok := props[field]; !ok {
				t.Fatalf("/serve RequestBody schema missing field %q", field)
			}
		}
		return
	}
	t.Fatal("Describe() did not include the /serve route")
}
