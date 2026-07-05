// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	core "dappco.re/go"
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

func TestProvider_Name_Good(t *testing.T) {
	p := NewProvider(nil)
	if got := p.Name(); got != "driver" {
		t.Fatalf("Name() = %q, want %q", got, "driver")
	}
}

func TestProvider_BasePath_Good(t *testing.T) {
	p := NewProvider(nil)
	if got := p.BasePath(); got != "/v1/driver" {
		t.Fatalf("BasePath() = %q, want %q", got, "/v1/driver")
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

// --- models ---

func TestProvider_Models_Good(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	modelsDir := core.PathJoin(home, "Lethean", "data", "models")
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
