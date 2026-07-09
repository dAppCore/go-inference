// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/compat"
	openai "dappco.re/go/inference/serving/provider/openai"
)

// freeListenAddr returns a "127.0.0.1:port" address that is free at the
// moment of the call — the probe listener is opened then immediately closed
// so Serve/RunServe can bind the same port. The TOCTOU gap is vanishingly
// small (the same trade-off httptest.NewServer makes internally).
func freeListenAddr(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("freeListenAddr: %v", err)
	}
	addr := l.Addr().String()
	l.Close()
	return addr
}

// waitForHTTPUp polls url until an HTTP GET succeeds or 2s pass, proving the
// listener is actually accepting connections rather than racing the boot
// goroutine with a fixed sleep.
func waitForHTTPUp(t *testing.T, url string) *http.Response {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	var lastErr error
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			return resp
		}
		lastErr = err
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("server never came up at %s: %v", url, lastErr)
	return nil
}

// noModelResolver is a Resolver that always reports no model loaded — enough
// for the /v1/health route the Serve/RunServe tests poll, without pulling in
// a real engine.
var noModelResolver = openai.ResolverFunc(func(context.Context, string) (inference.TextModel, error) {
	return nil, core.NewError("no model loaded")
})

// ---------------------------------------------------------------------------
// ServeOption setters — each is a bare closure over serveConfig, so these
// exercise the closures directly rather than through a full Serve() boot.
// ---------------------------------------------------------------------------

func TestServe_WithReadHeaderTimeout_Good(t *testing.T) {
	var cfg serveConfig
	WithReadHeaderTimeout(5 * time.Second)(&cfg)
	if cfg.readHeaderTimeout != 5*time.Second {
		t.Fatalf("readHeaderTimeout = %v, want 5s", cfg.readHeaderTimeout)
	}
}

func TestServe_WithReadHeaderTimeout_Bad(t *testing.T) {
	cfg := serveConfig{readHeaderTimeout: 30 * time.Second}
	WithReadHeaderTimeout(0)(&cfg)
	if cfg.readHeaderTimeout != 0 {
		t.Fatalf("readHeaderTimeout = %v, want 0 (an explicit zero overwrites the prior value)", cfg.readHeaderTimeout)
	}
}

func TestServe_WithReadHeaderTimeout_Ugly(t *testing.T) {
	var cfg serveConfig
	WithReadHeaderTimeout(-1 * time.Second)(&cfg)
	if cfg.readHeaderTimeout != -1*time.Second {
		t.Fatalf("readHeaderTimeout = %v, want -1s — the setter does not validate", cfg.readHeaderTimeout)
	}
}

func TestServe_WithWriteTimeout_Good(t *testing.T) {
	var cfg serveConfig
	WithWriteTimeout(2 * time.Minute)(&cfg)
	if cfg.writeTimeout != 2*time.Minute {
		t.Fatalf("writeTimeout = %v, want 2m", cfg.writeTimeout)
	}
}

func TestServe_WithWriteTimeout_Bad(t *testing.T) {
	cfg := serveConfig{writeTimeout: 5 * time.Minute}
	WithWriteTimeout(0)(&cfg)
	if cfg.writeTimeout != 0 {
		t.Fatalf("writeTimeout = %v, want 0 (an explicit zero overwrites the prior value)", cfg.writeTimeout)
	}
}

func TestServe_WithWriteTimeout_Ugly(t *testing.T) {
	var cfg serveConfig
	WithWriteTimeout(24 * time.Hour)(&cfg)
	if cfg.writeTimeout != 24*time.Hour {
		t.Fatalf("writeTimeout = %v, want 24h — the setter accepts any duration, however impractical", cfg.writeTimeout)
	}
}

func TestServe_WithShutdownTimeout_Good(t *testing.T) {
	var cfg serveConfig
	WithShutdownTimeout(15 * time.Second)(&cfg)
	if cfg.shutdownTimeout != 15*time.Second {
		t.Fatalf("shutdownTimeout = %v, want 15s", cfg.shutdownTimeout)
	}
}

func TestServe_WithShutdownTimeout_Bad(t *testing.T) {
	cfg := serveConfig{shutdownTimeout: 10 * time.Second}
	WithShutdownTimeout(0)(&cfg)
	if cfg.shutdownTimeout != 0 {
		t.Fatalf("shutdownTimeout = %v, want 0 (an explicit zero overwrites the prior value)", cfg.shutdownTimeout)
	}
}

func TestServe_WithShutdownTimeout_Ugly(t *testing.T) {
	var cfg serveConfig
	WithShutdownTimeout(-5 * time.Second)(&cfg)
	if cfg.shutdownTimeout != -5*time.Second {
		t.Fatalf("shutdownTimeout = %v, want -5s — the setter does not validate", cfg.shutdownTimeout)
	}
}

func TestServe_WithAdminToken_Good(t *testing.T) {
	var cfg serveConfig
	WithAdminToken("secret")(&cfg)
	if cfg.adminToken != "secret" {
		t.Fatalf("adminToken = %q, want %q", cfg.adminToken, "secret")
	}
}

func TestServe_WithAdminToken_Bad(t *testing.T) {
	cfg := serveConfig{adminToken: "secret"}
	WithAdminToken("")(&cfg)
	if cfg.adminToken != "" {
		t.Fatalf("adminToken = %q, want \"\" — an explicit empty token removes the wall", cfg.adminToken)
	}
}

func TestServe_WithAdminToken_Ugly(t *testing.T) {
	var cfg serveConfig
	odd := "tok\x00with\nembedded\tbytes"
	WithAdminToken(odd)(&cfg)
	if cfg.adminToken != odd {
		t.Fatalf("adminToken = %q, want the verbatim odd value %q — no sanitisation", cfg.adminToken, odd)
	}
}

func TestServe_WithAdminHandler_Good(t *testing.T) {
	var cfg serveConfig
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusTeapot) })
	WithAdminHandler(h)(&cfg)
	rec := httptest.NewRecorder()
	cfg.adminHandler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusTeapot {
		t.Fatalf("adminHandler status = %d, want %d", rec.Code, http.StatusTeapot)
	}
}

func TestServe_WithAdminHandler_Bad(t *testing.T) {
	cfg := serveConfig{adminHandler: http.NewServeMux()}
	WithAdminHandler(nil)(&cfg)
	if cfg.adminHandler != nil {
		t.Fatal("nil should overwrite a previously configured admin handler")
	}
}

func TestServe_WithAdminHandler_Ugly(t *testing.T) {
	var cfg serveConfig
	first := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	second := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusAccepted) })
	WithAdminHandler(first)(&cfg)
	WithAdminHandler(second)(&cfg) // last option wins
	rec := httptest.NewRecorder()
	cfg.adminHandler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusAccepted {
		t.Fatalf("adminHandler status = %d, want %d (the second WithAdminHandler should win)", rec.Code, http.StatusAccepted)
	}
}

func TestServe_WithAdminConfig_Good(t *testing.T) {
	var cfg serveConfig
	admin := compat.AdminConfig{Models: func() []string { return []string{"m1"} }}
	WithAdminConfig(admin)(&cfg)
	got := cfg.admin.Models()
	if len(got) != 1 || got[0] != "m1" {
		t.Fatalf("admin.Models() = %v, want [m1]", got)
	}
}

func TestServe_WithAdminConfig_Bad(t *testing.T) {
	cfg := serveConfig{admin: compat.AdminConfig{Models: func() []string { return []string{"stale"} }}}
	WithAdminConfig(compat.AdminConfig{})(&cfg)
	if cfg.admin.Models != nil {
		t.Fatal("a zero-value AdminConfig should clear the previously configured callbacks")
	}
}

func TestServe_WithAdminConfig_Ugly(t *testing.T) {
	var cfg serveConfig
	WithAdminConfig(compat.AdminConfig{Models: func() []string { return []string{"first"} }})(&cfg)
	WithAdminConfig(compat.AdminConfig{Models: func() []string { return []string{"second"} }})(&cfg)
	got := cfg.admin.Models()
	if len(got) != 1 || got[0] != "second" {
		t.Fatalf("admin.Models() = %v, want [second] (the second WithAdminConfig should win)", got)
	}
}

func TestServe_WithAuditLog_Good(t *testing.T) {
	var cfg serveConfig
	buf := core.NewBuffer()
	WithAuditLog(buf)(&cfg)
	core.Print(cfg.audit, "hello")
	if !core.Contains(buf.String(), "hello") {
		t.Fatalf("audit buffer = %q, want it to contain %q", buf.String(), "hello")
	}
}

func TestServe_WithAuditLog_Bad(t *testing.T) {
	cfg := serveConfig{audit: core.NewBuffer()}
	WithAuditLog(nil)(&cfg)
	if cfg.audit != nil {
		t.Fatal("nil should silence a previously configured audit log")
	}
}

func TestServe_WithAuditLog_Ugly(t *testing.T) {
	var cfg serveConfig
	first := core.NewBuffer()
	second := core.NewBuffer()
	WithAuditLog(first)(&cfg)
	WithAuditLog(second)(&cfg)
	if cfg.audit != io.Writer(second) {
		t.Fatal("the second WithAuditLog should win over the first")
	}
}

// ---------------------------------------------------------------------------
// Serve — full end-to-end boot / shutdown against a real listener.
// ---------------------------------------------------------------------------

func TestServe_Serve_Good(t *testing.T) {
	addr := freeListenAddr(t)
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- Serve(ctx, addr, noModelResolver) }()

	resp := waitForHTTPUp(t, "http://"+addr+"/v1/health")
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("health status = %d, want 200", resp.StatusCode)
	}

	cancel()
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("Serve returned %v after context cancel, want nil", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Serve did not shut down within 3s of context cancel")
	}
}

func TestServe_Serve_Bad(t *testing.T) {
	occupied, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("occupy a port: %v", err)
	}
	defer occupied.Close()
	addr := occupied.Addr().String()

	if err := Serve(context.Background(), addr, noModelResolver); err == nil {
		t.Fatal("Serve should fail to bind an address already in use")
	}
}

func TestServe_Serve_Ugly(t *testing.T) {
	addr := freeListenAddr(t)
	ctx, cancel := context.WithCancel(context.Background())
	adminMux := http.NewServeMux()
	adminMux.HandleFunc("/v1/admin/ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	errCh := make(chan error, 1)
	go func() {
		errCh <- Serve(ctx, addr, noModelResolver, WithAdminToken("s3cret"), WithAdminHandler(adminMux))
	}()

	waitForHTTPUp(t, "http://"+addr+"/v1/health").Body.Close()

	// Admin path without a token is denied.
	noAuth, err := http.Get("http://" + addr + "/v1/admin/ping")
	if err != nil {
		t.Fatalf("GET admin (no token): %v", err)
	}
	noAuth.Body.Close()
	if noAuth.StatusCode != http.StatusUnauthorized {
		t.Fatalf("admin without token = %d, want 401", noAuth.StatusCode)
	}

	// Admin path with the configured Bearer token succeeds; composition order
	// never leaves the admin subtree unauthenticated.
	req, _ := http.NewRequest(http.MethodGet, "http://"+addr+"/v1/admin/ping", nil)
	req.Header.Set("Authorization", "Bearer s3cret")
	authed, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET admin (with token): %v", err)
	}
	authed.Body.Close()
	if authed.StatusCode != http.StatusOK {
		t.Fatalf("admin with token = %d, want 200", authed.StatusCode)
	}

	cancel()
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("Serve returned %v after context cancel, want nil", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Serve did not shut down within 3s of context cancel")
	}
}

// ---------------------------------------------------------------------------
// RunServe — the full cmd/lem composition, model-less so no engine is needed.
// ---------------------------------------------------------------------------

func TestServe_RunServe_Good(t *testing.T) {
	addr := freeListenAddr(t)
	log := core.NewBuffer()
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- RunServe(ctx, ServeConfig{Addr: addr, Log: log}) }()

	resp := waitForHTTPUp(t, "http://"+addr+"/v1/health")
	resp.Body.Close()

	cancel()
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("RunServe returned %v after context cancel, want nil", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("RunServe did not shut down within 3s of context cancel")
	}

	if !core.Contains(log.String(), "starting model-less") {
		t.Fatalf("boot log = %q, want a model-less notice", log.String())
	}
}

func TestServe_RunServe_Bad(t *testing.T) {
	occupied, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("occupy a port: %v", err)
	}
	defer occupied.Close()
	addr := occupied.Addr().String()

	if err := RunServe(context.Background(), ServeConfig{Addr: addr}); err == nil {
		t.Fatal("RunServe should surface a listen failure")
	}
}

func TestServe_RunServe_Ugly(t *testing.T) {
	addr := freeListenAddr(t)
	log := core.NewBuffer()
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- RunServe(ctx, ServeConfig{
			Addr: addr,
			Log:  log,
			// No EnableContinuity injected — proves the degrade-to-stateless
			// edge rather than the happy no-continuity-requested path.
			StateConversations: true,
		})
	}()

	resp := waitForHTTPUp(t, "http://"+addr+"/v1/health")
	resp.Body.Close()

	cancel()
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("RunServe returned %v after context cancel, want nil", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("RunServe did not shut down within 3s of context cancel")
	}

	if !core.Contains(log.String(), "conversation continuity unavailable") {
		t.Fatalf("boot log = %q, want a continuity-unavailable notice", log.String())
	}
}
