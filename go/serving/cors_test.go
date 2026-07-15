// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func corsFixture(origins string) (http.Handler, *int) {
	hits := 0
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.WriteHeader(http.StatusTeapot) // distinctive: proves pass-through
	})
	return corsMiddleware(next, parseCORSOrigins(origins)), &hits
}

// TestCORS_Good pins the allowed-origin contract: the specific origin is
// echoed (with Vary: Origin), a preflight is answered 204 with the
// method/header/max-age set WITHOUT reaching the inner handler, and a normal
// request passes through with the headers stamped.
func TestCORS_Good(t *testing.T) {
	h, hits := corsFixture("http://localhost:4200,https://gui.example.com")

	// Preflight: answered at the middleware, never forwarded.
	pre := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
	pre.Header.Set("Origin", "http://localhost:4200")
	pre.Header.Set("Access-Control-Request-Method", "POST")
	pre.Header.Set("Access-Control-Request-Headers", "content-type")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, pre)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("preflight = %d, want 204", rec.Code)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "http://localhost:4200" {
		t.Fatalf("preflight allow-origin = %q, want the specific origin echoed", got)
	}
	if rec.Header().Get("Access-Control-Allow-Methods") == "" || rec.Header().Get("Access-Control-Max-Age") == "" {
		t.Fatalf("preflight missing method/max-age headers: %v", rec.Header())
	}
	if got := rec.Header().Get("Access-Control-Allow-Headers"); got != "content-type" {
		t.Fatalf("preflight allow-headers = %q, want the requested set echoed", got)
	}
	if *hits != 0 {
		t.Fatal("preflight leaked through to the inner handler")
	}

	// Actual request: headers stamped, handler reached.
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://gui.example.com")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusTeapot || *hits != 1 {
		t.Fatalf("request did not pass through: code %d hits %d", rec.Code, *hits)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "https://gui.example.com" {
		t.Fatalf("allow-origin = %q", got)
	}
	if got := rec.Header().Get("Vary"); got != "Origin" {
		t.Fatalf("Vary = %q, want Origin (caches must key on it)", got)
	}
}

// TestCORS_Bad pins the disallowed and non-browser paths: a disallowed origin
// gets NO CORS headers (the browser enforces the block; the server still
// answers), and a request with no Origin header is completely untouched.
func TestCORS_Bad(t *testing.T) {
	h, hits := corsFixture("http://localhost:4200")

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://evil.example.com")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusTeapot {
		t.Fatalf("disallowed origin must still be served: %d", rec.Code)
	}
	if rec.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatal("disallowed origin received CORS headers")
	}

	plain := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, plain)
	if rec.Header().Get("Access-Control-Allow-Origin") != "" || rec.Code != http.StatusTeapot {
		t.Fatalf("no-Origin request must be untouched: %d %v", rec.Code, rec.Header())
	}
	if *hits != 2 {
		t.Fatalf("hits = %d, want both requests forwarded", *hits)
	}
}

// TestCORS_Ugly pins the wildcard and the parse edges: * allows any origin as
// the literal * (no Vary needed), and empty/blank flag values disable the
// policy entirely.
func TestCORS_Ugly(t *testing.T) {
	h, _ := corsFixture("*")
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Origin", "http://anywhere.example")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Fatalf("wildcard allow-origin = %q, want *", got)
	}

	if parseCORSOrigins("") != nil || parseCORSOrigins("  ") != nil || parseCORSOrigins(" , ") != nil {
		t.Fatal("blank flag values must disable the policy")
	}
	p := parseCORSOrigins(" http://a.example , http://b.example ")
	if p == nil || !p.allows("http://a.example") || !p.allows("http://b.example") || p.allows("http://c.example") {
		t.Fatalf("comma parse with spaces broken: %+v", p)
	}
}
