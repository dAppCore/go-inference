// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
)

// TestAdminTokenPath_Good pins the canonical token location under $HOME — the
// same path lthn-mlx used, so cmd/lem and any launchd job reading the token
// file agree on where it lives.
func TestAdminTokenPath_Good(t *testing.T) {
	t.Setenv("HOME", "/home/tester")
	if got, want := AdminTokenPath(), core.PathJoin("/home/tester", "Lethean", "lem", "admin.token"); got != want {
		t.Fatalf("AdminTokenPath = %q, want %q", got, want)
	}
}

// TestGenerateAdminToken_Good pins the token shape: the secret-scanner prefix
// plus 256 bits of entropy, and proves two calls never collide.
func TestGenerateAdminToken_Good(t *testing.T) {
	tok, err := GenerateAdminToken()
	if err != nil {
		t.Fatalf("GenerateAdminToken: %v", err)
	}
	if !core.HasPrefix(tok, adminTokenPrefix) {
		t.Fatalf("token = %q, want %s prefix", tok, adminTokenPrefix)
	}
	other, err := GenerateAdminToken()
	if err != nil {
		t.Fatalf("GenerateAdminToken (2nd): %v", err)
	}
	if tok == other {
		t.Fatal("two generated tokens collided")
	}
}

// TestLoadAdminToken_Missing_Bad proves a missing token file reports
// exists=false with no error — the caller's "generate one" signal.
func TestLoadAdminToken_Missing_Bad(t *testing.T) {
	dir := t.TempDir()
	tok, exists, err := loadAdminToken(core.PathJoin(dir, "admin.token"))
	if err != nil || exists || tok != "" {
		t.Fatalf("loadAdminToken(missing) = (%q, %v, %v), want (\"\", false, nil)", tok, exists, err)
	}
}

// TestLoadAdminToken_Empty_Bad proves a blank (or whitespace-only) token file
// is treated as absent — never hands back an empty Bearer secret.
func TestLoadAdminToken_Empty_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "admin.token")
	if r := core.WriteFile(path, []byte("  \n"), 0o600); !r.OK {
		t.Fatalf("seed empty token: %v", r.Value)
	}
	tok, exists, err := loadAdminToken(path)
	if err != nil || exists || tok != "" {
		t.Fatalf("loadAdminToken(empty) = (%q, %v, %v), want (\"\", false, nil)", tok, exists, err)
	}
}

// TestLoadAdminToken_Valid_Good proves a real token file is read back trimmed.
func TestLoadAdminToken_Valid_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "admin.token")
	if r := core.WriteFile(path, []byte("lthn-mlx_abc123\n"), 0o600); !r.OK {
		t.Fatalf("seed token: %v", r.Value)
	}
	tok, exists, err := loadAdminToken(path)
	if err != nil || !exists || tok != "lthn-mlx_abc123" {
		t.Fatalf("loadAdminToken(valid) = (%q, %v, %v), want (\"lthn-mlx_abc123\", true, nil)", tok, exists, err)
	}
}

// TestWriteAdminToken_Good proves the token lands at 0600 with a trailing
// newline and that missing parent dirs are created.
func TestWriteAdminToken_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "nested", "admin.token")
	if err := WriteAdminToken(path, "tok-value"); err != nil {
		t.Fatalf("WriteAdminToken: %v", err)
	}
	res := core.ReadFile(path)
	if !res.OK {
		t.Fatalf("read back: %v", res.Value)
	}
	if got, want := string(res.Value.([]byte)), "tok-value\n"; got != want {
		t.Fatalf("token file = %q, want %q", got, want)
	}
	stat := core.Stat(path)
	if !stat.OK {
		t.Fatalf("stat: %v", stat.Value)
	}
	if info, ok := stat.Value.(interface{ Mode() core.FileMode }); ok {
		if perm := info.Mode().Perm(); perm != 0o600 {
			t.Fatalf("token file mode = %o, want 0600", perm)
		}
	} else {
		t.Fatal("core.Stat result does not expose Mode()")
	}
}

// TestWriteAdminToken_MkdirFails_Bad proves a parent-dir create failure (a
// path component already exists as a plain file) is surfaced rather than
// silently swallowed — the fail-closed checkpoint the doc comment promises.
func TestWriteAdminToken_MkdirFails_Bad(t *testing.T) {
	dir := t.TempDir()
	blocker := core.PathJoin(dir, "blocker")
	if r := core.WriteFile(blocker, []byte("not a dir"), 0o644); !r.OK {
		t.Fatalf("seed blocker file: %v", r.Value)
	}
	path := core.PathJoin(blocker, "child", "admin.token")
	if err := WriteAdminToken(path, "tok"); err == nil {
		t.Fatal("WriteAdminToken should fail when a parent path component is a file")
	}
}

// TestWriteAdminToken_WriteFails_Bad proves a write failure (the target path
// is itself a directory) is surfaced.
func TestWriteAdminToken_WriteFails_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "admin.token")
	if r := core.MkdirAll(path, 0o755); !r.OK {
		t.Fatalf("seed directory at token path: %v", r.Value)
	}
	if err := WriteAdminToken(path, "tok"); err == nil {
		t.Fatal("WriteAdminToken should fail when the token path is a directory")
	}
}

// TestEnsureAdminToken_GeneratesFresh_Good proves a first boot (no token file
// yet) generates, persists, and reports generated=true.
func TestEnsureAdminToken_GeneratesFresh_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "admin.token")
	tok, generated, err := EnsureAdminToken(path)
	if err != nil {
		t.Fatalf("EnsureAdminToken: %v", err)
	}
	if !generated {
		t.Fatal("first-boot EnsureAdminToken should report generated=true")
	}
	if !core.HasPrefix(tok, adminTokenPrefix) {
		t.Fatalf("token = %q, want %s prefix", tok, adminTokenPrefix)
	}
	res := core.ReadFile(path)
	if !res.OK || core.Trim(string(res.Value.([]byte))) != tok {
		t.Fatalf("persisted token does not match returned token (res.OK=%v)", res.OK)
	}
}

// TestEnsureAdminToken_LoadsExisting_Good proves a second boot reuses the
// already-persisted token rather than minting a new one.
func TestEnsureAdminToken_LoadsExisting_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "admin.token")
	if err := WriteAdminToken(path, "lthn-mlx_preexisting"); err != nil {
		t.Fatalf("seed token: %v", err)
	}
	tok, generated, err := EnsureAdminToken(path)
	if err != nil {
		t.Fatalf("EnsureAdminToken: %v", err)
	}
	if generated {
		t.Fatal("EnsureAdminToken should report generated=false for a pre-existing token")
	}
	if tok != "lthn-mlx_preexisting" {
		t.Fatalf("token = %q, want the pre-existing value", tok)
	}
}

// TestRequireBearerOnAdmin_NonAdminPath_Good proves paths outside /v1/admin/
// pass through unauthenticated — the localhost/tunnel-trust model still
// applies to inference.
func TestRequireBearerOnAdmin_NonAdminPath_Good(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	})
	h := RequireBearerOnAdmin(next, "secret", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil))
	if !called {
		t.Fatal("non-admin path should reach next without a Bearer token")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
}

// TestRequireBearerOnAdmin_MissingBearer_Bad proves an admin path with no
// Authorization header is denied with 401 and the WWW-Authenticate hint.
func TestRequireBearerOnAdmin_MissingBearer_Bad(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { called = true })
	h := RequireBearerOnAdmin(next, "secret", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil))
	if called {
		t.Fatal("next must not run without a valid Bearer token")
	}
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if got := rec.Header().Get("www-authenticate"); got == "" {
		t.Fatal("401 response should carry a www-authenticate hint")
	}
}

// TestRequireBearerOnAdmin_WrongBearer_Bad proves a mismatched Bearer token is
// denied, and that the deny is audit-emitted when a writer is supplied.
func TestRequireBearerOnAdmin_WrongBearer_Bad(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("next must not run on a wrong Bearer token")
	})
	audit := core.NewBuffer()
	h := RequireBearerOnAdmin(next, "secret", audit)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	req.Header.Set("Authorization", "Bearer wrong-token")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if !core.Contains(audit.String(), "auth deny") {
		t.Fatalf("audit log = %q, want an auth-deny line", audit.String())
	}
}

// TestRequireBearerOnAdmin_CorrectBearer_Good proves the matching token is let
// through to next.
func TestRequireBearerOnAdmin_CorrectBearer_Good(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	})
	h := RequireBearerOnAdmin(next, "secret", nil)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	req.Header.Set("Authorization", "Bearer secret")
	h.ServeHTTP(rec, req)
	if !called {
		t.Fatal("next should run on a correct Bearer token")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
}

// ---------------------------------------------------------------------------
// v0.9.0 shape triplets — Test<File>_<Symbol>_{Good,Bad,Ugly}
// ---------------------------------------------------------------------------

// TestServeAuth_AdminTokenPath_Good proves the canonical path under a normal
// $HOME — the same path lthn-mlx used.
func TestServeAuth_AdminTokenPath_Good(t *testing.T) {
	t.Setenv("HOME", "/home/tester")
	got := AdminTokenPath()
	want := core.PathJoin("/home/tester", "Lethean", "lem", "admin.token")
	if got != want {
		t.Fatalf("AdminTokenPath = %q, want %q", got, want)
	}
}

// TestServeAuth_AdminTokenPath_Bad proves an empty $HOME degrades to a
// relative path rather than panicking — AdminTokenPath has no error return,
// so this is the honest behaviour to pin down.
func TestServeAuth_AdminTokenPath_Bad(t *testing.T) {
	t.Setenv("HOME", "")
	got := AdminTokenPath()
	want := core.PathJoin("", "Lethean", "lem", "admin.token")
	if got != want {
		t.Fatalf("AdminTokenPath(empty HOME) = %q, want %q", got, want)
	}
}

// TestServeAuth_AdminTokenPath_Ugly proves a $HOME with a trailing separator
// is normalised — PathJoin cleans the join rather than doubling the slash.
func TestServeAuth_AdminTokenPath_Ugly(t *testing.T) {
	t.Setenv("HOME", "/home/tester/")
	got := AdminTokenPath()
	want := core.PathJoin("/home/tester/", "Lethean", "lem", "admin.token")
	if got != want {
		t.Fatalf("AdminTokenPath(trailing slash) = %q, want %q", got, want)
	}
	if core.Contains(got, "//") {
		t.Fatalf("AdminTokenPath should normalise doubled separators, got %q", got)
	}
}

// TestServeAuth_GenerateAdminToken_Good proves the token carries the
// secret-scanner prefix and a base64url-decodable payload.
func TestServeAuth_GenerateAdminToken_Good(t *testing.T) {
	tok, err := GenerateAdminToken()
	if err != nil {
		t.Fatalf("GenerateAdminToken: %v", err)
	}
	if !core.HasPrefix(tok, adminTokenPrefix) {
		t.Fatalf("token = %q, want %s prefix", tok, adminTokenPrefix)
	}
	if _, err := base64.RawURLEncoding.DecodeString(tok[len(adminTokenPrefix):]); err != nil {
		t.Fatalf("token payload does not decode as base64url: %v", err)
	}
}

// TestServeAuth_GenerateAdminToken_Bad proves the decoded payload carries the
// documented 256 bits (32 bytes) of entropy — not a truncated or padded
// value that would silently weaken the secret.
func TestServeAuth_GenerateAdminToken_Bad(t *testing.T) {
	tok, err := GenerateAdminToken()
	if err != nil {
		t.Fatalf("GenerateAdminToken: %v", err)
	}
	raw, err := base64.RawURLEncoding.DecodeString(tok[len(adminTokenPrefix):])
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(raw) != 32 {
		t.Fatalf("token payload = %d bytes, want 32", len(raw))
	}
}

// TestServeAuth_GenerateAdminToken_Ugly proves repeated calls never collide —
// the edge that matters for a secret minted once per boot forever after.
func TestServeAuth_GenerateAdminToken_Ugly(t *testing.T) {
	seen := make(map[string]bool, 100)
	for i := range 100 {
		tok, err := GenerateAdminToken()
		if err != nil {
			t.Fatalf("GenerateAdminToken: %v", err)
		}
		if seen[tok] {
			t.Fatalf("token collision after %d generations", i)
		}
		seen[tok] = true
	}
}

// TestServeAuth_WriteAdminToken_Good proves the token lands with a trailing
// newline, parent dirs created on demand.
func TestServeAuth_WriteAdminToken_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "nested", "admin.token")
	if err := WriteAdminToken(path, "tok-value"); err != nil {
		t.Fatalf("WriteAdminToken: %v", err)
	}
	res := core.ReadFile(path)
	if !res.OK || string(res.Value.([]byte)) != "tok-value\n" {
		t.Fatalf("read back = (%v, OK=%v), want \"tok-value\\n\"", res.Value, res.OK)
	}
}

// TestServeAuth_WriteAdminToken_Bad proves a blocked parent directory (a
// path component that already exists as a plain file) surfaces the failure
// rather than silently swallowing it — the fail-closed checkpoint.
func TestServeAuth_WriteAdminToken_Bad(t *testing.T) {
	dir := t.TempDir()
	blocker := core.PathJoin(dir, "blocker")
	if r := core.WriteFile(blocker, []byte("not a dir"), 0o644); !r.OK {
		t.Fatalf("seed blocker: %v", r.Value)
	}
	if err := WriteAdminToken(core.PathJoin(blocker, "child", "admin.token"), "tok"); err == nil {
		t.Fatal("WriteAdminToken should fail when a parent path component is a file")
	}
}

// TestServeAuth_WriteAdminToken_Ugly proves a second write overwrites the
// first rather than appending — the edge a naive append-mode write would
// get wrong.
func TestServeAuth_WriteAdminToken_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "admin.token")
	if err := WriteAdminToken(path, "first"); err != nil {
		t.Fatalf("first write: %v", err)
	}
	if err := WriteAdminToken(path, "second"); err != nil {
		t.Fatalf("second write: %v", err)
	}
	res := core.ReadFile(path)
	if !res.OK || string(res.Value.([]byte)) != "second\n" {
		t.Fatalf("token file = %v, want overwritten to \"second\\n\"", res.Value)
	}
}

// TestServeAuth_EnsureAdminToken_Good proves the first boot mints and
// persists a fresh token.
func TestServeAuth_EnsureAdminToken_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "admin.token")
	tok, generated, err := EnsureAdminToken(path)
	if err != nil {
		t.Fatalf("EnsureAdminToken: %v", err)
	}
	if !generated {
		t.Fatal("first boot should report generated=true")
	}
	if !core.HasPrefix(tok, adminTokenPrefix) {
		t.Fatalf("token = %q, want %s prefix", tok, adminTokenPrefix)
	}
}

// TestServeAuth_EnsureAdminToken_Bad proves a write failure (parent path
// blocked by a plain file) surfaces rather than returning a token that was
// never actually persisted.
func TestServeAuth_EnsureAdminToken_Bad(t *testing.T) {
	dir := t.TempDir()
	blocker := core.PathJoin(dir, "blocker")
	if r := core.WriteFile(blocker, []byte("not a dir"), 0o644); !r.OK {
		t.Fatalf("seed blocker: %v", r.Value)
	}
	if _, _, err := EnsureAdminToken(core.PathJoin(blocker, "child", "admin.token")); err == nil {
		t.Fatal("EnsureAdminToken should fail when the token file cannot be written")
	}
}

// TestServeAuth_EnsureAdminToken_Ugly proves a second call against the same
// path reuses the persisted token instead of minting another — the
// idempotency edge that keeps repeated boots agreeing on one secret.
func TestServeAuth_EnsureAdminToken_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "admin.token")
	first, _, err := EnsureAdminToken(path)
	if err != nil {
		t.Fatalf("first EnsureAdminToken: %v", err)
	}
	second, generated, err := EnsureAdminToken(path)
	if err != nil {
		t.Fatalf("second EnsureAdminToken: %v", err)
	}
	if generated {
		t.Fatal("second call should report generated=false")
	}
	if first != second {
		t.Fatalf("second call returned a different token: %q != %q", first, second)
	}
}

// TestServeAuth_RequireBearerOnAdmin_Good proves a matching Bearer token is
// let through to next.
func TestServeAuth_RequireBearerOnAdmin_Good(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { called = true })
	h := RequireBearerOnAdmin(next, "secret", nil)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	req.Header.Set("Authorization", "Bearer secret")
	h.ServeHTTP(rec, req)
	if !called {
		t.Fatal("matching Bearer token should reach next")
	}
}

// TestServeAuth_RequireBearerOnAdmin_Bad proves a mismatched token is denied
// with 401 and never reaches next.
func TestServeAuth_RequireBearerOnAdmin_Bad(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("next must not run on a wrong Bearer token")
	})
	h := RequireBearerOnAdmin(next, "secret", nil)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
}

// TestServeAuth_RequireBearerOnAdmin_Ugly proves the edge of an empty
// configured token: an Authorization header of exactly "Bearer " (empty
// secret) is treated as a match. Documents why Serve only ever wraps the
// admin subtree with RequireBearerOnAdmin when cfg.adminToken != "".
func TestServeAuth_RequireBearerOnAdmin_Ugly(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { called = true })
	h := RequireBearerOnAdmin(next, "", nil)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	req.Header.Set("Authorization", "Bearer ")
	h.ServeHTTP(rec, req)
	if !called {
		t.Fatal("an empty configured token matches an empty-secret Bearer header")
	}
}
