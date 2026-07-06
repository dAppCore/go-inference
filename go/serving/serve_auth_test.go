// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
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
