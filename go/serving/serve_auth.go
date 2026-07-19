// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"io"
	"net/http"

	core "dappco.re/go"
)

// adminTokenPrefix marks the token as a Lethean serve admin secret so secret
// scanners (gitleaks, trufflehog) recognise leaked tokens in repos. Matches the
// gh_pat_/sk-/ghp_ convention. Kept as the lthn-mlx_ value so a token already
// minted by lthn-mlx stays a drop-in secret for the lem CLI.
const adminTokenPrefix = "lthn-mlx_"

// AdminTokenPath returns ~/Lethean/lem/admin.token — the canonical location
// for the Bearer auth secret. Mode 0600 is enforced on write so other local
// users can't read it. This is the same path lthn-mlx used, so the lem CLI is a
// drop-in for any tool (lem.sh harness, launchd job) that reads the token file.
//
//	tok, _, _ := serving.EnsureAdminToken(serving.AdminTokenPath())
func AdminTokenPath() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "lem", "admin.token")
}

// GenerateAdminToken returns a fresh opaque 256-bit token, base64url-encoded,
// with the admin prefix. 256 bits of entropy is unbreakable in practice.
//
//	tok, err := serving.GenerateAdminToken() // → "lthn-mlx_K7gH..." (52 chars)
func GenerateAdminToken() (string, error) {
	var raw [32]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", core.E("serving.GenerateAdminToken", "rand", err)
	}
	return adminTokenPrefix + base64.RawURLEncoding.EncodeToString(raw[:]), nil
}

// loadAdminToken reads the existing token at path. Returns ("", false, nil) for
// any read failure including file-not-found — the caller treats that as "no
// token yet, generate one" rather than fatal.
func loadAdminToken(path string) (token string, exists bool, err error) {
	res := core.ReadFile(path)
	if !res.OK {
		return "", false, nil
	}
	tok := core.Trim(string(res.Bytes()))
	if tok == "" {
		return "", false, nil
	}
	return tok, true, nil
}

// WriteAdminToken writes the token to path with 0o600 perms; the parent dir is
// created if missing. This is the fail-closed checkpoint — the caller MUST
// abort serve startup if the write fails (better to refuse to boot than to bind
// a listener with an unprotected admin surface).
func WriteAdminToken(path, token string) error {
	if dir := core.PathDir(path); dir != "" {
		if r := core.MkdirAll(dir, 0o755); !r.OK {
			return core.E("serving.WriteAdminToken", "mkdir parent", r.Value.(error))
		}
	}
	if r := core.WriteFile(path, []byte(token+"\n"), 0o600); !r.OK {
		return core.E("serving.WriteAdminToken", "write", r.Value.(error))
	}
	return nil
}

// EnsureAdminToken loads the existing token or generates + writes a fresh one.
// It returns the token plus whether it was freshly generated (so serve can
// print a one-line notice the first time).
//
// TOCTOU defence: it re-reads after write. If two serve processes race on first
// boot, both see "absent", both generate, both write — last-writer-wins on the
// file content. The loser converges to the winning token via the re-read
// instead of returning a token nobody else will accept.
//
//	tok, fresh, err := serving.EnsureAdminToken(serving.AdminTokenPath())
func EnsureAdminToken(path string) (token string, generated bool, err error) {
	existing, exists, err := loadAdminToken(path)
	if err != nil {
		return "", false, err
	}
	if exists {
		return existing, false, nil
	}
	tok, err := GenerateAdminToken()
	if err != nil {
		return "", false, err
	}
	if err := WriteAdminToken(path, tok); err != nil {
		return "", false, err
	}
	after, afterExists, err := loadAdminToken(path)
	if err != nil {
		return "", false, err
	}
	if afterExists && after != tok {
		return after, false, nil
	}
	return tok, true, nil
}

// RequireBearerOnAdmin wraps next with Bearer-token auth on any path starting
// with /v1/admin/. Other paths (/v1/chat/completions, /v1/health, …) pass
// through unauthenticated — the localhost / tunnel-trust model still applies to
// inference, only admin verbs need explicit auth.
//
// It uses crypto/subtle constant-time compare to defeat timing side channels.
// Every 401 audit-emits to audit (when non-nil) so brute-force attempts against
// the token are visible in operator logs.
//
//	h := serving.RequireBearerOnAdmin(rootMux, token, os.Stderr)
func RequireBearerOnAdmin(next http.Handler, token string, audit io.Writer) http.Handler {
	expected := []byte("Bearer " + token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !core.HasPrefix(r.URL.Path, "/v1/admin/") {
			next.ServeHTTP(w, r)
			return
		}
		got := []byte(r.Header.Get("Authorization"))
		if len(got) != len(expected) || subtle.ConstantTimeCompare(got, expected) != 1 {
			if audit != nil {
				core.Print(audit, "serve admin: auth deny path=%s remote=%s", r.URL.Path, r.RemoteAddr)
			}
			w.Header().Set("www-authenticate", `Bearer realm="lthn-serve-admin"`)
			http.Error(w, "admin endpoint requires Authorization: Bearer <token>", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
