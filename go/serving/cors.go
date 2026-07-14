// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"net/http"

	core "dappco.re/go"
)

// Browser CORS for the serve — OFF by default (byte-identical responses when
// unconfigured). A browser app on a different origin (the GUI dev server, a
// hosted dashboard) cannot call lem without these headers; --cors names the
// origins allowed to. The middleware wraps OUTSIDE the admin Bearer wall
// because preflight OPTIONS requests carry no Authorization header — a
// preflight that hit the wall would 401 and the browser would report a CORS
// failure instead of the real story.

// corsPolicy is the resolved --cors flag: explicit origins, or wildcard.
type corsPolicy struct {
	wildcard bool
	origins  map[string]struct{}
}

// parseCORSOrigins builds the policy from the flag value: "*" allows any
// origin, otherwise a comma-separated exact-match origin list
// (e.g. "http://localhost:4200,https://gui.example.com"). Empty disables.
func parseCORSOrigins(flag string) *corsPolicy {
	flag = core.Trim(flag)
	if flag == "" {
		return nil
	}
	if flag == "*" {
		return &corsPolicy{wildcard: true}
	}
	p := &corsPolicy{origins: map[string]struct{}{}}
	for _, origin := range core.Split(flag, ",") {
		if origin = core.Trim(origin); origin != "" {
			p.origins[origin] = struct{}{}
		}
	}
	if len(p.origins) == 0 {
		return nil
	}
	return p
}

func (p *corsPolicy) allows(origin string) bool {
	if p.wildcard {
		return true
	}
	_, ok := p.origins[origin]
	return ok
}

// allowOriginValue is the Access-Control-Allow-Origin to emit for an allowed
// origin: the literal * under wildcard, otherwise the specific origin echoed
// back (with Vary: Origin so caches key on it).
func (p *corsPolicy) allowOriginValue(origin string) string {
	if p.wildcard {
		return "*"
	}
	return origin
}

// corsMiddleware answers preflights and stamps the CORS response headers for
// allowed origins; everything else passes through untouched. A disallowed
// origin gets no CORS headers (the browser enforces the block) and a
// non-browser request (no Origin header) is completely unaffected.
func corsMiddleware(next http.Handler, policy *corsPolicy) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin == "" || !policy.allows(origin) {
			next.ServeHTTP(w, r)
			return
		}
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", policy.allowOriginValue(origin))
		if !policy.wildcard {
			h.Add("Vary", "Origin")
		}
		if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
			// Preflight: answer here — it must never reach the Bearer wall or
			// the mux (it carries no auth and expects no body).
			h.Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
			reqHeaders := r.Header.Get("Access-Control-Request-Headers")
			if reqHeaders == "" {
				reqHeaders = "Content-Type, Authorization"
			}
			h.Set("Access-Control-Allow-Headers", reqHeaders)
			h.Set("Access-Control-Max-Age", "86400")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// WithCORS enables browser CORS for the listed origins ("*" = any; empty
// disables — the default). See corsMiddleware for the exact semantics.
func WithCORS(origins string) ServeOption {
	return func(c *serveConfig) { c.cors = parseCORSOrigins(origins) }
}
