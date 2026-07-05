// SPDX-Licence-Identifier: EUPL-1.2

package creds

import (
	"sync"

	core "dappco.re/go"
)

// Resolver resolves the credential for a request: it wraps a Store of remote
// provider secrets and a set of provider labels marked local (RFC §6.17 — local
// runtimes carry no credential). Resolve picks the right secret per request,
// letting a caller-supplied BYOK key override the stored one. Safe to share
// across request goroutines.
//
//	r := creds.New()
//	r.MarkLocal("local-metal", "local-gpu")              // on-device endpoints
//	r.Set(creds.Credential{Provider: "openai", Secret: key})
//	c, err := r.Resolve("openai", byok)                  // byok (if non-nil) wins
type Resolver struct {
	store Store

	mu    sync.RWMutex
	local map[string]struct{}
}

// New builds a Resolver over a fresh in-memory Store with no local providers.
//
//	r := creds.New()
func New() *Resolver {
	return NewResolver(NewStore())
}

// NewResolver builds a Resolver over an existing Store — use this to share one
// credential store across resolvers, or to back it with a persistent (encrypted
// at rest, RFC §6.17) implementation.
//
//	r := creds.NewResolver(myEncryptedStore)
func NewResolver(store Store) *Resolver {
	return &Resolver{store: store, local: make(map[string]struct{})}
}

// Set stores a remote provider's credential (delegates to the underlying Store).
//
//	r.Set(creds.Credential{Provider: "openrouter", Secret: "sk-or-…"})
func (r *Resolver) Set(cred Credential) error { return r.store.Set(cred) }

// Get returns the stored credential for provider and whether one is set
// (delegates to the underlying Store). Prefer Resolve on the request path —
// Get is the raw store read, with no BYOK or local handling.
func (r *Resolver) Get(provider string) (Credential, bool) { return r.store.Get(provider) }

// Delete removes a stored credential (delegates to the underlying Store).
func (r *Resolver) Delete(provider string) { r.store.Delete(provider) }

// MarkLocal records one or more provider labels as local runtimes that need no
// credential (RFC §6.17 "local needs nothing"). Resolve returns an empty
// credential for a local provider — and a local label always wins, so a stale
// stored secret for that label is never returned.
//
//	r.MarkLocal("local-metal", "local-gpu")
func (r *Resolver) MarkLocal(providers ...string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, p := range providers {
		r.local[p] = struct{}{}
	}
}

// UnmarkLocal removes a provider from the local set (it becomes a remote
// provider again, needing a credential). No-op if it wasn't local.
func (r *Resolver) UnmarkLocal(provider string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.local, provider)
}

// IsLocal reports whether provider is marked as a local runtime.
func (r *Resolver) IsLocal(provider string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.local[provider]
	return ok
}

// Resolve returns the credential to use for provider on this request (RFC
// §6.17). Resolution order:
//
//  1. A non-nil byok OVERRIDES everything — its secret is used for this call
//     only; the store is not mutated, so BYOK is per-request, not persistent.
//  2. A provider marked local needs no credential — an empty credential is
//     returned with no error. The local set wins over any stored secret, so a
//     local runtime never leaks an external key.
//  3. A stored credential for a remote provider is returned.
//  4. A remote provider with no stored credential and no byok is a typed error
//     (external providers need credentials).
//
// An empty provider name is rejected — it can never name an endpoint.
//
//	c, err := r.Resolve("openai", nil)          // stored key
//	c, err := r.Resolve("openai", byok)         // BYOK overrides for this call
//	c, err := r.Resolve("local-metal", nil)     // empty credential, no error
//	c, err := r.Resolve("openrouter", nil)      // err if none stored
func (r *Resolver) Resolve(provider string, byok *Credential) (Credential, error) {
	// BYOK wins outright — accounted as BYOK against the caller's key (RFC §6.6
	// is_byok). It works even for an otherwise-unknown provider.
	if byok != nil {
		return *byok, nil
	}

	if provider == "" {
		return Credential{}, core.E("creds", "resolve credential: empty provider", nil)
	}

	// Local wins over any stored secret — never return an external key for an
	// on-device runtime.
	if r.IsLocal(provider) {
		return Credential{Provider: provider}, nil
	}

	if cred, ok := r.store.Get(provider); ok {
		return cred, nil
	}

	// Remote provider, nothing stored, no BYOK → fail closed.
	return Credential{}, core.E("creds", "no credential for remote provider: "+provider, nil)
}
