// SPDX-Licence-Identifier: EUPL-1.2

// Package creds is provider credentials and BYOK from RFC §6.17. External
// providers (NVIDIA NIM, OpenAI, OpenRouter, …) each need their own key; local
// runtimes (go-mlx on the M3 Ultra, the CUDA/ROCm GPU) need none. The package
// holds those secrets, resolves the right one per request — honouring a
// caller-supplied BYOK key — and carries the per-API-key routing profile that
// lets different callers draw from different provider pools through the one
// surface (RFC §6.2).
//
// The secret is opaque and never logged: Credential.String() masks it, so a
// credential is safe to drop into a log line, an error, or a struct dump.
//
//	r := creds.New()
//	r.MarkLocal("local-metal")                                  // on-device, no key
//	r.Set(creds.Credential{Provider: "openai", Secret: key})    // stored, encrypted at rest by the caller
//	c, err := r.Resolve("openai", byok)                         // byok (if non-nil) wins
//	if err != nil { return err }
//	use(c.Secret)                                               // never log c.Secret — log c (masked)
package creds

import (
	"slices"
	"sync"

	core "dappco.re/go"
)

// Credential is one provider's secret (RFC §6.17 per-provider secrets). Secret
// is opaque — the inference stack never inspects it, only forwards it to the provider's wire
// translation (RFC §6.14). It is stored encrypted at rest by the caller and is
// NEVER logged: String() masks it.
//
//	c := creds.Credential{Provider: "openrouter", Secret: "sk-or-…"}
//	core.Print(c.String()) // "openrouter:****" — the secret is not exposed
type Credential struct {
	// Provider is the endpoint label this secret authenticates ("openai",
	// "openrouter", "nim", …). Empty for a local runtime's empty credential.
	Provider string `json:"provider"`
	// Secret is the opaque key / token. Empty for a local runtime. Never logged.
	Secret string `json:"-"`
}

// secretMask is the fixed redaction stand-in for a non-empty secret. A fixed
// mask (not a length-revealing one) leaks nothing about the secret — not even
// how long it is.
const secretMask = "****"

// String renders the credential for logs and diagnostics with the secret
// MASKED — the security guarantee of RFC §6.17 (credentials are never logged).
// A credential with a secret reads "provider:****"; an empty credential (a
// local runtime's) reads "provider:(none)" so it is distinguishable from a real
// one without revealing anything.
//
//	creds.Credential{Provider: "openai", Secret: "sk-x"}.String() // "openai:****"
//	creds.Credential{Provider: "local-metal"}.String()            // "local-metal:(none)"
func (c Credential) String() string {
	if c.Secret == "" {
		return c.Provider + ":(none)"
	}
	return c.Provider + ":" + secretMask
}

// HasSecret reports whether the credential carries a secret. A local runtime's
// resolved credential is empty (HasSecret false); a remote one is not.
//
//	if c.HasSecret() { authenticate(c) }
func (c Credential) HasSecret() bool { return c.Secret != "" }

// Store holds provider credentials. The in-memory implementation (New is in
// resolve.go via the Resolver, or NewStore for the bare store) is goroutine-safe
// so a runtime can read and rotate keys from multiple request goroutines.
//
//	var s creds.Store = creds.NewStore()
//	s.Set(creds.Credential{Provider: "openai", Secret: key})
//	c, ok := s.Get("openai")
type Store interface {
	// Get returns the stored credential for provider and whether one is set.
	Get(provider string) (Credential, bool)
	// Set stores (or replaces) the credential for cred.Provider.
	Set(cred Credential) error
	// Delete removes the credential for provider. Absent provider is a no-op.
	Delete(provider string)
}

// memStore is the goroutine-safe in-memory Store. Secrets live only in memory;
// encryption at rest (RFC §6.17) is the caller's responsibility when it
// persists them.
type memStore struct {
	mu sync.RWMutex
	by map[string]Credential
}

// NewStore builds an empty goroutine-safe in-memory credential store.
//
//	s := creds.NewStore()
func NewStore() Store {
	return &memStore{by: make(map[string]Credential)}
}

// Get returns the stored credential for provider, and false if none is set.
func (m *memStore) Get(provider string) (Credential, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	c, ok := m.by[provider]
	return c, ok
}

// Set stores cred under cred.Provider. An empty provider is rejected — a
// credential with no provider can never be resolved, so storing it is a bug.
func (m *memStore) Set(cred Credential) error {
	if cred.Provider == "" {
		return core.E("creds", "credential has empty provider", nil)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.by[cred.Provider] = cred
	return nil
}

// Delete removes provider's credential. Deleting an absent provider is a no-op.
func (m *memStore) Delete(provider string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.by, provider)
}

// KeyPolicy is the per-API-key routing profile from RFC §6.17: an API key
// carries its own allowed providers, price ceiling, ZDR requirement, and
// default model / preset, so different callers draw from different provider
// pools through the one surface (RFC §6.2). It is the per-key half of routing —
// the request-level provider preferences (§6.2) are filtered against it.
//
//	pol := creds.KeyPolicy{
//		AllowedProviders: []string{"local-metal", "openai"},
//		MaxPrice:         0.0,  // free-only
//		ZDR:              true, // zero-data-retention endpoints only
//		DefaultModel:     "gemma-4-31b",
//	}
//	if !pol.Allows(route.Provider) { return errDenied }
type KeyPolicy struct {
	// AllowedProviders is the allow-list of provider labels this key may route
	// to. EMPTY means every provider is allowed (the unrestricted default key).
	AllowedProviders []string `json:"allowed_providers,omitempty"`
	// MaxPrice is the per-request price ceiling (RFC §6.2 max_price); 0 means
	// free-only for this key.
	MaxPrice float64 `json:"max_price,omitempty"`
	// ZDR restricts this key to zero-data-retention endpoints (RFC §6.2 zdr).
	ZDR bool `json:"zdr,omitempty"`
	// DefaultModel is the model / preset this key falls back to when a request
	// names none (RFC §6.10 stored presets).
	DefaultModel string `json:"default_model,omitempty"`
}

// Allows reports whether this key's policy permits routing to provider. An empty
// AllowedProviders short-circuits to true (unrestricted key); otherwise provider
// must be a member of the allow-list. The empty provider string is only allowed
// by an empty (unrestricted) list.
//
//	KeyPolicy{}.Allows("openai")                               // true — unrestricted
//	KeyPolicy{AllowedProviders: []string{"local"}}.Allows("openai") // false
func (p KeyPolicy) Allows(provider string) bool {
	if len(p.AllowedProviders) == 0 {
		return true
	}
	return slices.Contains(p.AllowedProviders, provider)
}
