// SPDX-Licence-Identifier: EUPL-1.2

package creds

import (
	core "dappco.re/go"
)

// TestCreds_Resolve_Good covers the happy paths of credential resolution (RFC
// §6.17): a stored remote credential resolves to itself, a non-nil BYOK
// credential overrides the stored one for that call, and a provider marked
// local resolves to an empty credential with no error (local needs nothing).
func TestCreds_Resolve_Good(t *core.T) {
	r := New()
	r.MarkLocal("local-metal")
	core.AssertNoError(t, r.Set(Credential{Provider: "openai", Secret: "sk-stored"}))

	// Stored remote credential resolves to itself.
	got, err := r.Resolve("openai", nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "openai", got.Provider)
	core.AssertEqual(t, "sk-stored", got.Secret)

	// BYOK overrides the stored credential for this call only — the store is
	// left untouched, so the next plain resolve still sees the stored secret.
	byok := &Credential{Provider: "openai", Secret: "sk-byok"}
	got, err = r.Resolve("openai", byok)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "sk-byok", got.Secret)
	again, err := r.Resolve("openai", nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "sk-stored", again.Secret)

	// A local provider needs no credential — empty credential, no error.
	local, err := r.Resolve("local-metal", nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", local.Secret)
}

// TestCreds_Resolve_Bad covers the failure path: a remote provider with no
// stored credential and no BYOK is a typed error (RFC §6.17 — external
// providers need credentials), and Delete removes a stored credential so a
// resolve afterwards errors.
func TestCreds_Resolve_Bad(t *core.T) {
	r := New()

	// Missing credential for a remote provider → typed error, empty credential.
	got, err := r.Resolve("openrouter", nil)
	core.AssertError(t, err)
	core.AssertEqual(t, "", got.Secret)
	core.AssertContains(t, err.Error(), "openrouter")

	// Set then Delete → back to a missing-credential error.
	core.AssertNoError(t, r.Set(Credential{Provider: "openrouter", Secret: "sk-x"}))
	_, err = r.Resolve("openrouter", nil)
	core.AssertNoError(t, err)
	r.Delete("openrouter")
	_, err = r.Resolve("openrouter", nil)
	core.AssertError(t, err)
}

// TestCreds_Resolve_Ugly covers the edge cases: a BYOK credential resolves even
// for a provider that has no stored credential and isn't local (BYOK is enough
// on its own), an empty-provider resolve is a typed error rather than a panic,
// and a local provider with a stale stored credential still resolves to empty
// (local membership wins — no external secret is ever returned for a local
// runtime).
func TestCreds_Resolve_Ugly(t *core.T) {
	r := New()

	// BYOK alone satisfies an otherwise-unknown provider.
	byok := &Credential{Provider: "nim", Secret: "nvapi-byok"}
	got, err := r.Resolve("nim", byok)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "nvapi-byok", got.Secret)

	// Empty provider name is rejected, not panicked.
	_, err = r.Resolve("", nil)
	core.AssertError(t, err)

	// Local wins over a stale stored secret — a local runtime never leaks one.
	r.MarkLocal("local-gpu")
	core.AssertNoError(t, r.Set(Credential{Provider: "local-gpu", Secret: "leftover"}))
	local, err := r.Resolve("local-gpu", nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", local.Secret)
}

// TestCreds_Policy_Good covers the per-key routing profile (RFC §6.17): an
// empty AllowedProviders means every provider is allowed, and a populated list
// allows exactly its members.
func TestCreds_Policy_Good(t *core.T) {
	// Empty AllowedProviders = all allowed (the unrestricted default key).
	open := KeyPolicy{}
	core.AssertTrue(t, open.Allows("openai"))
	core.AssertTrue(t, open.Allows("anything"))

	// A populated allow-list permits its members.
	p := KeyPolicy{
		AllowedProviders: []string{"local-metal", "openai"},
		MaxPrice:         0.0,
		ZDR:              true,
		DefaultModel:     "gemma-4-31b",
	}
	core.AssertTrue(t, p.Allows("openai"))
	core.AssertTrue(t, p.Allows("local-metal"))
	core.AssertEqual(t, "gemma-4-31b", p.DefaultModel)
	core.AssertTrue(t, p.ZDR)
}

// TestCreds_Policy_Bad covers denial: a provider outside a populated allow-list
// is denied.
func TestCreds_Policy_Bad(t *core.T) {
	p := KeyPolicy{AllowedProviders: []string{"local-metal"}}
	core.AssertFalse(t, p.Allows("openai"))
	core.AssertFalse(t, p.Allows("openrouter"))
	core.AssertTrue(t, p.Allows("local-metal"))
}

// TestCreds_Policy_Ugly covers the edge cases: an empty provider is never
// allowed even by a populated list, and the empty-list "all allowed" rule still
// holds for the empty provider string (an empty list short-circuits to allow).
func TestCreds_Policy_Ugly(t *core.T) {
	// A populated list denies the empty provider — there is no "" member.
	p := KeyPolicy{AllowedProviders: []string{"openai"}}
	core.AssertFalse(t, p.Allows(""))

	// The unrestricted key allows everything, including the empty string — the
	// empty-list rule short-circuits before any membership check.
	core.AssertTrue(t, KeyPolicy{}.Allows(""))
}

// TestCreds_Redaction_Good proves the secret never appears in String() — the
// redaction is the security guarantee (RFC §6.17: credentials are never
// logged). String() shows the provider and a fixed mask, not the secret.
func TestCreds_Redaction_Good(t *core.T) {
	c := Credential{Provider: "openai", Secret: "sk-supersecret-value"}
	s := c.String()
	core.AssertContains(t, s, "openai")
	core.AssertNotContains(t, s, "sk-supersecret-value")
	core.AssertNotContains(t, s, "supersecret")
}

// TestCreds_Redaction_Bad proves a long secret is masked rather than partially
// leaked — no substring of the raw secret survives into String().
func TestCreds_Redaction_Bad(t *core.T) {
	c := Credential{Provider: "openrouter", Secret: "abcdefghijklmnopqrstuvwxyz0123456789"}
	s := c.String()
	core.AssertNotContains(t, s, "abcdefgh")
	core.AssertNotContains(t, s, "0123456789")
	core.AssertNotContains(t, s, "vwxyz")
	core.AssertContains(t, s, "openrouter")
}

// TestCreds_Redaction_Ugly covers the empty-secret edge: a credential with no
// secret reads as empty/unset rather than as a masked value, so an empty
// credential (a local provider's) is distinguishable from a real one.
func TestCreds_Redaction_Ugly(t *core.T) {
	empty := Credential{Provider: "local-metal"}
	s := empty.String()
	core.AssertContains(t, s, "local-metal")
	// An empty secret must not render as the masked form — it is genuinely unset.
	core.AssertNotContains(t, s, "****")
}

// TestCreds_HasSecret_Good covers the secret presence check: a remote
// credential carries a secret (true), a local runtime's empty credential does
// not (false).
func TestCreds_HasSecret_Good(t *core.T) {
	remote := Credential{Provider: "openai", Secret: "sk-x"}
	core.AssertTrue(t, remote.HasSecret(), "a remote credential carries a secret")

	local := Credential{Provider: "local-metal"}
	core.AssertFalse(t, local.HasSecret(), "a local credential has no secret")

	// A resolved local credential (via the Resolver) is likewise secret-less.
	r := New()
	r.MarkLocal("local-metal")
	got, err := r.Resolve("local-metal", nil)
	core.AssertNoError(t, err)
	core.AssertFalse(t, got.HasSecret(), "a resolved local credential has no secret")
}

// TestCreds_Store_Bad covers the store's input guard: storing a credential with
// an empty provider is rejected (an unkeyable credential is a bug), and the
// rejection leaves nothing behind to resolve.
func TestCreds_Store_Bad(t *core.T) {
	s := NewStore()
	err := s.Set(Credential{Secret: "sk-orphan"}) // no provider
	core.AssertError(t, err, "empty provider")
	core.AssertContains(t, err.Error(), "empty provider")

	// The Resolver delegates Set, so the same guard fires through it.
	r := New()
	core.AssertError(t, r.Set(Credential{Secret: "sk-orphan"}))
}

// TestCreds_ResolverGet_Good covers the raw store read exposed on the Resolver:
// Get returns a stored credential and true, or the zero credential and false
// for an absent provider — with no BYOK or local handling (that is Resolve's
// job).
func TestCreds_ResolverGet_Good(t *core.T) {
	r := New()
	core.AssertNoError(t, r.Set(Credential{Provider: "openai", Secret: "sk-stored"}))

	got, ok := r.Get("openai")
	core.AssertTrue(t, ok, "a stored provider is found")
	core.AssertEqual(t, "sk-stored", got.Secret)

	_, ok = r.Get("absent")
	core.AssertFalse(t, ok, "an unstored provider reports false")

	// Get is the raw read: it does NOT apply local masking. A local provider
	// with a stale stored secret still reads that secret through Get, whereas
	// Resolve would mask it — proving Get bypasses local handling.
	r.MarkLocal("local-gpu")
	core.AssertNoError(t, r.Set(Credential{Provider: "local-gpu", Secret: "leftover"}))
	raw, ok := r.Get("local-gpu")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "leftover", raw.Secret, "Get is raw — no local masking")
}

// TestCreds_UnmarkLocal_Good covers toggling a provider out of the local set: a
// provider marked local resolves to an empty credential, and after UnmarkLocal
// it is a remote provider again — needing a credential, so a bare resolve
// errors. Unmarking a provider that was never local is a harmless no-op.
func TestCreds_UnmarkLocal_Good(t *core.T) {
	r := New()
	r.MarkLocal("local-metal")
	core.AssertTrue(t, r.IsLocal("local-metal"))

	// While local, it resolves to an empty credential with no error.
	c, err := r.Resolve("local-metal", nil)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "", c.Secret)

	// Unmark it: now it is remote again and a bare resolve fails (no secret).
	r.UnmarkLocal("local-metal")
	core.AssertFalse(t, r.IsLocal("local-metal"))
	_, err = r.Resolve("local-metal", nil)
	core.AssertError(t, err, "no credential for remote provider")

	// Unmarking a provider that was never local is a no-op (no panic).
	r.UnmarkLocal("never-was-local")
	core.AssertFalse(t, r.IsLocal("never-was-local"))
}
