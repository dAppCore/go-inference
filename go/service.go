// SPDX-License-Identifier: EUPL-1.2

// Service registration for the inference package — exposes the canonical
// `NewService(opts)` + `RegisterCore(c)` shape per Mantis #1336, holding
// a thin Core handle over the package's global Backend registry.
//
// **Naming divergence from canon.** The canonical pattern uses
// `Register(c *core.Core) core.Result` for the imperative shorthand.
// This package already has `Register(b Backend)` — the well-known
// init-time backend-registration pattern (`inference.Register(metal.NewBackend())`
// from a backend's init()). Renaming it would break every backend
// package's init function. So the canonical Core registration is
// exposed as `RegisterCore(c *core.Core) core.Result` here, with the
// existing `Register(b Backend)` preserved untouched.
//
//	c, _ := core.New(core.WithService(inference.NewService(inference.Options{})))
//	svc := core.MustServiceFor[*inference.Service](c, "inference")
//	for name, b := range inference.All() { ... }
//
// The Backend interface, the global registry (Register(b), Get, List,
// All, snapshotBackends), and the package-level capability surface
// remain the source of truth — Service is a thin Core-side handle that
// gives the inference package a registerable identity the framework
// can discover via core.ServiceFor.

package inference

import (
	core "dappco.re/go"
)

// Options configures the inference service. v1 has no fields — the
// package's behaviour is entirely driven by which Backend
// implementations have called Register(Backend) at init time. Future
// fields (e.g. PreferredBackendOrder override, ProbeBus subscribers)
// land here as needed.
type Options struct{}

// Service is the registerable handle for the inference package — embeds
// *core.ServiceRuntime[Options] for typed options access. Backend
// lookups still go through the package-level Get / List / All — Service
// doesn't shadow the global registry, just provides a Core-discoverable
// identity for the package.
//
// Usage example: `svc := core.MustServiceFor[*inference.Service](c, "inference"); names := inference.List()`
type Service struct {
	*core.ServiceRuntime[Options]
}

// NewService returns a factory that registers the inference package as
// a Core service. v1 Options is empty; the underlying Backend registry
// (managed by the package-level Register(b) function called from each
// backend's init) is the real state.
//
//	core.WithService(inference.NewService(inference.Options{}))
func NewService(opts Options) func(*core.Core) core.Result {
	return func(c *core.Core) core.Result {
		return core.Ok(&Service{
			ServiceRuntime: core.NewServiceRuntime(c, opts),
		})
	}
}

// RegisterCore wires the inference service into the Core with default
// Options — the imperative-style alternative to NewService.
//
// Named RegisterCore (not Register) to avoid colliding with the
// existing package-level `func Register(b Backend)` used by backend
// implementations to self-register at init time. See the file-level
// docstring for why.
//
//	c := core.New()
//	if r := inference.RegisterCore(c); !r.OK { return r }
func RegisterCore(c *core.Core) core.Result {
	return NewService(Options{})(c)
}
