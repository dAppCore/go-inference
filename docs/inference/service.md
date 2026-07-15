<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# service.go — Core ServiceRuntime registration

**Package**: `dappco.re/go/inference`
**File**: `go/service.go`
**Mantis**: #1336 (canonical Service.go pattern)

## What this is

The Core-side handle for the `inference` package — exposes the canonical `NewService(opts) + RegisterCore(c)` shape so `dappco.re/go/core` can discover the inference package as a registerable framework service.

## The naming divergence

Canonical pattern across the rest of the Go canon:

```go
core.New(core.WithService(somepkg.Register))   // somepkg.Register is the registration fn
```

But `inference.Register(b Backend)` already exists — the init-time backend-registration call that every in-repo engine uses:

```go
// in engine/metal/inference_register.go
func init() { inference.Register(metalBackend{}) }
```

Renaming would break every backend. So this package exposes the canonical Core registration as **`RegisterCore(c *core.Core) core.Result`** instead, leaving the existing `Register(Backend)` untouched. Both names share a package; both keep their established consumers.

## Usage

```go
c, _ := core.New(core.WithService(inference.NewService(inference.Options{})))
svc := core.MustServiceFor[*inference.Service](c, "inference")

for name, b := range inference.All() {
    fmt.Printf("%s available=%v\n", name, b.Available())
}
```

## Options

```go
type Options struct{}
```

v1 has no fields. The package's behaviour is fully driven by which Backend implementations have called `Register(Backend)` at init time. Future fields land here as needed — preferred-backend-order override, ProbeBus subscribers, etc.

## Service

`*inference.Service` embeds `*core.ServiceRuntime[Options]` for typed Options access. The Service struct holds no state beyond Options + the Core handle; the real state (registered backends) lives in the package-global registry.

## Why a thin handle

The Service is **not the source of truth** — the global registry is. The Service is the Core-discovery surface that lets the framework's `core.ServiceFor` lookup find the package. This keeps the public-package shape stable while letting the framework treat inference like any other service for lifecycle (startup, shutdown, probes).

A backend's init-time `Register` does not need a Core handle. A consumer calling `inference.LoadModel(path)` does not need a Core handle. The Service is purely for framework-side discovery.

## Related

- `core/docs/service.md` — the canonical ServiceRuntime contract
- [inference.md](inference.md) — the global Backend registry the service surfaces
