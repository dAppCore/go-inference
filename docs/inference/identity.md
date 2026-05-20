<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# identity.go — aliases to state + sampler conversion

**Package**: `dappco.re/go/inference`
**File**: `go/identity.go`

## What this is

A thin re-export layer. The identity types (`ModelIdentity`, `TokenizerIdentity`, etc.), the `Bundle` envelope, and project-seed helpers live in the `state` subpackage; this file aliases them into the parent `inference` package so consumers importing only `dappco.re/go/inference` see the common names.

Two real bits of code on top: `SamplerConfigFromGenerateConfig` + `GenerateConfigFromSamplerConfig`.

## Aliases

```go
type ModelIdentity     = state.ModelIdentity
type TokenizerIdentity = state.TokenizerIdentity
type AdapterIdentity   = state.AdapterIdentity
type RuntimeIdentity   = state.RuntimeIdentity
type SamplerConfig     = state.SamplerConfig
type StateRef          = state.StateRef
type StateBundle       = state.Bundle
type ProjectSeed       = state.ProjectSeed
```

A consumer writes:

```go
import "dappco.re/go/inference"

func report(c inference.CapabilityReport) {
    if c.Adapter.Hash == "" { ... }           // AdapterIdentity from inference
    bundle := inference.StateBundle{ ... }    // Bundle from inference
}
```

— and never needs to import `inference/state` directly.

## SamplerConfigFromGenerateConfig

```go
state.SamplerConfig = inference.SamplerConfigFromGenerateConfig(cfg)
```

Lowers a live `GenerateConfig` (which carries Go-typed defaults and option-fn lineage) to the portable `SamplerConfig` that fits into a `Bundle`. Used when persisting a session: the bundle records the **outcome** of sampler options, not the option-fn chain that produced them.

`StopTokens` is cloned (separate slice ownership) so the bundle isn't mutated when the live cfg is.

## GenerateConfigFromSamplerConfig

The inverse:

```go
cfg := inference.GenerateConfigFromSamplerConfig(bundle.Sampler)
for tok := range model.Generate(ctx, prompt, withGenerateConfig(cfg)) { ... }
```

Restores a sampler config from a bundle and produces the matching `GenerateConfig`. Note: `StopSequences` (text-mode stop strings) is in `SamplerConfig` but **not** in `GenerateConfig` — the conversion drops it, because the runtime path uses token-id stops, not strings. A future GenerateOption could re-introduce it.

## Why this re-export layer exists at all

The `state` package was hoisted out so the wire shapes for state could be imported without dragging in the full backend-registry surface (see `state/README.md` for the why). Re-exporting through `inference` keeps existing consumers' imports stable — code written before the split compiles unchanged.

## Related

- [../state/identity.md](../state/identity.md) — the real DTOs
- [../state/project_seed.md](../state/project_seed.md) — project-seed helpers and wake compatibility checks
- [options.md](options.md) — `GenerateConfig` / `GenerateOption`
- [../state/agent_memory.md](../state/agent_memory.md) — bundles consume these identities at Sleep
