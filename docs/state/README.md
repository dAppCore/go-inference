<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# state/ — durable model-state contracts

**Package**: `dappco.re/go/inference/state`

## What this package owns

The portable, backend-neutral contracts for **storing live model state
to a durable medium and restoring it later** — what the wider stack
calls "agent memory" or "book state". Everything in here is interfaces
and DTOs; no runtime code. Backends in `go-mlx`, `go-rocm` (planned),
`go-cuda` (planned) implement these contracts; consumers in `go-ai`,
`go-ml`, `core/api` use them.

This package was hoisted out of `dappco.re/go/inference` so the wire
shapes for state — `Bundle`, `Ref`, `Wake/Sleep/Fork` — could be
imported without dragging in the full backend-registry surface. The
parent `inference` package re-exports the most common types as
aliases (`inference.ModelIdentity = state.ModelIdentity` etc.) so
existing callers keep compiling.

## File map

| File | Doc | What it owns |
|------|-----|--------------|
| `agent_memory.go` | [agent_memory.md](agent_memory.md) | Wake/Sleep/Fork lifecycle DTOs + `Session` + `Forker` interfaces |
| `identity.go`     | [identity.md](identity.md)     | `ModelIdentity` / `TokenizerIdentity` / `AdapterIdentity` / `RuntimeIdentity` / `SamplerConfig` / `StateRef` / `Bundle` |
| `store.go`        | [store.md](store.md)        | `Store` / `Resolver` / `Writer` interfaces + `Chunk` / `ChunkRef` DTOs + `Resolve*` free fns + codec constants |
| `memory.go`       | [memory.md](memory.md)       | `InMemoryStore` — in-process test/dev backend |
| `filestore/store.go` | [filestore.md](filestore.md) | Append-only file-log durable backend |

## Mental model

```
                ┌───────────────────────┐
                │  Bundle  (identity.go)│   ← what gets persisted
                └───────────┬───────────┘
                            │ contains
                ┌───────────┴───────────┐
                │  []StateRef           │
                │  Model/Tokenizer/etc  │
                └───────────────────────┘
                            ▲
                            │ written by
                            │
   ┌──────────────────┐     │     ┌──────────────────┐
   │  Session.        │─────┘     │  Session.        │
   │  SleepState()    │           │  WakeState()     │
   │  (agent_memory)  │           │  (agent_memory)  │
   └─────────┬────────┘           └────────▲─────────┘
             │ produces                    │ consumes
             ▼                             │
   ┌──────────────────┐         ┌──────────┴────────┐
   │  Store.PutBytes  │         │  Store.Resolve... │
   │  Writer.Put      │         │  Resolver         │
   │  (store.go)      │         │  URIResolver      │
   └─────────┬────────┘         └──────────▲────────┘
             │                             │
             ▼                             │
   ┌─────────────────────────────────────────┐
   │  InMemoryStore  /  filestore.Store      │
   │  memvid.FileStore  /  s3.Store (future) │
   └─────────────────────────────────────────┘
```

A sleep produces a `Bundle` whose `KVRefs` / `ProbeRefs` /
`MemvidRefs` point at chunks written to some `Store`. A wake reads the
bundle, then reads each chunk back through the same Store. The two
interfaces in `agent_memory.go` (`Session` + `Forker`) are the only
runtime contracts; everything else is data.

## Codec constants

```go
state.CodecMemory          = "memory/plaintext"   // InMemoryStore
state.CodecQRVideo         = "memvid/qr-video"    // memvid .mp4
filestore.CodecFile        = "memvid/file-log"    // append-only file
```

A `ChunkRef` carries its codec so the wake side knows which decoder to
run — same bundle index can refer to chunks across multiple codecs if
the writer chose to spread them (rare but supported).

## Why this package exists at all

Three forces pushed it out of `inference`:

1. **Cycle pressure.** `inference.Backend` wants to mention bundles
   (capability reports, model-pack inspection); bundles want to
   mention chunks; chunks want to mention bytes. Splitting state out
   gave a clean acyclic graph.

2. **Cross-package re-use.** `core/api` wants to serialise bundles
   over HTTP without importing the full backend surface. `core/ide`
   wants to display bundle indexes without linking go-mlx. Both can
   now `import "dappco.re/go/inference/state"` and get just the
   shapes.

3. **Lifecycle clarity.** Wake/Sleep/Fork are a small focused
   contract; storage interfaces are another. Putting them in their
   own package made the "what's the smallest implementation" question
   answerable without grep.

## See also

- [Parent inference docs](../inference/README.md) — how state is
  consumed by `Backend` / `TextModel`
- [openai/services.md](../openai/services.md) — wire types that carry
  `ModelIdentity` in capability reports
- `go-mlx/docs/memory/agent_memory.md` (planned) — the reference
  Metal-backed Session implementation
- `go-mlx/docs/memory/state_bundle.md` (planned) — bundle
  encode/decode round-trip
