---
title: go-inference
description: The sovereign local-inference repository for the Core Go ecosystem — contract, engines, serving, and the lem binary.
---

# go-inference

Module: `dappco.re/go/inference` · Go 1.26 · Licence EUPL-1.2.

go-inference is the sovereign local-inference repository for the Core Go ecosystem. It holds everything needed to run a local model in one place: the shared contract (`TextModel`, `Backend`, and supporting types), the GPU compute engines that implement it (`engine/metal`, `engine/hip`), the serving layer that exposes them over HTTP, and the `lem` binary.

## Why it exists

Earlier, this was a contract-only package that GPU backends in separate repositories (`go-mlx`, `go-rocm`) implemented. Those repositories are **retired** — their engines have been migrated in-tree, and `lem` now compiles from go-inference alone.

The contract still earns its keep: the root `inference` package defines the interfaces an engine implements and a consumer programs against, so a consumer loads a model and generates text without knowing which GPU runtime is underneath. What changed is that the engines now live in the same repository and register themselves against the contract at `init` time.

## Dependencies

The package consumes the Core externals (`dappco.re/go`, plus `api`, `cli`, `log`, `process`) and a handful of third-party libraries (Gin, the MCP SDK, DuckDB, parquet-go). It is **not** stdlib-only. Errors are constructed with `core.E(...)`; fallible calls return `core.Result`, not `(T, error)`.

## Quick start

```go
import "dappco.re/go/inference"

// Load a model (auto-detects the best available backend).
r := inference.LoadModel("/path/to/model/")
if !r.OK {
    log.Fatal(r.Error())
}
m := r.Value.(inference.TextModel)
defer m.Close()

// Stream tokens.
ctx := context.Background()
for tok := range m.Generate(ctx, "Once upon a time", inference.WithMaxTokens(128)) {
    fmt.Print(tok.Text)
}
if r := m.Err(); !r.OK {
    log.Fatal(r.Error())
}
```

`Generate` and `Chat` return an `iter.Seq[Token]` iterator; the trailing error is retrieved from `m.Err()`, which returns an OK Result on clean end-of-sequence.

## Engines

Two GPU engines live in-tree, each gated by build tags and registered via a blank import:

- **`engine/metal`** — Apple GPU (darwin/arm64), **no cgo**. Dispatches MLX's compiled Metal kernels directly through the objc runtime; the innovation is the Indirect Command Buffer (ICB) replay path for decode. Registers backend `"metal"`.
- **`engine/hip`** — AMD ROCm (linux/amd64), native HIP runtime. Registers backend `"rocm"`; a portable stub reports unavailable elsewhere.

```go
import (
    "dappco.re/go/inference"
    _ "dappco.re/go/inference/engine/metal" // registers "metal" on darwin/arm64
    _ "dappco.re/go/inference/engine/hip"   // registers "rocm" on linux/amd64
)
```

## Package layout

| Path | Purpose |
|------|---------|
| `inference.go` | `TextModel`, `Backend`, the registry, `LoadModel()` |
| `options.go` | `GenerateConfig`, `LoadConfig`, functional options (`WithMaxTokens`, `WithBackend`, …) |
| `training.go` | `TrainableModel`, `Adapter`, `LoRAConfig`, `LoadTrainable()` |
| `discover.go` | `Discover()` — recursive scan for model directories / GGUF files |
| `device.go` | `DeviceInfo`, `DeviceInfoProvider`, `BackendDeviceInfo()` |
| `engine/metal/` | Apple-GPU engine (package `native`, darwin/arm64, no cgo) |
| `engine/hip/` | AMD ROCm engine (package `hip`, linux/amd64) |
| `serving/` | OpenAI/Anthropic/Ollama-compatible HTTP servers over the engine |
| `model/`, `model/state/` | arch definitions; identity + agent-memory state |
| `cmd/lem/` | the `lem` binary — `serve`/`generate`/`ssd`/`sft`/`tune`/`pack`/`ebook`/`quant`/`spec` |

## Further reading

- [Documentation index](README.md) — the full doc tree (per-package pages under `inference/`, `state/`, `openai/`, …)
- [Architecture](architecture.md) — the repository as a whole
- [Interfaces](interfaces.md) — `TextModel`, `Backend`, `TrainableModel`, `Adapter`, optional capabilities
- [Types](types.md) — `Token`, `GenerateConfig`, `LoadConfig`, `LoRAConfig`, and supporting structs
- [Backends](backends.md) — the in-tree engines, the registry, implementing a new backend

## Stability contract

The root `inference` package is the shared contract. Changes there affect every engine, the serving layer, and consumers. The rules:

1. **Never change** existing method signatures on `TextModel` or `Backend`.
2. **Only add** methods when two or more call sites have a concrete need.
3. **New capability** is expressed as separate optional interfaces (`VisionModel`, `AttentionInspector`, `TrainableModel`, `DeviceInfoProvider`) discovered by type assertion — never by widening `TextModel`.
4. **New fields** on `GenerateConfig` or `LoadConfig` are safe — zero-value defaults preserve backwards compatibility.

## Requirements

- Go 1.26+ (uses `iter.Seq`, `maps`, `slices`)
- Consumes the Core externals via the `go.work` workspace (no `replace` directives)
- Engines are build-tag-gated: `engine/metal` needs darwin/arm64, `engine/hip` needs linux/amd64
- Licence: EUPL-1.2
</content>
