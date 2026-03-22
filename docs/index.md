---
title: go-inference
description: Shared interfaces for text generation backends in the Core Go ecosystem.
---

# go-inference

Module: `dappco.re/go/core/inference`

go-inference defines the shared contract between GPU-specific inference backends and their consumers. It contains the interfaces, types, and registry that let a consumer load a model and generate text without knowing which GPU runtime is underneath.

## Why it exists

The Core Go ecosystem has multiple inference backends:

- **go-mlx** — Apple Metal on macOS (darwin/arm64), native GPU memory access
- **go-rocm** — AMD ROCm on Linux (linux/amd64), llama-server subprocess
- **go-ml** — scoring engine, also wraps llama.cpp HTTP as a third backend path

And multiple consumers:

- **go-ai** — MCP hub exposing inference via 30+ agent tools
- **go-i18n** — domain classification via Gemma3-1B
- **go-ml** — training pipeline, scoring engine

Without a shared interface layer, every consumer would need to import every backend directly, dragging in CGO bindings, Metal frameworks, and ROCm libraries on platforms that cannot use them.

go-inference breaks that coupling. A backend imports go-inference and implements its interfaces. A consumer imports go-inference and programs against those interfaces. Neither needs to know about the other at compile time.

## Zero dependencies

The package imports only the Go standard library. The sole exception is `testify` in the test tree. This is a deliberate constraint — the package sits at the base of a dependency graph where backends pull in heavyweight GPU libraries. None of those concerns belong in the interface layer.

## Ecosystem position

```
go-inference (this package)
      |
      |── implemented by ────────────────────────
      |                                          |
   go-mlx                                    go-rocm
   (darwin/arm64, Metal GPU)        (linux/amd64, AMD ROCm)
      |                                          |
      └──────────── consumed by ─────────────────┘
                         |
                      go-ml
               (scoring engine, llama.cpp HTTP)
                         |
                      go-ai
                (MCP hub, 30+ tools)
                         |
                     go-i18n
              (domain classification)
```

## Package layout

| File | Purpose |
|------|---------|
| `inference.go` | `TextModel`, `Backend` interfaces, backend registry, `LoadModel()` entry point |
| `options.go` | `GenerateConfig`, `LoadConfig`, functional options (`WithMaxTokens`, `WithBackend`, etc.) |
| `training.go` | `TrainableModel`, `LoRAConfig`, `Adapter` interfaces, `LoadTrainable()` |
| `discover.go` | `Discover()` scans directories for model files (config.json + *.safetensors) |

## Quick start

```go
import "dappco.re/go/core/inference"

// Load a model (auto-detects the best available backend)
m, err := inference.LoadModel("/path/to/model/")
if err != nil {
    log.Fatal(err)
}
defer m.Close()

// Stream tokens
ctx := context.Background()
for tok := range m.Generate(ctx, "Once upon a time", inference.WithMaxTokens(128)) {
    fmt.Print(tok.Text)
}
if err := m.Err(); err != nil {
    log.Fatal(err)
}
```

## Further reading

- [Interfaces](interfaces.md) — `TextModel`, `Backend`, `TrainableModel`, `AttentionInspector`
- [Types](types.md) — `Token`, `GenerateConfig`, `LoadConfig`, `LoRAConfig`, and all supporting structs
- [Backends](backends.md) — How the registry works, how to implement a new backend

## Stability contract

This package is the shared contract. Changes here affect go-mlx, go-rocm, and go-ml simultaneously. The rules:

1. **Never change** existing method signatures on `TextModel` or `Backend`.
2. **Only add** methods when two or more consumers have a concrete need.
3. **New capability** is expressed as separate interfaces that embed `TextModel`, not by extending `TextModel` itself. Consumers opt in via type assertion.
4. **New fields** on `GenerateConfig` or `LoadConfig` are safe — zero-value defaults preserve backwards compatibility.

## Requirements

- Go 1.26+ (uses `iter.Seq`, `maps`, `slices`)
- No CGO, no build tags, no platform constraints
- Licence: EUPL-1.2
