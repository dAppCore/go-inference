# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Shared inference interfaces for the Core Go ecosystem. Module: `dappco.re/go/core/inference`

Zero external dependencies (stdlib only). Compiles on all platforms. See `docs/architecture.md` for design rationale.

## Commands

```bash
go test ./...                          # Run all tests
go test -run TestDefault_Good_Metal    # Run a single test by name
go vet ./...                           # Vet
golangci-lint run ./...                # Lint (govet, errcheck, staticcheck, gocritic, gofmt, etc.)
```

## Architecture

This is a pure interface package — it defines contracts but contains no backend implementations. The dependency flows one way: backends import this package, never the reverse.

**Core files:**
- `inference.go` — `TextModel` and `Backend` interfaces, `Token`/`Message`/`ClassifyResult`/`BatchResult` types, backend registry (`Register`/`Get`/`List`/`Default`), top-level `LoadModel()` router
- `options.go` — `GenerateConfig`/`LoadConfig` structs with functional options (`With*` functions) and `Apply*Opts` helpers
- `discover.go` — `Discover()` scans directories for model dirs (config.json + *.safetensors)
- `training.go` — `TrainableModel` interface (extends `TextModel` with LoRA), `Adapter` interface, `LoadTrainable()`

**Backend registry pattern:** Backends register via `init()` with build tags (e.g. `//go:build darwin && arm64`). `Default()` picks backends in priority order: metal > rocm > llama_cpp > any available. `LoadModel()` routes to explicit backend via `WithBackend()` or falls back to `Default()`.

**Optional interfaces via type assertion:** New capabilities are expressed as separate interfaces (e.g. `AttentionInspector`, `TrainableModel`) rather than extending `TextModel`. Consumers discover them with `model.(inference.AttentionInspector)`.

**Streaming uses `iter.Seq[Token]`:** Generate/Chat return Go 1.23+ range-over-function iterators. Errors are retrieved via `Err()` after the iterator finishes (follows `database/sql` `Row.Err()` pattern).

## Stability Rules

This package is the shared contract. Changes here affect go-mlx, go-rocm, and go-ml simultaneously.

- Never change existing method signatures on `TextModel` or `Backend`
- Only add methods when two or more consumers need them
- Prefer new interfaces that embed `TextModel` over extending `TextModel` itself
- New fields on `GenerateConfig` or `LoadConfig` are safe (zero-value defaults)
- All new interface methods require Virgil approval before merging

## Test Patterns

Tests use the `_Good`/`_Bad`/`_Ugly` suffix convention:
- `_Good` — happy path
- `_Bad` — expected error conditions
- `_Ugly` — edge cases, surprising-but-valid behaviour

Tests touching the global backend registry must call `resetBackends(t)` first (defined in `inference_test.go`, clears the registry map). Use existing `stubBackend`/`stubTextModel` from `inference_test.go` rather than creating new stubs.

Use `testify/assert` (general checks) and `testify/require` (preconditions). Use `assert.InDelta` for float comparisons.

## Coding Standards

- UK English (colour, organisation, serialise, licence)
- Zero external dependencies — stdlib only (testify permitted in tests)
- Error strings: `fmt.Errorf("inference: lowercase message without trailing period")`
- Conventional commits: `type(scope): description` — scopes: `inference`, `options`, `discover`
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Licence: EUPL-1.2

## Consumers

- **go-mlx**: implements `Backend` + `TextModel` for Apple Metal (darwin/arm64)
- **go-rocm**: implements `Backend` + `TextModel` for AMD ROCm (linux/amd64)
- **go-ml**: wraps inference backends into scoring engine, adds llama.cpp HTTP backend
- **go-ai**: MCP hub, exposes inference via MCP tools
- **go-i18n**: uses `TextModel` for Gemma3-1B domain classification
