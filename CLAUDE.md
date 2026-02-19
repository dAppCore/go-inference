# CLAUDE.md

## What This Is

Shared inference interfaces for the Core Go ecosystem. Module: `forge.lthn.ai/core/go-inference`

This package defines the contract between GPU-specific backends (go-mlx on macOS, go-rocm on Linux) and consumers (go-ml, go-ai, go-i18n). It has **zero dependencies** and compiles on all platforms.

## Commands

```bash
go test ./...        # Run all tests
go vet ./...         # Vet
```

## Architecture

```
go-inference (this package) ← defines TextModel, Backend, Token, Message
    ↑                    ↑
    │                    │
go-mlx (darwin/arm64)   go-rocm (linux/amd64)
    │                    │
    └────── go-ml ───────┘   (wraps backends into scoring engine)
             ↑
          go-ai (MCP hub)
```

### Key Types

| Type | Purpose |
|------|---------|
| `TextModel` | Core interface: Generate, Chat, Err, Close |
| `Backend` | Named engine that can LoadModel → TextModel |
| `Token` | Streaming token (ID + Text) |
| `Message` | Chat message (Role + Content) |
| `GenerateOption` | Functional option for generation (temp, topK, etc.) |
| `LoadOption` | Functional option for model loading (backend, GPU layers, etc.) |

### Backend Registry

Backends register via `init()` with build tags. Consumers call `LoadModel()` which auto-selects the best available backend:

```go
// Auto-detect: Metal on macOS, ROCm on Linux
m, err := inference.LoadModel("/path/to/model/")

// Explicit backend
m, err := inference.LoadModel("/path/", inference.WithBackend("rocm"))
```

## Coding Standards

- UK English
- Zero external dependencies — stdlib only
- Tests: testify assert/require
- Conventional commits
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Licence: EUPL-1.2

## Consumers

- **go-mlx**: Implements `Backend` + `TextModel` for Apple Metal (darwin/arm64)
- **go-rocm**: Implements `Backend` + `TextModel` for AMD ROCm (linux/amd64)
- **go-ml**: Wraps inference backends into scoring engine, adds llama.cpp HTTP backend
- **go-ai**: MCP hub, exposes inference via MCP tools
- **go-i18n**: Uses TextModel for Gemma3-1B domain classification

## Stability

This package is the shared contract. Changes here affect all backends and consumers. Keep the interface minimal and stable. Add new methods only when two or more consumers need them.

## Task Queue

See `TODO.md` for prioritised work.
See `FINDINGS.md` for research notes.
