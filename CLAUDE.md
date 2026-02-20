# CLAUDE.md

## What This Is

Shared inference interfaces for the Core Go ecosystem. Module: `forge.lthn.ai/core/go-inference`

Zero dependencies. Compiles on all platforms. See `docs/architecture.md` for design rationale.

## Commands

```bash
go test ./...        # Run all tests
go vet ./...         # Vet
```

## Stability Rules

This package is the shared contract. Changes here affect go-mlx, go-rocm, and go-ml simultaneously.

- Never change existing method signatures on `TextModel` or `Backend`
- Only add methods when two or more consumers need them
- Prefer new interfaces that embed `TextModel` over extending `TextModel` itself
- New fields on `GenerateConfig` or `LoadConfig` are safe (zero-value defaults)
- All new interface methods require Virgil approval before merging

## Coding Standards

- UK English
- Zero external dependencies — stdlib only (testify permitted in tests)
- Conventional commits: `type(scope): description`
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
- Licence: EUPL-1.2

## Consumers

- **go-mlx**: implements `Backend` + `TextModel` for Apple Metal (darwin/arm64)
- **go-rocm**: implements `Backend` + `TextModel` for AMD ROCm (linux/amd64)
- **go-ml**: wraps inference backends into scoring engine, adds llama.cpp HTTP backend
- **go-ai**: MCP hub, exposes inference via MCP tools
- **go-i18n**: uses `TextModel` for Gemma3-1B domain classification

## Documentation

- `docs/architecture.md` — interfaces, registry, options, design decisions
- `docs/development.md` — prerequisites, build, test patterns, coding standards
- `docs/history.md` — completed phases, commit log, known limitations
