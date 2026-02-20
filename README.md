# go-inference

Shared interface contract for text generation backends in the Core Go ecosystem. Defines `TextModel`, `Backend`, `Token`, `Message`, and associated configuration types that GPU-specific backends implement and consumers depend on. Zero external dependencies — stdlib only — and compiles on all platforms regardless of GPU availability. The backend registry supports automatic selection (Metal preferred on macOS, ROCm on Linux) and explicit pinning.

**Module**: `forge.lthn.ai/core/go-inference`
**Licence**: EUPL-1.2
**Language**: Go 1.25

## Quick Start

```go
import (
    "forge.lthn.ai/core/go-inference"
    _ "forge.lthn.ai/core/go-mlx"   // registers "metal" backend on darwin/arm64
)

model, err := inference.LoadModel("/path/to/safetensors/model/")
defer model.Close()

for tok := range model.Generate(ctx, "Hello", inference.WithMaxTokens(256)) {
    fmt.Print(tok.Text)
}
```

## Documentation

- [Architecture](docs/architecture.md) — interfaces, registry, options, stability contract, ecosystem position
- [Development Guide](docs/development.md) — prerequisites, build, test patterns, coding standards
- [Project History](docs/history.md) — completed phases, commit log, known limitations

## Build & Test

```bash
go test ./...
go vet ./...
go build ./...
```

## Licence

European Union Public Licence 1.2 — see [LICENCE](LICENCE) for details.
