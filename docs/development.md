# Development Guide — go-inference

## Prerequisites

- Go 1.25 or later (uses `iter.Seq` from Go 1.23 and range-over-function from 1.22)
- No CGO, no build tags, no external tools required
- The package compiles on macOS, Linux, and Windows without modification

## Commands

```bash
# Run all tests
go test ./...

# Run a single test by name
go test -run TestDefault_Good_Metal ./...

# Vet for common mistakes
go vet ./...

# View test coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

There is no Taskfile in this package; it is small enough that direct `go` invocations suffice. The parent workspace (`/Users/snider/Code/host-uk/core`) uses Task for cross-repo operations.

## Go Workspace

This package is part of the `host-uk/core` Go workspace. After adding or changing module dependencies:

```bash
go work sync
```

The workspace root is `/Users/snider/Code/host-uk/core`. The workspace file (`go.work`) includes this module alongside `cmd/core-gui`, `cmd/bugseti`, and others.

## Module Path

```
dappco.re/go/inference
```

Import it in consumers:

```go
import "dappco.re/go/inference"
```

Remote: `ssh://git@forge.lthn.ai:2223/core/go-inference.git`

## Repository Layout

```
go-inference/
├── inference.go        # TextModel, Backend, Token, Message, registry, LoadModel
├── options.go          # GenerateConfig, LoadConfig, all With* options
├── discover.go         # Discover() and DiscoveredModel
├── inference_test.go   # Tests for registry, LoadModel, all types
├── options_test.go     # Tests for GenerateConfig, LoadConfig, all options
├── discover_test.go    # Tests for Discover()
├── go.mod
├── go.sum
├── CLAUDE.md           # Agent instructions
├── README.md
└── docs/
    ├── architecture.md
    ├── development.md
    └── history.md
```

## Test Patterns

Tests follow the `_Good`, `_Bad`, `_Ugly` suffix convention used across the Core Go ecosystem:

- `_Good` — happy path; confirms the documented behaviour works correctly
- `_Bad` — expected error conditions; confirms errors are returned with useful messages
- `_Ugly` — edge cases, panics, surprising-but-valid behaviour (e.g. last-option-wins, registry overwrites)

```go
func TestDefault_Good_Metal(t *testing.T) { ... }
func TestDefault_Bad_NoBackends(t *testing.T) { ... }
func TestDefault_Ugly_SkipsUnavailablePreferred(t *testing.T) { ... }
```

### Backend Registry Isolation

Tests that touch the global backend registry call `resetBackends(t)` first. This helper clears the map and is defined in `inference_test.go`:

```go
func resetBackends(t *testing.T) {
    t.Helper()
    backendsMu.Lock()
    defer backendsMu.Unlock()
    backends = map[string]Backend{}
}
```

Because `resetBackends` is in the `inference` package (not `inference_test`), it has direct access to the unexported `backends` map. Tests must not rely on registration order across test functions; each test that uses the registry must call `resetBackends` at the top.

### Stub Implementations

`inference_test.go` provides `stubBackend` and `stubTextModel` — minimal implementations of `Backend` and `TextModel` for use in registry and routing tests. These are in the `inference` package itself (not a separate `_test` package) to allow access to unexported fields.

When writing new tests, use the existing stubs rather than creating new ones unless you need behaviour the stubs do not support.

### Table-Driven Tests

Prefer table-driven tests for options and configuration variants. The existing `TestApplyGenerateOpts_Good`, `TestWithTemperature_Good`, and `TestDefault_Good_PriorityOrder` tests demonstrate the pattern:

```go
tests := []struct {
    name string
    val  float32
    want float32
}{
    {"greedy", 0.0, 0.0},
    {"low", 0.3, 0.3},
}
for _, tt := range tests {
    t.Run(tt.name, func(t *testing.T) {
        cfg := ApplyGenerateOpts([]GenerateOption{WithTemperature(tt.val)})
        assert.InDelta(t, tt.want, cfg.Temperature, 0.0001)
    })
}
```

### Assertions

Use `testify/assert` and `testify/require`:

- `require` for preconditions where failure makes subsequent assertions meaningless (e.g. `require.NoError(t, err)` before using the returned value)
- `assert` for all other checks
- `assert.InDelta` for float32/float64 comparisons (never `==`)

## Coding Standards

### Language

UK English throughout: colour, organisation, centre, licence (noun), serialise, recognise. American spellings are not accepted in comments, documentation, or error messages.

### Formatting

Standard `gofmt` formatting. No custom style rules. Run `gofmt -w .` or `go fmt ./...` before committing.

### Error Messages

Error strings start with the package name and a colon, lowercase, no trailing period:

```go
fmt.Errorf("inference: no backends registered (import a backend package)")
fmt.Errorf("inference: backend %q not registered", cfg.Backend)
fmt.Errorf("inference: backend %q not available on this hardware", cfg.Backend)
```

This convention matches the Go standard library and makes `errors.Is`/`errors.As` wrapping straightforward.

### Strict Types

All parameters and return types are explicitly typed. No `interface{}` or `any` outside of test helpers where unavoidable.

### Dependencies

No new external dependencies may be added to the production code. The `go.mod` `require` block must remain stdlib-only for non-test code. `testify` is the only permitted test dependency.

If you find yourself wanting an external library, reconsider the approach. This package is intentionally minimal.

### Licence Header

Every new `.go` file must carry the EUPL-1.2 licence header:

```go
// Copyright (c) Lethean Technologies Ltd. All rights reserved.
// SPDX-License-Identifier: EUPL-1.2
```

Existing files without this header will be updated in a future housekeeping pass.

## Commit Guidelines

Use conventional commits:

```
type(scope): short imperative description

Longer explanation if needed. UK English. Wrap at 72 characters.
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`

Scope: `inference`, `options`, `discover`, or omit for cross-cutting changes.

Examples:

```
feat(inference): add WithParallelSlots load option
fix(discover): handle config.json with invalid JSON gracefully
test(options): add table-driven tests for WithTopP
docs: expand architecture section on registry priority
```

Always include the co-author trailer:

```
Co-Authored-By: Virgil <virgil@lethean.io>
```

## Implementing a Backend

To implement a new backend (e.g. `go-vulkan` for cross-platform GPU inference):

1. Import `dappco.re/go/inference` in the new module.
2. Implement `inference.Backend`:

```go
type vulkanBackend struct{}

func (b *vulkanBackend) Name() string { return "vulkan" }

func (b *vulkanBackend) Available() bool {
    // Check whether Vulkan runtime is present on this host.
    return vulkan.IsAvailable()
}

func (b *vulkanBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
    cfg := inference.ApplyLoadOpts(opts)
    // Load model using cfg.ContextLen, cfg.GPULayers, etc.
    return &vulkanModel{...}, nil
}
```

3. Implement `inference.TextModel` (all nine methods).
4. Register in `init()`, guarded by the appropriate build tag:

```go
//go:build linux && (amd64 || arm64)

func init() { inference.Register(&vulkanBackend{}) }
```

5. Write stub-based tests to confirm the backend registers and `LoadModel` routes correctly without requiring real GPU hardware in CI.

## Extending the Interface

Before adding a method to `TextModel` or `Backend`, consider:

- Do two or more existing consumers require this capability right now?
- Can the capability be expressed as a separate interface that embeds `TextModel`?
- Will adding this method break existing backend implementations that do not yet provide it?

If the answer to the first question is no, defer the addition. If a separate interface is sufficient, prefer that approach. See `docs/architecture.md` for the stability contract.

When a new method is genuinely necessary, coordinate with the owners of go-mlx, go-rocm, and go-ml before merging, since all three must implement the new method simultaneously or the interface will be broken at build time.
