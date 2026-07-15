# Development Guide — go-inference

go-inference is **the** sovereign inference repo for the Core Go ecosystem. It
carries the GPU engines, the OpenAI/Anthropic/Ollama-compatible server, the
training loops, the `lem` binary, and the LEM desktop GUI. go-mlx and go-rocm are
retired — everything lives here now.

For the `lem` verbs see [cmd-lem.md](cmd-lem.md); for the Metal build chain see
[build.md](build.md); for the desktop app see [gui.md](gui.md).

## Prerequisites

- **Go 1.26** (the modules declare `go 1.26.2`).
- Plain `go test ./...` and `go vet ./...` compile and run **without a GPU** —
  the engines are build-tagged (`metal_runtime` for Apple, cgo + linux/amd64 for
  HIP), so CI and routine dev do not need Metal or ROCm.
- Building the **Apple engine** binary needs macOS 26+, the Xcode command-line
  tools, CMake, and Task — see [build.md](build.md).
- The **HIP engine** (`engine/hip`) carries cgo and is built from this same repo
  on the AMD/linux box.

### Windows

`engine/hip`'s portable-stub build tag (`!linux || !amd64`, see
[backends.md](backends.md)) covers Windows, so the engine package itself
cross-compiles cleanly:

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build ./engine/hip/...   # OK
```

The full `lem` binary does not: `go build ./lem/... (from cli/)` under the same
`GOOS=windows CGO_ENABLED=0` fails in `github.com/marcboeker/go-duckdb`'s
generated Windows bindings (`undefined: bindings.Type` and friends) —
`eval/datapipe`, `serving/chathistory`, and the agent execute-history path all
import `go-duckdb` unconditionally, and go-duckdb needs cgo linked against
libduckdb; its pure-Go Windows binding stub does not stand alone. Two ways to
unblock a Windows `lem`, neither attempted here:

1. **Build on Windows with cgo** — `CGO_ENABLED=1` plus a Windows C toolchain
   and the DuckDB Windows library (`CGO_LDFLAGS` pointed at it), per
   go-duckdb's own cross-compile instructions.
2. **A pure-Go store behind a build tag** — swap the DuckDB-backed store for a
   pure-Go implementation on `!cgo`/Windows builds, so `CGO_ENABLED=0` stays
   viable. This is real work (a new store implementation), not a config flip.

## Module layout

The repository holds two Go modules plus vendored externals:

| Path | Module | What |
|------|--------|------|
| `go/` | `dappco.re/go/inference` | the whole inference stack (engines, serving, training, model, kv, decode, the `lem` binary) |
| `gui/` | `dappco.re/go/inference/gui` | the LEM desktop app (a side module — see [gui.md](gui.md)) |
| `external/` | (submodules) | core dependencies pulled locally for workspace builds |
| `patches/mlx/` | — | the lthn patch set applied to Apple MLX at build time |

```
go-inference/
├── go/                      # module dappco.re/go/inference
│   ├── (moved) cli/     # the `lem` binary (thin verb wiring — own module at the repo root)
│   ├── engine/
│   │   ├── metal/           # Apple GPU engine — NO cgo (tmc/apple bindings)
│   │   │   └── kernels/     # the fused *.metal sources
│   │   └── hip/             # AMD GPU engine — cgo, linux/amd64
│   ├── serving/             # OpenAI/Anthropic/Ollama HTTP + scheduler, sessions
│   ├── model/               # architectures, gguf, pack, quant, safetensors, …
│   ├── decode/              # generate, tokenizer, sampler, parser
│   ├── kv/                  # KV cache + portable snapshots
│   ├── train/               # LoRA SFT, self-distillation, tune, grpo
│   ├── eval/                # datapipe (Influx/DuckDB), probe, score, bench
│   ├── agent/               # the scoring agent loop
│   └── inference.go, …      # the TextModel/Backend/Token/Message contracts
├── gui/                     # module dappco.re/go/inference/gui (Wails v3)
├── external/                # core + third-party submodules (workspace)
├── patches/mlx/             # the 10 lthn MLX patches
├── Taskfile.yml             # metallib + build + build:embed
├── go.work                  # workspace: go/, gui/, external/*
└── docs/
```

## Go workspace

Development uses **workspace mode**. `go.work` at the repo root `use`s `./go`,
`./gui`, and every `external/<dep>/go` submodule, so local edits to the core
dependencies are picked up without a `replace` directive. After adding or
changing module dependencies:

```bash
go work sync
```

The `external/` submodules track the **`dev`** branch of the `github.com/dappcore`
repos (`go`, `go-io`, `api`, `cli`, `go-container`, `mcp`, `go-scm`, …), plus
Apple's `ml-explore/mlx` for the Metal build. Initialise them on a fresh clone
with `git submodule update --init --recursive`.

**CI** runs with `GOWORK=off`, which falls back to `go/go.mod`'s tagged
`require` versions for reproducible resolution.

## Remotes

Per house policy: **forge.lthn.sh** (homelab) is canonical, **forge.lthn.ai**
(de1) is the public mirror, and GitHub (`github.com/dappcore`) is downstream.
Note the local checkout's `origin` currently points at the mirror
(`ssh://git@forge.lthn.ai:2223/core/go-inference.git`), with a separate
`homelab` remote at `https://forge.lthn.sh/core/go-inference.git` — push to the
canonical remote, non-force.

## Dependencies

go-inference is no longer a stdlib-only contract package. `go/go.mod` consumes
the core primitives (`dappco.re/go`, `dappco.re/go/api`, `dappco.re/go/cli`,
`dappco.re/go/log`, `dappco.re/go/process`, and the `external/` family via the
workspace) plus third-party libraries where warranted (gin, go-duckdb, parquet,
the MCP Go SDK). The GUI additionally depends on Wails v3.

House rules for production code (enforced across the Core Go ecosystem):

- Errors via `core.E(...)`, never `fmt.Errorf`.
- Results are `core.Result` (`core.Ok` / `core.Fail`), not naked `(value, error)`
  pairs, on library boundaries.
- I/O through the core wrappers (`c.Fs()`, `c.Process()`, `coreio.Local`), not
  raw `os` / `os/exec`.
- Banned raw stdlib imports where a core wrapper exists: `os`, `os/exec`, `fmt`,
  `log`, `errors`, `strings`, `path/filepath`, `encoding/json`. (The
  `embed_metallib.go` build helper is a deliberate exception — it runs in
  `init()` before core is set up and uses raw `os`/`io`/`compress/gzip`.)

## Commands

```bash
go test ./...                              # all tests (no GPU needed)
go test -run TestBackend_Good_Metal ./...  # a single test by name
go vet ./...
golangci-lint run ./...                    # lint

# coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

For the GPU binary and kernel libraries, use Task — see [build.md](build.md):

```bash
task metallib      # build both Metal kernel libraries
task build         # -> bin/lem (external metallibs)
task build:embed   # -> bin/lem (self-contained, both metallibs baked in)
```

## Test patterns

Tests follow the ecosystem conventions:

- **One test per symbol per variant**, with the `_Good` / `_Bad` / `_Ugly`
  suffix convention:
  - `_Good` — happy path; the documented behaviour works.
  - `_Bad` — expected error conditions return useful errors.
  - `_Ugly` — edge cases and surprising-but-valid behaviour (last-option-wins,
    registry overwrites, …).
- File-per-concern testing: a `{file}.go` ships `{file}_test.go`, plus
  `{file}_example_test.go` (usage examples that double as AX documentation) and
  `{file}_bench_test.go` where a bench is meaningful.
- `testify/assert` for general checks, `testify/require` for preconditions where
  a failure makes later assertions meaningless. Use `assert.InDelta` for float
  comparisons, never `==`.
- Table-driven tests for option/config variants.

Tests that touch the global backend registry reset it first so registration
order across test functions does not leak.

## Coding standards

- **UK English throughout**: colour, organise, centre, licence (noun),
  serialise, recognise. American spellings are not accepted in comments,
  documentation, or error messages.
- **Formatting**: standard `gofmt`. Run `go fmt ./...` before committing.
- **Licence header**: every `.go` file carries the EUPL-1.2 SPDX line, in UK
  spelling:

  ```go
  // SPDX-Licence-Identifier: EUPL-1.2
  ```

- **Commits**: conventional commits (`type(scope): description`), UK English,
  wrapped at 72 characters. Always include the trailer:

  ```
  Co-Authored-By: Virgil <virgil@lethean.io>
  ```

## Adding an engine backend

An engine is a self-registering runtime package behind
`inference.Register` / `inference.LoadModel` (`WithBackend("<name>")`). To add
one:

1. Implement the `inference.Backend` and `inference.TextModel` contracts (plus
   any optional capability interfaces the engine supports — capabilities are
   discovered by type assertion, e.g. `model.(inference.AttentionInspector)`,
   rather than by widening `TextModel`).
2. Register in `init()`, guarded by the appropriate build tag for the platform.
3. Write stub-based tests that confirm registration and load routing without
   requiring real GPU hardware in CI.

Both current engines live in-repo (`engine/metal`, `engine/hip`), so extending
the contract no longer means coordinating across separate backend repositories —
add the capability as an optional interface and let engines opt in. See
[docs/architecture.md](architecture.md) for the stability contract and
[docs/backends.md](backends.md) for the engine designs.
