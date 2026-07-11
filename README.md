[![Go Reference](https://pkg.go.dev/badge/dappco.re/go/inference.svg)](https://pkg.go.dev/dappco.re/go/inference)
[![Licence: EUPL-1.2](https://img.shields.io/badge/Licence-EUPL--1.2-blue.svg)](LICENCE)
[![Go Version](https://img.shields.io/badge/Go-1.26-00ADD8?style=flat&logo=go)](go.mod)

# go-inference

**The one repo for local model inference in the Core Go ecosystem.** It carries the
whole stack — the GPU engines, the OpenAI/Anthropic/Ollama-compatible server, the
training loops, the `lem` command-line binary, and the desktop GUI. go-mlx and
go-rocm are retired; everything lives here now. The design goal: **you only need
go-inference** — one repo, and (with `task build:embed`) one self-contained binary.

**Module**: `dappco.re/go/inference` · **Licence**: EUPL-1.2 · **Go**: 1.26

## What's inside

| Area | Package | What it is |
|------|---------|-----------|
| **Engines** | `engine/metal` | Apple-GPU engine — **no cgo**, dispatches Apple MLX's compiled kernels + go-inference's own fused `lthn_` kernels via the Objective-C runtime; the ICB replay path replaces MLX's per-step re-encode (darwin/arm64) |
| | `engine/hip` | AMD-GPU engine (linux/amd64, ROCm) — built on the AMD box from this same repo |
| **Serving** | `serving/` | Native OpenAI / Anthropic / Ollama HTTP servers backed by the local engine (`/v1/chat/completions`, `/v1/messages`, `/api/chat`, …) + scheduler, sessions, chat history |
| **Binary** | `cmd/lem` | `lem` — `serve`, `generate`, `ssd`/`sft`/`tune` (training), `pack`/`ebook` |
| **Training** | `train/`, `eval/` | LoRA SFT, self-distillation (SSD), MTP tuning, the score cascade + capture, DuckDB/Influx metrics |
| **Core lib** | `inference`, `model/`, `kv/`, `decode/` | `TextModel`/`Backend`/`Token`/`Message` contracts, model loading, KV cache + portable snapshots, tokenizer + sampler |
| **GUI** | `gui/` | The LEM desktop app (Wails v3 — system tray + dashboard), a side module (`dappco.re/go/inference/gui`) |
| **State** | `state/`, `agent/` | Wake/Sleep/Fork agent memory, the scoring agent loop |

## The `lem` binary

```bash
task metallib          # build the Metal kernel libraries (once) -> build/dist/lib/
task build             # -> bin/lem  (resolves metallibs via MLX_METALLIB_PATH)
task build:embed       # -> bin/lem  SELF-CONTAINED (both metallibs baked in; runs anywhere)

lem serve --model ~/models/gemma-4-e2b-it-4bit      # OpenAI/Anthropic/Ollama HTTP on :36911
lem generate --max-tokens 256 --prompt "Hello" ~/models/gemma-4-e2b-it-4bit
lem sft   -model <bf16> -data train.jsonl -score-cascade    # LoRA fine-tune
```

Point any OpenAI or Ollama client at `http://localhost:36911`.

## The Metal build chain

The Apple engine dispatches two compiled kernel libraries, both **built from source in
this repo** (no go-mlx dependency):

- **`mlx.metallib`** — Apple's MLX kernels (`steel_gemm`, `affine_qmv`, `vv_*`, rms, rope).
  Built by CMake from `external/mlx` (Apple's `ml-explore/mlx` pinned at v0.31.2) with the
  10 **lthn patches** in `patches/mlx/` applied on top (decode-replay, `MLX_METALLIB_PATH`
  override, 512-dim sdpa). Patch-not-vendor: bump the pin + rebase to pull MLX updates.
- **`lthn_kernels.metallib`** — go-inference's own fused kernels (`engine/metal/kernels/*.metal`).

`task build:embed` gzips both into the binary so `lem` runs from any path with nothing
external to ship.

## Quick Start (library)

```go
import (
    "dappco.re/go/inference"
    _ "dappco.re/go/inference/engine/metal"   // registers the "metal" backend (darwin/arm64)
    _ "dappco.re/go/inference/model/builtin"  // registers gemma3/gemma4/mistral/qwen3
)

r := inference.LoadModel("/path/to/model/")   // core.Result
model := r.Value.(inference.TextModel)
defer model.Close()

for tok := range model.Generate(ctx, "Hello", inference.WithMaxTokens(256)) {
    fmt.Print(tok.Text)
}
```

## Documentation

- [Architecture](docs/architecture.md) — engines, serving, registry, contracts, ecosystem position
- [Backends](docs/backends.md) — the Metal + HIP engines
- [Serving](docs/openai/README.md) — OpenAI / Anthropic / Ollama compat
- [Inference](docs/inference/README.md) — contracts, options, training, gguf
- [State](docs/state/README.md) — agent memory, Wake/Sleep/Fork
- [Development](docs/development.md) — build, test, coding standards

## Build & Test

```bash
go test ./...     # tests compile + run without a GPU (engines are build-tagged)
go vet ./...
task metallib && task build     # the full GPU binary
```

## Licence

European Union Public Licence 1.2 — see [LICENCE](LICENCE).
