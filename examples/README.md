<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-inference examples

Minimal, self-contained programs — one folder per feature, one `main.go` per
folder, wails-style. Each shows the smallest honest use of one part of the
library; the comments in each file are the documentation.

## Layout

| Example | Shows |
|---------|-------|
| `pkg/chat/basic` | load a model, one chat turn, print the reply |
| `pkg/chat/stream` | per-token streaming (the iterator IS the stream) |
| `pkg/chat/thinking` | the Gemma 4 thought channel, split into thought + answer |
| `pkg/chat/vision` | attach an image to a turn, with the capability probe |
| `pkg/benchmark` | `Metrics()` — prefill/decode tok/s, token counts, peak GPU memory |
| `pkg/eval` | `Classify()` — batched one-token labels as a minimal eval harness |
| `pkg/train/ssd` | self-distillation sampling: capture a trace (no training) |
| `pkg/train/sft` | LoRA fine-tuning on `{"messages"}` rows, adapter out |

`cli/` is reserved for examples that drive the `lem` binary rather than the
library.

## Running

Apple Silicon (the metal engine) today; the hip engine (AMD/linux) joins at
feature parity. The engine resolves its shader library from
`MLX_METALLIB_PATH` — from a repo checkout:

```sh
task metallib                       # once: builds build/dist/lib/{mlx,lthn_kernels}.metallib
export MLX_METALLIB_PATH=$PWD/build/dist/lib/mlx.metallib

cd examples
go run ./pkg/chat/basic -model <snapshot dir>
```

`-model` (or `LEM_MODEL`) is a model snapshot directory — `config.json` +
`*.safetensors`, e.g. an `mlx-community/gemma-4-*` HF snapshot:

```sh
hf download mlx-community/gemma-4-e2b-it-4bit    # chat / benchmark / eval
hf download mlx-community/gemma-4-E2B-it-bf16    # train examples (train in bf16)
```

## Module resolution

Inside the repository the root `go.work` points these examples at the live
`./go` tree — you are always building against the code you are reading. As a
standalone module, `examples/go.mod` resolves the released
`dappco.re/go/inference` (tag `go/v0.12.0`).
