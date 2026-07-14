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
| `pkg/chat/multiturn` | conversation history — the message slice IS the memory |
| `pkg/chat/sampling` | temperature/top-k/top-p/min-p/seed, seeded reproducibility |
| `pkg/chat/stop` | stop tokens, suppression, min-tokens-before-stop |
| `pkg/chat/cancel` | cancelling generation: ctx cancel vs iterator break |
| `pkg/chat/budget` | `WithThinkingBudget` — capped thought channel |
| `pkg/chat/vision` | attach an image to a turn, with the capability probe |
| `pkg/chat/audio` | attach WAV audio, with the `AudioModel` probe |
| `pkg/chat/video` | attach video frames in time order |
| `pkg/chat/mtp` | MTP speculative pair (target + assistant drafter, darwin) |
| `pkg/generate` | raw completion — no chat template |
| `pkg/batch` | `BatchGenerate` with per-prompt error handling |
| `pkg/classify` | `Classify` + `WithLogits` — labels and margins |
| `pkg/discover` | scan a directory for loadable model snapshots |
| `pkg/info` | `Info()`, `ModelType()`, capability report |
| `pkg/trace` | per-token decode phase timing (`DecodePhases`) |
| `pkg/metrics-sink` | request-scoped usage under concurrent generations |
| `pkg/adapter` | load a LoRA adapter produced by `train/sft` |
| `pkg/tools` | function calling — declare, parse, execute, respond |
| `pkg/embed` | text embeddings via the host BERT encoder (no GPU) |
| `pkg/rerank` | query-vs-documents scoring (RAG's second half) |
| `pkg/tokenizer` | encode/decode and token counting (no GPU) |
| `pkg/quantise` | programmatic quantisation to GGUF recipes |
| `pkg/benchmark` | `Metrics()` — prefill/decode tok/s, token counts, peak GPU memory |
| `pkg/eval` | `Classify()` — batched one-token labels as a minimal eval harness |
| `pkg/backends` | the backend registry; pinning with `WithBackend` |
| `pkg/serve` | embed the OpenAI-compatible server in your own app |
| `pkg/serve/multimodel` | several resident models, aliases, memory ceiling |
| `pkg/state` | durable KV-state turns — the no-prompt-replay loop |
| `pkg/train/ssd` | self-distillation sampling: capture a trace (no training) |
| `pkg/train/sft` | LoRA fine-tuning on `{"messages"}` rows, adapter out |

Every folder carries its own README (generated from the `main.go` doc comment
— `python3 gen-readmes.py` after adding examples). Backend builds
(`<example>-{mlx,amd,cuda,cpu}`) are `make`-driven — see
[pkg/README.md](pkg/README.md) and the [Makefile](Makefile).

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
