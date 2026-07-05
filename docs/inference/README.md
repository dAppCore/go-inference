<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# inference/ — contract package root

**Package**: `dappco.re/go/inference`

## What this package owns

The **central contract** every backend and consumer in this repo speaks. Pure interfaces, DTOs, registries, and option types — the contract files import only `dappco.re/go` plus sibling `inference/*` subpackages, no CGO and no platform branches, so this package compiles everywhere. Backends (`engine/metal`, `engine/hip`) live in-repo behind build tags and register themselves at init time; go-mlx and go-rocm are retired and their proven code has migrated here — go-inference is now the sovereign inference repo.

Three categories:

| Category | What | Files |
|----------|------|-------|
| **Core runtime** | TextModel + Backend + registry + LoadModel | [inference.md](inference.md) |
| **Options** | GenerateOption + LoadOption + With* | [options.md](options.md) |
| **Extension** | Scheduler, Cache, Embedding, Rerank, ToolParse, ReasoningParse, ModelPackInspect | [contracts.md](contracts.md) |
| **Static intro** | CapabilityReport / AlgorithmProfile / RuntimeMemoryLimits | [capability.md](capability.md) |
| **Local setup** | MachineDiscoverer / TuningPlanner / model replace | [local_tuning.md](local_tuning.md) |
| **Dynamic observe** | ProbeEvent / ProbeSink | [probe.md](probe.md) |
| **Lifecycle** | Service + RegisterCore (Mantis #1336) | [service.md](service.md) |
| **Training** | TrainableModel + Adapter + LoRAConfig | [training.md](training.md) |
| **Discovery** | Discover() | [discover.md](discover.md) |
| **Format reader** | GGUFInfo | [gguf.md](gguf.md) |
| **Data shape** | DatasetSample + DatasetStream | [dataset.md](dataset.md) |
| **Re-export aliases** | identity types into the parent pkg | [identity.md](identity.md) |

## How the pieces fit

```
LoadModel(path, opts...)                  ← caller entry
   │
   ├──→ Default() / Get(name)             ← registry lookup
   │       │
   │       └──→ Backend.LoadModel(...)    ← native driver
   │              │
   │              └──→ returns TextModel  ← what the caller uses
   │
   └──→ Caller: model.Generate(ctx, prompt, WithMaxTokens(64))
                model.Chat(ctx, msgs, WithTemperature(0.7))
                model.Classify(ctx, prompts)
                model.BatchGenerate(ctx, prompts)
                ...

Optionally:
   if sched, ok := model.(SchedulerModel);    ok { ... }   ← contracts.go
   if cache, ok := model.(CacheService);      ok { ... }
   if embed, ok := model.(EmbeddingModel);    ok { ... }
   if train, ok := model.(TrainableModel);    ok { ... }   ← training.go
   if probe, ok := model.(CapabilityReporter);ok { report := probe.Capabilities() }
```

## Sibling packages

- [../state/](../state/README.md) — durable state DTOs + Wake/Sleep/Fork lifecycle (package `dappco.re/go/inference/model/state`)
- [../openai/](../openai/README.md) — OpenAI wire types + HTTP handlers
- [../anthropic/](../anthropic/anthropic.md) — Anthropic Messages wire types
- [../ollama/](../ollama/ollama.md) — Ollama-compatible wire types

The compat handlers themselves are served from `serving/` (`serving/compat`, `serving/provider/*`).

## Stability rules

This package is the shared contract. Changes here cascade to every backend and consumer.

- **No new methods on `TextModel` or `Backend`** without a Virgil review.
- **Prefer new interfaces over wider TextModel.** New capabilities land in `contracts.go` as opt-in extensions.
- **New fields on `GenerateConfig` / `LoadConfig` are safe** when zero-value defaults preserve old behaviour.
- **Wire DTOs in openai/anthropic/ollama track upstream** — adding fields is safe, renaming requires upstream rename first.

## Coding standards (this repo)

- UK English in code, comments, docs (colour, organisation, licence, serialise)
- SPDX header on every new file: `// SPDX-Licence-Identifier: EUPL-1.2`
- The root contract files depend only on `dappco.re/go` (core) plus sibling `inference/*` subpackages — no third-party imports, no CGO. (The wider module vendors serving/data-plane deps such as gin, duckdb and parquet; those live in `serving/`, `eval/` and `cmd/`, not the contract.)
- Errors go through `core.E(...)` / `core.Result`, not `fmt.Errorf`; messages start lowercase and end without punctuation: `"backend %q not registered"`
- Test triplets: `_Good` / `_Bad` / `_Ugly`
- Conventional commits scoped to `inference`, `state`, `openai`, `anthropic`, `ollama`, `options`, `discover`
- Co-Author trailer: `Co-Authored-By: Virgil <virgil@lethean.io>`

## Who imports this

Everything is in-repo now — these are packages under `dappco.re/go/inference`, not separate modules:

| Package | Why |
|---------|-----|
| `engine/metal` | implements Backend + TextModel for Apple GPU (no-cgo, `darwin && arm64`); registers backend `"metal"` at init |
| `engine/hip` | implements Backend + TextModel for AMD ROCm/HIP (`linux && amd64`) |
| `serving/` | mounts the OpenAI / Anthropic / Ollama compat handlers and the HTTP/llama fallback backend |
| `agent/` | wraps Backend + TextModel into the scoring/eval agent loop |
| `eval/` | benchmark + evaluation runners over `DatasetStream` |
| `cmd/lem` | the CLI: `serve`, `ask`, `sft`, `ssd`, `tune`, `pack` |
| `model/` | GGUF / safetensors loaders + quantisation feeding the backends |
