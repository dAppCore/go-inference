<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# inference/ — contract package root

**Package**: `dappco.re/go/inference`

## What this package owns

The **central contract** that every other tetrad repo speaks. Pure interfaces, DTOs, registries, and option types. Zero CGO. Zero platform branches. Compiles everywhere.

Three categories:

| Category | What | Files |
|----------|------|-------|
| **Core runtime** | TextModel + Backend + registry + LoadModel | [inference.md](inference.md) |
| **Options** | GenerateOption + LoadOption + With* | [options.md](options.md) |
| **Extension** | Scheduler, Cache, Embedding, Rerank, ToolParse, ReasoningParse, ModelPackInspect | [contracts.md](contracts.md) |
| **Static intro** | CapabilityReport / AlgorithmProfile / RuntimeMemoryLimits | [capability.md](capability.md) |
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

- [../state/](../state/README.md) — durable state DTOs + Wake/Sleep/Fork lifecycle
- [../openai/](../openai/README.md) — OpenAI wire types + HTTP handlers
- [../anthropic/](../anthropic/anthropic.md) — Anthropic Messages wire types
- [../ollama/](../ollama/ollama.md) — Ollama-compatible wire types

## Stability rules

This package is the shared contract. Changes here cascade to every backend and consumer.

- **No new methods on `TextModel` or `Backend`** without a Virgil review.
- **Prefer new interfaces over wider TextModel.** New capabilities land in `contracts.go` as opt-in extensions.
- **New fields on `GenerateConfig` / `LoadConfig` are safe** when zero-value defaults preserve old behaviour.
- **Wire DTOs in openai/anthropic/ollama track upstream** — adding fields is safe, renaming requires upstream rename first.

## Coding standards (this repo)

- UK English in code, comments, docs (colour, organisation, licence, serialise)
- SPDX header on every new file: `// SPDX-Licence-Identifier: EUPL-1.2`
- Zero external dependencies — stdlib + `dappco.re/go` only (testify in tests)
- Error strings start lowercase, end without punctuation: `"backend %q not registered"`
- Test triplets: `_Good` / `_Bad` / `_Ugly`
- Conventional commits scoped to `inference`, `state`, `openai`, `anthropic`, `ollama`, `options`, `discover`
- Co-Author trailer: `Co-Authored-By: Virgil <virgil@lethean.io>`

## Who imports this

| Module | Why |
|--------|-----|
| `dappco.re/go/mlx` | implements Backend + TextModel for Apple Metal |
| `dappco.re/go/rocm` (planned) | implements Backend + TextModel for AMD ROCm |
| `dappco.re/go/cuda` (planned) | implements Backend + TextModel for NVIDIA CUDA |
| `dappco.re/go/ml` | wraps Backend + TextModel into scoring/eval engine, adds HTTP/llama backends |
| `dappco.re/go/ai` | provider router, outbound OpenAI provider, BookState demo |
| `dappco.re/go/i18n` | TextModel for domain classification |
| `dappco.re/go/api` | mounts OpenAI / Anthropic / Ollama handlers |
| `dappco.re/go/ide` | reads CapabilityReport + bundle index for model picker |
