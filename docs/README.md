<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-inference — documentation index

**Module**: `dappco.re/go/inference`
**Role**: The contract package every backend and consumer in the tetrad imports.

## Tetrad position

```
                    ┌──────────────────────────────┐
                    │      dappco.re/go (core)     │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴────────────────┐
       you are here →   go-inference  (CONTRACT)    │   ← pure interfaces + wire types
                    │  • TextModel / Backend        │
                    │  • state/ (memvid lifecycle)  │
                    │  • openai/ anthropic/ ollama/ │
                    │  • capability / probe         │
                    └──┬─────────────┬──────────────┘
                       │             │ register via init()
              ┌────────┴───┐  ┌──────┴────────┐
              │  go-mlx    │  │  go-rocm /    │  ← native backends
              │  darwin/   │  │  go-cuda      │
              │  arm64     │  └───────────────┘
              └─────┬──────┘
                    │ consumed by
              ┌─────┴──────────┬────────────────┐
              │  go-ml         │  go-ai          │ ← consumers
              │  scoring/agent │  router/demos   │
              └────────────────┘ └───────────────┘
```

## Doc tree

```
docs/
├── README.md                ← you are here
├── inference/               ← root package
│   ├── README.md            — package overview + how the pieces fit
│   ├── inference.md         — TextModel + Backend + registry + LoadModel
│   ├── contracts.md         — extension interfaces (Scheduler, Cache, Embed, Rerank, ToolParse, …)
│   ├── options.md           — GenerateOption + LoadOption + With*
│   ├── capability.md        — CapabilityReport + AlgorithmProfile + RuntimeMemoryLimiter
│   ├── local_tuning.md      — MachineDiscoverer + TuningPlanner + model replace
│   ├── probe.md             — ProbeEvent + ProbeSink
│   ├── service.md           — Core ServiceRuntime registration (Mantis #1336)
│   ├── training.md          — TrainableModel + Adapter + LoRAConfig
│   ├── discover.md          — Discover() filesystem scan
│   ├── gguf.md              — GGUFInfo metadata reader
│   ├── dataset.md           — DatasetSample + DatasetStream
│   └── identity.md          — re-export aliases from state
│
├── state/                   ← state subpackage
│   ├── README.md            — package overview + mental model
│   ├── agent_memory.md      — Wake / Sleep / Fork lifecycle
│   ├── identity.md          — ModelIdentity / TokenizerIdentity / Adapter / Runtime / Sampler / Bundle
│   ├── project_seed.md      — project seed URI planning + compatibility checks
│   ├── store.md             — Store / Resolver / Writer interfaces
│   ├── memory.md            — InMemoryStore
│   └── filestore.md         — append-only file-backed store
│
├── openai/                  ← OpenAI wire types
│   ├── README.md            — package overview
│   ├── openai.md            — Chat Completions + Handler
│   ├── responses.md         — Responses API DTOs
│   └── services.md          — embeddings / rerank / cache / cancel / capabilities handlers
│
├── anthropic/
│   └── anthropic.md         — Messages API wire types
│
└── ollama/
    └── ollama.md            — Ollama-compatible wire types
```

## Where to start

- **"What's the basic loop?"** → [`inference/inference.md`](inference/inference.md)
- **"How do I add a backend?"** → [`inference/inference.md`](inference/inference.md) — Backend interface + Register pattern
- **"How does agent memory work?"** → [`state/agent_memory.md`](state/agent_memory.md) — Wake/Sleep/Fork
- **"How do project seeds reload safely?"** → [`state/project_seed.md`](state/project_seed.md) — project seed helpers + compatibility
- **"How does OpenAI compatibility work?"** → [`openai/openai.md`](openai/openai.md)
- **"What can a backend advertise?"** → [`inference/capability.md`](inference/capability.md)
- **"How does local setup/autotune work?"** → [`inference/local_tuning.md`](inference/local_tuning.md)
- **"How do I observe runtime?"** → [`inference/probe.md`](inference/probe.md)

## Legacy docs

`architecture.md`, `interfaces.md`, `backends.md`, `types.md`, `development.md`, `history.md`, `index.md`, `RFC.models.md`, `RFC-CORE-008-AGENT-EXPERIENCE.md` predate this per-file pass. They cover overlapping ground at a wider grain and may rot as the per-file docs evolve. Pending: collapse the still-useful bits into `inference/README.md` and the per-file pages, then mark the legacy docs deprecated.

## Standards

- UK English
- EUPL-1.2 licence (see [LICENCE](../LICENCE))
- SPDX header on every source file
- Conventional commits, scopes per package
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
