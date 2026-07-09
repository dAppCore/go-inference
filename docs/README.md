<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-inference — documentation index

**Module**: `dappco.re/go/inference`
**Role**: The sovereign local-inference repository — the shared contract, the in-tree GPU engines, the serving layer, and the `lem` binary, in one place.

## Repository position

`go-inference` sits on top of Core (`dappco.re/go`) and contains the whole local-inference stack. The engines that used to live in separate repositories (`go-mlx`, `go-rocm`) are retired and now live in-tree as `engine/metal` and `engine/hip`, registering against the contract at `init` time.

```
                 +------------------------------+
                 |      dappco.re/go (core)      |  core.Result, core.E, core.Fs, ...
                 +--------------+---------------+
                                |
   +----------------------------+-----------------------------+
   |  go-inference                                            |
   |                                                          |
   |   contract (root package)  - TextModel / Backend /       |
   |                              registry / options / types  |
   |        ^ register via init()                             |
   |   +----+--------------+    +-------------------+          |
   |   | engine/metal      |    | engine/hip        |  engines |
   |   | Apple GPU, no cgo |    | AMD ROCm          |          |
   |   | darwin/arm64      |    | linux/amd64       |          |
   |   +-------------------+    +-------------------+          |
   |                                                          |
   |   serving/  - OpenAI / Anthropic / Ollama HTTP           |
   |   cmd/lem/  - the lem binary                             |
   +----------------------------------------------------------+
                                |  consumed by
                 +--------------+---------------+
                 |  Core Go consumers           |  (agents, i18n, tooling)
                 +------------------------------+
```

## Doc tree

```
docs/
├── index.md                 ← package overview + quick start (landing page)
├── architecture.md          ← the repository as a whole: contract, engines, serving, binary
├── interfaces.md            ← TextModel / Backend / TrainableModel / Adapter / optional capabilities
├── types.md                 ← Token / config structs / options / DiscoveredModel / DeviceInfo
├── backends.md              ← the in-tree engines, the registry, adding a backend
│
├── inference/               ← root package, per-file
│   ├── README.md            — package overview + how the pieces fit
│   ├── inference.md         — TextModel + Backend + registry + LoadModel
│   ├── contracts.md         — extension interfaces (Scheduler, Cache, Embed, Rerank, ToolParse, …)
│   ├── options.md           — GenerateOption + LoadOption + With*
│   ├── capability.md        — CapabilityReport + AlgorithmProfile + RuntimeMemoryLimiter
│   ├── local_tuning.md      — MachineDiscoverer + TuningPlanner + model replace
│   ├── probe.md             — ProbeEvent + ProbeSink
│   ├── service.md           — Core ServiceRuntime registration
│   ├── training.md          — TrainableModel + Adapter + LoRAConfig
│   ├── discover.md          — Discover() filesystem scan
│   ├── gguf.md              — GGUFInfo metadata reader
│   ├── dataset.md           — DatasetSample + DatasetStream
│   └── identity.md          — re-export aliases from model/state
│
├── state/                   ← model/state subpackage
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

- **"What is this repo?"** → [`index.md`](index.md) — overview + quick start
- **"How does it fit together?"** → [`architecture.md`](architecture.md) — contract + engines + serving + binary
- **"What's the basic loop?"** → [`inference/inference.md`](inference/inference.md)
- **"How do I add a backend?"** → [`backends.md`](backends.md) — the registry + Register pattern
- **"How does the Metal engine work (no cgo)?"** → [`backends.md`](backends.md) — engine/metal + ICB replay
- **"How does agent memory work?"** → [`state/agent_memory.md`](state/agent_memory.md) — Wake/Sleep/Fork
- **"How do project seeds reload safely?"** → [`state/project_seed.md`](state/project_seed.md)
- **"How does OpenAI compatibility work?"** → [`openai/openai.md`](openai/openai.md)
- **"What can a backend advertise?"** → [`inference/capability.md`](inference/capability.md)

## Wider-grain docs

`index.md`, `architecture.md`, `interfaces.md`, `types.md`, and `backends.md` are the maintained reference set — kept accurate against the code. `development.md`, `history.md`, `RFC.models.md`, and `RFC-CORE-008-AGENT-EXPERIENCE.md` predate the per-file pass and cover overlapping ground at a wider grain; treat those four as background and verify against the code before relying on them.

## Standards

- UK English
- EUPL-1.2 licence (see [LICENCE](../LICENCE))
- SPDX header on every source file
- Conventional commits, scopes per package
- Co-Author: `Co-Authored-By: Virgil <virgil@lethean.io>`
</content>
