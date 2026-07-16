# Project History — go-inference

> **Where it is now:** go-inference is **the** sovereign inference repo for the
> Core Go ecosystem — engines, serving, training, the `lem` binary, and the GUI
> all live here. go-mlx and go-rocm are retired. The sections below trace the
> journey: the repo began (Feb 2026) as a tiny zero-dependency *contract* package
> shared by separate backend repos, and consolidated (2026) into the single
> repository it is today. Read the early "Origin"/"Phase" sections as history of
> the shared-contract era, and the "Consolidation" section for what happened
> since. Current design lives in [README](../README.md),
> [architecture.md](architecture.md), and [engine-merge.md](engine-merge.md).

## Origin (the shared-contract era)

`go-inference` was created on 19 February 2026 to solve a dependency inversion problem in the Core Go ecosystem.

`go-mlx` (Apple Metal inference on darwin/arm64) and `go-rocm` (AMD ROCm inference on linux/amd64) both needed to expose the same `TextModel` interface so that `go-ml` and `go-ai` could treat them interchangeably. The two backends cannot import each other — each carries platform-specific CGO or subprocess dependencies that would break cross-platform compilation. (Both backend repos were later retired and their engines pulled into this repo — see **Consolidation** below.)

Three options were considered:

1. **Duplicate interfaces** — Each backend defines its own `TextModel`. Simple to start, but the interfaces diverge over time as backends evolve without a shared contract. Rejected.
2. **Shared interface package** (chosen) — A new package with zero dependencies defines the contract. ~100 LOC at inception, compiles on all platforms. All backends import it; it imports nothing.
3. **Define in go-ml** — `go-ml` already had `Backend` and `StreamingBackend` types. Rejected because `go-ml` carries heavy dependencies (DuckDB, Parquet) that backends should not import.

## Commit History

### `fca0ed8` — Initial commit

Repository scaffolding. `go.mod`, empty `README.md`.

### `07cd917` — feat: define shared TextModel, Backend, Token, Message interfaces

First substantive commit. Defined `TextModel`, `Backend`, `Token`, `Message`, the `Register`/`Get`/`List`/`Default`/`LoadModel` registry functions, `GenerateConfig`, `LoadConfig`, and all `With*` options. Established the zero-dependency constraint and the `Default()` priority order (metal > rocm > llama_cpp).

### `3719734` — feat: add ParallelSlots to LoadConfig for concurrent inference

Added `WithParallelSlots` to `LoadConfig`. Required for llama.cpp backends that allocate inference slots at load time. Metal backends ignore the field.

### `2517b77` — feat: add batch inference API (Classify, BatchGenerate)

Added `Classify` and `BatchGenerate` to `TextModel`, along with `ClassifyResult` and `BatchResult`. `Classify` is a prefill-only fast path (single forward pass, no autoregressive decoding) for domain classification tasks in `go-i18n`. `BatchGenerate` runs full autoregressive decoding across multiple prompts in parallel.

### `df17676` — feat: add GenerateMetrics type and Metrics() to TextModel

Added `GenerateMetrics` and `TextModel.Metrics()`. Provides per-operation performance data: token counts, prefill and decode durations, throughput, and GPU memory usage. Required by the LEM Lab dashboard and future monitoring integrations.

### `28f444c` — feat: add ModelInfo type and Info() to TextModel

Added `ModelInfo` and `TextModel.Info()`. Provides static metadata about a loaded model: architecture, vocabulary size, layer count, hidden dimension, and quantisation details. Required by `go-ai` MCP tools that surface model information to agents.

### `884225d` — feat: add Discover() for scanning model directories

Added `Discover(baseDir string) ([]DiscoveredModel, error)` and `DiscoveredModel`. Scans a directory tree (one level deep) for model directories identified by the presence of `config.json` and `.safetensors` weight files. Used by LEM Lab's model picker UI and `go-ai`'s model listing MCP tool.

### `c61ec9f` — docs: expand package doc with workflow examples

Expanded the package-level godoc comment in `inference.go` with complete examples: streaming generation, chat, classification, batch generation, functional options, and model discovery.

### `15ee86e` — fix: add json struct tags to Message for API serialization

Added `json:"role"` and `json:"content"` tags to `Message`. Required for correct serialisation through `go-ai`'s MCP tool payloads and the agentic portal's REST API.

### `d76448d` — test(inference): add comprehensive tests for all exported API

1,074 lines of Pest-style tests (using Go's `testing` package and `testify`). Comprehensive coverage of:

- `Register`, `Get`, `List`, `Default`, `LoadModel` — all happy paths, error paths, and edge cases
- `Default()` priority order (metal > rocm > llama_cpp > any available)
- All `GenerateOption` and `LoadOption` functions
- `ApplyGenerateOpts` and `ApplyLoadOpts` — nil options, empty options, last-option-wins
- `Discover` — single models, multiple models, quantised models, base-dir-as-model, missing files, invalid JSON
- All struct types: `Token`, `Message`, `ClassifyResult`, `BatchResult`, `ModelInfo`, `GenerateMetrics`
- Compile-time interface compliance assertions

Dispatched to Charon (Linux build agent). Commit hash recorded in TODO.md as Phase 1 foundation marker.

### `85f587a` — docs: mark Phase 1 foundation tests complete (Charon d76448d)

Updated TODO.md to record Phase 1 completion and Charon's commit hash.

### `c91e305` — docs: mark Phase 2 integration complete — all 3 backends migrated

Updated TODO.md to record Phase 2 integration completion across go-mlx, go-rocm, and go-ml.

## Phase Summary

### Phase 1 — Foundation (complete)

Established the interface contract, registry, functional options, model discovery, and comprehensive tests. All exported API covered. No backend implementations in this package.

### Phase 2 — Integration (complete)

All three backends migrated to implement `inference.TextModel` and register via `inference.Register()`:

- **go-mlx** (`register_metal.go`, darwin/arm64): `metalBackend{}` + `metalAdapter{}` wrap the internal Metal model. Full `TextModel` coverage including `Classify`, `BatchGenerate`, `Info`, `Metrics`. Build-tagged `darwin && arm64`.
- **go-rocm** (`register_rocm.go`, linux/amd64): `rocmBackend{}` spawns and manages a `llama-server` subprocess. 5,794 LOC. Build-tagged `linux && amd64`.
- **go-ml** (`adapter.go`, `backend_http_textmodel.go`): Two-way bridge. `adapter.go` (118 LOC) wraps `inference.TextModel` into `go-ml`'s internal `Backend`/`StreamingBackend` interfaces. `backend_http_textmodel.go` (135 LOC) provides the reverse: wraps an HTTP llama.cpp server as `inference.TextModel`. `backend_mlx.go` collapsed from 253 to 35 LOC after migration.

### Phase 3 — Extended Interfaces (superseded)

The original plan deferred two speculative interfaces (`BatchModel`,
`StatsModel`) until multiple consumers demanded them, each to be added by
updating go-mlx and go-rocm in lockstep. That coordination model no longer
applies — the backends are retired and both engines now live in this repo, so a
new capability is added as an optional interface here and the in-repo engines opt
in directly. The specific interface sketches are left to the current design docs
rather than pinned in history.

## Consolidation — go-inference becomes the sovereign repo (2026)

The shared-contract package grew into the whole inference stack. go-mlx (Apple
Metal) and go-rocm (AMD ROCm) were **retired**, and everything they carried was
brought into go-inference:

- **Engines in-repo.** `engine/metal` is the Apple GPU engine — **no cgo**; it
  drives the Apple GPU API through pure-Go `tmc/apple` bindings and dispatches
  Apple MLX's compiled kernels plus go-inference's own fused `lthn_` kernels
  (`engine/metal/kernels/*.metal`). `engine/hip` is the AMD engine (linux/amd64,
  ROCm) and does carry cgo — no-cgo is a per-engine property, not a repo-wide
  rule. go-mlx's `pkg/metal` (the cgo engine) was **deleted, never ported**;
  `pkg/native` became `engine/metal`.
- **Model architectures stayed decoupled** from the engine — they live in the
  `model/` family (gemma3, gemma4, mistral, qwen3, …), which engines consume but
  never own.
- **Serving, training, and tooling consolidated here**: the
  OpenAI/Anthropic/Ollama HTTP servers (`serving/`), LoRA SFT + self-distillation
  + MTP tuning (`train/`), the `lem` binary (`cmd/lem`), and the LEM desktop GUI
  (`gui/`).
- **The Metal build chain moved in** too: `external/mlx` (Apple MLX pinned at
  v0.32.0) plus the lthn patch set in `patches/mlx/`, built by `task metallib`
  and optionally baked into a self-contained binary by `task build:embed`. See
  [build.md](build.md).
- **Go 1.26**, workspace-mode development against `external/` submodules, and the
  core house rules (`core.E` errors, `core.Result`, core I/O wrappers) — no
  longer the stdlib-only contract of the origin era.

The design that reconciled go-mlx's composition core into serving's shape is
recorded in [engine-merge.md](engine-merge.md). The endgame is captured in one
line: **you only need go-inference** — one repo, and with `task build:embed` one
self-contained binary.

## Known Limitations

> These describe the original shared-contract layer (the `inference` package
> itself). Some still hold for the contract; the engine and serving behaviour is
> documented in [architecture.md](architecture.md) and [backends.md](backends.md).

**Metrics on CPU backends** — `GenerateMetrics.PeakMemoryBytes` and `ActiveMemoryBytes` are zero for CPU-only backends. There is no protocol for backends to report CPU RAM usage; this was considered unnecessary at the time of design.

**`Discover` scan depth** — `Discover` scans only one level deep. Deeply nested model hierarchies (e.g. `models/org/repo/revision/`) are not found. The consumer is expected to call `Discover` on the correct parent directory.

**`Discover` and invalid JSON** — A `config.json` containing invalid JSON is silently tolerated: the directory is included with an empty `ModelType`. This prevents a single malformed file from hiding all other models in a directory, but it means the returned `DiscoveredModel` may be incomplete.

**No de-registration** — `Register` overwrites silently; there is no `Unregister`. This is intentional for simplicity. Backends registered in `init()` live for the lifetime of the process.

**`Default()` error message** — When all registered backends are unavailable, the error says "no backends registered" rather than "no backends available". This is slightly misleading but matches the no-backends case exactly, which simplifies error handling in consumers that treat both cases identically.

**`ParallelSlots` ignored by Metal** — Apple Metal manages concurrency internally. `WithParallelSlots` is accepted by `go-mlx` but has no effect. This is documented in `options.go` but not enforced.

## Future Considerations (origin-era — mostly overtaken by consolidation)

These were the forward-looking notes from the shared-contract era. Most have been
overtaken by the consolidation:

- Licence headers — now present: every `.go` file carries
  `// SPDX-Licence-Identifier: EUPL-1.2`.
- Single-package scope — long gone; the repo is now the full stack (see
  **Consolidation**).
- Error handling — production code now uses `core.E(...)` / `core.Result` rather
  than `fmt.Errorf` string matching, so the sentinel-error idea is moot.

The genuinely open contract questions (streaming batch, aggregated stats) now
follow the in-repo optional-interface pattern rather than a cross-repo rollout —
see [architecture.md](architecture.md).

## Release-day support: the gemma4 2026-07-15 update

Google refreshed the gemma4 family repos on 15 July 2026 (canonical chat
template + README/model-card). go-inference absorbed it same-day, ~12 hours
after the upstream commits:

- **Canonical chat template** (published 2026-07-09, shipped 07-15): the
  semantics overlapping our native dialect were ported — reasoning
  preservation on post-last-user turns (`reasoning`/`reasoning_content` now
  parse; `chat_template_kwargs.preserve_thinking` extends it to tool-call
  turns), adjacent-assistant turn folding (turn-tag balance), and the port
  flushed out a live bug: inbound `tool_calls` on replayed history were
  dropped by the hand-rolled decoder, so the stateless tool-history re-render
  could never fire. Null tool-argument rendering already matched.
- **OCR soft-token budget (1120)**: per-request budget override end-to-end —
  `mm_processor_kwargs.max_soft_tokens` (vLLM convention) and OpenAI-native
  `image_url.detail` (low→70, high→1120) → `WithVisionBudget` → the neutral
  `VisionBudgetTokenModel.ProjectImageAt` seam → the metal binding cloning the
  retained feature config per projection. Live receipt: the same 896×1152
  test sheet costs 290 prompt tokens at default and 1111 at `detail:"high"`
  on the 26B — the budget engages through every layer. The vendored reference
  docs confirm the sizing rule (budget×9 max patches, aspect-preserving
  resize, 3×3 mean-pool) matches the documented pipeline exactly.
- **FA4/Blackwell**: a transformers-side CUDA kernel lane, not an attention-
  scheme change — config.json and generation_config are byte-stable across
  the update, so the metal kernels (already implementing the declared scheme)
  need nothing.

The live receipt also flushed out — and fixed, same day — a long-standing
encoder-tower defect: the tower re-derived its patch grid from the flattened
patch COUNT (most-square factorisation, h ≤ w), transposing the split-axis
position field and the 3×3 pooling for every portrait image at every budget.
A giant-letter bisect (square reads, portrait misreads, both budgets) cornered
it; threading the true resize grid through VisionImagePatchesGrid →
ProjectImageFeaturesAt → VisionTower fixed it. Post-fix, the 26B transcribes
the same test sheet VERBATIM at 1120 — every line including the fine print —
where pre-fix it declared the image illegible. Still open, tracked: the 12B
unified (encoder-free) lane declines image prefill under chat's chunked
prefill entirely.

**2026-07-16 follow-up — the 12B unified decline closed (`1c60be1`).** The
decline was not vision code at all: q8 KV on the recorded-ICB lane (#367, the
default) declined ANY rowAttnCaps batch, and the bidirectional image-span lane
correctly refuses a sequential fallback — so the decline surfaced as a hard
error on every 12B image request (E2B dodged it only by geometry). The q8 gate
now accepts caps chunks (multiQ is structurally off under caps; the reads ride
the per-row q8 ladder against the chunk-wide rows-store landing, both
pre-checked), and span-free chunks the span cutter produces route through the
causal embedding lanes with their legal sequential fallback. Receipts: q8-on
answers the giant-letter probes byte-equal to the q8-off baseline at temp 0;
two new q8 bidir tests pin engagement and split routing; full metal suite
green.

## 2026-07-16 — Qwen MTP joins the ONE speculative route (#7)

The Qwen 3.5/3.6 multi-token-prediction head now pairs with its base
through the same declare/bind route gemma4's assistants use — four
slices on dev: `df7f79a` (composed Snapshot/Restore), `7d64182`
(declare + trained-shape pairing), `82d263d` (block-verify lane,
measured), `bd5ef1cc` (the serve seam). The drafter runs the trained
shape verified against vLLM's proposer — each head row pairs a token
with the base hidden that PRODUCED it, in a persistent head session at
true positions; the rejected earlier draft of the pairing had both
wrong, and the difference is 61-65% real acceptance on the 27B versus
the 7% a random head scores. Per-token verify is byte-identical to
plain greedy by construction; block-verify (one batched base forward
per round, Snapshot/Restore on reject) is mechanically proven — the
oracle fixture commits 20 tokens in 4 forwards — but ships OFF: on the
host lane a 4-row batched forward costs ~3x a single step, so the
forward-count win inverts (48 tokens: 15.7s per-token vs 21.1s block).
Its flip-levers, in order: truncate-on-restore for attention KV, #8-B
quant tails, an MoE base, the GPU decode lane. Live serve receipt:
`lem serve --model Qwen3.6-27B-4bit --draft Qwen3.6-27B-MTP-4bit`
announces the lane, answers ChatML at temp 0, and LTHN_MTP_DIAG shows
the pair carrying requests at 65.2% acceptance. Follow-ups filed in
the task: the in-checkpoint mtp.* layout (the 35B-A3B MoE convention)
as an auto-detect rung, and speculative continuity for BOTH pair
families.
