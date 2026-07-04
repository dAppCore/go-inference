# Engine merge — reconciling the go-mlx composition core

Resolves the open call named in `go-mlx/docs/MIGRATION.md`: *"reconcile go-mlx
composition into serving's shape, or the reverse."* Companion to that map;
this document lives in the receiving repo because go-inference owns the merge
design. Written 2026-07-04, after the Tier-0 contract diff below.

## The call: serving's shape wins

The `inference` contract layer is the boundary. Engines are self-registering
runtime packages behind `inference.Register` / `inference.LoadModel`
(`WithBackend("metal")`), exactly as `serving/backend_mlx.go` already assumes.
Nothing in serving reshapes toward go-mlx.

Why this direction and not the reverse:

1. **Engine count.** rocm and cuda engines follow metal (`engine/hip` is
   already named in the endgame). The registry pattern scales per engine;
   go-mlx's shape hard-binds one engine's types into the composition core.
2. **Type gravity.** go-mlx's composition core speaks `pkg/metal` types as its
   vocabulary — even `NativeModel` (the *native* engine's contract) is written
   in `metal.GenerateConfig` / `metal.Token` / `metal.ModelInfo`. Adopting
   that shape would drag the cgo engine's type namespace into the unified
   repo the moment pkg/metal is supposed to die.
3. **Independent design already converged.** serving's `InferenceAdapter` +
   `inference.TextModel` cover the same ground as go-mlx's `Model` facade with
   no engine imports. The overlap IS the contract; the residue is the diff
   below.

## Tier-0 contract diff (imports-verified, 2026-07-04)

`metal.X` references across the go-mlx composition core (`backend.go`,
`mlx.go`, `session.go`, `eval.go`, `speculative.go`, `tokenizer.go`,
`primitives.go`, `model_lora.go`, `native_model.go`,
`native_speculative_textmodel.go`, `register_metal.go`):

| go-mlx composition type (uses) | go-inference today | Disposition |
|---|---|---|
| `metal.ChatMessage` (13) | `inference.Message` | RECONCILE — rename onto `Message` |
| `metal.GenerateConfig` (8) | `inference` GenerateOptions/options.go | RECONCILE — one config type, engine converts inward |
| `metal.LoadConfig` (6) | `inference.LoadOption` | RECONCILE — functional options win |
| `metal.MTPMetrics` (5) | — | ADD to `inference` (speculative decode is engine-generic: metal MTP today, hip next) |
| `metal.KVSnapshot` + `CaptureOptions` (7) | `kv.Snapshot` (migrating up per map) | ADD capability interfaces to `inference`, expressed in `kv.Snapshot` — retires `kvconv` (map: DIES-WITH-METAL) |
| `metal.Token` (4) | `inference.Token` | RECONCILE |
| `metal.Model` / `InternalModel` (6) | `inference.TextModel` | RECONCILE — facade dissolves into TextModel + capability probes |
| `metal.LoRAAdapter` / `LoRAConfig` (4+) | — | ADD LoRA capability interface to `inference`; adapter stays engine-side |
| `metal.ModelInfo` (3) | `inference.ModelInfo` | RECONCILE |
| `metal.DeviceType` / `DeviceInfo` (4) | — | ADD neutral `inference.DeviceInfo`; engine reports it |
| `metal.SessionHandle` (1) | — | ADD session capability interface (conversation state is the LEM edge — first-class contract) |
| `metal.Tokenizer` (1) | tokenizer contract (go-inference) | RECONCILE |
| Raw array ops: `Zeros VJP ValueAndGrad Softmax SliceAxis Reshape Mul NewAdamW SeedRandom` (9, all in `eval.go` + `model_lora.go`) | — | ENGINE-SIDE — graph-level train/eval rides into `engine/metal`, never a contract |
| Memory verbs: `SetCacheLimit SetMemoryLimit SetWiredLimit GetActiveMemory GetPeakMemory GetCacheMemory ClearCache ResetPeakMemory RuntimeGC` | `inference.SetRuntimeMemoryLimits` (partial) | RECONCILE — extend the runtime-memory contract to cover the full verb set; serving already routes through it |

The capability-probe pattern is already the house style on both sides:
go-mlx `native_model.go` probes optional interfaces (`nativeKVSnapshotter`,
`nativePromptCacheWarmer`, `nativeChunkGenerator`…), and go-inference probes
`AttentionInspector`. The ADDs above are more of the same, not a new idea.

## Destinations (what happens to each composition file)

| File | Fate |
|---|---|
| `register_metal*.go`, `metal_capabilities.go` | DIES-WITH-METAL (per map) |
| `native_model.go`, `native_speculative_textmodel.go` | Ride into `engine/metal` as its `inference.Register` shim, re-expressed in `inference` types |
| `backend.go`, `mlx.go`, `session.go`, `tokenizer.go`, `primitives.go` | Dissolve: contract parts → `inference` root ADDs above; glue → engine/metal registration; aliases die |
| `speculative.go` | Engine-agnostic orchestration → go-inference (new `speculative` home or `inference` root); MTP internals stay engine-side |
| `eval.go`, `model_lora.go` | Graph-level work → `engine/metal`; any backend-agnostic eval semantics fold into `go/eval` |
| `split_cpu_ffn*.go`, `split_executor.go`, `split_remote_ffn.go` | MIGRATE-UP as-is (engine-import-free, per map) |
| `split_native_runtime.go` | Follows pkg/native into `engine/metal` |

## Execution tiers (each independently landable, tests green per tier)

- **Tier 1 — contract ADDs (this repo only).** `inference` grows: MTP/speculative
  metrics, KV-snapshot + session capability interfaces (in `kv.Snapshot`
  terms), LoRA capability, neutral `DeviceInfo`, full runtime-memory verb set.
  No go-mlx changes; go-mlx keeps compiling against the submodule pin.
- **Tier 2 — engine/metal scaffold (this repo).** `engine/metal` directory as
  a **separate cgo module** (darwin/arm64-only), wired via go.work — the
  root `dappco.re/go/inference` module stays pure-Go cross-platform. The
  scaffold hosts the registration shim contract-tested against `inference`.
- **Tier 3 — payload move (cross-repo, gated on endgame step 1).** pkg/native
  + pkg/model land in `engine/metal`; the go-mlx composition core dissolves
  per the table above; `lem` compiles from go-inference alone. Only after the
  native feature port is finished (pkg/metal is still the parity oracle).
- **Tier 4 — hip.** go-mlx becomes the quarantine sandbox; `engine/hip`
  lands by audit-then-land. Unsupervised agents never edit go-inference.

## Spine + session file-level triage (2026-07-04, imports-verified)

`spine` is not one disposition — it splits by file. Load-bearing find: **spine.go
IS the GenerateConfig home** (root aliases `type GenerateConfig =
spine.GenerateConfig`), so the "config reconcile" open call below and the spine
lift are the same work item, not two.

| spine file | mlx imports (non-test) | Wave |
|---|---|---|
| `prompt.go`, `token.go`, `tokenizer.go` | none | **A — lifted now** (partial `go/spine`) |
| `spine.go` (GenerateConfig/Options + conversions) | probe | **B — the config reconcile** (vs `inference` GenerateOption; Cladius, not mechanical) |
| `model_info.go` | bundle, lora, memory | **A-later** — after memory; bundle in flight, lora → `inference/lora.AdapterInfo` |
| `lora_config.go`, `metal_convert.go` | pkg/metal, probe | **engine-side** — ride into `engine/metal` |

| session file | mlx imports (non-test) | Blocked on |
|---|---|---|
| `defaults.go` | none | nothing |
| `artifact.go` | artifact | artifact lift (itself: bundle + kv — unlocks when bundle lands) |
| `agent_memory.go` | agent, bundle, kv, kvconv, spine | agent (→ memory), kvconv retirement (#259) |
| `session.go` | agent, blockcache, bundle, kv, kvconv, **pkg/metal**, spine | all of the above + `SessionHandle` contract re-home |
| `internal/sessionfake` | pkg/metal (`metal.KVSnapshot` field) | re-point to `kv.Snapshot` when session lifts |

Dependency-ordered execution:

- **Wave A (mechanical, agent-able):** bundle ✓in-flight · probe (leaf: core+coreio)
  ✓in-flight · blockcache (already inference-native imports) ✓in-flight ·
  spine prompt/token/tokenizer ✓in-flight · then artifact (after bundle) ·
  then memory+profile chain (memory also drags `pack`, which has the
  `model/pack` twin — RECONCILE gate before lifting) · then agent + spine
  model_info (after memory).
- **Wave B (reconcile, by hand):** spine.go GenerateConfig ↔ `inference`
  options — one config type survives; engines convert inward. **Root side
  DONE:** 13 of spine's 18 fields were already field-identical;
  `inference.GenerateConfig` grew the delta (Thinking policy, TraceTokenPhases,
  TraceTokenText, GenerationClearCache/-Interval) + `WithThinking`. The
  thinking trio (Config/Mode/Chunk) hoisted from `parser` into the root as
  ThinkingConfig/ThinkingMode/ThinkingChunk — parser aliases them back, zero
  consumer breakage — because parser imports the root (Token, parse results)
  and the config could not reference parser without a cycle. EnableThinking
  (API intent) and Thinking (resolved engine policy) coexist by design;
  serving resolves the former into the latter. Remaining: ProbeSink joins
  GenerateConfig once the probe lift merges; spine.go's conversions re-point
  at Tier 3. Then the session package: kvconv dies against the new KV
  contracts (#259 native implementation), `SessionHandle` re-homes as an
  inference capability, and session + sessionfake land speaking `kv.Snapshot`.
- **Engine-side (never lifts):** spine lora_config/metal_convert, kvconv.

## Open questions carried (not blockers for Tier 1)

- `serving.Backend`/`GenOpts` root types are currently skeletal (empty
  interfaces in `serving/inference.go`) — Tier 1 should firm them against the
  reconciled config type rather than grow a second config surface.
- `cmd/mlx` CLI verbs reconcile with `cmd/lthn-model-pack` at Tier 3 (per map).
- The daemon (`pkg/daemon` UDS/JSON-line) MIGRATE-UP lands beside serving —
  sequencing free between Tiers 1–3.
