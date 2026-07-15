# Composability design — the feature-vs-model seam

Status: design, grounded against the tree at `be79a1e`. Read-only research —
no code changes ship with this document. Two code lanes are live on this
branch's parent tree at time of writing (`model/qwen2` + `engine/metal` dense
attention) — anything below citing `decode_forward_arch.go` describes
*current behaviour*, not a fixed line count; that file is a moving target
this week.

## Principle

**Shared/engine code is named for the FEATURE — a type of attention, an MTP
variation, a model capability — never for the model that happened to
introduce it.** A file, type, or function that is model-named but sits in a
shared or engine location is a discovery signal. It marks one of two things:

1. **Hidden coupling to generalise** — the code secretly assumes one model's
   shape and needs its assumption turned into a declared feature.
2. **Scaffolding to move or retire** — the code was legitimately built
   against one model (a diagnostic, a regression pin) and belongs either in
   that model's own package or on the chopping block, not in the shared
   layer.

The engine reacts to what a config or checkpoint *declares* (weight
presence, an `ArchSpec` field, a `Features` struct) — never to a model name
string. When you're about to write `if arch == "gemma4"`, that's the tell:
stop, ask what FEATURE gemma4 is exhibiting, and declare that instead.

## Current state (grounded)

**The hard part is largely done.** This is the doc's headline finding: the
feared coupling that would make an engine reuse push painful mostly doesn't
exist. What's left is a short, concrete test-file residue (below), not a
redesign.

### The flat `model` package is the shared-primitive layer

`go/model/*.go` (root, no subdirectory) is clean of model-named files. Its
21 files are all feature-shaped:

| File | What it is |
|---|---|
| `arch_spec.go` | the reactive `ArchSpec`/`RegisterArch`/`LookupArch` registry — a model package declares itself once from `init()`, the loader reacts |
| `assistant_spec.go` | the MTP/speculative-decode twin of `arch_spec.go` — `AssistantSpec`/`RegisterAssistant`, `MTPMethod` as a declared enum (draft-model today; EAGLE/Medusa/in-model/n-gram each "earn their own constant" per the file's own comment) |
| `assemble.go` | `Assemble()` — the generic per-layer dense/MoE weight-gathering walk, driven by a `WeightNames` template (`%d`-templated tensor names) an arch supplies |
| `loaded.go` | the neutral output types: `LoadedModel`, `LoadedVision`, `LoadedVisionLayer`, `LoadedUnifiedVision`, `LoadedAudio`, `LoadedDiffusion` — every multimodal payload type lives here, model-neutral, regardless of which arch is the first (or only) one to populate it |
| `rope.go`, `residual.go`, `normalise.go`, `norm_bias.go`, `fused_qkv.go`, `mat.go`, `linear.go`, `sample.go`, `alibi.go`, `position.go` | the primitive building blocks every arch package composes |
| `quant.go`, `quant_config.go`, `quant_weight.go` | quantisation contracts, arch-neutral |
| `token.go`, `transformer_config.go`, `wrapper_names.go`, `backend.go`, `load.go`, `gguf_architecture.go` | loader plumbing shared across every arch |

Every arch package (`model/llama`, `model/qwen2`, `model/gemma3`, …) is a
thin registration shim reusing this layer. `llama` and `qwen2` are the clean
minimal case — 2 files each (`config.go` + `register.go`). `gemma3` is the
same shape. `qwen3` is 3 files (`config.go`, `register.go`, plus
`gated_delta.go` for its gated-delta feature — see below).

**Correction to prior recon:** not every arch package is uniformly 2-3
files. `model/gemma4` is 19 files / ~6,100 lines — the outlier, and
legitimately so: Gemma 4 is multimodal (vision tower, audio tower,
diffusion variant, MTP assistant with a `dflash` drafter), and each of those
is real per-arch checkpoint-parsing/tensor-gathering logic (`parse.go`,
`vision_assemble.go`, `audio_assemble.go`, `unified_vision_assemble.go`,
`diffusion.go`, `assistant.go`, `assistant_dflash.go`). Read closely,
though, the pattern holds even inside this larger package:

- `gemma4.go` declares `Features` (`Mixture`, `Vision`, `Audio`,
  `AttentionClass{SlidingWindow, SlidingPattern, SharedKVLayers}`) and
  `FeaturesOf(cfg)` — the file's own comment states the discipline
  explicitly: *"deliberately NOT a list of models... the engine reacts to
  what a config declares, never to a model name or quant... the engine
  never name-branches on gemma4."*
- `vision_assemble.go`'s own types (`LoadedVisionLinear`, `LoadedVisionLayer`,
  `LoadedVisionProjector`, `LoadedVision`) are all **type aliases to
  `model.*`** — the generic vision-tower shape lives in the flat `model`
  package; gemma4's file is only the SigLIP-checkpoint-specific tensor
  name→role mapping that populates it.

So gemma4's size is real per-checkpoint-format work, not model-named
scaffolding leaking into the shared layer — the shared *types* it fills are
still feature-named and multimodal-neutral.

### The feature-named backend pattern (engine/metal) — already correct

`go/engine/metal/*.go` (168 non-test files) contains **zero** model-named
production files. Confirmed by direct grep — no file name and no in-code
string comparison (`arch == "gemma4"` etc., searched repo-wide across
`engine/metal/*.go` excluding tests) matches a model name anywhere in
production code. The backend files are named for what they implement:

- `gated_delta_backend.go`, `mamba2_backend.go`, `rwkv7_backend.go`,
  `composed_backend.go`, `composed_quant_backend.go` — mixer **types**, not
  model names.
- `assistant_dflash*.go`, `assistant_draft_fused.go`, `assistant_gguf.go`,
  `assistant_load.go`, `mtp_*.go` — MTP/speculative-decode **variations**.
- `vision*.go`, `audio*.go`, `diffusion*.go` — multimodal **capabilities**,
  shared regardless of which arch first needed them.

This is the pattern the rest of the tree should follow, not a smell to
clean up — it's already the target state for a shared engine location.

### The generic decode path is generic in code, not just in aspiration

`engine/metal/decode_forward_arch.go` (2,161 lines currently) is the
clearest evidence: 35 occurrences of `gemma4`/`gemma3`/`llama`/`mistral`/
`qwen` in the file, and **every single one is a comment** explaining a
generic mechanism using a specific model as the worked example — never a
runtime string branch. Representative lines:

```go
if qNorm.buf != nil { // gemma4 per-head QK-norm before RoPE (sharers project...
```

```go
lff := s.dFF // per-layer FFN width (gemma4 E2B/E4B); falls back to the arch...
```

```go
// valueNormOnesBuf is the gemma4 value-norm weight: a [headDim] bf16 ones ve...
// RMSNormNoScale). Returns nil when off (non-gemma4) ⇒ the decode skips valu...
```

The branch condition is always **weight presence** (`qNorm.buf != nil`),
**spec-resolved geometry** (`s.dFF`, `headDimOf`/`kvHeadsOf` read off the
`ArchSpec`-derived layer), or a **nil-buffer feature gate**
(`valueNormOnesBuf` returns nil when the arch doesn't declare value-norm —
gemma4 does, Mistral doesn't). A repo-wide grep for `arch ==` /
`arch.Name ==` style string comparisons against a model name in
`engine/metal/*.go` (excluding tests) returns nothing. The gemma4 comments
are documentation of *why* the generic mechanism exists (it was gemma4 that
first needed per-head QK-norm, the value-norm ones-buffer, the E2B/E4B
per-layer FFN width), not evidence the code branches on gemma4's name.

### The "second consumer forces the abstraction" rule is already observed

`model/qwen3/gated_delta.go` declares `GatedDeltaConfig`/`GatedDeltaWeights`
and imports `model/deltanet` (the shared delta-recurrence primitive) and
`model/mamba2` — it is qwen3's package only because Qwen 3.6 is presently
the *only* consumer of the gated-delta mixer. This is the correct place for
a feature to live before a second consumer arrives, not a violation of the
principle — see the forward-discipline section below for the corollary this
sets up for NEEDLE.

## The residue

Everything above is production code (or, for the flat `model` types,
proven-generic). The actual residue is small and confined to
**engine/metal's model-specific *test* files** — 7 files:

```
gemma3_loader_test.go            gemma4_31b_op_diff_test.go
gemma4_12b_mtp_shapes_test.go    gemma4_31b_qmm_dims_test.go
gemma4_31b_layer_diag_test.go    mistral_session_test.go
qwen3_gated_delta_backend_test.go
```

### The load-bearing constraint: package boundary, not preference

Snider's stated default is: a model-specific test's resolution is to **move
it into that model's own package** (`model/{arch}/`, or
`model/{hf-org}/{arch}/` once the grouping below lands) — not a neutral
retire/parametrise/keep menu. That's the right default, but it runs into a
real Go constraint that decides whether a given test *can* actually move:

`engine/metal` is `package native`. A test relocated to `model/gemma4`
becomes `package gemma4` (or an external `gemma4_test` package). Either way
it can only reach `engine/metal`'s **exported** API — `LoadDir`,
`ArchSession.Generate`, and any capital-letter engine function a model
package's init() already wires as a hook (e.g. `qwen3.GatedDeltaInputDevice`
is filled in by `engine/metal/gated_delta_backend.go` at init time — a
declared seam, not an accident). It loses all access to unexported engine
internals: `shardBuffers`, `bufForNorm`, `stepToken`, `withAutoreleasePool`,
`bf16Size`, `icbDisabledForTest`, `encQMMTBF16At`, and so on.

So each test's resolution depends on what it actually touches, read from
its imports and call sites — not assumed:

| File | Lines | What it touches | Resolution |
|---|---:|---|---|
| `gemma3_loader_test.go` | 83 | `LoadDir` (public) **and** directly constructs the unexported `&shardBuffers{}`, calls unexported `bufFor`/`bufForNorm` | **PARAMETRISE.** The real subject is the engine's zero-copy norm-binding seam (folded-norm resident binding, #1851) — a generic capability any `NormBiasOne` arch exercises, gemma3 included only because its tied-embedding + folded-norm shape reaches the seam. Rename off "gemma3" (e.g. `norm_bind_resident_test.go`); stays `package native`. |
| `mistral_session_test.go` | 171 | `LoadDir`, `model.Assemble`, `ArchSession.Generate` (all public) **plus** an internal cross-check via unexported `buildBF16ArchLayerBufs`, `newArchDecodeState`, `stepToken`, `withAutoreleasePool`, `bf16Size` | **SPLIT.** The checkpoint→session→generate→dir-parity claim is achievable through public API alone and is a genuine model-integration test — **MOVE** that half to `model/mistral`. The "session output equals the manual decode-state chain" cross-check needs unexported internals and is itself a generic executor-parity claim (true for any arch, not Mistral-specific) — **PARAMETRISE** that half, keep it in `package native`, rename off "mistral". |
| `qwen3_gated_delta_backend_test.go` | 124 | Exclusively `qwen3.*` exported symbols (`GatedDeltaForwardF32`, `ProjMatMul`, `GatedDeltaInputDevice` hook vars) plus one direct call to `native`'s own exported `GatedDeltaInputDevice` | **PARAMETRISE.** Its production sibling is already correctly feature-named (`gated_delta_backend.go`); the test lags behind with a model name. `model/qwen3` does not import `engine/metal` (checked — no cycle), so a move is *technically* reachable via an external `qwen3_test` package, but the actual subject under test is the engine's device-vs-host GEMM parity for the gated-delta backend, which is engine-side by nature. Rename to match the production file (`gated_delta_backend_test.go`), stays `package native`. |
| `gemma4_31b_qmm_dims_test.go` | 116 | Unexported `encQMMTBF16At`, `ensureInit` — pure white-box; gated only on `MLX_METALLIB_PATH` (runs routinely, not skipped by default) | **PARAMETRISE.** A real, live regression guard for a boundary-class bug (#348: non-512-aligned `inDim`), mis-named after the model (31B) that first exposed it. Rename to describe the boundary class (e.g. `qmm_boundary_dims_test.go`); stays `package native`. |
| `gemma4_12b_mtp_shapes_test.go` | 274 | Unexported `bf16Size`, GEMM/gemv kernels directly; synthetic random data, no env-gate, no `model/gemma4` import at all | **RETIRE (flag for confirmation).** Point-in-time investigative sweep for #352 (a resolved-sounding NaN defect at specific K values). No ongoing regression framing in its own comments — the shapes were chosen to bisect a since-presumably-fixed bug, not to guard a boundary. |
| `gemma4_31b_layer_diag_test.go` | 463 | Unexported `icbDisabledForTest`; requires `GEMMA4_CROSS_ENGINE_MODEL` + `GEMMA4_IDS` + `GEMMA4_LAYER_DUMP` env vars (real checkpoint + a HIP-oracle dump) — skips entirely otherwise | **RETIRE (flag for confirmation).** Dormant cross-engine diagnostic scaffolding for #52; never runs in a normal suite. Owner should confirm #52's status before deletion — if still open, this is legitimate tooling, just not something CI ever exercises. |
| `gemma4_31b_op_diff_test.go` | 410 | `model` package (real weights) + `GEMMA4_SNAP`/`GEMMA4_OPS` env vars; skips otherwise | **RETIRE (flag for confirmation).** Same shape as the layer-diag file, for #348's per-op conviction pass. Same caveat: confirm issue status before deleting. |

**Classification counts: MOVE 0 clean / 1 partial (mistral, split) ·
PARAMETRISE 4 (gemma3_loader, mistral's other half, qwen3_gated_delta,
qmm_dims) · RETIRE-pending-confirmation 3 (mtp_shapes, layer_diag, op_diff).**

No file in this residue is a case of "genuinely model-coupled code" in the
sense the brief asked to watch for — none of them are a production
`if arch == "gemma4"` branch. They're all either white-box engine tests
wearing a model's name, or dormant point-in-time diagnostics. The house
rule applies to executing this list: **one test at a time, `go test` the
touched package after each move/rename, no bulk sed.**

## The HF-org grouping proposal (later)

Reorganise `model/{arch}` → `model/{hf-org}/{arch}` — `model/google/gemma4`,
`model/qwen/qwen3`, `model/mistralai/mistral`, and so on — so an arch's
package path names its origin automatically and same-origin models cluster
on disk.

**Honest costs:**

- **Import-path churn across every `register.go` and `model/builtin`.**
  `model/builtin/builtin.go` blank-imports all ~40 arch packages by path;
  every import line changes. (Aside, found in passing: `builtin.go`
  currently carries duplicate blank-import lines for roughly a third of its
  entries — e.g. `model/gemma4` is blank-imported 4 times, `model/bloom` 4
  times, `model/composed` 4 times. Harmless to the compiler, but since the
  HF-org rewrite touches every line of this file anyway, that's the natural
  place to fold the duplicates out as a side effect, not a reason to do it
  sooner.)
- **The Go package-name-vs-path distinction.** The package itself stays
  `package gemma4` (unqualified import names don't change); only the import
  *path* gains the org segment. Every call site (`gemma4.Config`, …)
  compiles unchanged — this is a path move, not a rename.
- **Purely organisational — zero functional change.** No behaviour differs
  before/after. That is *why* it's low-urgency: the coupling this doc
  exists to check is already clean, so this reshuffle buys navigability,
  not correctness.

**Sequencing:** wait for a quiet tree (no live model/engine lanes — the
qwen2/dense-attention lane currently touches exactly the files this move
would also touch), then move one arch at a time with `go vet` after each.
No bulk `sed` — the standing house rule for refactors of this shape.

## The forward discipline

New code names by feature from birth — the discipline above isn't a
one-time cleanup, it's the ongoing rule for anything landing next.

**The concrete near-term test: NEEDLE.** A 26M-parameter encoder-decoder
with cross-attention and a non-causal encoder — neither exists anywhere in
this tree today (confirmed: no cross-attention or non-causal-encoder
primitive in `model/` or `engine/metal/`). When NEEDLE lands, cross-attention
and the non-causal encoder pass **must** be built as shared attention
features — a `model/attn`-style seam, or extending the existing flat-package
pattern (`model/cross_attention.go`, alongside `model/rope.go` etc.) — reusable
by any future encoder-decoder or retrieval-augmented model, not scoped
inside a `model/needle` package.

This is the "second consumer forces the abstraction" rule already validated
above by `model/qwen3/gated_delta.go` (feature stayed with its sole
consumer until a second one arrives) — applied in reverse: NEEDLE is
*known in advance* to need a feature no current arch has, which makes it
the forcing function for building that feature in the shared layer from day
one rather than provisionally inside `model/needle` and extracting later.

## The bigger prize, cross-referenced not duplicated

The larger reuse opportunity is engine-level duplication, already fully
mapped in `docs/design-rocm.md` — do not re-derive it here. Headline (from
that doc): `engine/hip` was landed as a wholesale port from a
formerly-independent repo and grew its **own parallel copy** of
`engine/scheme`, the quant registry, and `model/gguf`'s parser instead of
consuming the shared engine-neutral layers `engine/metal` already uses (Part
B.1–B.4 of that doc). One detail worth flagging *because* it's this doc's
exact pattern: `design-rocm.md`'s own headline table notes
`engine/hip/gemma4_architecture_adapter.go` — a model-named file sitting in
a shared engine location, the textbook discovery signal this document
defines. That file is in scope for `design-rocm.md`'s campaign, not this
one; noted here only as confirmation the principle already caught a real
instance elsewhere.

Separately, the dense `ArchSession` and the `composed` session are two
decode engines with their own reuse question — real, but a harder, separate
campaign outside this doc's scope.

## Dispatch boundary / work-list

| Item | Tag | Notes |
|---|---|---|
| Rename `gemma3_loader_test.go` → norm-bind-resident test, `package native` | `[after-lanes]` | touches `engine/metal` |
| Split `mistral_session_test.go`: move public-path half to `model/mistral`, parametrise+rename the executor-parity half in `engine/metal` | `[after-lanes]` | touches both `model/` and `engine/metal` |
| Rename `qwen3_gated_delta_backend_test.go` → `gated_delta_backend_test.go`, `package native` | `[after-lanes]` | touches `engine/metal`; collides with the live qwen2/dense-attention lane's working set |
| Rename `gemma4_31b_qmm_dims_test.go` → boundary-class name | `[after-lanes]` | touches `engine/metal` |
| Confirm #352 status, then retire `gemma4_12b_mtp_shapes_test.go` if closed | `[after-lanes]` | touches `engine/metal`; needs an issue-tracker check this doc couldn't perform |
| Confirm #52 status, then retire `gemma4_31b_layer_diag_test.go` if closed | `[after-lanes]` | ditto |
| Confirm #348 status, then retire `gemma4_31b_op_diff_test.go` if closed | `[after-lanes]` | ditto |
| HF-org grouping (`model/{arch}` → `model/{hf-org}/{arch}`) | `[later]` | purely organisational; needs a fully quiet tree, one arch at a time, `go vet` after each |
| This document itself | `[now-safe]` | read-only, already delivered |

Every code-touching item above is `[after-lanes]` — all seven residue files
and the HF-org move sit in `engine/metal` and/or `model/`, the exact
directories the live qwen2/dense-attention lane is working in right now.
Nothing in this work-list is safe to execute until that lane (and any
sibling) clears.

---

*This repo's `docs/design-*.md` files are audited against the tree and
removed once implemented — this document should be re-verified against
current tree state before execution, not treated as frozen truth once the
lanes above clear.*
