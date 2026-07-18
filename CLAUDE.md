# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

The sovereign inference engine for the Core Go ecosystem. Module: `dappco.re/go/inference`, module root at **`go/`** (work from there). Two things live here:

1. **The shared contract** (`go/*.go`) — `TextModel`/`Backend` interfaces, options, registry, discovery. Backends import this; never the reverse.
2. **The engines** (`go/engine/`) — `engine/metal` (package `native`): the **no-cgo Apple GPU engine** (darwin/arm64, objc bridge via `github.com/tmc/apple`, ICB-replayed decode, MTP speculative pairs, paged SDPA, q8 KV); `engine/hip`: AMD ROCm (linux/amd64). Plus `cmd/lem` (the serving/bench binary), OpenAI/Anthropic/Ollama compat handlers, and conversation state (`-state`, no-prompt-replay).

`docs/` at repo root is the manual: `architecture.md`, `backends.md` (registry + **engine runtime levers**), `cmd-lem.md`, `build.md`, and **`handover.md` — read that one first; it is the working handover from the engine-perf campaigns.**

## Commands

```bash
# from go/ — the module root
go build -tags embed_metallib -o ../bin/lem ./cmd/lem     # build the binary
go vet ./...
MLX_METALLIB_PATH=<repo>/build/dist/lib/mlx.metallib go test ./...   # full suite (~10.5k tests)
go test -count=1 ...        # ALWAYS -count=1 for benchmarks — go test caches identical runs

# kernel changes (from repo root)
task metallib:kernels       # 31 .metal files -> build/dist/lib/lthn_kernels.metallib
gzip -9 -c build/dist/lib/lthn_kernels.metallib > go/cmd/lem/lthn_kernels.metallib.gz   # then rebuild lem

# bench shape (model path is POSITIONAL, last)
../bin/lem generate -draft <assistant-path> -temp 0 -max-tokens 400 [-context 20480 -prompt-file <f>] <model-path>
```

`MLX_METALLIB_PATH` must be inlined per command — it does not persist between shells here.

## Working discipline (how this engine got fast)

- **Instrument before assessment; receipt before claim.** Every perf commit carries its before→after numbers in the message. A theory without a live receipt gets built, measured, and **reverted if it loses** — falsifications are banked in the task tracker, not hidden.
- **One lever per commit.** Kill-switch env for anything wall-clock-adaptive (`LTHN_MTP_REENGAGE=0`, `LTHN_MTP_DRAFTLEN=0` — the repro anchors).
- **Bench→live transfer is not guaranteed** — micro-bench wins on >134MB buffers have reproducibly lost in live decode. The live A/B is the only receipt that counts.
- **No bulk perl/sed refactors** — one site at a time, vet after each.
- Branch **dev**; origin `github.com/dAppCore/go-inference` (push non-force). Task board via the session task tools; git log is the history of record.

## Composed lane (Qwen hybrid) — device levers + the standing gap

The `model/composed` lane serves the Qwen3.5 hybrid (gated-delta recurrence + MoE/dense FFN). Its device seams are declared in `composed.go` and bound in `engine/metal/composed_*_backend.go` (AX-8: lib declares the hook, backend binds it). Runtime levers + the load-bearing facts:

- **Batched MoE**: `composed.MoEExpertsDevice` collapses the routed top-K experts into ONE device dispatch per layer (was top-K×3 quant-seam command-buffer commits). Kill-switch **`LTHN_COMPOSED_MOE_DEVICE=0`** leaves the seam nil → the per-expert host loop (the A/B baseline + revert-safety).
- **Activation is per-arch, not per-kernel.** The batched kernel `MoEExpertsQuant` is gemma's **GELU** SwitchGLU; the composed Qwen lane is **SiLU** (`MoEExpertsQuantSiLU`, `encSiLUGateMulBF16`). Binding the wrong one produces *coherent-but-wrong* text — GELU≈SiLU stays below the greedy argmax threshold, so a model A/B won't catch it. Gate the fix on a **byte-level unit test** (`TestMoEExpertsQuantSiLU`: SiLU ≡ SiLU-ref AND ≠ GELU-sibling), never on "the output still reads fine".
- **A/B recipe** (from a worktree, GPU free): `task build` → `bin/lem`; `MLX_METALLIB_PATH=build/dist/lib/mlx.metallib bin/lem generate -temp 0 -max-tokens N <snapshot>` (lem finds `lthn_kernels.metallib` as its sibling). Run once default, once with the kill-switch; greedy makes the output a correctness oracle. **Two worktrees = two checkouts** — edits in one are invisible to a build in the other until committed + ff'd; consolidate onto one branch before you trust an A/B.
- **The standing gap (the real campaign).** On the mlx-community **Qwen3.5-35B-A3B-4bit** (M3 Ultra, greedy decode): batched-device **~7.6 tok/s** vs per-expert host **~3.1** (the 2.5× above) — but **mlx-lm does ~110 tok/s** on the same checkpoint. The composed lane is ~14× off the ceiling because decode is **host-orchestrated per token** (routing + per-layer submits on the CPU seam); closing it means a device-resident decode loop (ICB-class), not more per-op kernel folds. The batched MoE is a building block toward that, not the finish line.

## Stability Rules (root contract)

Changes to `go/*.go` interfaces affect every consumer simultaneously.

- Never change existing method signatures on `TextModel` or `Backend`
- New capabilities are **separate interfaces** discovered by type assertion (`AttentionInspector`, `VisionModel`, `engine.TrainerModel`) — never extend `TextModel`
- New fields on `GenerateConfig`/`LoadConfig` are safe (zero-value defaults)
- Streaming is `iter.Seq[Token]`; errors via `Err()` after the iterator (the `database/sql` pattern)

## Test Patterns

- `_Good`/`_Bad`/`_Ugly` suffixes (happy path / expected errors / surprising-but-valid) — house-wide
- One test per symbol per variant; names match the real code symbol; `X_test.go` only
- Root package: `resetBackends(t)` before registry tests; reuse `stubBackend`/`stubTextModel`; testify permitted in tests
- `engine/metal` tests skip cleanly without `MLX_METALLIB_PATH`; unit-style policy tests (e.g. `mtp_reengage_test.go`, `mtp_draftlen_test.go`) run host-side

## Coding Standards

- UK English (colour, organisation, serialise, licence) · Licence: EUPL-1.2
- Conventional commits `type(scope): description`, receipts in the body
- Commit trailer, exactly: `Co-Authored-By: Virgil <virgil@lethean.io>`

## Consumers

- **go-mlx** — the airlock/dev tree the metal engine graduated from (this repo is now canonical for `engine/metal`)
- **go-rocm** — AMD ROCm engine consuming the shared contract via `engine/hip`
- **go-ml / go-ai / go-i18n** — scoring, MCP hub, classification consumers of the root interfaces
