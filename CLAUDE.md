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

## Qwen hybrid — the FACTORY is the default route (#18); composed is the escape hatch

Since 2026-07-19 (`ae095829`) **qwen3_5 / qwen3_5_moe load through the factory** (`model.Assemble` + `arch_session`) with the fused whole-token chain decode (`engine/metal/arch_qwen_fused.go`): every layer (gated-delta / gated-attention, dense or MoE tail) encodes into ONE command buffer with resident state, CB-recorded and replayed per token where servable. It **beats the composed lane on both local hybrids** — 0.8B 217 vs 195 tok/s greedy, 35B-A3B 30 vs 19.

- **Routing** (`engine/metal/load.go`): factory default for qwen3_5* EXCEPT `LTHN_QWEN_COMPOSED=1` (the A/B + revert lever), sub-2-bit packs (Bonsai 1-bit — no 1-bit qmv width yet), MTP pair loads, and image turns (the 35B ships a vision tower composed serves; the factory answers image turns with the clean 400). `LTHN_QWEN_FUSED=0` forces the factory's host halves (the correctness reference).
- **The submit-ahead decode tails decline recurrent sessions** (`hasRecurrentLayers`): a speculative link past a stop cannot be unwound from a gated-delta recurrence. Serial tail only.
- **Replay bookkeeping**: `composedChainReplay` advances the `attnKVDeviceState.n` counters ITSELF — bumping them again host-side double-advances the position and silently corrupts KV slot addressing (coherent-but-wrong text; caught by the live-walk A/B).
- **Activation is per-arch, not per-kernel.** `MoEExpertsQuant` is gemma's **GELU**; the qwen lane is **SiLU** (`MoEExpertsQuantSiLU`). Wrong binding = *coherent-but-wrong* text below the greedy argmax threshold. Gate on the byte-level test (`TestMoEExpertsQuantSiLU`), never on "the output reads fine".
- **A/B recipe** (from a worktree, GPU free): `task build` → `bin/lem`; `MLX_METALLIB_PATH=build/dist/lib/mlx.metallib bin/lem generate -temp 0 -max-tokens N <snapshot>`; greedy output is the correctness oracle. Run default vs `LTHN_QWEN_COMPOSED=1` vs `LTHN_QWEN_FUSED=0`.
- **What keeps `model/composed` alive** (the #18 deletion checklist): engine/hip's consumer (codex-owned until ~Aug 10), the arch-zoo Composed registrations (mixtral/dbrx/olmoe/granitemoe/llama4/qwenmoe + qwen3_6/qwen3_next/composed/hybrid ids), the qwen MTP pair + vision tower, mamba2/rwkv7. Next gap to close: mlx-lm ~110 tok/s on the 35B vs our 30 — now a unified-engine campaign.

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
