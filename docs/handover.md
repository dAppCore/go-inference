---
title: Handover
description: Working notes from the engine-perf campaigns — for the next driver of this codebase.
---

# Handover — from the previous driver

You're inheriting a working engine. Everything below is the part that isn't
derivable from the code: how it got to this state, what was falsified along the
way, and the traps that cost real hours. Trust the receipts in `git log` over
anything here; the task tracker carries the live campaign state.

## State of the board (2026-07-10, M3 Ultra 96GB, 4-bit, tg512)

| model | plain | MTP pair | notes |
|-------|-------|----------|-------|
| E2B | ~172 | **~235** | trainer workhorse; PLE arch |
| E4B | ~120 | **~171** | qat-4bit drafter (dequantised at load) |
| 12B | ~72 | ~93 | bf16 drafter |
| 26B-A4B | ~142 | LOSS | MoE; its qat drafter loses — the MoE verify cost, not the drafter |
| 31B | ~34 | ~47 | orchestrator |

Depth (e2b, `generate`-measured): linear −0.61 tok/s per 1K context, no cliff;
~105 at 98K. Cold prefill is the deep-context pain (54s at 98K) — conversation
continuity (`serving/continuity`, shipped) makes it once-per-conversation:
turn 2 of a 10K chat is 0.4s vs 5.5s, book-bench prompt column flat at ~88
tok/turn while the client resends full history.

## The method (this is the important part)

1. **Instrument first.** Before forming any view, build or run the
   measurement: the anatomy bench (`LEM_SDPA_ANATOMY=1`), the MTP diag
   (`LTHN_MTP_DIAG=1`), the confidence capture (`LTHN_MTP_CONF=<path>`), or
   a plain live A/B. Every productive result this engine has came from an
   instrument saying something surprising; every wasted hour came from a
   theory that skipped the instrument.
2. **No receipt, no claim — and no keep.** If you build an optimisation and
   the live A/B doesn't show it, revert it *even if the design is beautiful
   and the suite is green*. It has happened here: a full cache rewrite,
   suite-green, reproducibly slower at 32K — reverted the same night, lesson
   banked. The discipline is what keeps the tree trustworthy.
3. **One lever per commit, receipt in the message.** Future-you greps
   commit messages for numbers. Make them greppable.
4. **Kill-switch anything adaptive.** Wall-clock-adaptive policies (the MTP
   re-engagement gate, the dynamic draft cap) make runs non-reproducible by
   design. Their env kill switches (`LTHN_MTP_REENGAGE=0`,
   `LTHN_MTP_DRAFTLEN=0`) are the reproducibility anchors — never ship an
   adaptive behaviour without one.

## The open campaigns (task tracker)

- **#372 verify-ICB** — the next big lever (+15-25 e2b MTP est). Spec is
  build-ready to the function level in the task metadata: record the
  pos-independent per-layer tail of the MTP verify fold into an ICB, replay
  between live attention encodes. Every unknown pre-answered (barrier
  mechanics, the `SupportIndirectCommandBuffers` PSO checklist, sink
  conversions, the cut line, validity keys). 3-5 focused hours.
- **#367 q8 KV (dense/ICB lane)** — priced: halves the depth slope, small-RAM
  fleet value. The 26B depth curve is still wanted before building.
- **#373 (closed — read its receipts before ANY fusion work)** — the fusion
  map: decode is GPU-busy at ~170GB/s of ~800; thin-stage fusion is EXHAUSTED
  (receipted flat); the 500-tok/s lane is fat-dispatch kernel architecture.
- **#371** — 13 pre-existing metal_runtime failures (q8 tolerance, ICB prefill
  serial-parity, RestoreKV budgets). They predate the campaigns above — do not
  panic when the FULL sweep fails; the targeted suites are green. Suspected
  mlx v0.32.0 bump fallout.
- **#366 / #360** — the frontier lanes (KV temporal coherence; Lemma v2
  training), waiting on their own starts.

## Falsified — do not re-dig

- MXU/NAX on M3 Ultra (needs GPU arch gen 17; this part is g15).
- Token early-exit on e2b (argmax decides at layer ~29/35; detector > skip).
- Weight-stream splitting at small K (the gather's CONCURRENT threadgroups
  already L2-amortise the stream; small-K matvecs are occupancy/launch-bound,
  not bandwidth-bound).
- TG-per-row fusion kernels (K threadgroups starve a 76-core GPU).
- In-kernel grid-barrier megakernels (Metal guarantees no TG co-residency —
  barriers spin-bail into wrong results; even `lthn_ffn_megakernel` is
  off-by-receipt in production).
- Thin-stage ICB fusion for speed (saves only ~3µs launch; the byte-identical
  PLE gate fusion shipped anyway since it costs nothing — `d00c526`).
- Draft-block ≠ 4 on e2b (3/6/8/12 all worse or noise).
- Serve two-point marginal for depth curves past ~16K (rides the variance of
  two full prefills — produced a false "98K collapse"; use
  `generate -prompt-file`).

## The house fusion rule (learned twice, keep it)

Fold only **O(output)** work into a matvec's loads/stores; per-element work
consumed by many tiles must stay its own dispatch. The receipted-off
`geluFoldEnabled` / `routerFusedEnabled` / `enableInputRMSFusion` all violate
it; the byte-identical `lthn_ple_gate_gelu_qmv` respects it (gelu once per
output row at lane 0). For byte-identity ports of MLX matvecs, replicate
`qmv_fast_impl` verbatim — `lthn_qmv_rows` and `lthn_ple_gate_gelu_qmv` are
the precedents, and `bytes.Equal` gates are achievable, not aspirational.

## Instruments (run these, don't re-derive)

- **Wall vs GPU split**: `TestDecodeWallGPUSplitRealE2B` (`LEM_REAL_E2B=1`) —
  chained-lane GPU-span accounting; the fixed-cost fork resolver.
- **Per-stage GPU buckets**: `LTHN_GPU_TRACE=1` on any serve (batched pass).
- **MTP internals**: `LTHN_MTP_DIAG=1` (draft fwd/head split, verify rows,
  acceptance per block).
- **Depth curve**: `lem-dev generate -prompt-file` per depth point.
- **Live harness**: `lem.sh bench/pair/book/smoke` — the book prompt column is
  the continuity receipt (~88 tok/turn flat = no replay).

## Traps that will bite you (each cost real time)

- `MLX_METALLIB_PATH` must be THIS repo's `build/dist/lib/mlx.metallib` —
  go-mlx's lacks the lthn kernels ("kernel not found" on runtime tests).
- `task metallib:kernels` rebuilds the lthn library; new `.metal` files under
  `engine/metal/kernels/` auto-include. The MLX include chain needs the fast
  variant's geometry gates (`inDim%512==0`) respected per kernel.
- Serve's default context now follows the checkpoint window capped at 32K
  (`resolveDefaultContext`); before `601ac4e` everything silently capped at
  4096 — the book-bench chapter-5 death class.
- The ICB op layout is dynamic per-feature (`opsPerLayer` accounting) —
  uniform shrinks are fine and the recorder self-asserts the final count.
- Continuity decline reasons log via `core.Error`, which does NOT reach the
  serve stderr — don't chase ghosts in `/tmp/lem-serve.log`.
- Codex (GPT-5.6-Sol) owns `go/engine/hip` on the homelab box (~a month); it
  has NO git access — snapshot its working tree to a `codex/...` branch (see
  `codex/hip-stint-2026-07-10`) and review for mitigation-instead-of-root-cause
  before anything reaches shared layers. Its RAM-offload lane (experts in
  system RAM, <6GB VRAM for 26B) amortises on MoE only — dense models must fit
  resident (12B fits a 16GB card; 31B does not).

## Where the money went (shipped receipts, greppable)

`c77f5be` serve arms MTP (was silently plain) · `b2b2dcf` drafter cluster head
on GPU (10-17ms → 0.1ms/draft tok) · `aa45e02` quant K-row verify head ·
`028766d` 3-dispatch verify epilogue (verify 11.5 → 9.7ms) · `96e1b83`
conversation continuity (13× turn-2) · `601ac4e` checkpoint-window context
default · `45e014f` the wall/GPU split probe · `d00c526` byte-identical PLE
gate fusion + the thin-stage receipt.
