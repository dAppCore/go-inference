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

- **#372 verify-ICB (closed — built, measured, FALSIFIED at break-even)** —
  the tail-only replay shipped opt-in (`LTHN_VERIFY_ICB=1`, default off):
  live verify fwd 10.3-10.9ms vs replay 10.8-11.2ms. Two banked findings:
  recordings must be **per-K** (the adaptive draft cap wobbles block width
  5/6/7 — a single-K ICB never replays), and at ONE executeCommands per
  14-op layer tail the execute + recorded-barrier-drain tax eats the
  recorded-op savings — the chained lane's ~3µs/op economics need ONE
  execute per pass. The remaining shape is whole-layer recording
  (attention + tail, pos/N rebinds), blocked on the staged-sliding landing's
  per-pass slot offsets. Also from this campaign: `c06cc8a` fixed
  `serve --draft` (dead since 601ac4e — LoadDir passed maxLen 0 through).
- **#367 q8 KV (dense/ICB lane)** — BUILT, opt-in (`LTHN_KV_Q8_ICB=1`),
  receipted on e2b: global layers store int8+scales on the recorded ICB
  (`0103bd9` kernels, `b41ce28` wiring). Depth receipt: 16K +3.9%, 32K +2.8%
  decode; slope −0.70→−0.52/1K — i.e. ~half of e2b's depth slope IS the
  global-KV bytes (now halved), the other half is #365's non-KV class. Zero
  shallow-depth cost. V1 tax: batched prefill + MTP verify + KV snapshots
  DECLINE on q8. Slice C (`bd1e314`) made the batched fold q8-aware
  (stage + `lthn_kv_q8_store_rows` landing, `lthn_sdpa_multiq_q8` read):
  prefill 146s→14.3s @32K (bf16 batched 8.2s — the gap is the GEMM lane
  forced to multiQ on q8), MTP verify engages at shallow depth. Still
  declining: KV save/restore, per-row shapes, the deep+small 2-pass corner.
  12B receipt CALIBRATES the campaign: decode 16K flat, 32K −2.5% — on
  compute-dominated tokens the current int8 read overhead ≈ the bandwidth
  saved; e2b gains +2.8-3.9%. `-state` works under q8 (`ee8632a`): the bf16
  snapshot mirror makes sleep/wake BYTE-IDENTICAL and .kv portable both
  directions (+0.9s wake at 16K = host mirror loops). The read-cost
  instrument (`TestDiagQ8ReadKernelCost`) shows the q8 2-pass at 105 GB/s
  effective vs bf16's 246 — char4 vectorisation FALSIFIED (no change); the
  cause is strided small-visit DRAM efficiency + the separate scale line.
  **THE TARGET STANCE (Snider): MTP + q8 + 256K context becomes the DEFAULT
  once q8 completes** (q8 ≈ bf16 at ~1%; q6 −6% / q4 −22% stay user-choice).
  The 2026-07-10 fresh-binary 2×2 (e2b-4bit @16K, HIGH-acceptance
  log-continuation prompt, greedy) rewrote the previous day's partial
  receipt — that 47.6-vs-112.9 framing was wrong twice over: (a) the q8+MTP
  loss was the DRAFTER's target-KV export host-dequantising the q8 mirrors
  every draft block (kvExport 108.9ms vs bf16's 0.4); the GPU mirror
  dequant/quant kernel pair fixed it (kvExport → 2.6ms steady, q8+MTP 85.6 →
  125.6 tok/s = bf16+MTP's 128.0 at the same 44% acceptance); (b) q8 plain
  decode @16K is at PARITY (152.0 vs 151.0), so the read-format fix does NOT
  gate MTP+q8. The read-efficiency micro-fixes are ALL FALSIFIED by
  instrument (2026-07-10): chunked key walk 105→95 GB/s (worse — the
  strided TGs already cooperate on aggregate), scale-stream stubbed to 1.0
  gives 105→107 (the separate scale plane is FREE), char4 already dead —
  the q8 2-pass is per-iteration ISSUE-bound; a fat-iteration redesign is
  the only shape that would move it, and the live matrix says it is NOT
  needed: q8 decode holds PARITY at 16/32/64/124K (clean 124K pair 100.6
  vs 103.2 — an earlier "−11% deep decode" was a pressured-machine
  artefact; depth receipts want a quiet machine and full decodes, never
  8-token samples). The q8 PREFILL tax (1.5→3.6× with depth, 307s vs 85s
  at 124K) was the GEMM lane declining q8 — fixed by the q8 GEMM prefix
  (`4751f9a`): the owner dequantises its attended prefix into the layer's
  snapshot mirrors in-encoder and the steel GEMM reads the mirrors.
  Prefill now 1.03-1.04× bf16 at every depth (9.9/26.9/70.8s at
  32/64/124K). MTP-at-depth economics: MTP@16K loses on BOTH formats
  (127-128 vs ~151 plain) even at 44% acceptance — draft AND verify pay
  the depth scan — and the re-engage gate correctly oscillates it off,
  which is exactly the adaptive default the stance needs. q8 is now at
  bf16 parity on every measured axis (decode, prefill, MTP, -state) while
  halving global-KV bytes. **THE DEFAULT LANDED (`ae0636e`)**: q8 KV on by
  default (kill switch `LTHN_KV_Q8_ICB=0` — the bf16 A/B anchor),
  defaultContextCap 32768→262144 (unset context follows the checkpoint
  window: e2b runs 128K, 26B/31B 256K), `-draft auto` already the serve
  default. Live receipt with ZERO flags: e2b 124K prompt → 104.6 tok/s
  decode, 69.6s prefill (the identical command errored at the old 32K
  cap). Known banked cost: the q8 GEMM prefix mirrors allocate
  full-cacheRows bf16 per global owner on deep prompts (e2b@128K ~1.9GB,
  31B@256K ~17GB) — free-after-prefill is the follow-up lever. Remaining:
  the 31B@256K flagship receipt, the N-bit knob (q6/q4 opt-in tiers).
- **#373 (closed — read its receipts before ANY fusion work)** — the fusion
  map: decode is GPU-busy at ~170GB/s of ~800; thin-stage fusion is EXHAUSTED
  (receipted flat); the 500-tok/s lane is fat-dispatch kernel architecture.
- **#371 (closed — the FULL metal_runtime sweep is GREEN again)** — the 16
  failures had three roots, none of them the suspected mlx bump (falsified:
  pre-bump Go + current metallib passes). (a) the q8 KV default quantised the
  paged pool on lanes whose decode never reads it — q8 now defaults only on
  the paged-decode lane (MoE/trace = 26B), `8fc251d`; (b) the fused PLE
  gate+gelu ICB op was reverted from the arch recording — byte-identical in
  every standalone probe yet drifting in the recorded replay from the second
  step (in-situ cause unresolved; its receipt was ~3µs), `96e0cc3`; (c) the
  coverage guards now pin the loader-owns-default context contract,
  `011a41d`. A red FULL sweep is a REGRESSION again — treat it as such.
  Residual seam for #367: on the 26B paged-q8 lane the batched pass attends
  its own fresh K/V rows pre-quantisation (serial attends post-round-trip) —
  unpinned, price a fresh-row round-trip if 26B MTP revives.
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

- **`bin/lem` is a build artifact, not the tree** — a full 2×2 A/B ran on a
  binary built 2h before the code it was supposed to measure (env opt-ins
  silently ignored; the q8 lanes were bf16 reruns identical to the ms).
  `task build` before ANY receipt run, and treat impossibly-identical lanes
  as a stale-binary tell, not a result.
- **objc `new*` buffers leak unless YOU release them** (third scar in this
  class: the MoE re-upload leak, the resident-cache eviction leak). The
  test sweep stacked ~10 GB of IOAccelerator-dirty per real-model test and
  drove machine memory pressure red — `653aef1` releases at ranged eviction.
  Two rules from its falsifications: release ONLY where the owner is
  provably done (releasing at reset-for-test dangled live fixtures — wrong
  tokens, then SIGSEGV), and don't chase ps-RSS — clean mmap pages inflate
  it harmlessly; `vmmap --summary`'s physical footprint + IOAccelerator
  DIRTY column is the pressure that matters.
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
