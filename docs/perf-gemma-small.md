# gemma4 small-model (E2B/E4B) decode underperformance — root-cause report

Campaign: `perf/gemma-small`, off `1d561d80`. M3 Ultra, greedy decode (temp 0),
256-token generations unless noted. Instrument-first per the house rule — every
number below is a live measurement, not an estimate.

## Headline numbers

CLI bench (`bin/lem generate -temp 0 -max-tokens 256 -trace -prompt "Write a
detailed technical essay about the history and future of computer
processors."`), decode-phase GPU-busy from the `-trace` phase budget:

| model | quant | tok/s | ms/token | GPU-busy % |
|---|---|---:|---:|---:|
| E2B | bf16 | 88.0 | 11.308 | 99.9% |
| E2B | qat-4bit (mixed 4/8-bit) | 140.7 | 7.071 | 99.9% |
| E2B | **plain 4bit (uniform)** | **170.9** | **5.823** | 99.8% |
| E4B | qat-4bit (mixed 4/8-bit) | 91.9 | 10.835 | 99.9% |
| 12B | 4bit (uniform) | 70.7 | 14.073 | 99.9% |
| 26B-A4B | 4bit (uniform, MoE) | 139.6 (task-measured) | — | — |

(Small variance vs the task brief's own numbers — 143.8/93.7/72.0 there vs
140.7/91.9/70.7 here — is normal run-to-run noise on a shared prompt/box, not
a discrepancy.)

**The load-bearing new data point: plain (non-QAT) uniform-4bit E2B hits
170.9 tok/s** — at/above the bf16 reference of ~174 — on the exact same CLI
binary and engine code that gets only 88.0 tok/s on E2B bf16. Same
architecture (same PLE tower, same 35 layers, same KV-sharing), same code
path, wildly different result. **This rules out "PLE tower is structurally
broken" as the root cause** — PLE is present and paying its cost in the fast
170.9 tok/s run too. The variable that actually predicts speed here is the
**weight representation** (uniform int4 vs mixed int4/int8 vs bf16), not PLE
presence.

Every row above shows GPU-busy ≥99.8% of wall time (`LTHN_GPU_BUSY=1`
corroborates: no meaningful host-stall on any of these). **This is a genuine
GPU-execution-time problem, not a host-orchestration/CPU-stall problem.** The
"chained GPU span" bucket in `-trace` (single bucket, no further split)
confirms all these runs take the production chained/pipelined ICB-replay
decode path — none of them fall back to a slower host loop.

## Isolating the seam: PLE projection is cheap; the ICB layer stack dominates

The `-trace` CLI flag only reports one GPU bucket for the chained/pipelined
path. The finer 3-way split (PLE projection / ICB layer stack / head) needs
`stepGreedyChainDisabled` forced (a process-global, no CLI flag), which the
repo already had wired for exactly this purpose in
`TestRealE2BChainedGPUParityAndSpeed` (`engine/metal/native_e2b_real_test.go`).
Ran it on the real **e2b-it-4bit** (uniform) checkpoint
(`LEM_REAL_E2B=1 MLX_METALLIB_PATH=... go test -tags metal_runtime -run TestRealE2BChainedGPUParityAndSpeed -v ./engine/metal/`):

```
real e2b-4bit decode tok/s (tg64): host 158.2  chained-GPU 164.1 (1.04x, gpu-busy 90%)  pipelined 179.3 (1.13x, gpu-busy 100%) — tokens identical
per-token GPU split (serial, ms): PLE 0.267  layer-stack 5.605  head 0.588  (sum 6.460)
barrier ceiling: pipelined no-barrier per-token GPU 1.789ms (wall 552.3 tok/s) vs barriered layer-stack 5.605ms — barrier cost headroom
FFN-fusion ceiling: drop gate/gelu/down barriers -> per-token GPU 5.206ms (191.2 tok/s) vs full 5.552ms — fused-FFN reclaim 0.346ms/token (~191 tok/s if realised)
fine-grained pipelined: 116.1 tok/s  8.574ms/token GPU  tokens-match-host=true
```

Reading this:

- **PLE projection is 4% of the token** (0.267ms of 6.460ms). Not the
  bottleneck by itself.
- **The ICB layer stack is 87%** (5.605ms) — this is where the time is,
  but it's the WHOLE per-layer op sequence (attention + FFN + the
  interleaved per-layer PLE gate + norms + the per-layer output scalar),
  not PLE specifically.
- **Removing every barrier drops the layer stack from 5.605ms to 1.789ms**
  (racy/invalid output — timing-only ceiling). ~3.8ms/token, roughly
  **68% of the layer stack and ~59% of the whole token, is pure
  barrier-drain**, not compute.
- **Removing only the FFN-internal barriers reclaims just 0.346ms** of that
  3.8ms available. The barrier cost is **spread across attention, FFN, PLE
  and norms roughly evenly** — it is not concentrated in one stage.
- **Fine-grained (resource-scoped) ICB barriers are slower**, not faster
  (116.1 vs 179.3 tok/s), confirming the codebase's own prior finding
  (`TestSpikeFineGrainedReplayMatchesCoarse`, same file): "fine-grained ICB
  replay is an invalid R&D spike: encoder memory barriers between ICB
  ranges do not enforce command dependencies." Already tried, already a
  documented dead end.

## Per-op histogram: many small dispatches, no fat kernel

`TestRealE2BWithinLayerOpCost` (same file) times each ICB op individually
(its own command buffer) on the real e2b-4bit checkpoint:

```
decode (tg48): 182.3 tok/s (5.48 ms/token, gpu-busy 99%)
--- per-op GPU µs, global owns-cache layer 4 (ops 88..109) ---
  op  9 (idx  97):   24.50 µs <- SDPA
  op 16 (idx 104):   33.62 µs
  (20 more ops, range 8.2–19.0 µs)
layer 4 Σ = 289.83 µs (× 35 layers ≈ 10.144 ms)
```

22 ops per layer, 8–34µs each, no single dominant kernel (SDPA is the
biggest *labelled* op at 24.5µs, comparable to several unlabelled ones).
Per the test's own framing: "if it lives in a few fat gemvs... projection
fusion is low-value; if it spreads across many skinny dispatches,
dispatch-count reduction pays" — **it spreads**. (Note: the isolated-CB Σ
here, 10.14ms, overshoots the fused barriered layer-stack, 5.6–5.8ms,
because timing each op via its OWN command buffer adds a fixed submit+wait
tax per op on top of its real GPU time — ~5µs/op × 770 ops ≈ 3.96ms. Not a
usable "true compute floor"; still valid for the RELATIVE per-op shape.)

## bf16's extra cost is fully explained by bandwidth + the *same* fixed tax — no missing optimisation

Structural check first: does bf16 decode skip an optimisation the quantised
path gets? No. `recordArchICBBF16` (`engine/metal/decode_forward_arch_icb.go:2224`)
wraps every bf16 weight into a "dense" `QuantWeight{Packed: b}` and calls
**`recordArchICBQuant` directly** — the identical recorder, identical op
count, identical barrier structure as the quantised path. The only
per-op difference is which kernel PSO gets bound (`emitGemv` for
dense/bf16 vs `emitQMV` for quantised, selected in `setQMV` at
`decode_forward_arch_icb_quant.go:565-575`). bf16 pays a bigger per-op
kernel cost (a 16-bit GEMV streams 4x the bytes of a 4-bit QMV) on an
**identical dispatch/barrier skeleton** — not a bug, a genuine cost.

Quantifying it — a 2-point linear model over the CLI `-trace` GPU-busy
numbers (fixed cost `F` + bit-width-proportional weight-bandwidth cost `W`,
in 4-bit-equivalent units):

```
int4 uniform: F + 1·W = 5.823 ms   (measured: plain 4bit CLI trace)
bf16 (4x the bit-width): F + 4·W = 11.308 ms   (measured: bf16 CLI trace)
  ⇒ W ≈ 1.83 ms      (weight-bandwidth-proportional share at 4-bit)
  ⇒ F ≈ 3.99 ms      (fixed: dispatch count / barrier-drain / attention / norms — width-independent)
```

**Even at int4, the fixed per-token dispatch/barrier tax (~4ms) is bigger
than the actual weight-streaming time (~1.83ms).** E2B decode is
dispatch-count-bound before it is bandwidth-bound. bf16 pays the identical
fixed tax plus a large, inherent 4x bandwidth premium on top of it — both
terms structural, neither a bug.

Sanity check against the third data point, qat-4bit (MLP tensors only at
8-bit, everything else 4-bit — confirmed via `config.json`'s
`quantization` block: 105 per-tensor overrides = 35 layers × {gate,up,down},
all `bits:8`, vs plain-4bit's 0 overrides): predicted total
`F + W·(1+m)` where `m` = MLP's share of baseline 4-bit weight bytes.
Measured qat total 7.071ms ⇒ `m ≈ 0.68` — a dense transformer's FFN
(gate+up+down) typically running 70-80% of a layer's parameters. No
unexplained residual; the two-term model holds up.

## Why 12B/26B don't pay this

12B has **no PLE** (0 extra ops/layer), **no KV-sharing indirection**
(every layer owns its own cache — `num_kv_shared_layers: 0` vs E2B/E4B's
20/18), and is **2.5x wider** (dModel=3840/dFF=15360 vs E2B's 1536/6144).
Same coarse-ICB-barrier architecture (`recordArchICBQuant`/`recordArchICBBF16`
are shared across every gemma4 size — there is no small-model-specific code
path), but each of 12B's ~18-20 ops/layer is proportionally much bigger, so
the *same* fixed per-op barrier-drain cost is a much smaller fraction of
each op's time. 12B's measured 14.073ms/token over 48 layers ≈ 293µs/layer
in absolute terms — bigger than E2B's ≈160-166µs/layer — yet 12B lands at
~97% of its reference (70.7 vs ~73) while E2B bf16 lands at ~50% of its
(88.0 vs ~174), because the fixed component is proportionally negligible
for 12B and dominant for E2B. 26B-A4B (MoE-routed, task-measured 139.6 vs
ref ~144) matches for the same reason: its routed-expert dispatch is large
per op.

**Conclusion: this is not a PLE-tower bug. It is E2B/E4B's narrow
dimensions making the shared, deliberately-coarse ICB barrier scheme's
fixed per-op cost a dominant fraction of decode time — a structural
property of small/narrow MatFormer models on this decode architecture,
not a defect isolated to the per-layer-input path.** PLE contributes
~4% directly and indirectly adds 2 more small ops/layer to an
already-unfavourable op-size distribution, but removing PLE entirely
would not close the gap — the same barrier tax would still dominate
attention + FFN alone at E2B's width (12B's dense-only, PLE-free ops
already show the identical shape, just diluted by size).

## Already tried and rejected in this codebase (not re-attempted here)

1. **PLE gate+GELU single-kernel fusion inside the ICB replay**
   (`lthn_ple_gate_gelu_qmv`, #373) — comment at
   `engine/metal/decode_forward_arch_icb_quant.go:598-606`: byte-identical
   to the composed two-stage path in every standalone context, but **caused
   token divergence from the second decode step onward when recorded into
   the replay** (#371, in-situ cause unresolved), and its receipt was only
   ~3µs/layer ("thin-stage"). Low-yield AND has an open correctness bug —
   correctly left reverted. Consistent with my own finding that the PLE
   piece alone is ~4% of the token.
2. **Fine-grained (resource-scoped) ICB memory barriers** instead of the
   coarse `SetBarrier` — `fineGrainedReplay`, measured slower on the real
   model (116.1 vs 179.3 tok/s) and documented as breaking dependency
   ordering (`TestSpikeFineGrainedReplayMatchesCoarse`). Dead end, already
   banked as a negative result.

## SHIPPED (slice 1) — value-norm barrier prune, +2.3–2.6% on E2B/E4B

Acting on the barrier-drain root cause, the first bounded slice removes **one
full-drain `SetBarrier` per gemma4 layer** from the attention half of the ICB
recorder (`decode_forward_arch_icb.go`, the owner + shared-KV branches).

The recorder already relaxes the big-gemv siblings to no-barrier
(`emitNB`): Q/K/V all read `normed`, so Q barriers and K+V "ride free". The
gemma4 **value-norm** (`vCache = rms(vCache)` per head) was left as its own
full-drain barrier because it followed the V-projection with nothing to flush
it. The fix reorders the K/V prep so V is projected **before** the K-norm-rope:
the K-norm-rope's barrier then flushes vCache, and the value-norm reads the
already-flushed row and rides free (`emitNB`). Byte-identical — same ops, same
data dependencies (SDPA still barriers on qr/kCache/vCache); only the recorded
order and one barrier flag change. Gated on `valueNormOnes != nil && !qkvBias`
so non-gemma arches (Qwen qkvBias / llama / mistral) keep the proven order
untouched.

**Before → after (M3 Ultra, temp 0, 256-token decode, GPU quiet, 3-run best):**

| model | quant | before tok/s | after tok/s | Δ |
|---|---|---:|---:|---:|
| E2B | qat-4bit | 140.7 | 144.3 | **+2.6%** |
| E2B | plain 4bit | 168.2 | 172.2 | **+2.4%** |
| E4B | qat-4bit | 91.7 | 93.8 | **+2.3%** |

Corroborated by the piece-split instrument: `TestRealE2BChainedGPUParityAndSpeed`
layer-stack dropped 5.605ms → 5.498ms/token (~0.107ms, ~1 barrier·15 owner
layers). Commit carries the full receipt.

**Correctness (all gates green):**
- Full `-run 'ICB|Parity|Chain'` metal suite passes (3 pre-existing
  allocation-budget failures in `icb.go`/`icb_layer.go` — `NormProjectICB`,
  `AttentionBlockICB` — fail identically on the clean base; untouched code).
- `TestRealE2BChainedGPUParityAndSpeed`: host / chained-GPU / pipelined tokens
  **identical** on the real e2b-4bit weights.
- Greedy "…is **Paris**." AND a 64-token greedy essay are **byte-identical**
  before→after on E2B qat-4bit, E2B plain-4bit, and E4B qat-4bit (md5-checked).

The fixed per-token dispatch/barrier tax is shared by every gemma size, so this
compounds fleet-wide (12B dense owner layers get the same saving).

## Slice 2 (k-norm-rope hoist) — FALSIFIED, net-zero, reverted

The obvious next lever was the **second** attention-half barrier: the three
post-projection ops — qk-norm-rope(q), k-norm-rope(k), value-norm(v) — are
*mutually* independent once Q/K/V are all projected, so hoisting the K and V
projections into the common prefix **ahead** of qk-norm-rope makes that single
barrier flush all three, letting BOTH k-norm-rope and value-norm ride free
(`emitNB`) — one more full-drain barrier/layer removed on top of slice 1.

Built it (owner + shared-KV branches, gated `useFusedQKRope && hasQN && fuseK &&
valueNorm && !qkvBias && !kvQ8.any()` so q8 KV stayed on the proven path). It is
**byte-perfect**: the full `ICB|Parity|Chain|Q8|KVShare|KEqV|PLE|OpLayout` suite
passed, `TestRealE2BChainedGPUParityAndSpeed` + `TestRealQuantVerifyBatchedHiddensParity`
reported host/chained/pipelined tokens identical, and greedy Paris + a 64-token
essay were md5-identical before→after on E2B qat-4bit, E2B plain-4bit, E4B
qat-4bit AND **12B** (the all-owner path), with the q8 CLI path unchanged.

But the **perf receipt was net-zero** (same-session A/B, GPU quiet, 3-run):

| model | slice-1 | slice-2 | Δ |
|---|---:|---:|---:|
| E2B qat-4bit | 144.2 | 143.5 | −0.5% |
| E2B plain-4bit | 172.8 | 172.9 | flat |
| E4B qat-4bit | 93.2 | 93.2 | flat |
| 12B 4bit | 72.6 | 72.5 | flat |

(E4B and 12B dead-flat across both measurement blocks ⇒ the machine was stable,
so this is a real result, not drift.)

**Why it loses (the mechanism):** hoisting the two big K/V matvecs *ahead* of
qk-norm-rope forces that barrier to wait for them — so the saved barrier drain is
paid back exactly as lost overlap. In slice 1, K/V project `emitNB` *after*
qk-norm-rope and overlap it plus the k-norm-rope; moving them earlier trades that
hidden matvec latency for the barrier. Net zero. This is the codebase's standing
"removing a barrier can cost the overlap it was hiding" caution, confirmed live.

**Reverted** per the working discipline (a lever measured net-zero is not
shipped into the parity-critical recorder for complexity's sake). Slice 1's clean
+2.4% stands. Banking the falsification here rather than in the tracker since the
whole root-cause thread lives in this doc.

## Remaining levers (lower priority)

- **FFN half is already at the `emitNB` floor** (gate barriers, up rides free;
  gelu·up → down → residual are a true sequential chain). No slack without a
  kernel fusion — and the two candidate fusions (input-rms→qmv, gelu-fold into
  down) are both measured net-zero-or-worse and reverted. Skip.
- **PLE tail (5 sequential barriered ops)** is a genuine dependency chain with
  no sibling slack; the one available fusion (post-norm residual) diverges
  ~2 ULP and is byte-parity-hostile (already reverted). Not worth pursuing.
- The attention half is now at its barrier floor for this ICB decode structure:
  slice 1 took the one free barrier, slice 2 proved the second isn't free. A
  further win needs a *kernel* change (e.g. a projection megakernel), not a
  barrier/ordering change — a different, larger campaign.

## Reproduce

- Bench: the recipe in the task brief (`bin/lem generate -temp 0
  -max-tokens 256 …`), read the `decode N tok/s` line; run on a quiet GPU
  (`pgrep -f "lem generate"` empty).
- Parity: `MLX_METALLIB_PATH=<repo>/build/dist/lib/mlx.metallib go test -tags metal_runtime ./engine/metal/ -run 'ICB|Parity|Chain' -count=1`
  (from `go/`).
- Piece split / real-model parity: add `LEM_REAL_E2B=1` and
  `-run 'TestRealE2BChainedGPUParityAndSpeed|TestRealE2BWithinLayerOpCost' -v`
  (resolves the plain e2b-it-4bit snapshot automatically; `LEM_PROFILE_DIR`
  aims the op-cost breakdown at any other dense-ICB gemma4 snapshot).
