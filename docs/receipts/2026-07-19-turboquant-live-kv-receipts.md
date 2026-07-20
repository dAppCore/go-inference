# TurboQuant live KV — merge receipts and the default decision (2026-07-19)

Method: `lem generate --prompt-file <needle-prompt> --max-tokens 24 [--kv-cache <mode>]`
on `mlx-community/gemma-4-e2b-it-4bit`, greedy, 16 384-token filler with the fact
planted ~10% in and the question at the end; decode tok/s from generate's own
report (prefill excluded); peak RSS from `/usr/bin/time -l`. One run per mode,
M3 Ultra 96 GB.

| mode | needle | decode tok/s | prefill tok/s | peak RSS |
|---|---|---|---|---|
| native (off) | retrieved | 130.2 | 157 | 5.78 GB |
| turboquant:4 | retrieved | 88.2 | 110 | 5.79 GB |
| turboquant:3.5 | retrieved | 85.5 | 109 | 5.73 GB |

**CAVEAT (added same day): these numbers ran under heavy fleet load** (a 27-core
metal test suite was resident; the off-mode 130 tok/s vs the known ~174 E2B
baseline is the tell). The needle/quality column stands; the speed and RSS
columns are not clean. Root-causing the gaps found BOTH to be v1 wiring
deferrals, not codec cost: (1) TQ sessions declined the submit-ahead peer ICB —
the #23 serial tail re-taxed every token — fixed in `e870c2ef` with the live
gates unchanged; (2) TQ prefill fell back to per-token sequential replay
instead of the batched pass — batched TQ prefill is in flight. A quiet-box
re-run of this table follows both; the default decision below is provisional
until then.

Correctness context (committed gates, same tree): teacher-forced top-1 agreement
62/64 with max |logit Δ| 6.31 at b=4 over 64 steps; in-test retrieval over 1 107
prefilled tokens passes; real-KV distortion vs the paper oracle in
`2026-07-19-turboquant-real-kv-distortion.md` (b=2 at oracle, b=3 +14–16%,
b=4 +4–8%; K/V asymmetry ≤3.7% and sign-inconsistent).

## Decision

**The default stays `native` — provisionally.** The pre-registered bar for
flipping the default to 3.5-bit was: needle retrieved AND decode tok/s at or
above parity AND RSS visibly down. Quality passes; the speed/RSS columns above
are load-contaminated and both speed gaps traced to fixable wiring (see
caveat) — the bar gets re-judged on the quiet-box re-run.

Why RSS is flat here: gemma-4 E2B carries only three distinct global-attention
caches (`num_kv_shared_layers` routes the other nominal full-attention layers
onto layer 14's cache; the remaining layers are sliding-window bounded). The
codes shrink a small fraction of residency while weights + transient prefill
scratch dominate, and the TQ read path (code dequant in-kernel, per-step q
rotation, per-layer prefill dequant scratch) prices every token.

## Where the opt-in applies

`--kv-cache turboquant[:2|:3|:3.5|:4]` remains available on dense
global-attention models. The configuration with a credible residency case is a
large dense model at very long context (31B, 256K window, where global KV
reaches tens of GB); that receipt has not been run. v1 declines MoE,
hybrid-recurrent, sinks models, paged/batch lanes, MTP pairing, and state
sleep — each loudly.

## What would earn the default

1. A 31B (or comparable many-global-layer) long-context receipt showing the
   residency win materialise with needle parity.
2. A perf-recovery pass on the TQ read path (fused q-rotation, wider code
   loads) closing the decode gap toward parity.

Until at least one of those lands with numbers, TurboQuant is a knob, not the
default.

## Quiet-box re-run (same day, all lanes drained)

| mode | needle | decode tok/s | prefill tok/s (TTFT) | peak RSS |
|---|---|---|---|---|
| native (off) | retrieved | 131.6 | 158 (77.8 s) | 5.80 GB |
| turboquant:4 | retrieved | 87.8 | 8,291 (1.49 s) | 5.68 GB |
| turboquant:3.5 | retrieved | 86.1 | 8,391 (1.47 s) | 5.68 GB |

Post peer-ICB fix + batched TQ prefill + single-pass fusion. Two findings:
(1) the TQ batched landing prefil ls 52x faster than the native lane at 16k —
and the native side of that ratio is its own pathology (158 tok/s on a quiet
box is per-token-class, not batched; tracked separately). (2) the decode gap
at 16k persists because kvLen>=1024 runs the UNFUSED 2-pass lane — the fusion
receipt's +16% lives on single-pass only; the pass-2 runtime-dim port is the
named lever for long-context decode parity.

**Decision confirmed: the default stays `native`.** Needle passes everywhere;
decode fails the parity bar at exactly the depths TQ targets; E2B residency
delta is real but small (three global caches). The opt-in stands, now with a
52x TTFT win at long context as its honest selling point.

## Addendum (same day): #54 landed — the native side of the TTFT ratio is fixed

The 158 tok/s native prefill above was the resident-session q8
canonical-landing guard forcing per-token prefill on one-shot callers
(`DisablePromptReuse` opt-out, merged 819957a1). Native E2B 12k prefill now
measures **9,585 tok/s median / 1.24s TTFT** (3 runs, idle box, <1.1%
variance) — faster than TQ's own 8,291–8,391. TQ's honest selling point is
therefore **memory** (codes-resident KV, 5.68 vs 5.80GB) and needle-clean
long-context storage, NOT TTFT; the 52x ratio is retired. The 2-pass decode
lever remains the open item.

## #48 long-context slice (2026-07-20): pass 2 folds the output unrotation — the named lever, closed

The 2-pass decode gap above (kvLen ≥ 1024, `sdpa2PassMinKV`) was the unfused
TQ tax: q pre-rotation, pass 1, pass 2, output unrotation — 4 recorded
ops/layer/token, where the single-pass lane (below the knee) pays 1. Pass 2
was MLX's shipped, precompiled `sdpa_vector_2pass_2` — shared kernel code
this lane could not add an epilogue to — so the unrotation stayed a fourth,
separate `lthn_tq_unrot_rows_bf16` dispatch reading a rotated-space bf16
scratch (`attnRotTQ`) pass 2 had just written.

**Mechanism**: `lthn_sdpa_vector_2pass_2_tq` (`kernels/lthn_tq_kv.metal`) —
TQ's OWNED fork of MLX's `sdpa_vector_2pass_2`, ported verbatim from the
vendored source (`external/mlx/mlx/backend/metal/kernels/sdpa_vector.h`) with
ONE addition: after the stock merge reduction produces the per-head
ROTATED-space output, the kernel stages it into threadgroup memory and folds
Πᵀ once before writing the FINAL value straight to `out` — no rotated-space
scratch, no fourth dispatch. The fold is legal under the house fusion rule
(O(output)-only folds) because this kernel dispatches ONE threadgroup per
head — grid `(nHeads,1,1)`, 1024 threads — identical geometry to the
single-pass kernel's own fused epilogue, so Πᵀy costs the same O(d²) once per
head, never once per block. Pass 1 (`blocks` threadgroups per kv-head) stays
genuinely unfused — the q pre-rotation remains the caller's separate
`lthn_tq_rot_rows_bf16` dispatch before it, exactly as before, because
folding it into pass 1 would multiply that O(d²) work by the block count.
Net: **4 recorded ops → 3** (rotate, pass 1, pass 2) for every TQ global
layer past the 2-pass knee.

The port was safe to write from scratch (no MLX header to cross-check
against previously blocked an unrelated runtime-dim port attempt at this
same seam, see `sdpa_rtdim.go`) because (a) `external/mlx` is vendored in
this tree, giving a byte-exact reference for the merge math, and (b) our OWN
pass 1 already defines the `(partials, sums, maxs)` contract pass 2 reads —
not reverse-engineered from MLX's binary.

**Correctness** (scoped gate, this box, `MLX_METALLIB_PATH` at
`build/dist/lib/mlx.metallib`, `go test -tags metal_runtime -run
'TQ|TurboQuant|Turboquant|SDPA|Sdpa' ./engine/metal/`): all cases green.
The 2-pass parity band vs the f64 dequantised-rows oracle **tightened** to
≤0.0006 across all 12 deep cases (hd ∈ {128,256,512}, modes 4/4-4/3-3/3-2/2)
— was ≤0.0008 when pass 2 wrote a rotated-space bf16 intermediate for a
separate dispatch to re-read; removing that round-trip is a strict precision
win, the same pattern the single-pass fusion measured. Shallow-N (mostly
empty pass-1 blocks) measures ≤0.0044, still ~18× inside the 0.08 assert.
Single-pass band unchanged (≤0.0020, untouched by this change). `go vet
./engine/metal/` clean; gofmt clean on all touched files.

**16k live A/B** (`mlx-community/gemma-4-e2b-it-4bit`, greedy, needle-prompt
16,507-token prefill, fact planted ~10% in, `--kv-cache turboquant:4`,
3-run medians, same session/box, idle at measurement time — no sibling
`lem`/`go test` processes observed before or after):

| lane | decode tok/s | needle |
|---|---|---|
| native (off, same-session reference) | 144.4 | retrieved |
| turboquant:4, unfused (base 7084cf14) | 102.5 | retrieved |
| turboquant:4, fused pass 2 (this change) | 113.9 | retrieved |

**+11.1% decode** (102.5 → 113.9 tok/s) at 16.5k, needle-clean before and
after. The native gap closes from 29.0% to 21.1% (7.9 points recovered) —
same-session numbers throughout, not a cross-day comparison against the
131.6/87.8 figures above (those ran on a different day under different load;
this table's three rows are the only mutually comparable set). The
remaining ~21% gap is pass 1's own per-block fan cost plus the q pre-rotation
dispatch — both already at the O(output) floor the fusion rule allows; the
receipts owner should treat this as the lane's steady state absent a
different pass-1 strategy, not a further "3 dispatches should equal 1"
target.

Files: `kernels/lthn_tq_kv.metal` (new kernel), `sdpa_vector_tq.go`
(pipeline resolver + emitter + `TurboQuantSDPADevice` rewire),
`decode_forward_arch_icb_tq.go` (ICB pipeline resolver),
`decode_forward_arch_icb.go` (TQ 2-pass emission site, `nTQOps` accounting,
`attnRotTQ` scratch removed), `sdpa_vector_tq_test.go` (new ABI test,
re-measured band doc).
