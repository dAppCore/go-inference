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

Correctness context (committed gates, same tree): teacher-forced top-1 agreement
62/64 with max |logit Δ| 6.31 at b=4 over 64 steps; in-test retrieval over 1 107
prefilled tokens passes; real-KV distortion vs the paper oracle in
`2026-07-19-turboquant-real-kv-distortion.md` (b=2 at oracle, b=3 +14–16%,
b=4 +4–8%; K/V asymmetry ≤3.7% and sign-inconsistent).

## Decision

**The default stays `native`.** The pre-registered bar for flipping the default
to 3.5-bit was: needle retrieved AND decode tok/s at or above parity AND RSS
visibly down. Quality passes; speed is −32/−34%; RSS is flat.

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
