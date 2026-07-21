# MTP verify economics + KV-mode matrix — receipts

Campaign: #53 (MTP pair speed) and #48 (KV cache modes), dev @ `41cc8c12`.
M3 Ultra 96GB, greedy decode (`--temp 0`), tg512 unless noted, E2B =
`mlx-community/gemma-4-e2b-it-4bit` + bf16 assistant drafter. Every number is
a live measurement from `lem-dev generate` (decode-only, prefill excluded).

## The 230 claim — REPRODUCED at 248 on today's dev

The reference-table ~230 (+32%) is real and lives on the original method:
`lem.sh pair` / the counting BENCH_PROMPT ("integers from 1 to 800, single
spaces"), fully greedy, no-think. Today's dev on that exact method:

| lane | tok/s |
|---|---:|
| plain (serve, `lem.sh pair e2b`) | 174.4 |
| **MTP (serve, same verb)** | **248.4 (+42%)** |
| MTP (`generate` CLI, same prompt) | 243.9 |

No serve-vs-CLI lane gap (248 vs 244 ≈ noise). The number is
prompt-CLASS-elastic, not harness- or regression-shaped: integer streams
248, line-formatted counting 186, free prose 172 (≈ plain parity). Every
historical board is beaten like-for-like today — the 07-10 board's own
commit measures 165 on prose where today's dev does 172; the 07-17 matrix's
counting 219 is now 244-248. Nothing was lost in the 600 commits.

The prose floor is the real frontier: acceptance drops below the verify
cost's fixed component there, which is what the decomposition below prices.
The fold engages everywhere (zero `mtp-diag` declines; fwd 11.6ms @ K=6 ≈ 2
plain steps, not the 27ms sequential shape).

## Where the verify round actually goes (K=6, LTHN_MTP_DIAG + LTHN_GPU_TRACE)

Steady round ≈ draft 4.8ms + verify 18.8ms for ~4-5 committed tokens:

| stage | ms | note |
|---|---:|---|
| fold forward (GPU) | 9.4 | 352 encoder segments across 35 layers — launch-bound |
| fold forward (host gap) | ~2.2 | encode + submit of those segments |
| rows-head (K-row fused) | 1.6–2.3 | |
| boundary head + host argmax rescan | ~2.3 | **eliminated** (below) |
| truncate/commit/host misc | ~1.5 | |

GPU stage split of the 9.4ms: MLP gate/up/down 4.2ms (vs ~0.9ms weight-read
floor), resid+epilogue 1.9ms (PLE chain, already #372-fused 5→3 dispatches),
attention ~3.0ms. The fold pays a ~8ms FIXED cost (fwd 9.4ms at K=2!) — at
real acceptance the fixed cost caps the win regardless of prompt.

## Shipped: content-keyed boundary-greedy cache (`41cc8c12`)

The K-row verify head already computes the boundary's argmax
(`rows[accepted-1]`); the next round re-derived it with a full-vocab
`headLogitsScratch` + 262k-entry host bf16 scan. Now cached keyed by VALUE
(exact hidden bytes + suppress set, `bytes.Equal` at use — no retained-state
write path can leave it stale; `BoundaryLogits` lazily recomputes if a
non-greedy consumer wants the bytes).

Receipt: verify 21.6→18.2ms, 20.2→17.9ms per round; pair 169.7→172.3–173.4
(3-run band); output byte-identical to pre-patch on the same command.

## Refuted on E2B: wide-M qmv rows (`LTHN_QMV_ROWS_WIDE=1`)

165.0/165.1 vs 172.3–173.4 baseline = −4%. Already live-refuted on the 26B
pair (#53, `0027e76a`); now refuted at the small end too. Stays banked
opt-in.

## The priced next step (unchanged, now re-evidenced)

`decode_verify_icb.go`'s receipt: per-layer tail ICB replay is break-even —
the win is recording the WHOLE layer stack (one `executeCommands` per verify
pass, ~3µs/op chained-decode economics, pos/N rebinds), blocked on the
staged-sliding landing's per-pass slot offsets. Today's 352-segment trace is
the same conclusion from the other side: the fold's remaining cost is launch
count, not arithmetic.

## #48 — KV-mode matrix (q8 IS the baseline)

`--kv-cache native` means q8-by-default on global-attention owners
(`LTHN_KV_Q8*` opt out). The old "TQ vs native zero RSS delta" confusion was
this: the native baseline was never bf16.

E2B @ ctx 32K (35 layers: 12 sliding-native + 3 global + 20 shared):

| mode | decode tok/s | plan (global layers) | peak RSS |
|---|---:|---|---:|
| q8 (default) | 174.7 | 102MiB | 5862MiB |
| bf16-forced | 84.2 | 198MiB | 5851MiB |
| turboquant | 134.8 | 42MiB | 5854MiB |

bf16-forced falls off the optimised q8-ICB lane entirely (−52% at ZERO cache
depth — a lane artefact, not bandwidth; not a usable quality A/B path).
TQ: −23% decode for −60MiB plan. At 18.5K-deep decode: q8 122.6 vs TQ 110.8
(−10%), RSS −36MiB.

31B @ ctx 32K, 18.5K-deep decode (60 layers: 50 sliding + 10 global):

| mode | decode tok/s | plan | peak RSS |
|---|---:|---|---:|
| q8 | 19.4 | q8 10×(1360MiB) + sliding 800MiB | 19,277MiB |
| turboquant | 15.4 | tq 10×(570MiB) + sliding 800MiB | **34,922MiB** |

**Defect: TQ's deep-prefill path allocates ~15.6GB above the q8 lane** —
~20× its own 790MiB cache saving — while decoding 21% slower. Until that
allocation is found and fixed, TQ loses on BOTH axes at exactly the depth it
was built for. q8 stays the default; #48 parks with these receipts.

## KV-mode ladder (the decided defaults)

The bf16-base TQ cells (E2B, same method): shallow 134.7 / deep 112.1 —
byte-identical numbers and cache plan to the q8-base TQ cells, because
`--kv-cache turboquant` already replaces q8 on the global layers and the
sliding owners are native bf16 in every mode. Against a bf16 baseline TQ's
compression is the more meaningful shift (~204→42MiB on the globals, ≈5×, vs
≈2.4× against q8) — but its decode cost is its own read path (the ~105 GB/s
effective vs bf16's 246, issue-bound), independent of what it replaced.

The ladder:
- **default = q8** — fastest decode AND half the bf16 footprint; ~1%-class
  fidelity cost. (Carries the open re-engagement bistability — the two-plane
  row/scale store race — which is a defect to fix IN q8, not a reason to
  change the default.)
- **`--kv-cache turboquant` = the no-q8 configuration of record** — 134.7 vs
  the bf16-force env's 84.2: TQ keeps the optimised ICB lane where forcing
  bf16 falls off it. Also the capacity mode once its deep-prefill allocation
  defect is fixed.
- **`LTHN_KV_Q8*=0` = dev instrument only** — the unoptimised bf16 lane
  exists for A/Bs and forensics, is not a product surface, and no CLI flag
  offers it.

## Reproduce

```sh
ML=$REPO/build/dist/lib/mlx.metallib
MLX_METALLIB_PATH=$ML lem-dev generate --temp 0 --max-tokens 512 \
  --draft <assistant-snapshot> <e2b-4bit-snapshot>        # pair
LTHN_MTP_DIAG=1 …                                         # round timings
LTHN_GPU_TRACE=1 …                                        # fold stage split
MLX_METALLIB_PATH=$ML /usr/bin/time -l lem-dev generate --temp 0 \
  --max-tokens 512 --context 32768 --kv-cache turboquant \
  --prompt-file deep.txt <snapshot>                       # KV matrix leg
```
