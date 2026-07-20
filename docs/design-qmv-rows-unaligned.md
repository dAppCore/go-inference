# Design: qmv_rows for unaligned dims — the qmv_impl M-variant (#53)

## Problem

The byte-exact multi-row qmv tier (`lthn_qmv_rows.metal` + `qmv_rows.go`) serves
rows ≥ 2 only on the qmv_fast envelope — `outDim%8==0 && inDim%512==0`, the rule
`qmvBF16KernelName` routes the per-row decode by. The real gemma4 26B-A4B MoE
geometry fails that envelope on **all four** block projections (hidden 2816,
expert FF 704, dense FF 2112 — none of the inDims is 512-aligned):

| projection      | outDim | inDim | fast? |
|-----------------|--------|-------|-------|
| expert gate/up  | 704    | 2816  | no (2816%512=256) |
| expert down     | 2816   | 704   | no (704%512=192)  |
| local gate/up   | 2112   | 2816  | no                |
| local down      | 2816   | 2112  | no (2112%512=64)  |

So `mtpRowsMoEEligible` declines up-front on the real checkpoint, the layer-major
MTP verify driver never engages, and greedy pair verify pays K sequential rows
(pair −41% vs plain). The per-row oracle at these dims is MLX's **`qmv_impl`** —
the non-fast twin `qmvBF16KernelName` routes to. This note designs its M-variant,
by the house method: VERBATIM structure, per-row byte identity, gated the same way
`lthn_qmv_rows`' fast M-variant is gated by `TestLthnQMVRowsParity`.

## What qmv_impl does differently from qmv_fast_impl

Vendored source: `external/mlx/mlx/backend/metal/kernels/quantized.h` (`qmv_impl`
at ~line 818, `qmv_fast_impl` at ~751). Same grid for both — the host dispatch
(`external/mlx/mlx/backend/metal/quantized.cpp`, `qmv()`) uses
`grid=(M, ceil(N/8), B)`, `group=(32, 2, 1)` for fast AND non-fast — so
`emitQMVRowsTiled`'s `(1, (outDim+7)/8, 1) × (32,2,1)` dispatch carries over
unchanged. The differences are all inside the kernel:

1. **`packs_per_thread = 1` always** (fast: 2 for bits 4/8). So
   `values_per_thread = pack_factor` (8 at 4-bit, 4 at 8-bit) and
   `block_size = 32·values_per_thread` (256 / 128) — half the fast twin's k-step.
   This is an accumulation-ORDER difference, not just a tiling one: the refuted
   2026-07-13 predecessor proved packs=1 vs packs=2 drift ~1 ulp value-dependently.
   The general M-variant therefore pairs ONLY with the qmv_impl per-row route
   (non-fast dims), exactly as the fast M-variant pairs only with fast dims.

2. **Tail-safe k-walk.** The plain loop runs `k < in_vec_size - block_size`; the
   FINAL block — full or partial — always goes through
   `load_vector_safe` + `qdot_safe` with
   `remaining = clamp(in_vec_size - k - simd_lid·vpt, 0, vpt)`. Lanes wholly past
   the tail get `remaining == 0` and skip. (Note the final block is safe-walked
   even when `inDim % block_size == 0` — byte identity requires reproducing that,
   not "optimising" it away.)

3. **Ragged out-tiles.** After the common `out_row >= out_vec_size → return`:
   - `out_vec_size < 8` (less than one grid tile): every per-row loop is bounded
     by `out_row + row < out_vec_size` — fully guarded reads and writes.
   - else: the last tile is **moved back** — `used_out_row =
     min(out_vec_size − 4, out_row)` — so a ragged outDim re-computes a few
     overlap rows instead of reading out of bounds. Overlapping threadgroups
     write identical bytes (per-row maths depends only on the row index and
     inputs), the vendored kernel's own idiom.

## What the M-variant must preserve per row

For a FIXED row m, the operation sequence touching row m's values must be
EXACTLY `qmv_impl`'s for that row: per k-block `load_vector(x_m)` (the `sums[m]`
add chain), then `qdot` per out-row in row order with `result[m][row] +=`
chaining; then the ONE safe tail block with the SAME `remaining` (identical for
every m — it depends only on k and `simd_lid`); then `simd_sum(result[m][row])`
and the write to `y[m·out_vec_size + row]`. The M loop only interleaves ACROSS
rows and rows never mix, so each row's FP chain is bit-deterministic — the same
argument the fast twin's flat tile is proven on (`TestLthnQMVRowsParity`).

Per-row safety doubles as cross-row correctness: the `remaining` clamp that
guards device bounds in the single-row kernel is exactly what stops row m's tail
read straying into row m+1's slice in the M-slab layout.

Both ragged-out branches port verbatim: at those outDims the branch IS what the
single-row oracle executes, so byte identity requires carrying it.

## Register / occupancy shape, M ∈ 2..4

`values_per_thread` is HALF the fast twin's (packs=1), so the flat register tile
is strictly lighter than the proven-live fast flat tile at every M:

| tile               | x_thread        | sums | result | total floats (4-bit, M=4) |
|--------------------|-----------------|------|--------|---------------------------|
| fast flat (live)   | M×16            | M    | M×4    | 84                        |
| general flat (new) | M×8             | M    | M×4    | 52                        |

No new occupancy risk at M ≤ 4. The safe tail adds one clamp + a zero-fill per
row per k-walk — thin.

## Wide (M 5..8): skipped, deliberately

- The fast wide twin (halved tile) held byte identity
  (`TestLthnQMVRowsWideByteBand`) but **lost its live A/B** on the 26B MTP pair
  (2026-07-19: wide on 120.2 vs off 122.7 tok/s) — it is a banked opt-in
  instrument (`LTHN_QMV_ROWS_WIDE=1`), not a default.
- On unaligned dims the only rows>4 consumer is the BYTE TIER (the grouped
  expert MoE lane and the chunked fold), which composes 2..4-row chunks via
  `qmvRowsChunks` — a general 5..8 single-dispatch tile would serve nothing the
  chunks don't already serve byte-exactly, and the fast twin's live refutation
  argues the throughput case is against it.
- The general tile's halved vpt means a wide variant COULD hold more rows in the
  same registers — noted for the record; no consumer, no port. Re-probe per
  geometry if a future model makes rows 5..8 a live unaligned dispatch.

## Host routing (one rule, consulted everywhere)

- `lthnQMVRowsKey` gains `general bool`; the PSO resolvers (plain + ICB) derive
  the host name from it: `lthn_qmv_rows[_general]_bfloat16_t_gs_%d_b_%d`. The
  ICB verify-tail recorder inherits through `plan.tiledKey` untouched.
- **`qmvRowsTiledKeyFor(m, outDim, inDim, gs, bits)`** becomes the single tiled
  eligibility rule: fast envelope → fast key at m ≤ `qmvRowsTiledCap()` (wide
  cases included when armed); anything else → general key at m ≤ 4 (flat cap
  only — no general wide). The plan gate, the chunked fold, the byte-exact
  encoder and the servability probe all consult it, so record==live and
  probe==encode stay one rule.
- `encQMVRowsBF16ChunkedAt` drops its fast-envelope gate: chunks are always
  2..4-row, so unaligned groups compose from general-tile dispatches.
- `encQMVByteExactAt` / `qmvByteExactServable`: the rows ≤ tiledCap branch now
  FALLS THROUGH to the chunked composition when no tiled plan serves (fixes the
  wide-armed hole where unaligned rows 5..8 sat under the raised cap and
  declined instead of chunking).
- Eligibility inherits automatically: `mtpRowsMoEEligible` probes
  `qmvByteExactServable` per rows 1..K over all four projections — with the
  general tier the real 26B geometry passes.

## Receipts (scope of this lane)

- Byte identity per row vs the single-row `qmv_impl` route (`QMVBF16`) across a
  non-fast dim sweep — the real 26B shapes (704/2816, 2816/704, 2112/2816,
  2816/2112) plus the 1408-class (2560/1408) and ragged-out branch shapes
  (outDim 706 → moved-back overlap, outDim 6 → small-out branch) — at every
  gs/bits pair the metallib instantiates ({32,64,128} × {4,8}), skipping pairs
  where inDim%gs≠0.
- `mtpRowsMoEBatched` byte-identity re-run green with a NEW fixture at the real
  26B projection dims (numExperts shrunk — expert COUNT is not qmv geometry).
- NO live serve/bench here: the 26B pair number is the orchestrator's. This lane
  ends at byte-proven + eligibility-passing.

## Residue

None expected by construction (every (gs,bits) the metallib instantiates uses
plain-indexed `qdot`/`qdot_safe` bodies at packs=1; the 3/5/6-bit pointer-walk
qdot idiom is not instantiated). If a combination fails its sweep, it is
excluded from `qmvRowsTiledKeyFor` and recorded here with the failing receipt.
