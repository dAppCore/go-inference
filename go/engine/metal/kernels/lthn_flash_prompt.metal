// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// lthn_flash_prompt_bf16_<D> — the prompt-scale streaming-softmax SDPA (#375).
//
// The GEMM composition it replaces materialises S[rows × nTotal] into device
// slabs between its three dispatches; the round-trip's cache eviction taxes
// every neighbouring GEMM ~6× its own bandwidth (the #367 fold-context
// conviction, TestDiagMLPChainWithSDPASlabTraffic). This kernel never forms
// S: per (head, query-tile) threadgroup, K/V stream through threadgroup
// memory in BK-row tiles shared by all BQ resident queries, and each query
// carries the online-softmax state (running max, running sum, f32 output
// accumulator) across the stream — flash-attention forward, shaped from the
// proven lthn_sdpa_multiq idioms rather than simdgroup-MMA novelty.
//
//   * grid (nHeads, ceil(K/BQ)): one threadgroup per (query head, BQ-row
//     query tile). NSG simdgroups per group; each simdgroup owns
//     BQ/NSG queries and every key of the tile against each of them.
//   * queries/out are QUERY-major ([s][h][D], the engine slab layout);
//     K/V are the layer caches ([n][kvDim], head at kvh·hd) — the same
//     bindings the multiQ kernel takes, plus kRows (the chunk width) for
//     the causal cap, which the multiQ derived from the grid.
//   * per-query causal cap: query s (tile-global) uses key i iff
//     i <= nTotal - kRows + s — identical to the multiQ/composition rule.
//     A tile past the LAST resident query's cap ends the stream.
//   * numerics: f32 dot/softmax/accumulate end to end — one rounding tier
//     BETTER than the composition (which stores S as bf16 between GEMMs);
//     accumulation ORDER differs from both the multiQ stride and the GEMM
//     composition, so the lane stays token-identity like the fold's qmm.
//
// TG memory at D=256: K tile 16×256 bf16 (8KB) + V tile (8KB) = 16KB.
// At D=512 the tile halves to 8 rows (8+8KB) — same budget.
template <typename T, int D, int BK, int NSG>
[[kernel]] void lthn_flash_prompt(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& nTotal [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const constant int& kRows [[buffer(11)]],
    const constant int& nHeads [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int elems = D / 32;      // per-lane slice of one row
  constexpr int QPS = 4;             // queries per simdgroup
  constexpr int BQ = NSG * QPS;      // queries per threadgroup

  typedef float U;

  threadgroup T kTile[BK * D];
  threadgroup T vTile[BK * D];

  const int head_idx = int(tid.x);
  const int kv_head_idx = head_idx / gqa_factor;
  const int qTileBase = int(tid.y) * BQ;

  // this simdgroup's queries: qTileBase + simd_gid*QPS + [0..QPS)
  const int q0 = qTileBase + int(simd_gid) * QPS;

  thread U q[QPS][elems];
  thread U o[QPS][elems];
  thread U maxScore[QPS];
  thread U sumExp[QPS];
  int cap[QPS]; // per-query causal cap (last usable key index)

  for (int s = 0; s < QPS; s++) {
    const int qi = q0 + s;
    const bool live = qi < kRows;
    cap[s] = live ? (nTotal - kRows + qi) : -1;
    maxScore[s] = -MAXFLOAT;
    sumExp[s] = 0;
    const device T* qp =
        queries + (size_t(qi < kRows ? qi : 0) * size_t(nHeads) + size_t(head_idx)) * D +
        simd_lid * elems;
    for (int j = 0; j < elems; j++) {
      q[s][j] = live ? static_cast<U>(scale) * qp[j] : U(0);
      o[s][j] = 0;
    }
  }

  // the whole tile's stream ends at the LAST resident query's cap
  int tileCap = qTileBase + BQ - 1;
  if (tileCap > kRows - 1) {
    tileCap = kRows - 1;
  }
  const int nStream = nTotal - kRows + tileCap + 1; // keys [0, nStream)

  const int threadsPerTG = NSG * 32;
  const int lin = int(simd_gid) * 32 + int(simd_lid);

  for (int base = 0; base < nStream; base += BK) {
    const int tileRows = min(BK, nStream - base);
    // cooperative tile load: BK rows × D cols, all threads striding the tile
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = lin; idx < tileRows * D; idx += threadsPerTG) {
      const int r = idx / D;
      const int c = idx % D;
      kTile[r * D + c] =
          keys[size_t(kv_head_idx) * k_head_stride + size_t(base + r) * k_seq_stride + c];
      vTile[r * D + c] =
          values[size_t(kv_head_idx) * v_head_stride + size_t(base + r) * v_seq_stride + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = 0; r < tileRows; r++) {
      const int key = base + r;
      for (int s = 0; s < QPS; s++) {
        if (key > cap[s]) {
          continue; // masked for this query (later queries in the tile may still use it)
        }
        U score = 0;
        for (int j = 0; j < elems; j++) {
          score += q[s][j] * static_cast<U>(kTile[r * D + simd_lid * elems + j]);
        }
        score = simd_sum(score);
        const U newMax = max(maxScore[s], score);
        const U factor = fast::exp(maxScore[s] - newMax);
        const U expScore = fast::exp(score - newMax);
        maxScore[s] = newMax;
        sumExp[s] = sumExp[s] * factor + expScore;
        for (int j = 0; j < elems; j++) {
          o[s][j] = o[s][j] * factor +
              expScore * static_cast<U>(vTile[r * D + simd_lid * elems + j]);
        }
      }
    }
  }

  // each simdgroup owns its queries outright — normalise and write
  for (int s = 0; s < QPS; s++) {
    const int qi = q0 + s;
    if (qi >= kRows) {
      continue;
    }
    device T* op = out + (size_t(qi) * size_t(nHeads) + size_t(head_idx)) * D + simd_lid * elems;
    const U denom = sumExp[s];
    for (int j = 0; j < elems; j++) {
      const U val = denom == 0 ? U(0) : o[s][j] / denom;
      op[j] = static_cast<T>(val);
    }
  }
}

#define LTHN_FLASH_PROMPT_INST(D, BK, NSG)                                   \
  template [[host_name("lthn_flash_prompt_bf16_" #D)]] [[kernel]] void      \
  lthn_flash_prompt<bfloat, D, BK, NSG>(                                     \
      const device bfloat*, const device bfloat*, const device bfloat*,     \
      device bfloat*, const constant int&, const constant int&,             \
      const constant size_t&, const constant size_t&,                       \
      const constant size_t&, const constant size_t&,                       \
      const constant float&, const constant int&, const constant int&,      \
      uint3, uint, uint);

LTHN_FLASH_PROMPT_INST(256, 16, 4)
LTHN_FLASH_PROMPT_INST(512, 8, 4)
