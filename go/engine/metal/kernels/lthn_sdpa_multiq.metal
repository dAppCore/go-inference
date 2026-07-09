// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// lthn_sdpa_multiq_bf16_<D> — MLX's sdpa_vector loop (lib/mlx .../kernels/sdpa_vector.h) with the
// query batch on grid Y, specialised for the engine's batched dense pass:
//
//   * K causal query rows at positions nBase..nBase+K-1 run in ONE dispatch — grid (nHeads, K).
//     N binds the TOTAL live length (nBase+K); each query s uses key i iff i <= N-K+s — the
//     per-row length cap (upstream's do_causal), so causality needs no mask storage. Valid only
//     when the K rows' K/V occupy rows [nBase..nBase+K) of the SAME cache buffer (global layers,
//     or a sliding ring with no eviction inside the batch — the fold's `direct` case).
//   * queries AND out are QUERY-major ([s][h][D] — the engine's slab layout, feeding the batched
//     O-projection gemv). Upstream writes out head-major; that is the ONE divergence.
//   * the mask/sink/query_transposed function-constant branches are stripped — the engine's
//     decode builds the upstream pipeline with all six constants false and never binds them.
//
// Per-(head,query) accumulation order is IDENTICAL to sdpa_vector single-query: skipped keys
// touch no accumulator, used keys stride i = simd_gid, +BN, … in the same sequence, and the
// simdgroup/threadgroup reduction tail is copied verbatim — so each row's output is byte-identical
// to K single-query dispatches (the fold's parity bar, pinned by the batched parity tests).
template <typename T, int D>
[[kernel]] void lthn_sdpa_multiq(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions — QUERY-major rows: query s, head h at (s*nHeads + h)*D.
  const int head_idx = int(tid.x);
  const int q_seq_idx = int(tid.y);
  const int kv_head_idx = head_idx / gqa_factor;
  const int qo_offset = q_seq_idx * int(tpg.x) + head_idx;
  queries += qo_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  out += qo_offset * D + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -MAXFLOAT;
  U sum_exp_score = 0;

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    // the per-query causal cap: query s attends keys [0 .. N-K+s]
    bool use_key = i <= (N - int(tpg.y) + q_seq_idx);
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

#define LTHN_SDPA_MULTIQ_INST(D)                                            \
  template [[host_name("lthn_sdpa_multiq_bf16_" #D)]] [[kernel]] void      \
  lthn_sdpa_multiq<bfloat, D>(                                              \
      const device bfloat*, const device bfloat*, const device bfloat*,    \
      device bfloat*, const constant int&, const constant int&,             \
      const constant size_t&, const constant size_t&,                       \
      const constant size_t&, const constant size_t&,                       \
      const constant float&, uint3, uint3, uint, uint);

LTHN_SDPA_MULTIQ_INST(64)
LTHN_SDPA_MULTIQ_INST(128)
LTHN_SDPA_MULTIQ_INST(256)
LTHN_SDPA_MULTIQ_INST(512)
