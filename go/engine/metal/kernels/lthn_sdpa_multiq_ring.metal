// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// lthn_sdpa_multiq_ring_bf16_<D> — the staged sliding tail's multi-query SDPA: K causal query
// rows over a FULL ring window in ONE dispatch, reading TWO segments per query:
//
//   * the PRE-BATCH ring (slide_w slots, positions nBase-slide_w .. nBase-1), excluding — per
//     query s — the run of s+1 slots starting at slot_base: exactly the rows query s's window
//     has evicted (the slots this batch will overwrite when it lands).
//   * the STAGED batch rows [0..s] (roped/normed in the staging slab; the ring is landed by a
//     bulk copy AFTER every layer's attention has read the pre-batch state — the ordering that
//     keeps eviction semantics without per-row interleaving, and gives shared-KV layers the
//     owner's true pre-batch window).
//
// The caller guarantees the ring is FULL (nBase >= slide_w, every row evicts) so the exclusion
// arithmetic is uniform. Softmax is order-invariant but fp accumulation is not: this lane is
// token-identity with the sequential oracle (like the steel GEMM prefill), engaged only at
// large row counts — small batches keep the byte-identical per-row interleave.
template <typename T, int D>
[[kernel]] void lthn_sdpa_multiq_ring(
    const device T* queries [[buffer(0)]],   // QUERY-major rows [K × nHeads·D]
    const device T* ring_k [[buffer(1)]],    // the pre-batch ring, slide_w rows
    const device T* ring_v [[buffer(2)]],
    device T* out [[buffer(3)]],             // QUERY-major rows [K × nHeads·D]
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& slide_w [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device T* stage_k [[buffer(11)]],  // this batch's K rows, staged (roped), K rows
    const device T* stage_v [[buffer(12)]],
    const constant int& slot_base [[buffer(13)]], // nBase % slide_w — the first slot this batch lands in
    const constant int& ring_live [[buffer(14)]], // min(nBase, slide_w) — valid pre-batch ring rows (0 = fresh ring)
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

  const int head_idx = int(tid.x);
  const int q_seq_idx = int(tid.y);
  const int kv_head_idx = head_idx / gqa_factor;
  const int qo_offset = q_seq_idx * int(tpg.x) + head_idx;
  queries += qo_offset * D + simd_lid * qk_per_thread;
  out += qo_offset * D + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -MAXFLOAT;
  U sum_exp_score = 0;

  // --- segment 1: the pre-batch ring, skipping query s's evicted run [slot_base, slot_base+s] ---
  {
    const device T* keys = ring_k + kv_head_idx * k_head_stride +
        simd_gid * k_seq_stride + simd_lid * qk_per_thread;
    const device T* values = ring_v + kv_head_idx * v_head_stride +
        simd_gid * v_seq_stride + simd_lid * v_per_thread;
    const int excl_len = q_seq_idx + 1;
    for (int i = simd_gid; i < ring_live; i += BN) {
      int d = i - slot_base;
      if (d < 0) {
        d += slide_w;
      }
      bool use_key = d >= excl_len;
      if (use_key) {
        for (int j = 0; j < qk_per_thread; j++) {
          k[j] = keys[j];
        }
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * k[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * values[j];
        }
      }
      keys += inner_k_stride;
      values += inner_v_stride;
    }
  }

  // --- segment 2: the staged batch rows [0..s] (causal) ---
  {
    const device T* keys = stage_k + kv_head_idx * k_head_stride +
        simd_gid * k_seq_stride + simd_lid * qk_per_thread;
    const device T* values = stage_v + kv_head_idx * v_head_stride +
        simd_gid * v_seq_stride + simd_lid * v_per_thread;
    const int rows = int(tpg.y);
    for (int i = simd_gid; i < rows; i += BN) {
      bool use_key = i <= q_seq_idx && i + slide_w > q_seq_idx; // causal cap + the sliding window lower bound (binds only when K > slide_w)
      if (use_key) {
        for (int j = 0; j < qk_per_thread; j++) {
          k[j] = keys[j];
        }
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * k[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * values[j];
        }
      }
      keys += inner_k_stride;
      values += inner_v_stride;
    }
  }

  // --- combine (verbatim sdpa_vector reduction tail) ---
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

#define LTHN_SDPA_MULTIQ_RING_INST(D)                                       \
  template [[host_name("lthn_sdpa_multiq_ring_bf16_" #D)]] [[kernel]] void \
  lthn_sdpa_multiq_ring<bfloat, D>(                                         \
      const device bfloat*, const device bfloat*, const device bfloat*,    \
      device bfloat*, const constant int&, const constant int&,             \
      const constant size_t&, const constant size_t&,                       \
      const constant size_t&, const constant size_t&,                       \
      const constant float&, const device bfloat*, const device bfloat*,   \
      const constant int&, const constant int&, uint3, uint3, uint, uint);

LTHN_SDPA_MULTIQ_RING_INST(64)
LTHN_SDPA_MULTIQ_RING_INST(128)
LTHN_SDPA_MULTIQ_RING_INST(256)
LTHN_SDPA_MULTIQ_RING_INST(512)
