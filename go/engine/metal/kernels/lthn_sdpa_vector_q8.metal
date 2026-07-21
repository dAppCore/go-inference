// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

// lthn_sdpa_vector_q8 — the LINEAR-cache q8 decode SDPA pair (#367): MLX's
// sdpa_vector / sdpa_vector_2pass_1 bodies with the K/V reads swapped from
// bf16 to int8 + f32 group scales (kvQ8GroupSize = 64 elements per scale,
// symmetric — the same format lthn_kv_q8_store_bf16 writes and the paged q8
// kernels read). This is the dense family's deep-context lever: the GLOBAL
// layers' unbounded KV scan is the measured depth slope (e2b: 14 MiB per 1K
// context per token ≈ the -0.61 tok/s per 1K receipt), and int8 halves the
// bytes.
//
// Structure is MLX-verbatim where it matters: same loop order, same
// accumulator stations (fp32 online softmax, fast::exp), same simd_sum
// points — the ONLY change is each lane's K/V element loads dequantise
// int8·scale in fp32. A lane's qk_per_thread (=D/32) elements sit inside one
// 64-element scale group for every instantiated D (128/256/512 — per-lane
// group = simd_lid·per/64, and D/32 ≤ 64 holds throughout), so each row
// costs a lane exactly ONE scale load — the lthn_sdpa_paged q8 pattern. The decode form is fixed: no
// masks, no sinks, not causal, queries not transposed (the recorded arch ICB
// binds none of those), batch 1.
//
// ABI mirrors the MLX kernels with the scale planes appended, so the emit
// wrappers stay shape-identical:
//   single:  q=0 k=1 v=2 out=3 gqa=4 N=5 khs=6 kss=7 vhs=8 vss=9 scale=10
//            kscales=11 vscales=12
//   2pass_1: q=0 k=1 v=2 partials=3 sums=4 maxs=5 N=7 khs=8 kss=9 vhs=10
//            vss=11 scale=12 kscales=13 vscales=14  (blocks = fc 26, as MLX)
// Strides are in ELEMENTS (int8 elements = the bf16 element counts); the
// scale-plane strides are stride/64, computed in-kernel (wiring gates
// stride%64==0 — headDim 512/256 and kvDim multiples hold it).
//
// Pass 2 stays MLX's sdpa_vector_2pass_2 unchanged — the merge reads f32
// partials and never touches K/V.

constant int lthn_q8_blocks [[function_constant(26)]];

template <int D>
[[kernel]] void lthn_sdpa_vector_q8(
    const device bf16* queries [[buffer(0)]],
    const device char* keys [[buffer(1)]],
    const device char* values [[buffer(2)]],
    device bf16* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device float* k_scales [[buffer(11)]],
    const device float* v_scales [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  const int inner_k_stride = BN * int(k_seq_stride);
  const int inner_v_stride = BN * int(v_seq_stride);
  const size_t ks_head_stride = k_head_stride / 64;
  const size_t ks_seq_stride = k_seq_stride / 64;
  const size_t vs_head_stride = v_head_stride / 64;
  const size_t vs_seq_stride = v_seq_stride / 64;
  const int inner_ks_stride = BN * int(ks_seq_stride);
  const int inner_vs_stride = BN * int(vs_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // decode form: one query row per head, batch 1
  const int head_idx = tid.x;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  k_scales += kv_head_idx * ks_head_stride + simd_gid * ks_seq_stride +
      (simd_lid * qk_per_thread) / 64;
  v_scales += kv_head_idx * vs_head_stride + simd_gid * vs_seq_stride +
      (simd_lid * v_per_thread) / 64;
  out += head_idx * D + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * static_cast<U>(queries[i]);
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -3.0e38f;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < N; i += BN) {
    // score: per-lane int8 dot, the lane's ONE group scale applied before
    // the cross-lane simd_sum (lanes carry different groups).
    U partial = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      partial += q[j] * U(keys[j]);
    }
    U score = simd_sum(partial * k_scales[0]);

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    const U vsf = exp_score * v_scales[0];
    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + vsf * U(values[j]);
    }

    keys += inner_k_stride;
    values += inner_v_stride;
    k_scales += inner_ks_stride;
    v_scales += inner_vs_stride;
  }

  // cross-simdgroup combine — MLX sdpa_vector verbatim
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
      out[i] = static_cast<bf16>(o[i]);
    }
  }
}

template <int D>
[[kernel]] void lthn_sdpa_vector_2pass_1_q8(
    const device bf16* queries [[buffer(0)]],
    const device char* keys [[buffer(1)]],
    const device char* values [[buffer(2)]],
    device bf16* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device float* k_scales [[buffer(13)]],
    const device float* v_scales [[buffer(14)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  const size_t ks_head_stride = k_head_stride / 64;
  const size_t ks_seq_stride = k_seq_stride / 64;
  const size_t vs_head_stride = v_head_stride / 64;
  const size_t vs_seq_stride = v_seq_stride / 64;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // decode form: batch 1, q_seq_len 1 — grid (nKVHeads, 1, blocks) of
  // (32, gqa, 1), exactly the recorded arch ICB's 2-pass dispatch.
  const int kv_head_idx = tid.x;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int o_offset = q_head_idx;

  queries += o_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  k_scales += kv_head_idx * ks_head_stride + block_idx * ks_seq_stride +
      (simd_lid * qk_per_thread) / 64;
  v_scales += kv_head_idx * vs_head_stride + block_idx * vs_seq_stride +
      (simd_lid * v_per_thread) / 64;
  out += o_offset * lthn_q8_blocks * D + block_idx * D +
      simd_lid * v_per_thread;
  sums += o_offset * lthn_q8_blocks + block_idx;
  maxs += o_offset * lthn_q8_blocks + block_idx;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * static_cast<U>(queries[i]);
  }

  U max_score = -3.0e38f;
  U sum_exp_score = 0;

  for (int i = block_idx; i < N; i += lthn_q8_blocks) {
    U partial = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      partial += q[j] * U(keys[j]);
    }
    U score = simd_sum(partial * k_scales[0]);

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    const U vsf = exp_score * v_scales[0];
    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + vsf * U(values[j]);
    }

    keys += lthn_q8_blocks * int(k_seq_stride);
    values += lthn_q8_blocks * int(v_seq_stride);
    k_scales += lthn_q8_blocks * int(ks_seq_stride);
    v_scales += lthn_q8_blocks * int(vs_seq_stride);
  }

  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<bf16>(o[i]);
  }
}

#define instantiate_lthn_sdpa_vector_q8(dim)                              \
  template [[host_name("lthn_sdpa_vector_q8_bf16_" #dim)]] [[kernel]]     \
  void lthn_sdpa_vector_q8<dim>(                                          \
      const device bf16*, const device char*, const device char*,        \
      device bf16*, const constant int&, const constant int&,            \
      const constant size_t&, const constant size_t&,                    \
      const constant size_t&, const constant size_t&,                    \
      const constant float&, const device float*, const device float*,   \
      uint3, uint, uint);                                                 \
  template                                                                \
      [[host_name("lthn_sdpa_vector_2pass_1_q8_bf16_" #dim)]] [[kernel]]  \
  void lthn_sdpa_vector_2pass_1_q8<dim>(                                  \
      const device bf16*, const device char*, const device char*,        \
      device bf16*, device float*, device float*, const constant int&,   \
      const constant size_t&, const constant size_t&,                    \
      const constant size_t&, const constant size_t&,                    \
      const constant float&, const device float*, const device float*,   \
      uint3, uint3, uint3, uint3, uint);

instantiate_lthn_sdpa_vector_q8(128)
instantiate_lthn_sdpa_vector_q8(256)
instantiate_lthn_sdpa_vector_q8(512)

// lthn_sdpa_multiq_q8 — the batched dense pass's multi-query causal SDPA
// (lthn_sdpa_multiq: grid.y carries K causal query rows, per-row length cap
// in-kernel) with the q8 K/V reads — the prompt-prefill / MTP-verify read for
// q8 global caches. Same loop, same stations, same combine as the bf16 twin;
// only the K/V element loads dequantise int8·scale (one lane-group scale per
// row, exactly the single-query q8 kernel above).
template <int D>
[[kernel]] void lthn_sdpa_multiq_q8(
    const device bf16* queries [[buffer(0)]],
    const device char* keys [[buffer(1)]],
    const device char* values [[buffer(2)]],
    device bf16* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device float* k_scales [[buffer(11)]],
    const device float* v_scales [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  const int inner_k_stride = BN * int(k_seq_stride);
  const int inner_v_stride = BN * int(v_seq_stride);
  const size_t ks_head_stride = k_head_stride / 64;
  const size_t ks_seq_stride = k_seq_stride / 64;
  const size_t vs_head_stride = v_head_stride / 64;
  const size_t vs_seq_stride = v_seq_stride / 64;
  const int inner_ks_stride = BN * int(ks_seq_stride);
  const int inner_vs_stride = BN * int(vs_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // QUERY-major rows: query s, head h at (s*nHeads + h)*D.
  const int head_idx = int(tid.x);
  const int q_seq_idx = int(tid.y);
  const int kv_head_idx = head_idx / gqa_factor;
  const int qo_offset = q_seq_idx * int(tpg.x) + head_idx;
  queries += qo_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  k_scales += kv_head_idx * ks_head_stride + simd_gid * ks_seq_stride +
      (simd_lid * qk_per_thread) / 64;
  v_scales += kv_head_idx * vs_head_stride + simd_gid * vs_seq_stride +
      (simd_lid * v_per_thread) / 64;
  out += qo_offset * D + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * static_cast<U>(queries[i]);
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -MAXFLOAT;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < N; i += BN) {
    // the per-query causal cap: query s attends keys [0 .. N-K+s]
    bool use_key = i <= (N - int(tpg.y) + q_seq_idx);
    if (use_key) {
      U partial = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        partial += q[j] * U(keys[j]);
      }
      U score = simd_sum(partial * k_scales[0]);

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      const U vsf = exp_score * v_scales[0];
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + vsf * U(values[j]);
      }
    }

    keys += inner_k_stride;
    values += inner_v_stride;
    k_scales += inner_ks_stride;
    v_scales += inner_vs_stride;
  }

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
      out[i] = static_cast<bf16>(o[i]);
    }
  }
}

// lthn_kv_q8_store_rows_bf16 — the rows-widened landing hop: `rows` contiguous
// bf16 staging rows (kvDim elements each) quantise into contiguous int8 cache
// rows + f32 scale rows in ONE dispatch (grid.y carries the row) — the batched
// prefill's landing (the single-row kernel would cost K launches per layer).
// Per-group math identical to lthn_kv_q8_store_bf16.
struct KVQ8StoreRowsDims {
  uint kvDim;
};

[[kernel]] void lthn_kv_q8_store_rows_bf16(
    const device bf16* rows   [[buffer(0)]],
    device char*       outp   [[buffer(1)]],
    device float*      scales [[buffer(2)]],
    const constant KVQ8StoreRowsDims& D [[buffer(3)]],
    uint2 g    [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
  const uint base = g.x * 64;
  if (base >= D.kvDim) return;
  const uint rowElem = g.y * D.kvDim;
  const uint rowGroup = g.y * (D.kvDim / 64);
  const float a = float(rows[rowElem + base + lane * 2]);
  const float b = float(rows[rowElem + base + lane * 2 + 1]);
  const float m = simd_max(max(abs(a), abs(b)));
  const float scale = m / 127.0f;
  const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
  outp[rowElem + base + lane * 2] = char(clamp(rint(a * inv), -127.0f, 127.0f));
  outp[rowElem + base + lane * 2 + 1] = char(clamp(rint(b * inv), -127.0f, 127.0f));
  if (lane == 0) {
    scales[rowGroup + g.x] = scale;
  }
}

// lthn_kv_q8_dequant_rows_bf16 — the mirror hop's inverse: `rows` contiguous
// int8 cache rows + f32 scale rows expand back into bf16 rows in one dispatch.
// This is the GPU twin of the host dequantise loop behind the -state snapshot
// mirrors and the MTP drafter's target-KV export (the host loop costs ~100ms
// per draft block at 16K). Bit-identical to the host path: float(code)·scale
// in fp32, bf16 store rounds to nearest even — same two operations both sides.
[[kernel]] void lthn_kv_q8_dequant_rows_bf16(
    const device char*  codes  [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    device bf16*        outp   [[buffer(2)]],
    const constant KVQ8StoreRowsDims& D [[buffer(3)]],
    uint2 g    [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
  const uint base = g.x * 64;
  if (base >= D.kvDim) return;
  const float scale = scales[g.y * (D.kvDim / 64) + g.x];
  const uint i0 = g.y * D.kvDim + base + lane * 2;
  outp[i0] = bf16(float(codes[i0]) * scale);
  outp[i0 + 1] = bf16(float(codes[i0 + 1]) * scale);
}

#define instantiate_lthn_sdpa_multiq_q8(dim)                              \
  template [[host_name("lthn_sdpa_multiq_q8_bf16_" #dim)]] [[kernel]]     \
  void lthn_sdpa_multiq_q8<dim>(                                          \
      const device bf16*, const device char*, const device char*,        \
      device bf16*, const constant int&, const constant int&,            \
      const constant size_t&, const constant size_t&,                    \
      const constant size_t&, const constant size_t&,                    \
      const constant float&, const device float*, const device float*,   \
      uint3, uint3, uint, uint);

instantiate_lthn_sdpa_multiq_q8(128)
instantiate_lthn_sdpa_multiq_q8(256)
instantiate_lthn_sdpa_multiq_q8(512)
