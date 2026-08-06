// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

// lthn_sdpa_rtdim.metal — the RUNTIME-HEAD-DIM decode SDPA fallback (#28).
//
// The shipped MLX metallib only instantiates sdpa_vector for a fixed class of head dims
// (64/96/128/256 — a C++ `template <int D>` resolved at .metal COMPILE time, one kernel function
// per width). head_dim 32 has no pipeline anywhere in that metallib; before this file, every call
// site's absent-width behaviour differed (some errored immediately, some fell back to a DIFFERENT
// kernel family that also lacked the width and then errored) — never a working decode, and never
// the SAME failure twice. lthn_sdpa_vector_rtdim_bf16 / lthn_sdpa_vector_2pass_1_rtdim_bf16 close
// that gap the honest way: head_dim becomes a RUNTIME buffer parameter instead of a template
// argument, so this ONE compiled kernel pair serves every width the fixed metallib does not carry
// — not just 32 — without growing a new fixed instantiation per missing width. The Go driver
// (../sdpa_rtdim.go) routes here ONLY when the fixed per-headDim pipeline lookup fails, logs the
// fallback once per distinct absent width, and never returns an "absent pipeline" error for a
// missing width again.
//
// This is the FALLBACK lane, not the fast path: correctness first. The fixed 64/96/128/256
// pipelines are completely untouched (byte-identical dispatch, same PSOs, same cache) — this file
// only ever runs for a head_dim the fixed metallib does not carry.
//
// CONSTRAINT: head_dim must be a POSITIVE MULTIPLE OF 32 (BD below — the SIMD-group width every
// MLX/lthn sdpa_vector variant divides the head dimension across; Apple GPU simdgroups are
// hardware-fixed at 32 lanes, so BD=32 is not a choice this kernel makes, it is what "simdgroup"
// means on this hardware) and at most LTHN_SDPA_RTDIM_MAX_D. Both are the SAME constraint every
// fixed instantiation already satisfies (64/96/128/256 are themselves all multiples of 32) — this
// kernel does not relax it, it just resolves the width at PSO-build/dispatch time instead of at
// .metal compile time. sdpa_rtdim.go enforces the guard BEFORE dispatch (sdpaVectorRTDimValidHeadDim)
// and never calls either kernel here with a head_dim outside it, so the per-thread accumulator
// arrays below are sized to the MAX and a smaller runtime head_dim only ever shortens the active
// loop trip count — it never under- or over-runs the fixed-size array.
//
// Structure, accumulator stations, and simd_sum/simd_max reduction points are MLX's sdpa_vector /
// sdpa_vector_2pass_1 verbatim — the same body lthn_sdpa_vector_q8.metal already ports for int8
// K/V (that port is proven byte-for-byte against the fixed bf16 kernels in
// TestSDPAVectorQ8Parity); this file makes the SAME port, reading plain bf16 K/V (like
// lthn_sdpa_multiq.metal's read style) instead of int8+scale, with head_dim resolved at runtime
// instead of via C++ template. Decode form: no masks, no sinks, not causal, queries not
// transposed, batch 1 for the 2-pass kernel (see its own comment) — the recorded arch ICB binds
// none of those, matching every other lthn sdpa_vector port in this package.

// LTHN_SDPA_RTDIM_MAX_D bounds the per-thread register arrays: the largest head_dim this fallback
// will ever serve. It equals the largest FIXED sdpa_vector instantiation the shipped MLX metallib
// carries (256) — any head_dim that size or smaller already either has a fixed pipeline (the fast
// path, untouched) or lands here; sdpa_rtdim.go refuses a larger head_dim with a named error
// rather than silently truncating it against this cap.
#define LTHN_SDPA_RTDIM_MAX_D 256

// lthn_sdpa_vector_rtdim_bf16 — the single-pass runtime-head-dim decode SDPA. Buffer ABI mirrors
// MLX's sdpa_vector / this package's own emitSDPAAt exactly (q=0 k=1 v=2 out=3 gqa_factor=4 N=5
// k_head_stride=6 k_seq_stride=7 v_head_stride=8 v_seq_stride=9 scale=10) with ONE addition —
// head_dim=11 — appended after scale, so the fixed-kernel emit helpers and this one differ by
// exactly one binding. One threadgroup per (batch·head) — BN=32 simdgroups of BD=32 lanes
// (1024 threads), the same dispatch shape as every fixed/q8 sdpa_vector variant — batch is folded
// into gqa_factor/N by the caller exactly as it is for the fixed kernel, so this kernel is batch-
// general with no special-casing (see sdpa_rtdim.go's dispatch comment).
[[kernel]] void lthn_sdpa_vector_rtdim_bf16(
    const device bf16* queries [[buffer(0)]],
    const device bf16* keys [[buffer(1)]],
    const device bf16* values [[buffer(2)]],
    device bf16* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const constant int& head_dim [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int MAX_PER_THREAD = LTHN_SDPA_RTDIM_MAX_D / BD;
  const int D = head_dim;
  const int per_thread = D / BD; // sdpa_rtdim.go guarantees D is a positive multiple of BD, <= MAX_D
  const int inner_k_stride = BN * int(k_seq_stride);
  const int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[MAX_PER_THREAD];
  thread U o[MAX_PER_THREAD];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // decode form: one query row per head, batch folded into head_idx by the caller — identical
  // addressing to sdpa_vector / lthn_sdpa_vector_q8, only D is resolved at runtime.
  const int head_idx = tid.x;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride + simd_lid * per_thread;
  out += head_idx * D + simd_gid * per_thread;

  for (int i = 0; i < per_thread; i++) {
    q[i] = static_cast<U>(scale) * static_cast<U>(queries[i]);
  }
  for (int i = 0; i < per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -3.0e38f;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < N; i += BN) {
    U partial = 0;
    for (int j = 0; j < per_thread; j++) {
      partial += q[j] * static_cast<U>(keys[j]);
    }
    U score = simd_sum(partial);

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    for (int j = 0; j < per_thread; j++) {
      o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
    }

    keys += inner_k_stride;
    values += inner_v_stride;
  }

  // cross-simdgroup combine — MLX sdpa_vector verbatim (a BNxBD threadgroup-memory transpose: BN
  // and BD are both fixed at 32 regardless of head_dim, so this section is completely unaffected
  // by D being a runtime value — only the per_thread loop bound above changes).
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < per_thread; i++) {
      out[i] = static_cast<bf16>(o[i]);
    }
  }
}

// lthn_sdpa_vector_2pass_1_rtdim_bf16 — the runtime-head-dim pass 1 sibling, for decode past the
// single-pass knee (sdpa2PassMinKV). BATCH-1 ONLY: unlike the single-pass kernel above, MLX's
// 2-pass pass-1 addresses kv_head_idx straight off the threadgroup grid (tid.x), not derived from
// a batch-folded head_idx, so a batch dimension needs its OWN grid axis — the fixed/q8 2-pass-1
// kernels in this package make the same batch=1 simplification (q8's own comment: "decode form:
// batch 1"). sdpa_rtdim.go therefore wires this kernel ONLY into the always-batch-1 decode
// chokepoint (encSDPADecodeAt / encSDPA2PassStrided); the batch-general public 2-pass entry point
// (SDPA2Pass/SDPA2PassInto) keeps today's error-on-absent-width behaviour — a named, deliberate
// boundary, not a silent one.
//
// Buffer ABI (tight, no inherited MLX gaps — this is a from-scratch kernel): q=0 k=1 v=2
// partials(out)=3 sums=4 maxs=5 N=6 k_head_stride=7 k_seq_stride=8 v_head_stride=9 v_seq_stride=10
// scale=11 head_dim=12 blocks=13. blocks is a runtime buffer value here (not the fixed kernel's
// function-constant 26) — the fallback caches ONE pipeline for every (head_dim, blocks) pair
// instead of building a new PSO per blocks value. Grid: (nKVHeads, 1, blocks) threadgroups of
// (32, gqa_factor, 1) threads — identical shape to the fixed/q8 2-pass-1 dispatch.
[[kernel]] void lthn_sdpa_vector_2pass_1_rtdim_bf16(
    const device bf16* queries [[buffer(0)]],
    const device bf16* keys [[buffer(1)]],
    const device bf16* values [[buffer(2)]],
    device bf16* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant size_t& k_head_stride [[buffer(7)]],
    const constant size_t& k_seq_stride [[buffer(8)]],
    const constant size_t& v_head_stride [[buffer(9)]],
    const constant size_t& v_seq_stride [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const constant int& head_dim [[buffer(12)]],
    const constant int& blocks [[buffer(13)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int MAX_PER_THREAD = LTHN_SDPA_RTDIM_MAX_D / BD;
  const int D = head_dim;
  const int per_thread = D / BD; // sdpa_rtdim.go guarantees D is a positive multiple of BD, <= MAX_D

  typedef float U;

  thread U q[MAX_PER_THREAD];
  thread U o[MAX_PER_THREAD];
  for (int i = 0; i < per_thread; i++) {
    o[i] = 0;
  }

  // decode form: batch 1, q_seq_len 1 — grid (nKVHeads, 1, blocks) of (32, gqa, 1), exactly the
  // recorded arch ICB's 2-pass dispatch (mirrors lthn_sdpa_vector_2pass_1_q8).
  const int kv_head_idx = tid.x;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int o_offset = q_head_idx;

  queries += o_offset * D + simd_lid * per_thread;
  keys += kv_head_idx * k_head_stride + block_idx * k_seq_stride + simd_lid * per_thread;
  values += kv_head_idx * v_head_stride + block_idx * v_seq_stride + simd_lid * per_thread;
  out += o_offset * blocks * D + block_idx * D + simd_lid * per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  for (int i = 0; i < per_thread; i++) {
    q[i] = static_cast<U>(scale) * static_cast<U>(queries[i]);
  }

  U max_score = -3.0e38f;
  U sum_exp_score = 0;

  for (int i = block_idx; i < N; i += blocks) {
    U partial = 0;
    for (int j = 0; j < per_thread; j++) {
      partial += q[j] * static_cast<U>(keys[j]);
    }
    U score = simd_sum(partial);

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    for (int j = 0; j < per_thread; j++) {
      o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
    }

    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
  }

  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i = 0; i < per_thread; i++) {
    out[i] = static_cast<bf16>(o[i]);
  }
}
