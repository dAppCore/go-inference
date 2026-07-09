// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_softmax_causal_rows_bf16 — in-place causal row softmax over the prompt attention's
// score slab S[rows × N] (bf16, rows packed at ldd = N). One threadgroup per slab row. The
// grid may stack several heads' K-row score blocks ([head][K][N], the batched-GEMM layout),
// so the query index within the chunk is row % K.
//
// Query s attends keys [0 .. N-K+s] — the same per-row cap the multiQ vector kernel applies
// (query s uses key i iff i <= N - K + s), expressed as a valid-prefix length so causality
// needs no mask storage. Entries at or beyond the cap are written 0.0 so the following
// P @ V steel GEMM reads zero weight from the masked tail.
//
// Maths is f32 throughout (scores load bf16 → f32, scale applied at read — equivalent to the
// vector kernel scaling q before the dot). Three sweeps: max → exp+sum (exp stored bf16
// in place) → normalise. The stored-P double rounding (unnormalised exp then normalise) is
// bounded by bf16's 2⁻⁸ relative step — the same tier as the score slab itself.
kernel void lthn_softmax_causal_rows_bf16(
    device bfloat* S [[buffer(0)]],
    const constant int& N [[buffer(1)]],
    const constant int& K [[buffer(2)]],
    const constant float& scale [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  device bfloat* rowPtr = S + size_t(row) * size_t(N);
  const int s = int(row) % K;    // query index within the chunk (grid may stack heads)
  const int valid = N - K + s + 1; // keys [0, valid)
  const int simds = int(lsize) / 32;

  threadgroup float tgRed[32];
  threadgroup float tgOut[2];

  // sweep 1: row max over the valid prefix (scaled scores)
  float m = -MAXFLOAT;
  for (int i = int(lid); i < valid; i += int(lsize)) {
    m = max(m, float(rowPtr[i]) * scale);
  }
  m = simd_max(m);
  if (simd_lid == 0) {
    tgRed[simd_gid] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = (int(lid) < simds) ? tgRed[lid] : -MAXFLOAT;
  m = simd_max(m);
  if (lid == 0) {
    tgOut[0] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = tgOut[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // sweep 2: exponentiate in place + accumulate the sum; zero the masked tail
  float sum = 0.0f;
  for (int i = int(lid); i < N; i += int(lsize)) {
    if (i < valid) {
      float p = fast::exp(float(rowPtr[i]) * scale - m);
      rowPtr[i] = bfloat(p);
      sum += p;
    } else {
      rowPtr[i] = bfloat(0.0f);
    }
  }
  sum = simd_sum(sum);
  if (simd_lid == 0) {
    tgRed[simd_gid] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = (int(lid) < simds) ? tgRed[lid] : 0.0f;
  sum = simd_sum(sum);
  if (lid == 0) {
    tgOut[1] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = tgOut[1];

  // sweep 3: normalise the valid prefix
  const float inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
  for (int i = int(lid); i < valid; i += int(lsize)) {
    rowPtr[i] = bfloat(float(rowPtr[i]) * inv);
  }
}
