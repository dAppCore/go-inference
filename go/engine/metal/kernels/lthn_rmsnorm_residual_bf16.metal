// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// lthn_rmsnorm_residual_bf16 — gemma4's post-attention / post-FF tail fused into ONE dispatch:
//
//     out = residual + RMSNorm(x, w)
//
// Collapses two ICB ops (the rms-norm + the residual add) and the barrier between them into one,
// recovering the per-op barrier-serialisation idle that dominates the single-token decode wall.
//
// BYTE-IDENTITY (the gate): the reduction is a verbatim copy of MLX's rms_single_row<bfloat>
// (lib/mlx/.../rms_norm.metal — N_READS=4, simd_sum, precise::rsqrt), so the normed value matches
// the standalone rms kernel bit-for-bit. The normed product is rounded to bf16 BEFORE the add —
// exactly as the composed path stores rms→attnOut(bf16) then reads it back for the add — so the
// fused output equals the composed rms→bf16→add→bf16 result byte-for-byte. Gated against the
// composed path on random data in the parity test, and end-to-end by the ICB≡re-encode tests.
//
// Dispatch matches the standalone rms single-row path: one threadgroup per row (gid),
// ceil(axis/N_READS)→simd-rounded threads (lid). Single-token decode ⇒ one row.
kernel void lthn_rmsnorm_residual_bf16(
    const device bfloat* x         [[buffer(0)]],  // value to normalise (e.g. Wo·attn)
    const device bfloat* w         [[buffer(1)]],  // rms norm weight
    const device bfloat* res       [[buffer(2)]],  // residual addend (the layer input)
    device bfloat*       out       [[buffer(3)]],  // out = res + rmsnorm(x, w)
    constant float&      eps       [[buffer(4)]],
    constant uint&       axis_size [[buffer(5)]],
    constant uint&       w_stride  [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  // --- reduction: Σ x²  (verbatim MLX rms_single_row) ---
  float acc = 0;
  uint base = gid * axis_size + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      float xi = x[base + i];
      acc += xi * xi;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        float xi = x[base + i];
        acc += xi * xi;
      }
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- output: out = res + bf16(w · bf16(x·inv_mean)) ---
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      bfloat normed = static_cast<bfloat>(float(w[w_stride * (lid * N_READS + i)]) * float(static_cast<bfloat>(x[base + i] * local_inv_mean[0])));
      out[base + i] = static_cast<bfloat>(float(res[base + i]) + float(normed));
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        bfloat normed = w[w_stride * (lid * N_READS + i)] * static_cast<bfloat>(x[base + i] * local_inv_mean[0]);
        out[base + i] = res[base + i] + normed;
      }
    }
  }
}
