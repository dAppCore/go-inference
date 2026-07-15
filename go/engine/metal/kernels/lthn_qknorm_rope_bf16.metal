// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
using namespace metal;

// lthn_qknorm_rope_bf16 — gemma4's per-head QK-norm + RoPE fused into ONE dispatch:
//
//     normed = RMSNorm(x_head, w)          (per head, over head_dim)
//     out    = RoPE(normed)                (rotate the first rotary_dim dims; tail passes through)
//
// Replaces the two barriered ICB ops (per-head rms-rows + rope) the probe showed are the bulk of the
// element-wise barrier idle (per-head norms ~+7.5, rope ~+5.5 tok/s on e2b-4bit). One threadgroup per
// head, head_dim threads.
//
// Math copied from MLX's rms_single_row + rope_single (lib/mlx): non-traditional RoPE rotates pairs
// (i, i+rotary_dim/2) by theta = scale·offset·inv_freq, inv_freq = exp2(-(i/(rotary_dim/2))·base)
// [base path, base = log2(theta)] OR 1/periods[i] [freqs/YaRN path]. The normed value is rounded to
// bf16 BEFORE the rotation — exactly as the composed path stores the rms output then re-reads it for
// rope — so the result tracks the composed rms→bf16→rope path. NOT bit-exact (native-Metal `bfloat`
// rounds tie-cases ~1 ULP off MLX's bfloat16_t — the documented fused-kernel gap, cosine ~1.0); a
// deliberate fp32-internal, lockstep numerics decision. Gated cosine-tight in the parity test.
//
// HEAD_DIM_MAX caps the threadgroup normed scratch (gemma4 global head_dim = 512).
constant int HEAD_DIM_MAX = 512;

kernel void lthn_qknorm_rope_bf16(
    const device bfloat* x          [[buffer(0)]],  // [n_heads * head_dim] projection output
    const device bfloat* w          [[buffer(1)]],  // [head_dim] qk-norm weight
    device bfloat*       out        [[buffer(2)]],  // [n_heads * head_dim] roped, normed output
    constant float&      eps        [[buffer(3)]],
    constant int&        head_dim   [[buffer(4)]],
    constant int&        rotary_dim [[buffer(5)]],  // dims rotated (<= head_dim); tail passes through
    constant float&      scale      [[buffer(6)]],
    const device int*    offset     [[buffer(7)]],  // position (int32, one element)
    constant float&      base       [[buffer(8)]],  // log2(theta); used when use_freqs == 0
    const device float*  periods    [[buffer(9)]],  // 1/inv_freq per rotated dim; used when use_freqs != 0
    constant int&        use_freqs  [[buffer(10)]],
    uint head [[threadgroup_position_in_grid]],
    uint d    [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];
  threadgroup bfloat normed[HEAD_DIM_MAX];

  uint hbase = head * uint(head_dim);

  // --- per-head RMSNorm reduction: Σ x² over head_dim (one element per thread) ---
  float acc = 0;
  if (int(d) < head_dim) {
    float xi = x[hbase + d];
    acc = xi * xi;
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
      local_inv_mean[0] = precise::rsqrt(acc / head_dim + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- normed[d] = bf16(w[d] · bf16(x[d] · inv_mean)) — same rounding the standalone rms stores ---
  if (int(d) < head_dim) {
    normed[d] = static_cast<bfloat>(float(w[d]) * float(static_cast<bfloat>(x[hbase + d] * local_inv_mean[0])));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (int(d) >= head_dim) {
    return;
  }

  // --- RoPE: rotate pairs (i, i+rotary_dim/2); dims >= rotary_dim pass through (partial rotary) ---
  int hrot = rotary_dim / 2;
  if (int(d) >= rotary_dim) {
    out[hbase + d] = normed[d]; // tail: the normed value, unrotated
    return;
  }
  int i = (int(d) < hrot) ? int(d) : int(d) - hrot; // pair index for this dim
  float inv_freq;
  if (use_freqs != 0) {
    inv_freq = 1.0f / periods[i];
  } else {
    float dfrac = float(i) / float(hrot);
    inv_freq = metal::exp2(-dfrac * base);
  }
  float L = scale * float(offset[0]);
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);
  float x1 = float(normed[i]);        // the low half of the pair
  float x2 = float(normed[i + hrot]); // the high half of the pair
  if (int(d) < hrot) {
    out[hbase + d] = static_cast<bfloat>(x1 * costheta - x2 * sintheta); // rx1
  } else {
    out[hbase + d] = static_cast<bfloat>(x1 * sintheta + x2 * costheta); // rx2
  }
}

// lthn_qknorm_rope_rows_bf16 — the SAME fused per-head QK-norm + RoPE across a batch of rows in
// ONE dispatch: grid Y carries the row, x/out advance by a caller-supplied ELEMENT stride per row,
// and the position comes from offset[row] — the batched dense pass's packed per-row positions
// buffer (the per-row dispatches read the same buffer one int at a time). Per-(row, head) math is
// the single-row kernel's body verbatim, so each row's output is byte-identical to a per-row
// dispatch at the same offsets.
kernel void lthn_qknorm_rope_rows_bf16(
    const device bfloat* x          [[buffer(0)]],  // rows of [n_heads * head_dim], x_row_stride apart
    const device bfloat* w          [[buffer(1)]],  // [head_dim] qk-norm weight (shared by every row)
    device bfloat*       out        [[buffer(2)]],  // rows of [n_heads * head_dim], out_row_stride apart
    constant float&      eps        [[buffer(3)]],
    constant int&        head_dim   [[buffer(4)]],
    constant int&        rotary_dim [[buffer(5)]],
    constant float&      scale      [[buffer(6)]],
    const device int*    offset     [[buffer(7)]],  // per-row positions (int32, one per row)
    constant float&      base       [[buffer(8)]],
    const device float*  periods    [[buffer(9)]],
    constant int&        use_freqs  [[buffer(10)]],
    constant int&        x_row_stride   [[buffer(11)]], // elements between consecutive rows of x
    constant int&        out_row_stride [[buffer(12)]], // elements between consecutive rows of out
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 dpos [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];
  threadgroup bfloat normed[HEAD_DIM_MAX];

  const uint d = dpos.x;
  const uint head = tg.x;
  const uint row = tg.y;
  x += row * uint(x_row_stride);
  out += row * uint(out_row_stride);
  uint hbase = head * uint(head_dim);

  // --- per-head RMSNorm reduction: Σ x² over head_dim (one element per thread) ---
  float acc = 0;
  if (int(d) < head_dim) {
    float xi = x[hbase + d];
    acc = xi * xi;
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
      local_inv_mean[0] = precise::rsqrt(acc / head_dim + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- normed[d] = bf16(w[d] · bf16(x[d] · inv_mean)) — same rounding the standalone rms stores ---
  if (int(d) < head_dim) {
    normed[d] = static_cast<bfloat>(float(w[d]) * float(static_cast<bfloat>(x[hbase + d] * local_inv_mean[0])));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (int(d) >= head_dim) {
    return;
  }

  // --- RoPE: rotate pairs (i, i+rotary_dim/2); dims >= rotary_dim pass through (partial rotary) ---
  int hrot = rotary_dim / 2;
  if (int(d) >= rotary_dim) {
    out[hbase + d] = normed[d]; // tail: the normed value, unrotated
    return;
  }
  int i = (int(d) < hrot) ? int(d) : int(d) - hrot; // pair index for this dim
  float inv_freq;
  if (use_freqs != 0) {
    inv_freq = 1.0f / periods[i];
  } else {
    float dfrac = float(i) / float(hrot);
    inv_freq = metal::exp2(-dfrac * base);
  }
  float L = scale * float(offset[row]);
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);
  float x1 = float(normed[i]);        // the low half of the pair
  float x2 = float(normed[i + hrot]); // the high half of the pair
  if (int(d) < hrot) {
    out[hbase + d] = static_cast<bfloat>(x1 * costheta - x2 * sintheta); // rx1
  } else {
    out[hbase + d] = static_cast<bfloat>(x1 * sintheta + x2 * costheta); // rx2
  }
}
