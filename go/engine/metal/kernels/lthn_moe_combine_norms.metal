// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_moe_combine_norms_bf16 — the MoE block's norm/combine tail in ONE dispatch:
//
//   a = rms(xL)·w1 ; b = rms(xE)·w2 ; c = a + b ; out = h + rms(c)·w3
//
// replacing five dispatches (rms, rms, add, rms, add), each a bubble on the serial
// decode encoder. Byte parity with the chain it replaces: the per-thread element
// windows, the fp32 simd/threadgroup reduction tree, precise::rsqrt, and EVERY
// rounding point mirror MLX's single-row `rms` kernel (N_READS = 4,
// out = w · T(x·inv) — T-cast then bfloat multiply) and vv_Addbfloat16
// (bf16(float + float)). Safe math mode keeps the register-resident bf16 rounds
// from being contracted away (the weighted-sum kernel's one-ULP lesson).
// Single-row only: axis_size ≤ 4096 (the Go side falls back to the chain above it).
#pragma METAL fp math_mode(safe)

kernel void lthn_moe_combine_norms_bf16(
    const device bfloat* xL  [[buffer(0)]],  // local MLP output
    const device bfloat* w1  [[buffer(1)]],
    const device bfloat* xE  [[buffer(2)]],  // expert weighted sum
    const device bfloat* w2  [[buffer(3)]],
    const device bfloat* w3  [[buffer(4)]],
    const device bfloat* h   [[buffer(5)]],  // residual input
    device bfloat*       out [[buffer(6)]],
    constant float&      eps [[buffer(7)]],
    constant uint&  axis_size [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  constexpr int SIMD_SIZE = 32;
  // batched rows: each threadgroup owns one token row (decode dispatches one threadgroup =
  // row 0; the batched tail dispatches rows*tg threads). The norm weights are shared.
  const uint row_off = tgid * axis_size;
  xL += row_off;
  xE += row_off;
  h += row_off;
  out += row_off;

  threadgroup float inv_shared[2];
  threadgroup float sumsL[SIMD_SIZE];
  threadgroup float sumsE[SIMD_SIZE];

  const uint base = lid * N_READS;

  // The two independent input norms' fp32 square-sums, per-element order as rms.
  float accL = 0;
  float accE = 0;
  if (base + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      const float l = xL[base + i];
      const float e = xE[base + i];
      accL += l * l;
      accE += e * e;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (base + i < axis_size) {
        const float l = xL[base + i];
        const float e = xE[base + i];
        accL += l * l;
        accE += e * e;
      }
    }
  }
  accL = simd_sum(accL);
  accE = simd_sum(accE);
  if (simd_group_id == 0) {
    sumsL[simd_lane_id] = 0;
    sumsE[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    sumsL[simd_group_id] = accL;
    sumsE[simd_group_id] = accE;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    accL = simd_sum(sumsL[simd_lane_id]);
    accE = simd_sum(sumsE[simd_lane_id]);
    if (simd_lane_id == 0) {
      inv_shared[0] = metal::precise::rsqrt(accL / axis_size + eps);
      inv_shared[1] = metal::precise::rsqrt(accE / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float invL = inv_shared[0];
  const float invE = inv_shared[1];

  // c = a + b at the chain's exact rounds (two rms output rounds, then the add round),
  // held in registers for the third norm — the same bf16 values rms3 read from device.
  bfloat c[N_READS];
  float accC = 0;
  for (int i = 0; i < N_READS; i++) {
    if (base + uint(i) < axis_size) {
      const bfloat a = w1[base + i] * static_cast<bfloat>(xL[base + i] * invL);
      const bfloat b = w2[base + i] * static_cast<bfloat>(xE[base + i] * invE);
      const bfloat ci = bfloat(float(a) + float(b));
      c[i] = ci;
      const float cf = ci;
      accC += cf * cf;
    } else {
      c[i] = bfloat(0.0f);
    }
  }
  accC = simd_sum(accC);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    sumsL[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    sumsL[simd_group_id] = accC;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    accC = simd_sum(sumsL[simd_lane_id]);
    if (simd_lane_id == 0) {
      inv_shared[0] = metal::precise::rsqrt(accC / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float invC = inv_shared[0];

  // out = h + rms(c)·w3, both rounds as the chain.
  for (int i = 0; i < N_READS; i++) {
    if (base + uint(i) < axis_size) {
      const bfloat d = w3[base + i] * static_cast<bfloat>(c[i] * invC);
      out[base + i] = bfloat(float(h[base + i]) + float(d));
    }
  }
}
