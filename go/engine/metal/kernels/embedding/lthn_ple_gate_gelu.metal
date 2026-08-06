// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

// lthn_ple_gate_gelu_rows — the PLE epilogue's gate projection with the gelu·slab fused into the
// store: gated[r,o] = bf16( gelu(bf16( qgemv(gate_w[o], x[r]) )) · ple[r,o] ), one simdgroup per
// (row, output) pair over ALL K rows in one dispatch. Replaces the batched epilogue's gate qmm_t
// + gelu-gate-mul pair (two dispatches and the hazard hop between them) — at MTP-verify K the
// stage is launch-bound (#372: resid+epilogue bucket fixed at 3.8ms for K=5 and K=33). An
// ordinary hazard-tracked dispatch: no cross-threadgroup handoff, no grid barrier (the in-kernel
// barrier variant measured broken co-residency — see #372's negative results).
//
// NUMERICS — the composed pair's stations exactly: the gate sum rounds to bf16 (the qmm_t store)
// before the fp32-internal gelu (lthn_gelu_gate_mul_bf16's form, byte-for-byte), and the product
// with the bf16 ple value rounds once at the store. The qgemv accumulation order (lane-strided
// k, simd_sum) differs from qmm_t's simdgroup-MMA — the token-identity tier the verify fold's
// projections already trade at. Deterministic: each output is owned by exactly one simdgroup.
//
// Quant width bakes as function constant 0 (4 = nibble LSB-first, 8 = byte codes — the
// lthn_ffn_megakernel specialisation pattern); group size is a runtime argument.

constant uint lthn_ple_gg_bits_fc [[function_constant(0)]];
constant uint lthn_ple_gg_bits = is_function_constant_defined(lthn_ple_gg_bits_fc) ? lthn_ple_gg_bits_fc : 4u;

static inline float lthn_ple_gg_qcode_at(const device uint8_t* prow, uint k) {
  if (lthn_ple_gg_bits == 8u) {
    return float(prow[k]);
  }
  const uint8_t pb = prow[k >> 1];
  return (k & 1u) == 0u ? float(pb & 0x0F) : float(pb >> 4);
}

kernel void lthn_ple_gate_gelu_rows(
    const device uint8_t* gateP [[buffer(0)]],  // [pliDim × dModel] packed
    const device bf16*    gateS [[buffer(1)]],
    const device bf16*    gateB [[buffer(2)]],
    const device bf16*    x     [[buffer(3)]],  // [rows × dModel] — the layer's residual rows
    const device bf16*    ple   [[buffer(4)]],  // [rows × pliDim] at the layer's slab base
    device bf16*          gated [[buffer(5)]],  // [rows × pliDim]
    const constant uint& dModel [[buffer(6)]],
    const constant uint& pliDim [[buffer(7)]],
    const constant uint& rows   [[buffer(8)]],
    const constant uint& groupSize [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint f = tid.y * 2u + simd_gid;
  if (f >= rows * pliDim) {
    return;
  }
  const uint r = f / pliDim;
  const uint o = f % pliDim;
  const uint rowBytes = (dModel * lthn_ple_gg_bits) / 8u;
  const uint groups = dModel / groupSize;

  const device uint8_t* prow = gateP + o * rowBytes;
  const device bf16* srow = gateS + o * groups;
  const device bf16* brow = gateB + o * groups;
  const device bf16* xrow = x + r * dModel;
  float partial = 0.0f;
  for (uint k = lane; k < dModel; k += 32u) {
    const uint g = k / groupSize;
    partial += (float(srow[g]) * lthn_ple_gg_qcode_at(prow, k) + float(brow[g])) * float(xrow[k]);
  }
  const float gsum = simd_sum(partial);
  if (lane == 0u) {
    const float g = float(bf16(gsum));
    const float inner = g + 0.044715f * (g * g * g);
    const float t = precise::tanh(0.7978845608028654f * inner);
    const float gelu = 0.5f * g * (1.0f + t);
    gated[f] = bf16(gelu * float(ple[f]));
  }
}
