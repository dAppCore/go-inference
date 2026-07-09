// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_gelu_gate_mul_bf16 — gemma's MLP gate gelu(gate)·up, fused into ONE kernel so the Metal
// compiler FMA-contracts the polynomial and keeps every intermediate in an fp32 register: a single
// bf16 rounding at the store (vs the composed chain's ~10 dispatches, each rounded to bf16).
//
// IMPORTANT — this is the fp32-internal gelu. It is byte-identical to mlx-c's *compiled* GELUGateMul
// on fp32 inputs, and MORE accurate than the engine's production path. But production
// (metal.GeluGateMul, enableNativeGELUGateMul=false) runs the COMPOSED bf16 path (each op rounded),
// so on bf16 this kernel differs from production by ~34% of elements at the 1-ulp level. It is a
// capability, NOT a drop-in: wiring it into the serve decode is a deliberate "compute fp32, store
// bf16" decision that must move both engines (metal + native) in lockstep, not a native-only swap.
//
// gelu_approx(x) = 0.5·x·(1 + tanh(0.7978845608028654·(x + 0.044715·x³)))
kernel void lthn_gelu_gate_mul_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up   [[buffer(1)]],
    device bfloat*       out  [[buffer(2)]],
    constant uint&       n    [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  float g = float(gate[i]);
  float inner = g + 0.044715f * (g * g * g);
  float t = precise::tanh(0.7978845608028654f * inner);
  float gelu = 0.5f * g * (1.0f + t);
  out[i] = bfloat(gelu * float(up[i]));
}
