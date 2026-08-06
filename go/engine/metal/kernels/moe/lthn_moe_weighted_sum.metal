// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// Byte-parity requires the intermediate bf16 rounds to SURVIVE: under the library's default
// fast math the compiler contracts the multiply + add across the register-resident bf16 cast
// (one-ULP drift vs the chain). Safe mode scopes to this file only.
#pragma METAL fp math_mode(safe)

// lthn_moe_weighted_sum_bf16 — the MoE expert combine (acc = Σ_r w_r · down_r) in ONE
// dispatch, replacing the decode tail's per-route scale + add chain (2·top_k − 1 dispatches
// per layer, each a dispatch bubble on the serial encoder). The rounding follows the chain
// it replaces EXACTLY — bf16 round after each multiply AND after each add (lthn_bf16_mul_scalar
// then vv_Addbfloat16 semantics) — so the fused output is byte-identical to the loop's.
// top_k stays a runtime bound: the accumulator is a scalar (no local arrays to spill).
kernel void lthn_moe_weighted_sum_bf16(
    device const bfloat* rows    [[buffer(0)]],  // [top_k × n] per-route rows, row-major
    device const bfloat* weights [[buffer(1)]],  // [top_k] route weights
    device bfloat*       out     [[buffer(2)]],  // [n]
    constant int&        n       [[buffer(3)]],
    constant int&        top_k   [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
  const uint i = pos.x;
  if (i >= uint(n) || top_k <= 0) return;
  // batched rows: grid y selects the token (decode dispatches height 1 = row 0); the route
  // rows are token-major [row][top_k][n] and the weights [row][top_k].
  const uint base = pos.y * uint(top_k);
  bfloat acc = bfloat(float(rows[base * uint(n) + i]) * float(weights[base]));
  for (int r = 1; r < top_k; r++) {
    const bfloat scaled = bfloat(float(rows[(base + uint(r)) * uint(n) + i]) * float(weights[base + uint(r)]));
    acc = bfloat(float(acc) + float(scaled));
  }
  out[pos.y * uint(n) + i] = acc;
}
