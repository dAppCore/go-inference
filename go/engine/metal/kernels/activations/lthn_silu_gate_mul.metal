// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_silu_gate_mul_bf16 — the SwiGLU gate silu(gate)·up (llama/mistral/qwen/olmoe), fused into
// ONE kernel: every intermediate stays in an fp32 register with a single bf16 rounding at the
// store. The composed chain (σ rounded to bf16, ·gate rounded, ·up rounded) loses ~1% sumAbs and
// up to ~0.33 absolute per element per MLP pass (#67's layer-0 seed measurement) — enough to
// compound into the qwen3 mid-stack drift and the layer-35 cancellation blowup.
//
// silu(x) = x·σ(x) = x / (1 + e⁻ˣ)
kernel void lthn_silu_gate_mul_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up   [[buffer(1)]],
    device bfloat*       out  [[buffer(2)]],
    constant uint&       n    [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  float g = float(gate[i]);
  float s = g / (1.0f + precise::exp(-g));
  out[i] = bfloat(s * float(up[i]));
}
