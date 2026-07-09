// SPDX-Licence-Identifier: EUPL-1.2

// lthn_qgemv — a self-contained 4-bit affine quantised gemv: out[o] = Σ_k (scale_og·code_ok + bias_og)·x[k].
// The decode's matmuls use MLX's steel affine_quantized gemv (simd-cooperative, tiled) — which a megakernel
// can't call. This inlines the SAME affine dequant (scale·code + bias, the embed-gather's verified 4-bit
// path) with a plain per-output f32 reduction, so it's token-identical (cosine~1) to the steel kernel though
// not byte-identical (different reduction order). This is the gemv the full-layer megakernel inlines on the
// grid-barrier pattern. One thread per output row; affine params hoisted per group. 4-bit only.
// ABI: x(0) packed(1) scales(2) biases(3) out(4) outDim(5) inDim(6) groupSize(7) rowPacked(8) rowSB(9).
#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

[[kernel]] void lthn_qgemv(
    const device bf16*    x       [[buffer(0)]],   // [inDim]
    const device uint8_t* packed  [[buffer(1)]],   // [outDim × inDim/2] 4-bit, row-major
    const device bf16*    scales  [[buffer(2)]],   // [outDim × inDim/groupSize]
    const device bf16*    biases  [[buffer(3)]],
    device bf16*          out     [[buffer(4)]],   // [outDim]
    const constant uint& outDim   [[buffer(5)]],
    const constant uint& inDim    [[buffer(6)]],
    const constant uint& groupSize [[buffer(7)]],
    const constant uint& rowPacked [[buffer(8)]],  // inDim/2
    const constant uint& rowSB     [[buffer(9)]],  // inDim/groupSize
    uint o [[thread_position_in_grid]]) {
  if (o >= outDim) {
    return;
  }
  const device uint8_t* prow = packed + (uint)o * rowPacked;
  const device bf16* srow = scales + (uint)o * rowSB;
  const device bf16* brow = biases + (uint)o * rowSB;
  const uint groups = inDim / groupSize;
  float acc = 0.0f;
  for (uint g = 0; g < groups; g++) {
    const float s = float(srow[g]);
    const float b = float(brow[g]);
    const uint base = g * groupSize;
    for (uint j = 0; j < groupSize; j++) {
      const uint k = base + j;
      const uint8_t pb = prow[k >> 1];
      const float code = (k & 1u) == 0u ? float(pb & 0x0F) : float(pb >> 4);
      acc += (s * code + b) * float(x[k]);
    }
  }
  out[o] = bf16(acc);
}
