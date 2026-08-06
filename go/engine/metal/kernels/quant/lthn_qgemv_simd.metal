// SPDX-Licence-Identifier: EUPL-1.2

// lthn_qgemv_simd — the simd-cooperative form of lthn_qgemv: ONE 32-lane simd group per output row. Each
// lane accumulates a strided slice of the reduction (k = lane, lane+32, …) then simd_sum combines them — a
// SIMD-tree reduction instead of the sequential one-thread-per-output sum, so it tracks MLX's steel
// affine_quantized gemv's reduction ORDER far more closely (the sequential sum diverged to ~0.99 cosine on
// ill-conditioned/cancelling inputs like the FFN's gated activations; the tree reduction holds ~1.0). Same
// affine dequant (scale·code + bias). This is the gemv the full-layer megakernel inlines for robust
// token-identity. Dispatch outDim·32 threads (threadgroup a multiple of 32). 4-bit only.
// ABI identical to lthn_qgemv: x(0) packed(1) scales(2) biases(3) out(4) outDim(5) inDim(6) groupSize(7)
//      rowPacked(8) rowSB(9).
#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

[[kernel]] void lthn_qgemv_simd(
    const device bf16*    x       [[buffer(0)]],
    const device uint8_t* packed  [[buffer(1)]],
    const device bf16*    scales  [[buffer(2)]],
    const device bf16*    biases  [[buffer(3)]],
    device bf16*          out     [[buffer(4)]],
    const constant uint& outDim   [[buffer(5)]],
    const constant uint& inDim    [[buffer(6)]],
    const constant uint& groupSize [[buffer(7)]],
    const constant uint& rowPacked [[buffer(8)]],
    const constant uint& rowSB     [[buffer(9)]],
    uint gid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint o = gid / 32u;
  if (o >= outDim) {
    return;
  }
  const device uint8_t* prow = packed + (uint)o * rowPacked;
  const device bf16* srow = scales + (uint)o * rowSB;
  const device bf16* brow = biases + (uint)o * rowSB;
  float partial = 0.0f;
  for (uint k = lane; k < inDim; k += 32u) {
    const uint g = k / groupSize;
    const float s = float(srow[g]);
    const float b = float(brow[g]);
    const uint8_t pb = prow[k >> 1];
    const float code = (k & 1u) == 0u ? float(pb & 0x0F) : float(pb >> 4);
    partial += (s * code + b) * float(x[k]);
  }
  const float acc = simd_sum(partial);
  if (lane == 0u) {
    out[o] = bf16(acc);
  }
}
