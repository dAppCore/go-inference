// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// The batched PLE slab builder's device-side bookends (the middle — steel GEMM + scale/rms/
// combine — reuses the shared kernels). Together they keep the whole K-token slab on the GPU:
// no host gather, no token→layer-major host scatter, no re-upload.

// lthn_ple_gather_rows_bf16 — gather K tokens' per-layer embedding rows and scale:
// out[i·plDim + c] = bf16(float(table[ids[i]·plDim + c]) · embScale). One thread per element,
// grid (plDim, K). The bf16 twin of the quant lthn_embed_gather (E2B's PLE table is bf16).
kernel void lthn_ple_gather_rows_bf16(
    const device int* ids       [[buffer(0)]],  // K token ids
    const device bfloat* table  [[buffer(1)]],  // [vocabPLI × plDim]
    device bfloat* out          [[buffer(2)]],  // [K × plDim] token-major
    constant int& plDim         [[buffer(3)]],
    constant float& embScale    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x);
  const int i = int(gid.y);
  if (c >= plDim) {
    return;
  }
  const int tok = ids[i];
  out[i * plDim + c] = static_cast<bfloat>(float(table[tok * plDim + c]) * embScale);
}

// lthn_ple_relayout_bf16 — token-major → layer-major: out[(li·K + i)·pliDim + d] =
// in[(i·numLayers + li)·pliDim + d]. One thread per element over K·plDim; pure copy, so the
// landed slab bytes equal the token-major tensor exactly.
kernel void lthn_ple_relayout_bf16(
    const device bfloat* in  [[buffer(0)]],  // [K × numLayers·pliDim] token-major
    device bfloat* out       [[buffer(1)]],  // [numLayers × K·pliDim] layer-major
    constant int& rows       [[buffer(2)]],  // K
    constant int& numLayers  [[buffer(3)]],
    constant int& pliDim     [[buffer(4)]],
    uint g [[thread_position_in_grid]]) {
  const int total = rows * numLayers * pliDim;
  if (int(g) >= total) {
    return;
  }
  const int d = int(g) % pliDim;
  const int rest = int(g) / pliDim;
  const int li = rest % numLayers;
  const int i = rest / numLayers;
  out[(li * rows + i) * pliDim + d] = in[int(g)];
}

// lthn_ple_gather_rows_quant — the QUANT twin of lthn_ple_gather_rows_bf16: gather + dequantise
// K tokens' per-layer embedding rows in ONE dispatch (the per-token gather loop paid ~14µs a
// dispatch — ~100ms of an 8K e2b prefill). out[i·plDimOut + c] = bf16((s·code + b)·embScale).
// plDimOut may be a PREFIX of the packed row (a shared-suffix prefill chunk needs only the
// owner layers' slices, #381); rowPacked/rowSB carry the FULL row strides so codes and scale
// groups land exactly as the per-token kernel read them. Row offsets are LONG: an 8-bit e4b
// table's packed bytes exceed 2^31. One thread per element, grid (plDimOut, K).
kernel void lthn_ple_gather_rows_quant(
    const device int* ids        [[buffer(0)]],  // K token ids
    const device uint8_t* packed [[buffer(1)]],
    const device bfloat* scales  [[buffer(2)]],
    const device bfloat* biases  [[buffer(3)]],
    device bfloat* out           [[buffer(4)]],  // [K × plDimOut] token-major
    constant int& plDimOut       [[buffer(5)]],
    constant int& groupSize      [[buffer(6)]],
    constant float& embScale     [[buffer(7)]],
    constant int& rowPacked      [[buffer(8)]],  // FULL packed row bytes
    constant int& rowSB          [[buffer(9)]],  // FULL row scale/bias groups
    constant int& bits           [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x);
  const int i = int(gid.y);
  if (c >= plDimOut) {
    return;
  }
  const long tok = long(ids[i]);
  const int g = c / groupSize;
  const float s = float(scales[tok * long(rowSB) + long(g)]);
  const float b = float(biases[tok * long(rowSB) + long(g)]);
  const int bitOff = c * bits;
  const long byteIdx = tok * long(rowPacked) + long(bitOff >> 3);
  const int shift = bitOff & 7;
  uint v = uint(packed[byteIdx]) >> shift;
  if (shift + bits > 8) {
    v |= uint(packed[byteIdx + 1]) << (8u - uint(shift));
  }
  const float code = float(v & ((1u << uint(bits)) - 1u));
  out[i * plDimOut + c] = static_cast<bfloat>((s * code + b) * embScale);
}

// lthn_ple_gather_rows_bf16_pfx — lthn_ple_gather_rows_bf16 with the row STRIDE split from
// the gathered width, so a shared-suffix prefill chunk gathers only the owner layers' prefix
// of each bf16 PLE row (#381). plDimOut == plDimRow reproduces the original kernel's bytes.
kernel void lthn_ple_gather_rows_bf16_pfx(
    const device int* ids      [[buffer(0)]],  // K token ids
    const device bfloat* table [[buffer(1)]],  // [vocabPLI × plDimRow]
    device bfloat* out         [[buffer(2)]],  // [K × plDimOut] token-major
    constant int& plDimOut     [[buffer(3)]],
    constant int& plDimRow     [[buffer(4)]],
    constant float& embScale   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int c = int(gid.x);
  const int i = int(gid.y);
  if (c >= plDimOut) {
    return;
  }
  const long tok = long(ids[i]);
  out[i * plDimOut + c] = static_cast<bfloat>(float(table[tok * long(plDimRow) + long(c)]) * embScale);
}
