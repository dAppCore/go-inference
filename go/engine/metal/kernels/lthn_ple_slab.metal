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
