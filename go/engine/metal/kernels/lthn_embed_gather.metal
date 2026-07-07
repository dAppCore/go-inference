// SPDX-Licence-Identifier: EUPL-1.2

// lthn_embed_gather_bf16 — gather + dequantise ONE token's embedding row on the GPU: out[c] =
// bf16((scale_g·code_c + bias_g)·embedScale), the token id read from a GPU buffer (the LM-head argmax
// output). This is the seam that lets the chained decode step produce the NEXT step's input embedding
// without a host round-trip — so token t's step can submit before t's token is read back (the
// submit-ahead pipeline). Affine codes are bit-packed LSB-first contiguous (MLX's packing — the 4-bit
// nibble-low-first layout generalised), spanning byte boundaries for 3/5/6-bit; the unpack mirrors the
// host extractAffineCode oracle bit for bit, so for 4-bit the code values (and therefore the bf16
// output bytes) are identical to the old nibble read. A spanning code never ends past its row's last
// byte, so the second byte is read only when the width demands it — never out of bounds.
// ABI: token(0) packed(1) scales(2) biases(3) out(4) dModel(5) groupSize(6) embedScale(7) rowPacked(8)
//      rowSB(9) bits(10). One thread per output element.
#include <metal_stdlib>
using namespace metal;

typedef bfloat bfloat16_t;

[[kernel]] void lthn_embed_gather_bf16(
    const device int* token [[buffer(0)]],
    const device uint8_t* packed [[buffer(1)]],
    const device bfloat16_t* scales [[buffer(2)]],
    const device bfloat16_t* biases [[buffer(3)]],
    device bfloat16_t* out [[buffer(4)]],
    const constant int& dModel [[buffer(5)]],
    const constant int& groupSize [[buffer(6)]],
    const constant float& embedScale [[buffer(7)]],
    const constant int& rowPacked [[buffer(8)]],
    const constant int& rowSB [[buffer(9)]],
    const constant int& bits [[buffer(10)]],
    uint c [[thread_position_in_grid]]) {
  if (int(c) >= dModel) {
    return;
  }
  const int tok = token[0];
  const int g = int(c) / groupSize;
  const float s = float(scales[tok * rowSB + g]);
  const float b = float(biases[tok * rowSB + g]);
  const int bitOff = int(c) * bits;
  const int byteIdx = tok * rowPacked + (bitOff >> 3);
  const int shift = bitOff & 7;
  uint v = uint(packed[byteIdx]) >> shift;
  if (shift + bits > 8) {
    v |= uint(packed[byteIdx + 1]) << (8u - uint(shift));
  }
  const float code = float(v & ((1u << uint(bits)) - 1u));
  out[c] = static_cast<bfloat16_t>((s * code + b) * embedScale);
}
