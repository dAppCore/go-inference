// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_mul_rows_bf16 — elementwise a·b across `rows` contiguous rows of row_len where b is ONE
// row broadcast to every a row: out[i] = a[i] * b[i % row_len]. The batched dense pass's
// per-layer output scalar (a dModel-wide broadcast buffer) applied to all K rows in one dispatch
// — per-element math identical to K per-row vv_mul dispatches.
kernel void lthn_mul_rows_bf16(
    const device bfloat* a       [[buffer(0)]],
    const device bfloat* b       [[buffer(1)]],  // one row of row_len, shared by every a row
    device bfloat*       out     [[buffer(2)]],
    constant uint&       n       [[buffer(3)]],  // rows * row_len
    constant uint&       row_len [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[i] = static_cast<bfloat>(float(a[i]) * float(b[i % row_len]));
}
