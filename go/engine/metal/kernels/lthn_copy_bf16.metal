// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_copy_bf16 — a plain contiguous bf16 copy: out[i] = in[i]. The staged sliding tail lands
// its roped/normed stage rows into the ring in (at most) two contiguous slot runs AFTER every
// layer's attention has read the pre-batch ring — this is the landing. Offsets come from the
// buffer bindings; per-element identity, so the landed bytes equal the staged bytes exactly.
kernel void lthn_copy_bf16(
    const device bfloat* in  [[buffer(0)]],
    device bfloat*       out [[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[i] = in[i];
}
