// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_attn_vappend_f32 — append L freshly-projected V rows into the device KV cache at the
// slot named by a POSITION BUFFER: cacheV[(pos0+t)·rowDim + i] = v[t·rowDim + i]. The compute
// twin of the blit copy the chain used (an ICB carries compute commands only, and a blit's
// destination offset cannot follow a per-token position anyway — this kernel reads pos0 from
// buffer(3), so a recorded command replays at whatever position the host has bumped it to).
kernel void lthn_attn_vappend_f32(
    const device float* v      [[buffer(0)]], // [L, rowDim]
    device float*       cacheV [[buffer(1)]], // [cap, rowDim]
    constant uint&      rowDim [[buffer(2)]], // KVH*HD
    constant int&       pos0   [[buffer(3)]],
    constant uint&      total  [[buffer(4)]], // L*rowDim
    uint i [[thread_position_in_grid]]) {
  if (i >= total) return;
  cacheV[uint(pos0) * rowDim + i] = v[i];
}
