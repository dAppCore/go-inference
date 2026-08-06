// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_bf16_mul_scalar multiplies a bf16 vector by one bf16 scalar. This is the
// native equivalent of pkg/metal's scalar multiply path for hot decode sites
// that previously built a dense broadcast vector just to use vv_Multiplybfloat16.
kernel void lthn_bf16_mul_scalar(
    device const bfloat* in     [[buffer(0)]],
    device const bfloat* scalar [[buffer(1)]],
    device bfloat*       out    [[buffer(2)]],
    constant int&        n      [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= uint(n)) return;
  out[i] = bfloat(float(in[i]) * float(scalar[0]));
}
