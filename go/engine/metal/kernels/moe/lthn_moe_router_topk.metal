// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

#include "lthn_router_topk_impl.h"

// lthn_moe_router_topk_bf16 mirrors pkg/metal's NativeMoERouterTopK kernel for
// the native engine: select the top-k router scores, softmax only those scores,
// and optionally apply the per-expert scale. One dispatch handles one decode
// token's score row; the Go side gates top_k to <= 32. The selection body lives
// in lthn_router_topk_impl.h, shared verbatim with the fused router kernel.
kernel void lthn_moe_router_topk_bf16(
    device const bfloat* scores           [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device int*          top_indices      [[buffer(2)]],
    device bfloat*       top_weights      [[buffer(3)]],
    constant int&        num_experts      [[buffer(4)]],
    constant int&        top_k            [[buffer(5)]],
    constant int&        has_scale        [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  // batched rows: each 32-lane threadgroup owns one score row (decode dispatches one
  // threadgroup = row 0; the batched router dispatches rows*32 threads).
  lthn_router_topk_impl(
      scores, per_expert_scale, top_indices, top_weights,
      num_experts, top_k, has_scale, tgid, lane);
}
