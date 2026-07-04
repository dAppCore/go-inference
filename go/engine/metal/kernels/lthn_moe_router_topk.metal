// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_moe_router_topk_bf16 mirrors pkg/metal's NativeMoERouterTopK kernel for
// the native engine: select the top-k router scores, softmax only those scores,
// and optionally apply the per-expert scale. One dispatch handles one decode
// token's score row; the Go side gates top_k to <= 32.
kernel void lthn_moe_router_topk_bf16(
    device const bfloat* scores           [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device int*          top_indices      [[buffer(2)]],
    device bfloat*       top_weights      [[buffer(3)]],
    constant int&        num_experts      [[buffer(4)]],
    constant int&        top_k            [[buffer(5)]],
    constant int&        has_scale        [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]]) {
  if (top_k <= 0 || top_k > 32 || top_k > num_experts || lane >= 32) return;

  float local_values[32];
  uint local_indices[32];
  for (int i = 0; i < top_k; i++) {
    local_values[i] = -3.402823466e+38f;
    local_indices[i] = 0u;
  }

  for (uint expert = lane; expert < uint(num_experts); expert += 32u) {
    float score = float(scores[expert]);
    for (int slot = 0; slot < top_k; slot++) {
      bool better = score > local_values[slot] ||
                    (score == local_values[slot] && expert < local_indices[slot]);
      if (!better) continue;
      for (int move = top_k - 1; move > slot; move--) {
        local_values[move] = local_values[move - 1];
        local_indices[move] = local_indices[move - 1];
      }
      local_values[slot] = score;
      local_indices[slot] = expert;
      break;
    }
  }

  threadgroup float lane_values[32 * 32];
  threadgroup uint lane_indices[32 * 32];
  uint base = lane * 32u;
  for (int i = 0; i < top_k; i++) {
    lane_values[base + uint(i)] = local_values[i];
    lane_indices[base + uint(i)] = local_indices[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane != 0) return;

  float best_values[32];
  uint best_indices[32];
  for (int i = 0; i < top_k; i++) {
    best_values[i] = -3.402823466e+38f;
    best_indices[i] = 0u;
  }
  for (uint src_lane = 0; src_lane < 32u; src_lane++) {
    uint src_base = src_lane * 32u;
    for (int src = 0; src < top_k; src++) {
      float score = lane_values[src_base + uint(src)];
      uint expert = lane_indices[src_base + uint(src)];
      for (int slot = 0; slot < top_k; slot++) {
        bool better = score > best_values[slot] ||
                      (score == best_values[slot] && expert < best_indices[slot]);
        if (!better) continue;
        for (int move = top_k - 1; move > slot; move--) {
          best_values[move] = best_values[move - 1];
          best_indices[move] = best_indices[move - 1];
        }
        best_values[slot] = score;
        best_indices[slot] = expert;
        break;
      }
    }
  }

  float max_value = best_values[0];
  float denom = 0.0f;
  for (int i = 0; i < top_k; i++) {
    denom += exp(best_values[i] - max_value);
  }
  for (int i = 0; i < top_k; i++) {
    uint expert = best_indices[i];
    float weight = exp(best_values[i] - max_value) / denom;
    if (has_scale != 0) {
      weight *= float(per_expert_scale[expert]);
    }
    top_indices[i] = int(expert);
    top_weights[i] = bfloat(weight);
  }
}
