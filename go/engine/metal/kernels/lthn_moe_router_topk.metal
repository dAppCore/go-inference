// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_moe_router_topk_bf16 mirrors pkg/metal's NativeMoERouterTopK kernel for
// the native engine: select the top-k router scores, softmax only those scores,
// and optionally apply the per-expert scale. One dispatch handles one decode
// token's score row; the Go side gates top_k to <= 32.
//
// top_k is a FUNCTION CONSTANT (PSO-specialised per model), not the buffer(5)
// value: with a runtime bound the selection loops cannot unroll, so the 32-slot
// local arrays spill to device-memory stack and this trivial kernel measured
// 4.78 ms/token on the 26B decode — 43% of ALL GPU time. Compile-time K keeps
// the slots in registers. buffer(5) stays bound as a coherence check only.
constant uint lthn_router_topk_k [[function_constant(0)]];

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
  const int K = int(lthn_router_topk_k);
  if (K <= 0 || K > 32 || K > num_experts || K != top_k || lane >= 32) return;
  // batched rows: each 32-lane threadgroup owns one score row (decode dispatches one
  // threadgroup = row 0; the batched router dispatches rows*32 threads).
  const uint row = tgid;
  scores += row * uint(num_experts);
  top_indices += row * uint(K);
  top_weights += row * uint(K);

  float local_values[32];
  uint local_indices[32];
  for (int i = 0; i < K; i++) {
    local_values[i] = -3.402823466e+38f;
    local_indices[i] = 0u;
  }

  for (uint expert = lane; expert < uint(num_experts); expert += 32u) {
    float score = float(scores[expert]);
    for (int slot = 0; slot < K; slot++) {
      bool better = score > local_values[slot] ||
                    (score == local_values[slot] && expert < local_indices[slot]);
      if (!better) continue;
      for (int move = K - 1; move > slot; move--) {
        local_values[move] = local_values[move - 1];
        local_indices[move] = local_indices[move - 1];
      }
      local_values[slot] = score;
      local_indices[slot] = expert;
      break;
    }
  }

  // Merge the 32 per-lane sorted lists with K rounds of simd reduction — no threadgroup
  // memory, no barrier, no serial lane-0 scan. Each round: simd_max picks the best score,
  // simd_min over the max-holders' expert ids applies the same (score desc, index asc)
  // order the insertion sort used, so the selection — and the softmax bytes — match the
  // old merge exactly. The winner lane shifts its list down one slot (static K-bounded
  // loops keep everything in registers; a data-dependent cursor would re-spill).
  float best_values[32];
  uint best_indices[32];
  for (int slot = 0; slot < K; slot++) {
    float v = local_values[0];
    uint idx = local_indices[0];
    float bestV = simd_max(v);
    uint cand = (v == bestV) ? idx : 0xFFFFFFFFu;
    uint bestI = simd_min(cand);
    if (v == bestV && idx == bestI) {
      for (int m = 0; m < 31; m++) {
        if (m >= K - 1) break;
        local_values[m] = local_values[m + 1];
        local_indices[m] = local_indices[m + 1];
      }
      if (K >= 1) {
        local_values[K - 1] = -3.402823466e+38f;
        local_indices[K - 1] = 0xFFFFFFFFu;
      }
    }
    best_values[slot] = bestV;
    best_indices[slot] = bestI;
  }

  if (lane != 0) return;

  float max_value = best_values[0];
  float denom = 0.0f;
  for (int i = 0; i < K; i++) {
    denom += exp(best_values[i] - max_value);
  }
  for (int i = 0; i < K; i++) {
    uint expert = best_indices[i];
    float weight = exp(best_values[i] - max_value) / denom;
    if (has_scale != 0) {
      weight *= float(per_expert_scale[expert]);
    }
    top_indices[i] = int(expert);
    top_weights[i] = bfloat(weight);
  }
}
