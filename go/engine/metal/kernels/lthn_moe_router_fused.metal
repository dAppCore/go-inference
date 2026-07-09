// SPDX-Licence-Identifier: EUPL-1.2

// lthn_moe_router_fused — the whole gemma4 MoE router in ONE dispatch:
// RMSNorm(x, normW) → 4-bit affine qmv score projection → top-k + softmax
// (+ optional per-expert scale). Replaces the 3-dispatch chain
// (rms_single_row | affine_qmv | lthn_moe_router_topk) the decode ran per
// MoE layer per token — 2 launches saved per layer (#340).
//
// BYTE-EXACTNESS BY STRUCTURAL REPLICATION — the router feeds a top-8
// selection over 128 scores, where ~1 ULP of drift can flip a near-tie and
// change EXPERT SELECTION, so every phase reproduces its standalone kernel's
// arithmetic order exactly:
//
//   phase 1  the kernel is LAUNCHED at rms_single_row's own threadgroup shape
//            (32·ceil(ceil(axis/4)/32) threads, host-guarded axis ≤ 4096) and
//            replicates its body verbatim (same N_READS=4 per-thread squares,
//            same simd/threadgroup reduction tree, same bf16 store) — the
//            normed bytes match the standalone dispatch bit-for-bit.
//   phase 2  simdgroup PAIRS emulate affine qmv's virtual (32,2) threadgroups:
//            each pair calls MLX's qmv_impl VERBATIM (same include) with a
//            constructed tid.y — per-lane k-split, qdot and simd_sum order are
//            qmv_impl's own, so the score bytes match the standalone dispatch.
//   phase 3  simdgroup 0 runs lthn_router_topk_impl — the SAME body the
//            standalone top-k kernel calls (shared header), K a function
//            constant as before.
//
// Phases hand off through the device scratch buffers the chain already owns
// (normed, scores) with mem_device barriers, so the values seen by each phase
// are the exact bytes the chain's kernels would have read.
//
// Same include chain as MLX's quantized.metal (built with -I external/mlx).
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
// clang-format on

#include "lthn_router_topk_impl.h"

template <typename T, int group_size, int bits>
[[kernel]] void lthn_moe_router_fused(
    const device T* x [[buffer(0)]],
    const device T* norm_w [[buffer(1)]],
    device T* normed [[buffer(2)]],
    const device uint32_t* wq [[buffer(3)]],
    const device T* scales [[buffer(4)]],
    const device T* biases [[buffer(5)]],
    device T* scores [[buffer(6)]],
    device const bfloat* per_expert_scale [[buffer(7)]],
    device int* top_indices [[buffer(8)]],
    device bfloat* top_weights [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    constant int& axis_size [[buffer(11)]],
    constant int& num_experts [[buffer(12)]],
    constant int& has_scale [[buffer(13)]],
    constant int& top_k [[buffer(14)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = RMS_N_READS;
  constexpr int SIMD_SIZE = 32;

  // ---- phase 1: rms_single_row, replicated verbatim (gid = 0, w_stride = 1) ----
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  {
    const device T* xr = x + lid * N_READS;
    const device T* wr = norm_w + lid * N_READS;
    float acc = 0;
    if (lid * N_READS + N_READS <= uint(axis_size)) {
      for (int i = 0; i < N_READS; i++) {
        float xi = xr[i];
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((lid * N_READS + i) < uint(axis_size)) {
          float xi = xr[i];
          acc += xi * xi;
        }
      }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
      local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
      local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device T* outr = normed + lid * N_READS;
    if (lid * N_READS + N_READS <= uint(axis_size)) {
      for (int i = 0; i < N_READS; i++) {
        outr[i] = wr[i] * static_cast<T>(xr[i] * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((lid * N_READS + i) < uint(axis_size)) {
          outr[i] = wr[i] * static_cast<T>(xr[i] * local_inv_mean[0]);
        }
      }
    }
  }
  // The normed bytes must be visible to every simdgroup's qmv reads.
  threadgroup_barrier(mem_flags::mem_device);

  // ---- phase 2: affine qmv via qmv_impl, virtual (32,2) threadgroups ----
  // qmv dispatches ceil(N/8) threadgroups of 2 simdgroups (4 rows each); here
  // simdgroup pair p covers virtual threadgroups {p, p+pairs, ...} with its own
  // constructed tid.y. qmv_impl guards out-of-range rows internally.
  {
    const uint num_sgs = tg_size / SIMD_SIZE;
    const uint pairs = num_sgs / 2;
    const uint blocks = uint((num_experts + 7) / 8);
    if (pairs == 0) {
      // Degenerate single-simdgroup launch (tiny axis): emulate both virtual
      // simdgroups sequentially.
      for (uint v = 0; v < blocks; v++) {
        qmv_impl<T, group_size, bits>(
            wq, scales, biases, normed, scores, axis_size, num_experts,
            uint3(0, v, 0), 0, simd_lane_id);
        qmv_impl<T, group_size, bits>(
            wq, scales, biases, normed, scores, axis_size, num_experts,
            uint3(0, v, 0), 1, simd_lane_id);
      }
    } else if (simd_group_id / 2 < pairs) {
      for (uint v = simd_group_id / 2; v < blocks; v += pairs) {
        qmv_impl<T, group_size, bits>(
            wq, scales, biases, normed, scores, axis_size, num_experts,
            uint3(0, v, 0), simd_group_id % 2, simd_lane_id);
      }
    }
  }
  // The score bytes must be visible to simdgroup 0's selection.
  threadgroup_barrier(mem_flags::mem_device);

  // ---- phase 3: top-k + softmax (+ scale), the shared selection body ----
  if (simd_group_id == 0) {
    lthn_router_topk_impl(
        scores, per_expert_scale, top_indices, top_weights,
        num_experts, top_k, has_scale, 0, simd_lane_id);
  }
}

#define instantiate_lthn_router_fused(group_size, bits)                          \
  template [[host_name("lthn_moe_router_fused_bfloat16_t_gs_" #group_size       \
                       "_b_" #bits)]] [[kernel]] void                           \
  lthn_moe_router_fused<bfloat16_t, group_size, bits>(                          \
      const device bfloat16_t*, const device bfloat16_t*, device bfloat16_t*,   \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, \
      device bfloat16_t*, device const bfloat*, device int*, device bfloat*,    \
      constant float&, constant int&, constant int&, constant int&,             \
      constant int&, uint, uint, uint, uint);

instantiate_lthn_router_fused(32, 2)
instantiate_lthn_router_fused(32, 3)
instantiate_lthn_router_fused(32, 4)
instantiate_lthn_router_fused(32, 5)
instantiate_lthn_router_fused(32, 6)
instantiate_lthn_router_fused(32, 8)
instantiate_lthn_router_fused(64, 2)
instantiate_lthn_router_fused(64, 3)
instantiate_lthn_router_fused(64, 4)
instantiate_lthn_router_fused(64, 5)
instantiate_lthn_router_fused(64, 6)
instantiate_lthn_router_fused(64, 8)
instantiate_lthn_router_fused(128, 2)
instantiate_lthn_router_fused(128, 3)
instantiate_lthn_router_fused(128, 4)
instantiate_lthn_router_fused(128, 5)
instantiate_lthn_router_fused(128, 6)
instantiate_lthn_router_fused(128, 8)
