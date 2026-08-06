// SPDX-Licence-Identifier: EUPL-1.2
//
// lthn_moe_grouped.metal — the grouped-GEMM MoE prefill's plumbing kernels (#347).
//
// The expert projections themselves ride MLX's affine_gather_qmm_rhs_nax steel kernel
// (rows pre-sorted by expert, each expert's weights read once per 64-row block instead
// of once per routed pair). These kernels supply the sorted order:
//
//   lthn_moe_pair_sort         — bucket-sort the K·topK routed pairs by expert id
//   lthn_moe_gather_rows_bf16  — xSorted[i] = x[map(sortedPair[i])] row gather
//   lthn_moe_scatter_rows_bf16 — dst[sortedPair[i]] = src[i] row scatter (back to pair order)
//   lthn_moe_gelu_gate_up_bf16 — gated[i,j] = gelu(gu[i,j]) · gu[i, dff+j] over the fused
//                                 [pairs, 2·dff] gate_up rows → contiguous [pairs, dff]
//
// Pair order within one expert's bucket is NOT stable (atomic ranking) — irrelevant to the
// maths: the scatter restores exact pair order before the weighted-sum combine.

#include <metal_stdlib>
using namespace metal;

// One threadgroup: histogram (threadgroup atomics) → exclusive scan → atomic rank scatter.
// pairs ≤ a few tens of thousands, experts ≤ 1024. expert ids arrive as int32 route indices.
kernel void lthn_moe_pair_sort(
    const device int* pair_expert [[buffer(0)]],   // pairs: expert id per pair (the router's idx)
    device int* sorted_pair [[buffer(1)]],         // pairs: pair id at sorted position
    device uint* sorted_expert [[buffer(2)]],      // pairs: expert id at sorted position (qmm_rhs indices)
    constant int& pairs [[buffer(3)]],
    constant int& num_experts [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  threadgroup atomic_int counts[1024];
  threadgroup int offsets[1024];

  for (int e = int(tid); e < num_experts; e += int(tg_size)) {
    atomic_store_explicit(&counts[e], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int p = int(tid); p < pairs; p += int(tg_size)) {
    atomic_fetch_add_explicit(&counts[pair_expert[p]], 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // exclusive scan of counts → offsets (single thread: experts ≤ 1024, negligible)
  if (tid == 0) {
    int acc = 0;
    for (int e = 0; e < num_experts; e++) {
      offsets[e] = acc;
      acc += atomic_load_explicit(&counts[e], memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = int(tid); e < num_experts; e += int(tg_size)) {
    atomic_store_explicit(&counts[e], offsets[e], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int p = int(tid); p < pairs; p += int(tg_size)) {
    int e = pair_expert[p];
    int pos = atomic_fetch_add_explicit(&counts[e], 1, memory_order_relaxed);
    sorted_pair[pos] = p;
    sorted_expert[pos] = uint(e);
  }
}

// xSorted[i, :] = x[rowmap[sorted_pair[i]], :]   (has_map=0 ⇒ x[sorted_pair[i], :])
// One threadgroup per sorted row; threads stride the row in uint4 (8 bf16) steps with a
// scalar ushort tail, so any row width works.
kernel void lthn_moe_gather_rows_bf16(
    const device uchar* x [[buffer(0)]],
    const device int* sorted_pair [[buffer(1)]],
    const device int* rowmap [[buffer(2)]],
    device uchar* out [[buffer(3)]],
    constant int& row_bytes [[buffer(4)]],
    constant int& has_map [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  int i = int(tid.y);
  int p = sorted_pair[i];
  int src_row = has_map != 0 ? rowmap[p] : p;
  const device uchar* src = x + size_t(src_row) * size_t(row_bytes);
  device uchar* dst = out + size_t(i) * size_t(row_bytes);
  int vecs = row_bytes / 16;
  for (int v = int(tid.x); v < vecs; v += int(grid.x)) {
    ((device uint4*)dst)[v] = ((const device uint4*)src)[v];
  }
  for (int b = vecs * 16 + int(tid.x); b < row_bytes; b += int(grid.x)) {
    dst[b] = src[b];
  }
}

// dst[sorted_pair[i], :] = src[i, :] — the inverse of the gather, back to pair order.
kernel void lthn_moe_scatter_rows_bf16(
    const device uchar* src [[buffer(0)]],
    const device int* sorted_pair [[buffer(1)]],
    device uchar* dst [[buffer(2)]],
    constant int& row_bytes [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  int i = int(tid.y);
  int p = sorted_pair[i];
  const device uchar* s = src + size_t(i) * size_t(row_bytes);
  device uchar* d = dst + size_t(p) * size_t(row_bytes);
  int vecs = row_bytes / 16;
  for (int v = int(tid.x); v < vecs; v += int(grid.x)) {
    ((device uint4*)d)[v] = ((const device uint4*)s)[v];
  }
  for (int b = vecs * 16 + int(tid.x); b < row_bytes; b += int(grid.x)) {
    d[b] = s[b];
  }
}

// gated[i, j] = gelu_tanh(gu[i, j]) · gu[i, dff + j] over fused gate_up rows [rows, 2·dff]
// → contiguous [rows, dff]. The same tanh gelu approximation as lthn_gelu_gate_mul.
kernel void lthn_moe_gelu_gate_up_bf16(
    const device bfloat* gu [[buffer(0)]],
    device bfloat* gated [[buffer(1)]],
    constant int& dff [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
  int i = int(tid.y);
  int j = int(tid.x);
  if (j >= dff) {
    return;
  }
  float g = float(gu[size_t(i) * size_t(2 * dff) + size_t(j)]);
  float u = float(gu[size_t(i) * size_t(2 * dff) + size_t(dff + j)]);
  float g3 = g * g * g;
  float inner = 0.7978845608028654f * (g + 0.044715f * g3);
  float gelu = 0.5f * g * (1.0f + precise::tanh(inner));
  gated[size_t(i) * size_t(dff) + size_t(j)] = bfloat(gelu * u);
}
