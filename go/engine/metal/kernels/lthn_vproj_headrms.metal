// SPDX-Licence-Identifier: EUPL-1.2

// lthn_vproj_headrms — the whole gemma4 V path in ONE kernel: input-RMSNorm(x, inNormW) → 4-bit V
// projection → per-head value-norm (RMS over head_dim) → write the cache row. Removes BOTH the separate
// input-rms barrier (folded as a prologue, like lthn_rms_affine_qmv_fast) AND the separate value-norm
// barrier (folded as an epilogue) before SDPA — the superlinear input-rms+value-norm batch the relax
// probe measured. The value-norm can't ride the fast qmv (its per-head rms needs ALL of head_dim, which
// the fast variant splits across ~32 threadgroups); here ONE threadgroup owns one KV head, so the
// reduction lives in threadgroup memory. The cache stays stored NORMED (no SDPA-side change).
//
// Layout: grid = nKVHeads threadgroups, head_dim threads each. Thread d computes V output row
// (head·head_dim + d) with a plain per-thread 4-bit dot (small matmul — head_dim×in_vec_size — so the
// per-thread dot is fine and keeps the rms reductions trivially threadgroup-local). bf16 intermediates
// are rounded to track the composed rms→bf16→qmv→bf16→value-norm path (cosine ~1.0, lockstep).
//
// ABI: w(0) scales(1) biases(2) x(3) inNormW(4) out(5) in_vec_size(6) eps(8). head_dim == the
// threadgroup size (we dispatch exactly head_dim threads per head), so it needs no buffer.
#include <metal_stdlib>
using namespace metal;

typedef bfloat bfloat16_t; // matches MLX bf16.h (bfloat16_t IS native bfloat)

template <typename T, int group_size, int bits>
[[kernel]] void lthn_vproj_headrms(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device T* inNormW [[buffer(4)]],
    device T* out [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant float& eps [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint d [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = 32 / bits; // 8 vals per uint32 for 4-bit
  const int groups = in_vec_size / group_size;
  const int packs_per_row = in_vec_size / pack_factor;
  const int head_dim = int(tg_size); // we dispatch exactly head_dim threads per head
  const int n_simd = int(tg_size) / 32;

  threadgroup float red[1024]; // reused: input-rms partials, then V outputs for the value-norm

  // ---- input RMSNorm over the whole x row (threadgroup-reduced inv_mean) ----
  float ss = 0;
  for (int k = int(d); k < in_vec_size; k += int(tg_size)) {
    float xi = float(x[k]);
    ss += xi * xi;
  }
  red[d] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint off = tg_size >> 1; off > 0; off >>= 1) {
    if (d < off) {
      red[d] += red[d + off];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv_mean = precise::rsqrt(red[0] / float(in_vec_size) + eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- V projection (simd-cooperative): each simdgroup owns output rows {simd_id, +n_simd, …}; its 32
  // lanes reduce the inDim dot. ~32× more parallel than one-thread-per-output. ----
  for (int o = int(simd_id); o < head_dim; o += n_simd) {
    const int orow = int(head) * head_dim + o;
    const device uint32_t* wrow = w + orow * packs_per_row;
    const device T* srow = scales + orow * groups;
    const device T* brow = biases + orow * groups;
    float acc = 0;
    for (int k = int(lane); k < in_vec_size; k += 32) {
      uint32_t pack = wrow[k / pack_factor];
      float q = float((pack >> (bits * (k % pack_factor))) & ((1u << bits) - 1));
      float wv = q * float(srow[k / group_size]) + float(brow[k / group_size]);
      T xn_t = inNormW[k] * static_cast<T>(float(x[k]) * inv_mean); // bf16(inNormW · bf16(x·inv_mean))
      acc += wv * float(xn_t);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      red[o] = float(static_cast<T>(acc)); // bf16-rounded V (matches composed qmv→bf16)
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- per-head value-norm: RMS over head_dim ----
  // sum of squares over [0, head_dim) — reduce only the active lanes.
  float vss = 0;
  for (int i = int(d); i < head_dim; i += int(tg_size)) {
    float vi = red[i];
    vss += vi * vi;
  }
  threadgroup float vred[1024];
  vred[d] = vss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint off = tg_size >> 1; off > 0; off >>= 1) {
    if (d < off) {
      vred[d] += vred[d + off];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv_vrms = precise::rsqrt(vred[0] / float(head_dim) + eps);

  if (int(d) < head_dim) {
    out[int(head) * head_dim + int(d)] = static_cast<T>(red[d] * inv_vrms);
  }
}

#define instantiate_vproj_headrms(group_size, bits)                            \
  template [[host_name("lthn_vproj_headrms_bfloat16_t_gs_" #group_size         \
                       "_b_" #bits)]] [[kernel]] void                          \
  lthn_vproj_headrms<bfloat16_t, group_size, bits>(                            \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, \
      const device bfloat16_t*, const device bfloat16_t*, device bfloat16_t*,  \
      const constant int&, const constant float&,                              \
      uint, uint, uint, uint, uint);

instantiate_vproj_headrms(32, 4)
instantiate_vproj_headrms(64, 4)
instantiate_vproj_headrms(128, 4)
