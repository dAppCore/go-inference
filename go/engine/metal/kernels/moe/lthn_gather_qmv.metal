// SPDX-Licence-Identifier: EUPL-1.2

// lthn_gather_qmv[_fast] — MLX's affine_gather_qmv[_fast] with the runtime shape/stride
// buffer machinery retired (#280). The MLX kernel resolves each route's expert offsets
// through adjust_matrix_offsets: 9 constant-buffer bindings (x_shape/x_strides,
// w_shape/w_strides, s/b_strides, batch_shape, lhs/rhs_strides) + 3 ndim scalars, read
// per threadgroup. For the MoE gather the engine dispatches, every one of those values
// is a pure function of the pipeline's geometry:
//
//	x_idx    = lhs_indices[tid.z]          (batched-x; shared-x never reads lhs)
//	w_idx    = rhs_indices[tid.z]
//	w       += w_idx · expert_rows · in_vec_size·bits/32
//	scales  += w_idx · expert_rows · in_vec_size/group_size   (biases identically)
//	x       += x_idx · in_vec_size          (batched-x only)
//	y       += tid.z · out_vec_size
//
// so the extents bake as FUNCTION CONSTANTS (expert_rows, batched_x) on the already
// geometry-specialised template (group_size/bits in the name), and the dispatch ABI
// shrinks from 14 buffers + 6 scalars to 7 buffers + 2 scalars. The dot-product body
// IS MLX's qmv_fast_impl / qmv_impl (same include), so the arithmetic — and therefore
// the output bytes — are identical to the affine_gather_qmv path by construction; the
// parity test gates it anyway.
//
// Same include chain as MLX's quantized.metal (built with -I external/mlx).
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
// clang-format on

// expert_rows: weight rows per expert in the packed slab (in-proj: dFF·2 or dFF;
// down-proj: dModel). batched_x: the lhs map selects a per-route x row (the down
// projection and the prefill's pair->token gathers); false shares one x row (decode
// gate/up). Both always specialised — Metal refuses an unspecialised build once a
// function constant is declared, which is the point: no runtime fallback lane exists.
constant int lthn_gather_expert_rows [[function_constant(0)]];
constant bool lthn_gather_batched_x [[function_constant(1)]];

template <typename T, int group_size, int bits>
[[kernel]] void lthn_gather_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int64_t w_idx = rhs_indices[tid.z];
  // (in_vec_size·bits) is guaranteed ≡ 0 mod 32 by the host's packed-row alignment
  // guard; int64 keeps the expert offset safe past 2^31 packed words on big slabs.
  w += w_idx * lthn_gather_expert_rows * (int64_t(in_vec_size) * bits / 32);
  const int64_t sb_stride = int64_t(lthn_gather_expert_rows) * (in_vec_size / group_size);
  scales += w_idx * sb_stride;
  biases += w_idx * sb_stride;
  if (lthn_gather_batched_x) {
    x += int64_t(lhs_indices[tid.z]) * in_vec_size;
  }
  y += tid.z * out_vec_size;
  qmv_fast_impl<T, group_size, bits>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void lthn_gather_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int64_t w_idx = rhs_indices[tid.z];
  w += w_idx * lthn_gather_expert_rows * (int64_t(in_vec_size) * bits / 32);
  const int64_t sb_stride = int64_t(lthn_gather_expert_rows) * (in_vec_size / group_size);
  scales += w_idx * sb_stride;
  biases += w_idx * sb_stride;
  if (lthn_gather_batched_x) {
    x += int64_t(lhs_indices[tid.z]) * in_vec_size;
  }
  y += tid.z * out_vec_size;
  qmv_impl<T, group_size, bits>(
      w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

#include "lthn_gelu_qmv_impl.h"

// lthn_gather_qmv_gelu — the routed experts' DOWN gather with the MLP gate
// fused into its x-load (#341 phase 1): each route reads its expert's gate/up
// activation rows and computes gelu(gate)·up at load, byte-identical to the
// chain's gated buffer (lthn_gelu_qmv_impl.h), so the expert gelu dispatch and
// its barrier disappear. Same fc prologue as lthn_gather_qmv; gate and up ride
// the same lhs indexing.
template <typename T, int group_size, int bits>
[[kernel]] void lthn_gather_qmv_gelu(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* gate [[buffer(3)]],
    const device T* up [[buffer(4)]],
    const device uint32_t* lhs_indices [[buffer(5)]],
    const device uint32_t* rhs_indices [[buffer(6)]],
    device T* y [[buffer(7)]],
    const constant int& in_vec_size [[buffer(8)]],
    const constant int& out_vec_size [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int64_t w_idx = rhs_indices[tid.z];
  w += w_idx * lthn_gather_expert_rows * (int64_t(in_vec_size) * bits / 32);
  const int64_t sb_stride = int64_t(lthn_gather_expert_rows) * (in_vec_size / group_size);
  scales += w_idx * sb_stride;
  biases += w_idx * sb_stride;
  if (lthn_gather_batched_x) {
    const int64_t xo = int64_t(lhs_indices[tid.z]) * in_vec_size;
    gate += xo;
    up += xo;
  }
  y += tid.z * out_vec_size;
  qmv_gelu_impl<T, group_size, bits>(
      w, scales, biases, gate, up, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

#define instantiate_lthn_gather_qmv_gelu(group_size, bits)                          \
  template [[host_name("lthn_gather_qmv_gelu_bfloat16_t_gs_" #group_size           \
                       "_b_" #bits)]] [[kernel]] void                              \
  lthn_gather_qmv_gelu<bfloat16_t, group_size, bits>(                              \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*,  \
      const device bfloat16_t*, const device bfloat16_t*, const device uint32_t*,  \
      const device uint32_t*, device bfloat16_t*,                                  \
      const constant int&, const constant int&, uint3, uint, uint);

instantiate_lthn_gather_qmv_gelu(32, 4)
instantiate_lthn_gather_qmv_gelu(32, 8)
instantiate_lthn_gather_qmv_gelu(64, 4)
instantiate_lthn_gather_qmv_gelu(64, 8)
instantiate_lthn_gather_qmv_gelu(128, 4)
instantiate_lthn_gather_qmv_gelu(128, 8)

#define instantiate_lthn_gather_qmv(group_size, bits)                              \
  template [[host_name("lthn_gather_qmv_fast_bfloat16_t_gs_" #group_size           \
                       "_b_" #bits)]] [[kernel]] void                              \
  lthn_gather_qmv_fast<bfloat16_t, group_size, bits>(                              \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*,  \
      const device bfloat16_t*, const device uint32_t*, const device uint32_t*,    \
      device bfloat16_t*, const constant int&, const constant int&,                \
      uint3, uint, uint);                                                          \
  template [[host_name("lthn_gather_qmv_bfloat16_t_gs_" #group_size                \
                       "_b_" #bits)]] [[kernel]] void                              \
  lthn_gather_qmv<bfloat16_t, group_size, bits>(                                   \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*,  \
      const device bfloat16_t*, const device uint32_t*, const device uint32_t*,    \
      device bfloat16_t*, const constant int&, const constant int&,                \
      uint3, uint, uint);

instantiate_lthn_gather_qmv(32, 2)
instantiate_lthn_gather_qmv(32, 3)
instantiate_lthn_gather_qmv(32, 4)
instantiate_lthn_gather_qmv(32, 5)
instantiate_lthn_gather_qmv(32, 6)
instantiate_lthn_gather_qmv(32, 8)
instantiate_lthn_gather_qmv(64, 2)
instantiate_lthn_gather_qmv(64, 3)
instantiate_lthn_gather_qmv(64, 4)
instantiate_lthn_gather_qmv(64, 5)
instantiate_lthn_gather_qmv(64, 6)
instantiate_lthn_gather_qmv(64, 8)
instantiate_lthn_gather_qmv(128, 2)
instantiate_lthn_gather_qmv(128, 3)
instantiate_lthn_gather_qmv(128, 4)
instantiate_lthn_gather_qmv(128, 5)
instantiate_lthn_gather_qmv(128, 6)
instantiate_lthn_gather_qmv(128, 8)
