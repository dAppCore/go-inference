// SPDX-Licence-Identifier: EUPL-1.2

// lthn_gelu_qmv — the MoE local expert's DOWN projection with the MLP gate
// fused into its x-load (#341 phase 1): y = dequant(W_down) · gelu(gate)·up,
// one dispatch where the chain ran lthn_gelu_gate_mul + affine_qmv with a
// dependency hop between them. Values are byte-identical to the chain (see
// lthn_gelu_qmv_impl.h); the gelu dispatch and its barrier disappear.
//
// Same include chain as MLX's quantized.metal (built with -I external/mlx).
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
// clang-format on

#include "lthn_gelu_qmv_impl.h"

template <typename T, int group_size, int bits>
[[kernel]] void lthn_gelu_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* gate [[buffer(3)]],
    const device T* up [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_gelu_impl<T, group_size, bits>(
      w, scales, biases, gate, up, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

#define instantiate_lthn_gelu_qmv(group_size, bits)                              \
  template [[host_name("lthn_gelu_qmv_bfloat16_t_gs_" #group_size               \
                       "_b_" #bits)]] [[kernel]] void                           \
  lthn_gelu_qmv<bfloat16_t, group_size, bits>(                                  \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, \
      const device bfloat16_t*, const device bfloat16_t*, device bfloat16_t*,   \
      const constant int&, const constant int&, uint3, uint, uint);

instantiate_lthn_gelu_qmv(32, 4)
instantiate_lthn_gelu_qmv(32, 8)
instantiate_lthn_gelu_qmv(64, 4)
instantiate_lthn_gelu_qmv(64, 8)
instantiate_lthn_gelu_qmv(128, 4)
instantiate_lthn_gelu_qmv(128, 8)
