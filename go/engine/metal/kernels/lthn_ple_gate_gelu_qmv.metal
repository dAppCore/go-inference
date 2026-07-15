// SPDX-Licence-Identifier: EUPL-1.2

// lthn_ple_gate_gelu_qmv — the PLE gate projection with the gelu(gate)·pli product applied at the
// STORE: one op where the decode recorded two barriered stages (gate qmv → gelu·pli). The matmul
// body is MLX's qmv_fast_impl VERBATIM (same load_vector pre-scaling, qdot, simd_sum order), so the
// gate row sums are byte-identical to the composed qmv's; the epilogue rounds each sum to bf16 —
// exactly the composed op's store — before the fp32-internal gelu (lthn_gelu_gate_mul_bf16's form)
// and one bf16 store of gelu·pli. Output bytes therefore equal the composed pair's bytes, so the
// ICB and the re-encode path stay byte-equal without touching the re-encode twin — gated by the
// kernel byte-parity test and the ICB≡re-encode suites.
//
// This is the stage-count lever, not a bandwidth one (#373's corrected cost model): each dependent
// ICB stage drains ~9-10µs; the PLE chain was five serial stages, this makes the first two one.
// gelu is O(output) work applied once per row at lane 0 — the house rule the receipted-off x-load
// folds (geluFoldEnabled) encode is respected: nothing per-element is re-evaluated per tile.
//
// Same include chain as MLX's quantized.metal (built with -I lib/mlx).
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
// clang-format on

template <typename T, int group_size, int bits>
METAL_FUNC void ple_gate_gelu_qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device T* pli,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;
  pli += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      // the composed pair's stations: the qmv stores bf16, the gelu op reads
      // that bf16, computes the fp32-internal gelu, multiplies the bf16 pli
      // value in fp32 and stores bf16 — replicated exactly.
      const float g = float(static_cast<T>(result[row]));
      const float inner = g + 0.044715f * (g * g * g);
      const float t = precise::tanh(0.7978845608028654f * inner);
      y[row] = static_cast<T>((0.5f * g * (1.0f + t)) * float(pli[row]));
    }
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void lthn_ple_gate_gelu_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device T* pli [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& in_vec_size [[buffer(6)]],
    const constant int& out_vec_size [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  ple_gate_gelu_qmv_fast_impl<T, group_size, bits>(
      w, scales, biases, x, pli, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

#define instantiate_ple_gate_gelu_qmv(group_size, bits)                            \
  template [[host_name("lthn_ple_gate_gelu_qmv_bfloat16_t_gs_" #group_size        \
                       "_b_" #bits)]] [[kernel]] void                              \
  lthn_ple_gate_gelu_qmv<bfloat16_t, group_size, bits>(                            \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, \
      const device bfloat16_t*, const device bfloat16_t*, device bfloat16_t*,     \
      const constant int&, const constant int&, uint3, uint, uint);

instantiate_ple_gate_gelu_qmv(32, 4)
instantiate_ple_gate_gelu_qmv(32, 8)
instantiate_ple_gate_gelu_qmv(64, 4)
instantiate_ple_gate_gelu_qmv(64, 8)
instantiate_ple_gate_gelu_qmv(128, 4)
instantiate_ple_gate_gelu_qmv(128, 8)
