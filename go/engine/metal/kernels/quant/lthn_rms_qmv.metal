// SPDX-Licence-Identifier: EUPL-1.2

// lthn_rms_affine_qmv_fast — fused per-row RMSNorm + 4-bit affine_qmv: the input-rms folded into the
// projection so the rms is no longer a separate barriered ICB op before the matmul. This is the
// matmul-fusion tier toward 311 — the element-wise rms can only overlap the projection correctly when
// fused INTO it (a standalone fused-pair can't remove the rms→qmv barrier).
//
// It is qmv_fast_impl (MLX, quantized.h) with two changes: (1) an rms PRE-PASS over x computing
// inv_mean (each simdgroup covers the whole row, so its simd_sum is the full Σx² — no cross-simd
// reduce); (2) the matmul pass normalises each loaded x element — normed = bf16(normW · bf16(x·inv_mean))
// — exactly as the composed rms→bf16→qmv path stores then re-reads it, before qdot. bfloat16_t IS native
// bfloat (bf16.h), so the qmv arithmetic is byte-identical to the composed qmv; only the rms reduction
// order differs (~1 ULP, cosine ~1.0 — the lockstep fused-kernel gap). Gated in the parity test.
//
// Same include chain as MLX's quantized.metal (built with -I lib/mlx).
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
// clang-format on

template <typename T, int group_size, int bits>
METAL_FUNC void rms_qmv_fast_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    const device T* normW,
    device T* y,
    const constant float& eps,
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

  // --- RMS pre-pass: inv_mean over the whole x row (one simdgroup spans it) ---
  const device T* xr = x + tid.x * in_vec_size + simd_lid * values_per_thread;
  float ss = 0;
  for (int k = 0; k < in_vec_size; k += block_size) {
    for (int i = 0; i < values_per_thread; i++) {
      float xi = xr[k + i];
      ss += xi * xi;
    }
  }
  ss = simd_sum(ss);
  float inv_mean = precise::rsqrt(ss / in_vec_size + eps);

  // --- positions (as qmv_fast_impl) ---
  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  const device T* nw = normW + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  // --- matmul pass: normalise each x element, then apply load_vector's bits==4 pre-scaling
  // (x_thread[i+j] /= 16^j to compensate the weight packing in qdot; sum is Σ of the RAW normed) ---
  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = 0;
    for (int i = 0; i < values_per_thread; i += 4) {
      U u0 = static_cast<U>(nw[i + 0] * static_cast<T>(static_cast<U>(x[i + 0]) * inv_mean));
      U u1 = static_cast<U>(nw[i + 1] * static_cast<T>(static_cast<U>(x[i + 1]) * inv_mean));
      U u2 = static_cast<U>(nw[i + 2] * static_cast<T>(static_cast<U>(x[i + 2]) * inv_mean));
      U u3 = static_cast<U>(nw[i + 3] * static_cast<T>(static_cast<U>(x[i + 3]) * inv_mean));
      sum += u0 + u1 + u2 + u3;
      x_thread[i + 0] = u0;
      x_thread[i + 1] = u1 / 16.0f;
      x_thread[i + 2] = u2 / 256.0f;
      x_thread[i + 3] = u3 / 4096.0f;
    }

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
    nw += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void lthn_rms_affine_qmv_fast(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const device T* normW [[buffer(7)]],
    const constant float& eps [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  rms_qmv_fast_impl<T, group_size, bits>(
      w, scales, biases, x, normW, y, eps, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

#define instantiate_rms_qmv(group_size, bits)                                  \
  template [[host_name("lthn_rms_affine_qmv_fast_bfloat16_t_gs_" #group_size   \
                       "_b_" #bits)]] [[kernel]] void                          \
  lthn_rms_affine_qmv_fast<bfloat16_t, group_size, bits>(                      \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, \
      const device bfloat16_t*, device bfloat16_t*, const constant int&,       \
      const constant int&, const device bfloat16_t*, const constant float&,    \
      uint3, uint, uint);

instantiate_rms_qmv(32, 4)
instantiate_rms_qmv(64, 4)
instantiate_rms_qmv(128, 4)
