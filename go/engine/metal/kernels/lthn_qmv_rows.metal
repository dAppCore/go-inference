// SPDX-Licence-Identifier: EUPL-1.2

// lthn_qmv_rows — the M-row qmv: one dispatch projects M activation rows
// through a quant weight with the weight stream read ONCE. MLX's own batched
// qmv (and the lean gather ridden by qmv_rows.go) put the batch on the grid,
// so every (row, out-tile) threadgroup re-streams the weight bytes — the MTP
// verify measured ~2× the single-sweep floor at M=5. Here the batch lives
// INSIDE the threadgroup: each thread holds M x-slices in registers, the
// weight pack is loaded per k-block once and qdot'ed against all M rows.
//
// Byte identity: this is qmv_fast_impl's M-variant — packs_per_thread, loop
// structure, qdot and simd_sum order match qmv_fast_impl row for row, so each
// output row is byte-identical to the per-row decode qmv EXACTLY where the
// per-row path itself routes fast: outDim%8==0 && inDim%512==0 (the
// qmvBF16KernelName rule; %512 also keeps the plain k-loop tail-free at both
// packs). Other dims route per-row to qmv_impl — a DIFFERENT twin (packs=1,
// safe-loaded final block, moved-back last out-tile) this kernel does not
// reproduce; such dims take lthn_qmv_rows_general below (qmv_impl's M-variant),
// and the Go plan gate (qmvRowsTiledKeyFor) picks the twin per envelope.
// History: this kernel originally shipped at packs_per_thread=1 claiming
// qmv_impl parity — refuted 2026-07-13 (value-dependent ~1 ulp accumulation
// drift on production dims, which route fast); matched to the fast twin since.
//
// M bakes as a function constant (2..8 — the MTP verify's draft block +
// carry; larger blocks keep the gather path). M ≤ 4 rides the flat register
// tile below; M 5..8 ride lthn_qmv_rows_wide_impl's HALVED tile — the flat
// tile's x registers (M×16 floats) collapse occupancy past M=4 (12B verify:
// M=3 18.7ms vs 20 gather, M=5 35.3ms vs 27.6 gather), so the wide variant
// keeps only ceil(M/2) x-slices live and walks the rows in two halves per
// k-block. The second half re-touches the k-block's weight bytes just read by
// the first (same threadgroup, ~2KB — L1-resident), so the DEVICE weight
// stream still happens once per threadgroup. Per-row byte identity is
// preserved by construction: a fixed row's operation sequence (load_vector →
// qdot per out-row, accumulator chaining, simd_sum) is exactly the flat
// tile's — halving only reorders ACROSS rows, and rows never mix.
// Same include chain as lthn_gather_qmv.metal (built with -I the mlx headers).
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
// clang-format on

constant int lthn_qmv_rows_m [[function_constant(0)]];

template <typename T, int group_size, int bits, int M>
METAL_FUNC void lthn_qmv_rows_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2; // qmv_fast_impl's choice — byte parity
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[M][values_per_thread];
  thread U sums[M];
  thread U result[M][results_per_simdgroup] = {{0}};

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
  x += simd_lid * values_per_thread;
  y += out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    for (int m = 0; m < M; m++) {
      sums[m] = load_vector<T, U, values_per_thread, bits>(
          x + m * in_vec_size, x_thread[m]);
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      for (int m = 0; m < M; m++) {
        result[m][row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread[m], s, b, sums[m]);
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int m = 0; m < M; m++) {
      result[m][row] = simd_sum(result[m][row]);
      if (simd_lid == 0) {
        y[m * out_vec_size + row] = static_cast<T>(result[m][row]);
      }
    }
  }
}

// lthn_qmv_rows_wide_impl — the M=5..8 register-lean variant: x lives in a
// ceil(M/2)-row tile refilled per half, results[M][4] persist across halves.
// packs_per_thread / loop structure / qdot / simd_sum match qmv_fast_impl row
// for row exactly as the flat tile does — the byte-parity constraint that
// refuted the packs=1 predecessor applies here unchanged, gated by
// TestLthnQMVRowsWideByteBand.
template <typename T, int group_size, int bits, int M>
METAL_FUNC void lthn_qmv_rows_wide_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2; // qmv_fast_impl's choice — byte parity
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int MH = (M + 1) / 2; // live half-tile — the occupancy lever

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[MH][values_per_thread];
  thread U sums[MH];
  thread U result[M][results_per_simdgroup] = {{0}};

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
  x += simd_lid * values_per_thread;
  y += out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    for (int h = 0; h < 2; h++) {
      const int m0 = h * MH;
      const int mn = (h == 0) ? MH : (M - MH);
      for (int m = 0; m < mn; m++) {
        sums[m] = load_vector<T, U, values_per_thread, bits>(
            x + (m0 + m) * in_vec_size, x_thread[m]);
      }

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        for (int m = 0; m < mn; m++) {
          result[m0 + m][row] +=
              qdot<U, values_per_thread, bits>(wl, x_thread[m], s, b, sums[m]);
        }
      }
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int m = 0; m < M; m++) {
      result[m][row] = simd_sum(result[m][row]);
      if (simd_lid == 0) {
        y[m * out_vec_size + row] = static_cast<T>(result[m][row]);
      }
    }
  }
}

// lthn_qmv_rows_general_impl — the M-row twin of qmv_impl, for every dim the
// fast envelope (outDim%8==0 && inDim%512==0) rejects: the real gemma4 26B-A4B
// MoE projections (2816/704/2112 — no inDim 512-aligned) all land here. The
// per-row oracle at such dims is qmv_impl (the kernel qmvBF16KernelName routes
// to), a DIFFERENT twin from qmv_fast_impl: packs_per_thread=1 always (so
// values_per_thread is HALF the fast tile's — an accumulation-order difference,
// the exact ~1 ulp drift that refuted this file's packs=1 predecessor against
// FAST dims), a tail-safe k-walk (the plain loop stops at
// in_vec_size - block_size; the final block — full or partial — always goes
// load_vector_safe + qdot_safe with remaining clamped per lane), and ragged
// out-tiles (out_vec_size < 8 bounds every row loop; otherwise the LAST tile is
// moved back to used_out_row = min(out_vec_size-4, out_row), overlap rows
// recomputed identically — the vendored kernel's own idiom). Structure follows
// qmv_impl verbatim; the batch lives INSIDE the threadgroup exactly as the flat
// fast tile above: each thread holds M x-slices, the weight pack is read once
// per k-block and qdot'ed against all M rows.
//
// Byte identity per row: for a fixed m the operation sequence (load_vector →
// sums[m] chain → qdot per out-row with result[m][row] += chaining → the ONE
// safe tail block with the SAME remaining (it depends only on k and simd_lid,
// never on m) → simd_sum → write) is exactly qmv_impl's for that row; the M
// loop interleaves only ACROSS rows and rows never mix. The remaining clamp
// that guards device bounds in the single-row kernel is also exactly what keeps
// row m's tail read out of row m+1's slice in the M-slab layout. Gated by
// TestLthnQMVRowsGeneralParity (qmv_rows_test.go).
//
// Register tile (M ≤ 4 only — no general wide variant, see
// docs/design-qmv-rows-unaligned.md): x_thread M×values_per_thread with vpt
// HALF the fast tile's (packs=1: 8 at 4-bit, 4 at 8-bit), sums M, result M×4 —
// at M=4/4-bit that is 52 live floats vs the proven-live fast flat tile's 84,
// so occupancy is strictly safer than the tile already shipping.
template <typename T, int group_size, int bits, int M>
METAL_FUNC void lthn_qmv_rows_general_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1; // qmv_impl's choice — byte parity with the non-fast per-row route
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[M][values_per_thread];
  thread U sums[M];
  thread U result[M][results_per_simdgroup] = {{0}};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // Less than one 8-row grid tile in the matrix: guard every read AND write on
  // out_row + row < out_vec_size, exactly as qmv_impl's first branch does.
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += simd_lid * values_per_thread;
    y += out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      for (int m = 0; m < M; m++) {
        sums[m] = load_vector<T, U, values_per_thread, bits>(
            x + m * in_vec_size, x_thread[m]);
      }

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        for (int m = 0; m < M; m++) {
          result[m][row] +=
              qdot<U, values_per_thread, bits>(wl, x_thread[m], s, b, sums[m]);
        }
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      for (int m = 0; m < M; m++) {
        sums[m] = load_vector_safe<T, U, values_per_thread, bits>(
            x + m * in_vec_size, x_thread[m], remaining);
      }

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        for (int m = 0; m < M; m++) {
          result[m][row] += qdot_safe<U, values_per_thread, bits>(
              wl, x_thread[m], s, b, sums[m], remaining);
        }
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      for (int m = 0; m < M; m++) {
        result[m][row] = simd_sum(result[m][row]);
        if (simd_lid == 0) {
          y[m * out_vec_size + row] = static_cast<T>(result[m][row]);
        }
      }
    }
  }

  // Otherwise the last tile is moved back to used_out_row — overlap rows are
  // recomputed with identical per-row maths (deterministic double-write),
  // exactly as qmv_impl's second branch does.
  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += simd_lid * values_per_thread;
    y += used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      for (int m = 0; m < M; m++) {
        sums[m] = load_vector<T, U, values_per_thread, bits>(
            x + m * in_vec_size, x_thread[m]);
      }

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        for (int m = 0; m < M; m++) {
          result[m][row] +=
              qdot<U, values_per_thread, bits>(wl, x_thread[m], s, b, sums[m]);
        }
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      for (int m = 0; m < M; m++) {
        sums[m] = load_vector_safe<T, U, values_per_thread, bits>(
            x + m * in_vec_size, x_thread[m], remaining);
      }

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        for (int m = 0; m < M; m++) {
          result[m][row] += qdot_safe<U, values_per_thread, bits>(
              wl, x_thread[m], s, b, sums[m], remaining);
        }
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      for (int m = 0; m < M; m++) {
        result[m][row] = simd_sum(result[m][row]);
        if (simd_lid == 0) {
          y[m * out_vec_size + row] = static_cast<T>(result[m][row]);
        }
      }
    }
  }
}

template <typename T, int group_size, int bits>
[[kernel]] void lthn_qmv_rows(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  switch (lthn_qmv_rows_m) {
    case 2:
      lthn_qmv_rows_impl<T, group_size, bits, 2>(
          w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
      break;
    case 3:
      lthn_qmv_rows_impl<T, group_size, bits, 3>(
          w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
      break;
    case 4:
      lthn_qmv_rows_impl<T, group_size, bits, 4>(
          w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
      break;
    case 5:
      lthn_qmv_rows_wide_impl<T, group_size, bits, 5>(
          w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
      break;
    case 6:
      lthn_qmv_rows_wide_impl<T, group_size, bits, 6>(
          w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
      break;
    case 7:
      lthn_qmv_rows_wide_impl<T, group_size, bits, 7>(
          w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
      break;
    case 8:
      lthn_qmv_rows_wide_impl<T, group_size, bits, 8>(
          w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
      break;
    default:
      break;
  }
}

// lthn_qmv_rows_general — the non-fast-dim entry: M ∈ 2..4 flat tiles only
// (qmvRowsTiledKeyFor's general cap; larger row groups compose from 2..4-row
// chunks host-side). Shares the lthn_qmv_rows_m function constant.
template <typename T, int group_size, int bits>
[[kernel]] void lthn_qmv_rows_general(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  switch (lthn_qmv_rows_m) {
    case 2:
      lthn_qmv_rows_general_impl<T, group_size, bits, 2>(
          w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
      break;
    case 3:
      lthn_qmv_rows_general_impl<T, group_size, bits, 3>(
          w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
      break;
    case 4:
      lthn_qmv_rows_general_impl<T, group_size, bits, 4>(
          w, scales, biases, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
      break;
    default:
      break;
  }
}

#define instantiate_lthn_qmv_rows(group_size, bits)                                \
  template [[host_name("lthn_qmv_rows_bfloat16_t_gs_" #group_size                  \
                       "_b_" #bits)]] [[kernel]] void                              \
  lthn_qmv_rows<bfloat16_t, group_size, bits>(                                     \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*,  \
      const device bfloat16_t*, device bfloat16_t*,                                \
      const constant int&, const constant int&, uint3, uint, uint);                \
  template [[host_name("lthn_qmv_rows_general_bfloat16_t_gs_" #group_size          \
                       "_b_" #bits)]] [[kernel]] void                              \
  lthn_qmv_rows_general<bfloat16_t, group_size, bits>(                             \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*,  \
      const device bfloat16_t*, device bfloat16_t*,                                \
      const constant int&, const constant int&, uint3, uint, uint);

instantiate_lthn_qmv_rows(32, 4)
instantiate_lthn_qmv_rows(32, 8)
instantiate_lthn_qmv_rows(64, 4)
instantiate_lthn_qmv_rows(64, 8)
instantiate_lthn_qmv_rows(128, 4)
instantiate_lthn_qmv_rows(128, 8)
