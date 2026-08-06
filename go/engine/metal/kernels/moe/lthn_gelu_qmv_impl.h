// SPDX-Licence-Identifier: EUPL-1.2

// lthn_gelu_qmv_impl — MLX's qmv_impl with the x-load fused with gemma's MLP
// gate: instead of reading a pre-computed gated buffer (the lthn_gelu_gate_mul
// dispatch's output), each element is computed AT LOAD as
//
//	x[i] = T( gelu_approx(float(gate[i])) * float(up[i]) )
//
// — the exact expression, type flow and single bf16 rounding of the standalone
// gelu kernel — so the values entering qdot are byte-identical to the chain's
// gated buffer read back, and the whole gelu dispatch (plus its barrier hop)
// disappears from the MoE layer (#341 phase 1).
//
// EVERYTHING below is a verbatim structural copy of quantized.h's
// load_vector / load_vector_safe / qmv_impl with only the element reads
// swapped: the T-typed sums (`sum += v0 + v1 + ...` adds bfloats before
// widening, exactly like the originals' device loads), the per-width x_thread
// scalings, the block walk, the guarded tails and the used_out_row redo branch
// are unchanged. Include after quantized.h (qdot/qdot_safe/get_pack_factor).

#pragma once

template <typename T>
METAL_FUNC T lthn_gelu_gate_mul_one(T gate, T up) {
  // gelu_approx(x) = 0.5·x·(1 + tanh(0.7978845608028654·(x + 0.044715·x³))) —
  // fp32 internals, ONE bf16 rounding at the return, mirroring
  // lthn_gelu_gate_mul_bf16's store.
  float g = float(gate);
  float inner = g + 0.044715f * (g * g * g);
  float t = metal::precise::tanh(0.7978845608028654f * inner);
  float gelu = 0.5f * g * (1.0f + t);
  return T(gelu * float(up));
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_gelu(
    const device T* gate,
    const device T* up,
    thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      const T v1 = lthn_gelu_gate_mul_one<T>(gate[i + 1], up[i + 1]);
      const T v2 = lthn_gelu_gate_mul_one<T>(gate[i + 2], up[i + 2]);
      const T v3 = lthn_gelu_gate_mul_one<T>(gate[i + 3], up[i + 3]);
      sum += v0 + v1 + v2 + v3;
      x_thread[i] = v0;
      x_thread[i + 1] = v1 / 4.0f;
      x_thread[i + 2] = v2 / 16.0f;
      x_thread[i + 3] = v3 / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      const T v1 = lthn_gelu_gate_mul_one<T>(gate[i + 1], up[i + 1]);
      const T v2 = lthn_gelu_gate_mul_one<T>(gate[i + 2], up[i + 2]);
      const T v3 = lthn_gelu_gate_mul_one<T>(gate[i + 3], up[i + 3]);
      const T v4 = lthn_gelu_gate_mul_one<T>(gate[i + 4], up[i + 4]);
      const T v5 = lthn_gelu_gate_mul_one<T>(gate[i + 5], up[i + 5]);
      const T v6 = lthn_gelu_gate_mul_one<T>(gate[i + 6], up[i + 6]);
      const T v7 = lthn_gelu_gate_mul_one<T>(gate[i + 7], up[i + 7]);
      sum += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
      x_thread[i] = v0;
      x_thread[i + 1] = v1 / 8.0f;
      x_thread[i + 2] = v2 / 64.0f;
      x_thread[i + 3] = v3 / 2.0f;
      x_thread[i + 4] = v4 / 16.0f;
      x_thread[i + 5] = v5 / 128.0f;
      x_thread[i + 6] = v6 / 4.0f;
      x_thread[i + 7] = v7 / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      const T v1 = lthn_gelu_gate_mul_one<T>(gate[i + 1], up[i + 1]);
      const T v2 = lthn_gelu_gate_mul_one<T>(gate[i + 2], up[i + 2]);
      const T v3 = lthn_gelu_gate_mul_one<T>(gate[i + 3], up[i + 3]);
      sum += v0 + v1 + v2 + v3;
      x_thread[i] = v0;
      x_thread[i + 1] = v1 / 16.0f;
      x_thread[i + 2] = v2 / 256.0f;
      x_thread[i + 3] = v3 / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      const T v1 = lthn_gelu_gate_mul_one<T>(gate[i + 1], up[i + 1]);
      const T v2 = lthn_gelu_gate_mul_one<T>(gate[i + 2], up[i + 2]);
      const T v3 = lthn_gelu_gate_mul_one<T>(gate[i + 3], up[i + 3]);
      const T v4 = lthn_gelu_gate_mul_one<T>(gate[i + 4], up[i + 4]);
      const T v5 = lthn_gelu_gate_mul_one<T>(gate[i + 5], up[i + 5]);
      const T v6 = lthn_gelu_gate_mul_one<T>(gate[i + 6], up[i + 6]);
      const T v7 = lthn_gelu_gate_mul_one<T>(gate[i + 7], up[i + 7]);
      sum += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
      x_thread[i] = v0;
      x_thread[i + 1] = v1 / 32.0f;
      x_thread[i + 2] = v2 / 4.0f;
      x_thread[i + 3] = v3 / 128.0f;
      x_thread[i + 4] = v4 / 16.0f;
      x_thread[i + 5] = v5 / 2.0f;
      x_thread[i + 6] = v6 / 64.0f;
      x_thread[i + 7] = v7 / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      const T v1 = lthn_gelu_gate_mul_one<T>(gate[i + 1], up[i + 1]);
      const T v2 = lthn_gelu_gate_mul_one<T>(gate[i + 2], up[i + 2]);
      const T v3 = lthn_gelu_gate_mul_one<T>(gate[i + 3], up[i + 3]);
      sum += v0 + v1 + v2 + v3;
      x_thread[i] = v0;
      x_thread[i + 1] = v1 / 64.0f;
      x_thread[i + 2] = v2 / 16.0f;
      x_thread[i + 3] = v3 / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      sum += v0;
      x_thread[i] = v0;
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_gelu_safe(
    const device T* gate,
    const device T* up,
    thread U* x_thread,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      const T v1 = lthn_gelu_gate_mul_one<T>(gate[i + 1], up[i + 1]);
      const T v2 = lthn_gelu_gate_mul_one<T>(gate[i + 2], up[i + 2]);
      const T v3 = lthn_gelu_gate_mul_one<T>(gate[i + 3], up[i + 3]);
      sum += v0 + v1 + v2 + v3;
      x_thread[i] = v0;
      x_thread[i + 1] = v1 / 4.0f;
      x_thread[i + 2] = v2 / 16.0f;
      x_thread[i + 3] = v3 / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      const T v1 = lthn_gelu_gate_mul_one<T>(gate[i + 1], up[i + 1]);
      const T v2 = lthn_gelu_gate_mul_one<T>(gate[i + 2], up[i + 2]);
      const T v3 = lthn_gelu_gate_mul_one<T>(gate[i + 3], up[i + 3]);
      const T v4 = lthn_gelu_gate_mul_one<T>(gate[i + 4], up[i + 4]);
      const T v5 = lthn_gelu_gate_mul_one<T>(gate[i + 5], up[i + 5]);
      const T v6 = lthn_gelu_gate_mul_one<T>(gate[i + 6], up[i + 6]);
      const T v7 = lthn_gelu_gate_mul_one<T>(gate[i + 7], up[i + 7]);
      sum += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
      x_thread[i] = v0;
      x_thread[i + 1] = v1 / 8.0f;
      x_thread[i + 2] = v2 / 64.0f;
      x_thread[i + 3] = v3 / 2.0f;
      x_thread[i + 4] = v4 / 16.0f;
      x_thread[i + 5] = v5 / 128.0f;
      x_thread[i + 6] = v6 / 4.0f;
      x_thread[i + 7] = v7 / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      const T v1 = lthn_gelu_gate_mul_one<T>(gate[i + 1], up[i + 1]);
      const T v2 = lthn_gelu_gate_mul_one<T>(gate[i + 2], up[i + 2]);
      const T v3 = lthn_gelu_gate_mul_one<T>(gate[i + 3], up[i + 3]);
      sum += v0 + v1 + v2 + v3;
      x_thread[i] = v0;
      x_thread[i + 1] = v1 / 16.0f;
      x_thread[i + 2] = v2 / 256.0f;
      x_thread[i + 3] = v3 / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      const T v1 = lthn_gelu_gate_mul_one<T>(gate[i + 1], up[i + 1]);
      const T v2 = lthn_gelu_gate_mul_one<T>(gate[i + 2], up[i + 2]);
      const T v3 = lthn_gelu_gate_mul_one<T>(gate[i + 3], up[i + 3]);
      const T v4 = lthn_gelu_gate_mul_one<T>(gate[i + 4], up[i + 4]);
      const T v5 = lthn_gelu_gate_mul_one<T>(gate[i + 5], up[i + 5]);
      const T v6 = lthn_gelu_gate_mul_one<T>(gate[i + 6], up[i + 6]);
      const T v7 = lthn_gelu_gate_mul_one<T>(gate[i + 7], up[i + 7]);
      sum += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
      x_thread[i] = v0;
      x_thread[i + 1] = v1 / 32.0f;
      x_thread[i + 2] = v2 / 4.0f;
      x_thread[i + 3] = v3 / 128.0f;
      x_thread[i + 4] = v4 / 16.0f;
      x_thread[i + 5] = v5 / 2.0f;
      x_thread[i + 6] = v6 / 64.0f;
      x_thread[i + 7] = v7 / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      const T v1 = lthn_gelu_gate_mul_one<T>(gate[i + 1], up[i + 1]);
      const T v2 = lthn_gelu_gate_mul_one<T>(gate[i + 2], up[i + 2]);
      const T v3 = lthn_gelu_gate_mul_one<T>(gate[i + 3], up[i + 3]);
      sum += v0 + v1 + v2 + v3;
      x_thread[i] = v0;
      x_thread[i + 1] = v1 / 64.0f;
      x_thread[i + 2] = v2 / 16.0f;
      x_thread[i + 3] = v3 / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      const T v0 = lthn_gelu_gate_mul_one<T>(gate[i], up[i]);
      sum += v0;
      x_thread[i] = v0;
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

// qmv_gelu_impl is qmv_impl verbatim with x replaced by (gate, up) — every
// pointer walk that advanced x advances both, and the two load calls become
// their gelu-fused twins. Same guards, same used_out_row redo branch, same
// qdot arithmetic, same simd_sum + bf16 store.
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_gelu_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* gate,
    const device T* up,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    gate += tid.x * in_vec_size + simd_lid * values_per_thread;
    up += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum =
          load_vector_gelu<T, U, values_per_thread, bits>(gate, up, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      gate += block_size;
      up += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_gelu_safe<T, U, values_per_thread, bits>(
          gate, up, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    gate += tid.x * in_vec_size + simd_lid * values_per_thread;
    up += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum =
          load_vector_gelu<T, U, values_per_thread, bits>(gate, up, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      gate += block_size;
      up += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_gelu_safe<T, U, values_per_thread, bits>(
          gate, up, x_thread, remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}
