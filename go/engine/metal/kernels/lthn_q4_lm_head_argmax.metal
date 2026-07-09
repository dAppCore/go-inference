// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant constexpr uint lthn_bf16_lm_head_rows_per_tile = 8;
constant constexpr uint lthn_bf16_logits_argmax_rows_per_tile = 256;
constant constexpr uint lthn_topk_max_k = 64;
constant constexpr uint lthn_q4_lm_head_topk_simdgroups = 4;
constant constexpr uint lthn_q4_lm_head_topk_subtiles = 8;
constant constexpr uint lthn_q4_lm_head_results_per_simdgroup = 4;
constant constexpr uint lthn_q4_lm_head_topk_rows_per_tile =
    lthn_q4_lm_head_topk_simdgroups * lthn_q4_lm_head_results_per_simdgroup * lthn_q4_lm_head_topk_subtiles;
constant constexpr uint lthn_q4_lm_head_values_per_thread = 16;
constant constexpr uint lthn_q4_lm_head_block_size = lthn_q4_lm_head_values_per_thread * 32;
constant constexpr uint lthn_logits_sample_lanes = 256;
constant constexpr uint lthn_logits_topk_lanes = 32;

static inline bool lthn_lm_head_row_suppressed(uint row, device const int* suppress, int suppress_count) {
  for (int i = 0; i < suppress_count; i++) {
    if (suppress[i] == int(row)) {
      return true;
    }
  }
  return false;
}

static inline float lthn_sample_softcap(float score, float soft_cap) {
  score = float(bfloat(score));
  if (soft_cap > 0.0f) {
    score = float(bfloat(score / soft_cap));
    score = float(bfloat(tanh(score)));
    score = float(bfloat(score * soft_cap));
  }
  return score;
}

static inline bool lthn_row_in_history(uint row, device const int* history, int history_count) {
  for (int i = 0; i < history_count; ++i) {
    if (history[i] == int(row)) {
      return true;
    }
  }
  return false;
}

static inline float lthn_repeat_penalized_logit(uint row, float value, device const int* history, int history_count, float penalty) {
  if (penalty <= 1.0f || history_count <= 0 || !lthn_row_in_history(row, history, history_count)) {
    return value;
  }
  if (value > 0.0f) {
    return value / penalty;
  }
  return value * penalty;
}

static inline void lthn_insert_topk(
    thread float* values,
    thread int* indices,
    int top_k,
    float value,
    int index) {
  if (index < 0) {
    return;
  }
  for (int pos = 0; pos < top_k; ++pos) {
    int current = indices[pos];
    if (value > values[pos] || (value == values[pos] && (current < 0 || index < current))) {
      for (int shift = top_k - 1; shift > pos; --shift) {
        values[shift] = values[shift - 1];
        indices[shift] = indices[shift - 1];
      }
      values[pos] = value;
      indices[pos] = index;
      return;
    }
  }
}

static inline bool lthn_ranked_logits_after(float score, int id, float prev_score, int prev_id) {
  return prev_id < 0 || score < prev_score || (score == prev_score && id > prev_id);
}

static inline bool lthn_ranked_logits_better(float score, int id, float best_score, int best_id) {
  return id >= 0 && (best_id < 0 || score > best_score || (score == best_score && id < best_id));
}

static inline void lthn_insert_topk_threadgroup(
    threadgroup float* values,
    threadgroup int* indices,
    int top_k,
    float value,
    int index) {
  if (index < 0) {
    return;
  }
  for (int pos = 0; pos < top_k; ++pos) {
    int current = indices[pos];
    if (value > values[pos] || (value == values[pos] && (current < 0 || index < current))) {
      for (int shift = top_k - 1; shift > pos; --shift) {
        values[shift] = values[shift - 1];
        indices[shift] = indices[shift - 1];
      }
      values[pos] = value;
      indices[pos] = index;
      return;
    }
  }
}

static inline float lthn_q4_lm_head_load_vector4(const device bfloat* x, thread float* x_thread) {
  float sum = 0.0f;
  for (uint i = 0; i < lthn_q4_lm_head_values_per_thread; i += 4u) {
    sum += float(x[i]) + float(x[i + 1u]) + float(x[i + 2u]) + float(x[i + 3u]);
    x_thread[i] = float(x[i]);
    x_thread[i + 1u] = float(x[i + 1u]) / 16.0f;
    x_thread[i + 2u] = float(x[i + 2u]) / 256.0f;
    x_thread[i + 3u] = float(x[i + 3u]) / 4096.0f;
  }
  return sum;
}

static inline float lthn_q4_lm_head_qdot4(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(w);
  float accum = 0.0f;
  for (uint i = 0; i < (lthn_q4_lm_head_values_per_thread / 4u); ++i) {
    uint16_t packed = ws[i];
    accum +=
        x_thread[4u * i] * float(packed & 0x000f) +
        x_thread[4u * i + 1u] * float(packed & 0x00f0) +
        x_thread[4u * i + 2u] * float(packed & 0x0f00) +
        x_thread[4u * i + 3u] * float(packed & 0xf000);
  }
  return scale * accum + sum * bias;
}

// BF16 direct greedy path. It scores up to eight vocab rows per tile against
// one bf16 hidden vector and writes only the tile-local best row. Scores are
// rounded to bf16 before comparison so the selected token matches model.Greedy
// over the existing full BF16 logits row.
kernel void lthn_bf16_lm_head_argmax_tiles_bf16(
    device const bfloat* x       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device float*        values  [[buffer(2)]],
    device int*          indices [[buffer(3)]],
    constant int&        d_model [[buffer(4)]],
    constant int&        vocab   [[buffer(5)]],
    device const int*    suppress [[buffer(6)]],
    constant int&        suppress_count [[buffer(7)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint row_in_tile [[simdgroup_index_in_threadgroup]]) {
  if (d_model <= 0 || vocab <= 0) return;

  uint row = tile * lthn_bf16_lm_head_rows_per_tile + row_in_tile;
  float partial = 0.0f;
  if (row < uint(vocab)) {
    for (uint col = lane; col < uint(d_model); col += 32u) {
      partial += float(x[col]) * float(weight[row * uint(d_model) + col]);
    }
  }

  float score = simd_sum(partial);
  if (lane == 0u) {
    bool masked = row >= uint(vocab) || lthn_lm_head_row_suppressed(row, suppress, suppress_count);
    score = !masked ? float(bfloat(score)) : -INFINITY;
  }

  threadgroup float tile_values[lthn_bf16_lm_head_rows_per_tile];
  threadgroup int tile_indices[lthn_bf16_lm_head_rows_per_tile];
  if (lane == 0u) {
    tile_values[row_in_tile] = score;
    tile_indices[row_in_tile] = (row < uint(vocab)) ? int(row) : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (row_in_tile == 0u && lane == 0u) {
    float best = -INFINITY;
    int best_idx = -1;
    for (uint i = 0; i < lthn_bf16_lm_head_rows_per_tile; i++) {
      float v = tile_values[i];
      int idx = tile_indices[i];
      if (idx >= 0 && (v > best || (v == best && (best_idx < 0 || idx < best_idx)))) {
        best = v;
        best_idx = idx;
      }
    }
    values[tile] = best;
    indices[tile] = best_idx;
  }
}

// Argmax over an already-materialised bf16 logits row. Quant direct greedy uses
// this after the proven MLX affine_qmv_bfloat16_t projection, so native avoids
// full-logit host readback without duplicating q4 dot-product numerics.
kernel void lthn_bf16_logits_argmax_tiles_bf16(
    device const bfloat* logits [[buffer(0)]],
    device float*        values [[buffer(1)]],
    device int*          indices [[buffer(2)]],
    constant int&        vocab [[buffer(3)]],
    device const int*    suppress [[buffer(4)]],
    constant int&        suppress_count [[buffer(5)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  if (vocab <= 0 || lane >= 32u) return;

  uint start = tile * lthn_bf16_logits_argmax_rows_per_tile;
  uint end = min(start + lthn_bf16_logits_argmax_rows_per_tile, uint(vocab));
  float local_best = -INFINITY;
  int local_idx = -1;
  for (uint row = start + lane; row < end; row += 32u) {
    if (lthn_lm_head_row_suppressed(row, suppress, suppress_count)) {
      continue;
    }
    float score = float(logits[row]);
    if (score > local_best || (score == local_best && (local_idx < 0 || int(row) < local_idx))) {
      local_best = score;
      local_idx = int(row);
    }
  }

  threadgroup float lane_values[32];
  threadgroup int lane_indices[32];
  lane_values[lane] = local_best;
  lane_indices[lane] = local_idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane != 0u) return;
  float best = -INFINITY;
  int best_idx = -1;
  for (uint i = 0; i < 32u; i++) {
    float v = lane_values[i];
    int idx = lane_indices[i];
    if (idx >= 0 && (v > best || (v == best && (best_idx < 0 || idx < best_idx)))) {
      best = v;
      best_idx = idx;
    }
  }
  values[tile] = best;
  indices[tile] = best_idx;
}

// Stage 2: merge tile-local candidates to one token id. One threadgroup scans
// the small candidate vector; only the final token id is copied to the host.
kernel void lthn_argmax_merge_f32(
    device const float* values  [[buffer(0)]],
    device const int*   indices [[buffer(1)]],
    device int*         out     [[buffer(2)]],
    constant int&       n       [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]) {
  if (n <= 0 || lane >= 32) return;

  float local_best = -INFINITY;
  int local_idx = -1;
  for (uint i = lane; i < uint(n); i += 32u) {
    float v = values[i];
    int idx = indices[i];
    if (idx >= 0 && (v > local_best || (v == local_best && (local_idx < 0 || idx < local_idx)))) {
      local_best = v;
      local_idx = idx;
    }
  }

  threadgroup float lane_values[32];
  threadgroup int lane_indices[32];
  lane_values[lane] = local_best;
  lane_indices[lane] = local_idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane != 0) return;
  float best = -INFINITY;
  int best_idx = -1;
  for (uint i = 0; i < 32u; i++) {
    float v = lane_values[i];
    int idx = lane_indices[i];
    if (idx >= 0 && (v > best || (v == best && (best_idx < 0 || idx < best_idx)))) {
      best = v;
      best_idx = idx;
    }
  }
  out[0] = best_idx;
}

// K-row merge: one threadgroup per row r reduces its tile slice
// values/indices[r*n .. r*n+n) to out[r] — the K-row head's K merges in a
// single dispatch (K sequential single-row merges each paid a full dispatch
// over the ~32k tile entries).
kernel void lthn_argmax_merge_rows_f32(
    device const float* values  [[buffer(0)]],
    device const int*   indices [[buffer(1)]],
    device int*         out     [[buffer(2)]],
    constant int&       n       [[buffer(3)]],
    uint r    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  if (n <= 0 || lane >= 32) return;
  device const float* row_values = values + r * uint(n);
  device const int* row_indices = indices + r * uint(n);

  float local_best = -INFINITY;
  int local_idx = -1;
  for (uint i = lane; i < uint(n); i += 32u) {
    float v = row_values[i];
    int idx = row_indices[i];
    if (idx >= 0 && (v > local_best || (v == local_best && (local_idx < 0 || idx < local_idx)))) {
      local_best = v;
      local_idx = idx;
    }
  }

  threadgroup float lane_values[32];
  threadgroup int lane_indices[32];
  lane_values[lane] = local_best;
  lane_indices[lane] = local_idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane != 0) return;
  float best = -INFINITY;
  int best_idx = -1;
  for (uint i = 0; i < 32u; i++) {
    float v = lane_values[i];
    int idx = lane_indices[i];
    if (idx >= 0 && (v > best || (v == best && (best_idx < 0 || idx < best_idx)))) {
      best = v;
      best_idx = idx;
    }
  }
  out[r] = best_idx;
}

// Sampling top-k stage for a BF16 resident head. It emits one candidate per
// vocab row (kept on GPU) after final RMSNorm + optional sampling soft-cap.
// The generic lthn_topk_merge_f32 stage then keeps only K values for the host.
kernel void lthn_bf16_lm_head_candidates_bf16(
    device const bfloat* x       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device float*        values  [[buffer(2)]],
    device int*          indices [[buffer(3)]],
    constant int&        d_model [[buffer(4)]],
    constant int&        vocab   [[buffer(5)]],
    device const int*    suppress [[buffer(6)]],
    constant int&        suppress_count [[buffer(7)]],
    device const int*    history [[buffer(8)]],
    constant int&        history_count [[buffer(9)]],
    constant float&      repeat_penalty [[buffer(10)]],
    constant float&      soft_cap [[buffer(11)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint row_in_tile [[simdgroup_index_in_threadgroup]]) {
  uint slot = tile * lthn_bf16_lm_head_rows_per_tile + row_in_tile;
  if (d_model <= 0 || vocab <= 0) {
    if (lane == 0u) {
      values[slot] = -INFINITY;
      indices[slot] = -1;
    }
    return;
  }

  uint row = slot;
  float partial = 0.0f;
  if (row < uint(vocab)) {
    for (uint col = lane; col < uint(d_model); col += 32u) {
      partial += float(x[col]) * float(weight[row * uint(d_model) + col]);
    }
  }

  float score = simd_sum(partial);
  if (lane == 0u) {
    bool masked = row >= uint(vocab) || lthn_lm_head_row_suppressed(row, suppress, suppress_count);
    float sampled = lthn_sample_softcap(score, soft_cap);
    values[slot] = !masked ? lthn_repeat_penalized_logit(row, sampled, history, history_count, repeat_penalty) : -INFINITY;
    indices[slot] = !masked ? int(row) : -1;
  }
}

// Sampling top-k stage for already materialised BF16 logits. Quant heads use
// the proven qmv kernel into scratch, then this row candidate pass, so the full
// vocab row still never crosses to the host.
kernel void lthn_bf16_logits_candidates_bf16(
    device const bfloat* logits [[buffer(0)]],
    device float*        values [[buffer(1)]],
    device int*          indices [[buffer(2)]],
    constant int&        vocab [[buffer(3)]],
    device const int*    suppress [[buffer(4)]],
    constant int&        suppress_count [[buffer(5)]],
    constant float&      soft_cap [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= uint(vocab)) {
    return;
  }
  bool masked = lthn_lm_head_row_suppressed(row, suppress, suppress_count);
  values[row] = !masked ? lthn_sample_softcap(float(logits[row]), soft_cap) : -INFINITY;
  indices[row] = !masked ? int(row) : -1;
}

// Quant sampled path: after qmv writes BF16 logits to scratch, reduce each
// 256-row tile to top-k candidates before the global merge. This keeps the
// expensive merge over tile_count*K instead of the full vocab.
kernel void lthn_bf16_logits_topk_tiles_bf16(
    device const bfloat* logits [[buffer(0)]],
    device float*        values [[buffer(1)]],
    device int*          indices [[buffer(2)]],
    constant int&        vocab [[buffer(3)]],
    device const int*    suppress [[buffer(4)]],
    constant int&        suppress_count [[buffer(5)]],
    device const int*    history [[buffer(6)]],
    constant int&        history_count [[buffer(7)]],
    constant float&      repeat_penalty [[buffer(8)]],
    constant float&      soft_cap [[buffer(9)]],
    constant int&        top_k [[buffer(10)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  if (lane >= 32u || vocab <= 0 || top_k <= 0 || top_k > int(lthn_topk_max_k)) {
    return;
  }

  uint start = tile * lthn_bf16_logits_argmax_rows_per_tile;
  uint end = min(start + lthn_bf16_logits_argmax_rows_per_tile, uint(vocab));
  float local_values[lthn_topk_max_k];
  int local_indices[lthn_topk_max_k];
  for (int i = 0; i < int(lthn_topk_max_k); ++i) {
    local_values[i] = -INFINITY;
    local_indices[i] = -1;
  }

  for (uint row = start + lane; row < end; row += 32u) {
    if (lthn_lm_head_row_suppressed(row, suppress, suppress_count)) {
      continue;
    }
    float sampled = lthn_sample_softcap(float(logits[row]), soft_cap);
    sampled = lthn_repeat_penalized_logit(row, sampled, history, history_count, repeat_penalty);
    lthn_insert_topk(local_values, local_indices, top_k, sampled, int(row));
  }

  threadgroup float group_values[32 * lthn_topk_max_k];
  threadgroup int group_indices[32 * lthn_topk_max_k];
  for (int i = 0; i < int(lthn_topk_max_k); ++i) {
    uint off = lane * lthn_topk_max_k + uint(i);
    group_values[off] = local_values[i];
    group_indices[off] = local_indices[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane != 0u) {
    return;
  }

  float merged_values[lthn_topk_max_k];
  int merged_indices[lthn_topk_max_k];
  for (int i = 0; i < int(lthn_topk_max_k); ++i) {
    merged_values[i] = -INFINITY;
    merged_indices[i] = -1;
  }
  for (int lane_i = 0; lane_i < 32; ++lane_i) {
    for (int pos = 0; pos < top_k; ++pos) {
      int off = lane_i * int(lthn_topk_max_k) + pos;
      lthn_insert_topk(merged_values, merged_indices, top_k, group_values[off], group_indices[off]);
    }
  }
  for (int i = 0; i < top_k; ++i) {
    uint out = tile * uint(top_k) + uint(i);
    values[out] = merged_values[i];
    indices[out] = merged_indices[i];
  }
}

// Fused quant LM-head sampled path copied from pkg/metal's q4 top-k bridge,
// adapted to native buffers and the existing native merge stage. It computes a
// tile-local TopK straight from packed q4 weights, so eligible quant heads avoid
// materialising a full BF16 vocab row before sampling.
kernel void lthn_q4_lm_head_topk_tiles_bf16(
    device const bfloat* x [[buffer(0)]],
    device const uint8_t* w [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* values [[buffer(4)]],
    device int* indices [[buffer(5)]],
    constant int& d_model [[buffer(6)]],
    constant int& vocab [[buffer(7)]],
    constant int& group_size [[buffer(8)]],
    device const int* suppress [[buffer(9)]],
    constant int& suppress_count [[buffer(10)]],
    device const int* history [[buffer(11)]],
    constant int& history_count [[buffer(12)]],
    constant float& repeat_penalty [[buffer(13)]],
    constant float& soft_cap [[buffer(14)]],
    constant int& top_k [[buffer(15)]],
    constant int& candidate_count [[buffer(16)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (d_model <= 0 || vocab <= 0 || top_k <= 0 || top_k > int(lthn_topk_max_k)) {
    return;
  }
  if (candidate_count <= 0 || candidate_count > top_k) {
    return;
  }
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    return;
  }
  if ((d_model % int(lthn_q4_lm_head_block_size)) != 0 || (d_model % group_size) != 0) {
    return;
  }

  const int row_packed = d_model / 2;
  const int row_sb = d_model / group_size;
  const int scale_step_per_thread = group_size / int(lthn_q4_lm_head_values_per_thread);
  const int tile_base = int(tile) * int(lthn_q4_lm_head_topk_rows_per_tile);

  threadgroup float top_values[lthn_topk_max_k];
  threadgroup int top_indices[lthn_topk_max_k];
  threadgroup float cand_values[lthn_q4_lm_head_topk_simdgroups * lthn_q4_lm_head_results_per_simdgroup];
  threadgroup int cand_indices[lthn_q4_lm_head_topk_simdgroups * lthn_q4_lm_head_results_per_simdgroup];

  if (simd_gid == 0u && simd_lid == 0u) {
    for (int i = 0; i < top_k; ++i) {
      top_values[i] = -INFINITY;
      top_indices[i] = -1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float x_thread[lthn_q4_lm_head_values_per_thread];
  for (uint subtile = 0; subtile < lthn_q4_lm_head_topk_subtiles; ++subtile) {
    int out_row = tile_base + int(subtile) *
        int(lthn_q4_lm_head_topk_simdgroups * lthn_q4_lm_head_results_per_simdgroup) +
        int(simd_gid) * int(lthn_q4_lm_head_results_per_simdgroup);
    float result[lthn_q4_lm_head_results_per_simdgroup] = {0.0f};

    const device uint8_t* w_base = w + out_row * row_packed + int(simd_lid) * 8;
    const device bfloat* scales_base = scales + out_row * row_sb + int(simd_lid) / scale_step_per_thread;
    const device bfloat* biases_base = biases + out_row * row_sb + int(simd_lid) / scale_step_per_thread;

    for (int k_block = 0; k_block < d_model; k_block += int(lthn_q4_lm_head_block_size)) {
      const device bfloat* x_ptr = x + k_block + int(simd_lid) * int(lthn_q4_lm_head_values_per_thread);
      float x_sum = lthn_q4_lm_head_load_vector4(x_ptr, x_thread);
      const device uint8_t* w_block = w_base + k_block / 2;
      const device bfloat* scales_block = scales_base + k_block / group_size;
      const device bfloat* biases_block = biases_base + k_block / group_size;

      for (uint row = 0; row < lthn_q4_lm_head_results_per_simdgroup; ++row) {
        int n = out_row + int(row);
        if (n < vocab) {
          const device uint8_t* wl = w_block + int(row) * row_packed;
          const device bfloat* sl = scales_block + int(row) * row_sb;
          const device bfloat* bl = biases_block + int(row) * row_sb;
          result[row] += lthn_q4_lm_head_qdot4(wl, x_thread, float(sl[0]), float(bl[0]), x_sum);
        }
      }
    }

    float summed[lthn_q4_lm_head_results_per_simdgroup];
    for (uint row = 0; row < lthn_q4_lm_head_results_per_simdgroup; ++row) {
      summed[row] = simd_sum(result[row]);
    }
    if (simd_lid == 0u) {
      for (uint row = 0; row < lthn_q4_lm_head_results_per_simdgroup; ++row) {
        int cand = int(simd_gid) * int(lthn_q4_lm_head_results_per_simdgroup) + int(row);
        int n = out_row + int(row);
        bool valid = n < vocab && !lthn_lm_head_row_suppressed(uint(n), suppress, suppress_count);
        float sampled = lthn_sample_softcap(summed[row], soft_cap);
        sampled = lthn_repeat_penalized_logit(uint(n), sampled, history, history_count, repeat_penalty);
        cand_values[cand] = valid ? sampled : -INFINITY;
        cand_indices[cand] = valid ? n : -1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0u && simd_lid == 0u) {
      for (uint cand = 0; cand < lthn_q4_lm_head_topk_simdgroups * lthn_q4_lm_head_results_per_simdgroup; ++cand) {
        lthn_insert_topk_threadgroup(top_values, top_indices, top_k, cand_values[cand], cand_indices[cand]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_gid == 0u && simd_lid == 0u) {
    for (int i = 0; i < candidate_count; ++i) {
      uint out = tile * uint(candidate_count) + uint(i);
      values[out] = top_values[i];
      indices[out] = top_indices[i];
    }
  }
}

// Merge an on-GPU candidate vector down to sorted top-k values/indices. One
// threadgroup scans the candidate row with 32 lanes, each lane holding a local
// top-k, then lane 0 merges the 32 local windows. K is capped at 64 to match
// pkg/metal's q4 lm-head top-k contract.
kernel void lthn_topk_merge_f32(
    device const float* values [[buffer(0)]],
    device const int*   indices [[buffer(1)]],
    device float*       out_values [[buffer(2)]],
    device int*         out_indices [[buffer(3)]],
    constant int&       n [[buffer(4)]],
    constant int&       top_k [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]]) {
  if (lane >= 32u || n <= 0 || top_k <= 0 || top_k > int(lthn_topk_max_k)) {
    return;
  }

  float local_values[lthn_topk_max_k];
  int local_indices[lthn_topk_max_k];
  for (int i = 0; i < int(lthn_topk_max_k); ++i) {
    local_values[i] = -INFINITY;
    local_indices[i] = -1;
  }

  for (uint i = lane; i < uint(n); i += 32u) {
    lthn_insert_topk(local_values, local_indices, top_k, values[i], indices[i]);
  }

  threadgroup float group_values[32 * lthn_topk_max_k];
  threadgroup int group_indices[32 * lthn_topk_max_k];
  for (int i = 0; i < int(lthn_topk_max_k); ++i) {
    uint off = lane * lthn_topk_max_k + uint(i);
    group_values[off] = local_values[i];
    group_indices[off] = local_indices[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane != 0u) {
    return;
  }

  float merged_values[lthn_topk_max_k];
  int merged_indices[lthn_topk_max_k];
  for (int i = 0; i < int(lthn_topk_max_k); ++i) {
    merged_values[i] = -INFINITY;
    merged_indices[i] = -1;
  }
  for (int lane_i = 0; lane_i < 32; ++lane_i) {
    for (int pos = 0; pos < top_k; ++pos) {
      int off = lane_i * int(lthn_topk_max_k) + pos;
      lthn_insert_topk(merged_values, merged_indices, top_k, group_values[off], group_indices[off]);
    }
  }
  for (int i = 0; i < top_k; ++i) {
    out_values[i] = merged_values[i];
    out_indices[i] = merged_indices[i];
  }
}

struct lthn_topk_sample_params {
  int n;
  int top_k;
  float temperature;
  float top_p;
  float min_p;
  float draw;
};

struct lthn_logits_sample_params {
  int vocab;
  int suppress_count;
  int history_count;
  int top_k;
  float temperature;
  float top_p;
  float min_p;
  float draw;
  float repeat_penalty;
};

static inline int lthn_sample_topk_window_raw(
    thread const float* values,
    thread const int* indices,
    int top_k,
    float temperature,
    float top_p,
    float min_p,
    float draw) {
  if (top_k <= 0 || top_k > int(lthn_topk_max_k) || temperature <= 0.0f) {
    return -1;
  }

  float scaled[lthn_topk_max_k];
  float probs[lthn_topk_max_k];
  float max_v = -INFINITY;
  int valid = 0;
  for (int i = 0; i < int(lthn_topk_max_k); ++i) {
    scaled[i] = -INFINITY;
    probs[i] = 0.0f;
  }
  for (int i = 0; i < top_k; ++i) {
    int id = indices[i];
    if (id < 0) {
      continue;
    }
    float v = values[i] / temperature;
    scaled[i] = v;
    max_v = max(max_v, v);
    valid++;
  }
  if (valid == 0) {
    return -1;
  }

  float total = 0.0f;
  for (int i = 0; i < top_k; ++i) {
    if (indices[i] < 0) {
      continue;
    }
    float p = exp(scaled[i] - max_v);
    probs[i] = p;
    total += p;
  }

  int keep = top_k;
  if (top_p > 0.0f && top_p < 1.0f) {
    float cum = 0.0f;
    int n = 0;
    for (n = 0; n < keep; ++n) {
      if (indices[n] < 0) {
        break;
      }
      cum += probs[n];
      if (cum >= top_p * total) {
        n++;
        break;
      }
    }
    keep = max(n, 1);
  }
  if (min_p > 0.0f && keep > 0) {
    float threshold = probs[0] * min_p;
    int n = 0;
    while (n < keep && indices[n] >= 0 && probs[n] >= threshold) {
      n++;
    }
    if (n > 0) {
      keep = n;
    }
  }

  float kept_sum = 0.0f;
  for (int i = 0; i < keep; ++i) {
    if (indices[i] >= 0) {
      kept_sum += probs[i];
    }
  }
  if (kept_sum <= 0.0f) {
    return indices[0];
  }

  float target = clamp(draw, 0.0f, 0.99999994f) * kept_sum;
  float acc = 0.0f;
  int fallback = indices[0];
  for (int i = 0; i < keep; ++i) {
    int id = indices[i];
    if (id < 0) {
      continue;
    }
    fallback = id;
    acc += probs[i];
    if (acc >= target) {
      return id;
    }
  }
  return fallback;
}

static inline int lthn_sample_topk_window(
    thread const float* values,
    thread const int* indices,
    constant lthn_topk_sample_params& params) {
  return lthn_sample_topk_window_raw(
      values, indices, params.top_k, params.temperature, params.top_p, params.min_p, params.draw);
}

// Merge candidate rows down to TopK and sample that window in one dispatch.
// This is the sampled native-session route: it preserves the device-only TopK
// reduction while avoiding a second sampler dispatch and the K-logit host readback.
kernel void lthn_topk_merge_sample_f32(
    device const float* values [[buffer(0)]],
    device const int*   indices [[buffer(1)]],
    device int*         out [[buffer(2)]],
    constant lthn_topk_sample_params& params [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]) {
  const int n = params.n;
  const int top_k = params.top_k;
  if (lane >= 32u || n <= 0 || top_k <= 0 || top_k > int(lthn_topk_max_k)) {
    if (lane == 0u) {
      out[0] = -1;
    }
    return;
  }

  float local_values[lthn_topk_max_k];
  int local_indices[lthn_topk_max_k];
  for (int i = 0; i < int(lthn_topk_max_k); ++i) {
    local_values[i] = -INFINITY;
    local_indices[i] = -1;
  }

  for (uint i = lane; i < uint(n); i += 32u) {
    lthn_insert_topk(local_values, local_indices, top_k, values[i], indices[i]);
  }

  threadgroup float group_values[32 * lthn_topk_max_k];
  threadgroup int group_indices[32 * lthn_topk_max_k];
  for (int i = 0; i < int(lthn_topk_max_k); ++i) {
    uint off = lane * lthn_topk_max_k + uint(i);
    group_values[off] = local_values[i];
    group_indices[off] = local_indices[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane != 0u) {
    return;
  }

  float merged_values[lthn_topk_max_k];
  int merged_indices[lthn_topk_max_k];
  for (int i = 0; i < int(lthn_topk_max_k); ++i) {
    merged_values[i] = -INFINITY;
    merged_indices[i] = -1;
  }
  for (int lane_i = 0; lane_i < 32; ++lane_i) {
    for (int pos = 0; pos < top_k; ++pos) {
      int off = lane_i * int(lthn_topk_max_k) + pos;
      lthn_insert_topk(merged_values, merged_indices, top_k, group_values[off], group_indices[off]);
    }
  }
  out[0] = lthn_sample_topk_window(merged_values, merged_indices, params);
}

// Device categorical over a full BF16 vocab row. This is the no-TopK sampled
// native-session route: the resident head materialises logits in shared GPU
// memory, this kernel consumes them in-place, and the host reads one int token.
// TopP is intentionally not handled here because it needs rank ordering over
// the distribution; the TopK path uses lthn_topk_merge_sample_f32 for that.
kernel void lthn_logits_sample_bf16(
    device const bfloat* logits [[buffer(0)]],
    device const int* suppress [[buffer(1)]],
    device const int* history [[buffer(2)]],
    device int* out [[buffer(3)]],
    constant lthn_logits_sample_params& params [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]]) {
  const int vocab = params.vocab;
  if (lane >= lthn_logits_sample_lanes || vocab <= 0 || params.temperature <= 0.0f) {
    if (lane == 0u) {
      out[0] = -1;
    }
    return;
  }

  threadgroup float block_values[lthn_logits_sample_lanes];
  threadgroup int block_indices[lthn_logits_sample_lanes];
  threadgroup float group_top_values[lthn_logits_topk_lanes * lthn_topk_max_k];
  threadgroup int group_top_indices[lthn_logits_topk_lanes * lthn_topk_max_k];
  threadgroup float shared_max;
  threadgroup float shared_total;
  threadgroup float shared_target;
  threadgroup float shared_prev_score;
  threadgroup float shared_acc;
  threadgroup float shared_final_sum;
  threadgroup float shared_first_weight;
  threadgroup int shared_prev_id;
  threadgroup int shared_keep_count;
  threadgroup int shared_min_keep_count;
  threadgroup int shared_final_keep_count;
  threadgroup int chosen_lane;
  threadgroup int shared_done;
  if (lane == 0u) {
    shared_max = -INFINITY;
    shared_total = 0.0f;
    shared_target = 0.0f;
    shared_prev_score = INFINITY;
    shared_acc = 0.0f;
    shared_final_sum = 0.0f;
    shared_first_weight = 0.0f;
    shared_prev_id = -1;
    shared_keep_count = 0;
    shared_min_keep_count = 0;
    shared_final_keep_count = 0;
    chosen_lane = -1;
    shared_done = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int chunk = (vocab + int(lthn_logits_sample_lanes) - 1) / int(lthn_logits_sample_lanes);
  const int start = int(lane) * chunk;
  const int end = min(start + chunk, vocab);

  const int top_k = params.top_k;
  if (top_k > 0) {
    const bool full_vocab_top_p =
        top_k == vocab && params.top_p > 0.0f && params.top_p < 1.0f;
    if (top_k > int(lthn_topk_max_k)) {
      if (!full_vocab_top_p) {
        if (lane == 0u) {
          out[0] = -1;
        }
        return;
      }
    } else if (top_k > vocab) {
      if (lane == 0u) {
        out[0] = -1;
      }
      return;
    }
    if (!full_vocab_top_p) {
      if (lane < lthn_logits_topk_lanes) {
        const int topk_chunk = (vocab + int(lthn_logits_topk_lanes) - 1) / int(lthn_logits_topk_lanes);
        const int topk_start = int(lane) * topk_chunk;
        const int topk_end = min(topk_start + topk_chunk, vocab);
        float local_values[lthn_topk_max_k];
        int local_indices[lthn_topk_max_k];
        for (int i = 0; i < int(lthn_topk_max_k); ++i) {
          local_values[i] = -INFINITY;
          local_indices[i] = -1;
        }
        for (int i = topk_start; i < topk_end; ++i) {
          if (lthn_lm_head_row_suppressed(uint(i), suppress, params.suppress_count)) {
            continue;
          }
          float raw = lthn_repeat_penalized_logit(uint(i), float(logits[i]), history, params.history_count, params.repeat_penalty);
          lthn_insert_topk(local_values, local_indices, top_k, raw, i);
        }
        for (int i = 0; i < int(lthn_topk_max_k); ++i) {
          uint off = lane * lthn_topk_max_k + uint(i);
          group_top_values[off] = local_values[i];
          group_top_indices[off] = local_indices[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (lane != 0u) {
        return;
      }
      float merged_values[lthn_topk_max_k];
      int merged_indices[lthn_topk_max_k];
      for (int i = 0; i < int(lthn_topk_max_k); ++i) {
        merged_values[i] = -INFINITY;
        merged_indices[i] = -1;
      }
      for (uint lane_i = 0u; lane_i < lthn_logits_topk_lanes; ++lane_i) {
        for (int pos = 0; pos < top_k; ++pos) {
          uint off = lane_i * lthn_topk_max_k + uint(pos);
          lthn_insert_topk(merged_values, merged_indices, top_k, group_top_values[off], group_top_indices[off]);
        }
      }
      out[0] = lthn_sample_topk_window_raw(
          merged_values, merged_indices, top_k, params.temperature, params.top_p, params.min_p, params.draw);
      return;
    }
  }

  const bool full_vocab_top_p =
      top_k == vocab && params.top_p > 0.0f && params.top_p < 1.0f;

  float local_max = -INFINITY;
  for (int i = start; i < end; ++i) {
    if (lthn_lm_head_row_suppressed(uint(i), suppress, params.suppress_count)) {
      continue;
    }
    float raw = lthn_repeat_penalized_logit(uint(i), float(logits[i]), history, params.history_count, params.repeat_penalty);
    float v = raw / params.temperature;
    local_max = max(local_max, v);
  }
  block_values[lane] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane == 0u) {
    float max_v = -INFINITY;
    for (uint i = 0; i < lthn_logits_sample_lanes; ++i) {
      max_v = max(max_v, block_values[i]);
    }
    shared_max = max_v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float local_sum = 0.0f;
  for (int i = start; i < end; ++i) {
    if (lthn_lm_head_row_suppressed(uint(i), suppress, params.suppress_count)) {
      continue;
    }
    float raw = lthn_repeat_penalized_logit(uint(i), float(logits[i]), history, params.history_count, params.repeat_penalty);
    float p = exp(raw / params.temperature - shared_max);
    if (!full_vocab_top_p && params.min_p > 0.0f && p < params.min_p) {
      continue;
    }
    local_sum += p;
  }
  block_values[lane] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane == 0u) {
    float total = 0.0f;
    for (uint i = 0; i < lthn_logits_sample_lanes; ++i) {
      total += block_values[i];
    }
    shared_total = total;
    if (total <= 0.0f) {
      out[0] = -1;
      chosen_lane = -1;
      shared_target = 0.0f;
    } else {
      float target = clamp(params.draw, 0.0f, 0.99999994f) * total;
      float prefix = 0.0f;
      int pick_lane = int(lthn_logits_sample_lanes) - 1;
      for (uint i = 0; i < lthn_logits_sample_lanes; ++i) {
        float next = prefix + block_values[i];
        if (next >= target) {
          pick_lane = int(i);
          shared_target = target - prefix;
          break;
        }
        prefix = next;
      }
      chosen_lane = pick_lane;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (full_vocab_top_p) {
    if (lane == 0u) {
      shared_prev_score = INFINITY;
      shared_prev_id = -1;
      shared_keep_count = 0;
      shared_min_keep_count = 0;
      shared_acc = 0.0f;
      shared_final_sum = 0.0f;
      shared_first_weight = 0.0f;
      shared_target = params.top_p * shared_total;
      shared_done = shared_total <= 0.0f ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int rank = 0; rank < vocab; ++rank) {
      if (shared_done != 0) {
        break;
      }
      float local_best = -INFINITY;
      int local_id = -1;
      const float prev_score = shared_prev_score;
      const int prev_id = shared_prev_id;
      for (int i = start; i < end; ++i) {
        if (lthn_lm_head_row_suppressed(uint(i), suppress, params.suppress_count)) {
          continue;
        }
        float raw = lthn_repeat_penalized_logit(uint(i), float(logits[i]), history, params.history_count, params.repeat_penalty);
        if (!lthn_ranked_logits_after(raw, i, prev_score, prev_id)) {
          continue;
        }
        if (lthn_ranked_logits_better(raw, i, local_best, local_id)) {
          local_best = raw;
          local_id = i;
        }
      }
      block_values[lane] = local_best;
      block_indices[lane] = local_id;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (lane == 0u) {
        float best_score = -INFINITY;
        int best_id = -1;
        for (uint i = 0; i < lthn_logits_sample_lanes; ++i) {
          if (lthn_ranked_logits_better(block_values[i], block_indices[i], best_score, best_id)) {
            best_score = block_values[i];
            best_id = block_indices[i];
          }
        }
        if (best_id < 0) {
          shared_done = 1;
        } else {
          float weight = exp(best_score / params.temperature - shared_max);
          if (shared_keep_count == 0) {
            shared_first_weight = weight;
          }
          shared_acc += weight;
          shared_keep_count += 1;
          if (params.min_p > 0.0f && weight >= shared_first_weight * params.min_p) {
            shared_final_sum += weight;
            shared_min_keep_count += 1;
          }
          shared_prev_score = best_score;
          shared_prev_id = best_id;
          if (shared_acc >= shared_target) {
            shared_done = 1;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
      shared_final_keep_count = shared_keep_count;
      if (params.min_p > 0.0f && shared_min_keep_count > 0) {
        shared_final_keep_count = shared_min_keep_count;
      } else {
        shared_final_sum = shared_acc;
      }
      shared_prev_score = INFINITY;
      shared_prev_id = -1;
      shared_acc = 0.0f;
      shared_target = clamp(params.draw, 0.0f, 0.99999994f) * shared_final_sum;
      shared_done = shared_final_keep_count <= 0 || shared_final_sum <= 0.0f ? 1 : 0;
      if (shared_done != 0) {
        out[0] = -1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int rank = 0; rank < vocab; ++rank) {
      if (rank >= shared_final_keep_count || shared_done != 0) {
        break;
      }
      float local_best = -INFINITY;
      int local_id = -1;
      const float prev_score = shared_prev_score;
      const int prev_id = shared_prev_id;
      for (int i = start; i < end; ++i) {
        if (lthn_lm_head_row_suppressed(uint(i), suppress, params.suppress_count)) {
          continue;
        }
        float raw = lthn_repeat_penalized_logit(uint(i), float(logits[i]), history, params.history_count, params.repeat_penalty);
        if (!lthn_ranked_logits_after(raw, i, prev_score, prev_id)) {
          continue;
        }
        if (lthn_ranked_logits_better(raw, i, local_best, local_id)) {
          local_best = raw;
          local_id = i;
        }
      }
      block_values[lane] = local_best;
      block_indices[lane] = local_id;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (lane == 0u) {
        float best_score = -INFINITY;
        int best_id = -1;
        for (uint i = 0; i < lthn_logits_sample_lanes; ++i) {
          if (lthn_ranked_logits_better(block_values[i], block_indices[i], best_score, best_id)) {
            best_score = block_values[i];
            best_id = block_indices[i];
          }
        }
        if (best_id < 0) {
          out[0] = -1;
          shared_done = 1;
        } else {
          float weight = exp(best_score / params.temperature - shared_max);
          shared_acc += weight;
          shared_prev_score = best_score;
          shared_prev_id = best_id;
          out[0] = best_id;
          if (shared_acc >= shared_target) {
            shared_done = 1;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return;
  }

  if (int(lane) != chosen_lane || chosen_lane < 0) {
    return;
  }

  float acc = 0.0f;
  int fallback = -1;
  for (int i = start; i < end; ++i) {
    if (lthn_lm_head_row_suppressed(uint(i), suppress, params.suppress_count)) {
      continue;
    }
    float raw = lthn_repeat_penalized_logit(uint(i), float(logits[i]), history, params.history_count, params.repeat_penalty);
    float p = exp(raw / params.temperature - shared_max);
    if (params.min_p > 0.0f && p < params.min_p) {
      continue;
    }
    fallback = i;
    acc += p;
    if (acc >= shared_target) {
      out[0] = i;
      return;
    }
  }
  out[0] = fallback;
}

// K-row BF16 direct greedy. Scores each tile's vocab rows against K normed
// hidden rows in ONE weight pass — the MTP verify head previously re-read the
// full lm_head weight once per row. K arrives as a function constant so the
// row loop fully unrolls and the accumulators live in registers (a runtime
// bound left the array stack-allocated, stalling the load pipeline at ~82GB/s
// — the first cut of this kernel). Weight and hidden rows read as bfloat4.
// Per-row tile bests write strided as values/indices[r * tile_count + tile];
// scores round to bf16 before comparison exactly as the single-row kernel.
constant constexpr uint lthn_bf16_lm_head_max_batch_rows = 8;
constant uint lthn_lm_head_rows_k [[function_constant(10)]];

kernel void lthn_bf16_lm_head_argmax_tiles_rows_bf16(
    device const bfloat* x       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device float*        values  [[buffer(2)]],
    device int*          indices [[buffer(3)]],
    constant int&        d_model [[buffer(4)]],
    constant int&        vocab   [[buffer(5)]],
    device const int*    suppress [[buffer(6)]],
    constant int&        suppress_count [[buffer(7)]],
    constant int&        tile_count [[buffer(8)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint row_in_tile [[simdgroup_index_in_threadgroup]]) {
  if (d_model <= 0 || vocab <= 0) return;
  const uint k = min(lthn_lm_head_rows_k, lthn_bf16_lm_head_max_batch_rows);

  uint row = tile * lthn_bf16_lm_head_rows_per_tile + row_in_tile;
  float partial[lthn_bf16_lm_head_max_batch_rows];
  for (uint r = 0; r < lthn_bf16_lm_head_max_batch_rows; ++r) partial[r] = 0.0f;

  if (row < uint(vocab)) {
    const uint d4 = uint(d_model) / 4u;
    device const bfloat4* w4 = reinterpret_cast<device const bfloat4*>(weight + row * uint(d_model));
    uint c = lane;
    for (; c + 32u < d4; c += 64u) {
      float4 w0 = float4(w4[c]);
      float4 w1 = float4(w4[c + 32u]);
      for (uint r = 0; r < k; ++r) {
        device const bfloat4* xr = reinterpret_cast<device const bfloat4*>(x + r * uint(d_model));
        partial[r] += dot(float4(xr[c]), w0) + dot(float4(xr[c + 32u]), w1);
      }
    }
    for (; c < d4; c += 32u) {
      float4 w = float4(w4[c]);
      for (uint r = 0; r < k; ++r) {
        float4 xv = float4(reinterpret_cast<device const bfloat4*>(x + r * uint(d_model))[c]);
        partial[r] += dot(xv, w);
      }
    }
    for (uint col = d4 * 4u + lane; col < uint(d_model); col += 32u) {
      float w = float(weight[row * uint(d_model) + col]);
      for (uint r = 0; r < k; ++r) {
        partial[r] += float(x[r * uint(d_model) + col]) * w;
      }
    }
  }

  threadgroup float tile_values[lthn_bf16_lm_head_max_batch_rows][lthn_bf16_lm_head_rows_per_tile];
  threadgroup int tile_indices[lthn_bf16_lm_head_rows_per_tile];
  bool masked = true;
  if (lane == 0u) {
    masked = row >= uint(vocab) || lthn_lm_head_row_suppressed(row, suppress, suppress_count);
    tile_indices[row_in_tile] = (row < uint(vocab)) ? int(row) : -1;
  }
  for (uint r = 0; r < k; ++r) {
    float score = simd_sum(partial[r]);
    if (lane == 0u) {
      tile_values[r][row_in_tile] = !masked ? float(bfloat(score)) : -INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (row_in_tile == 0u && lane == 0u) {
    for (uint r = 0; r < k; ++r) {
      float best = -INFINITY;
      int best_idx = -1;
      for (uint i = 0; i < lthn_bf16_lm_head_rows_per_tile; i++) {
        float v = tile_values[r][i];
        int idx = tile_indices[i];
        if (idx >= 0 && (v > best || (v == best && (best_idx < 0 || idx < best_idx)))) {
          best = v;
          best_idx = idx;
        }
      }
      values[r * uint(tile_count) + tile] = best;
      indices[r * uint(tile_count) + tile] = best_idx;
    }
  }
}
