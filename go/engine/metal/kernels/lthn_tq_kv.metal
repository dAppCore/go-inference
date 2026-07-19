// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

// lthn_tq_kv — the TurboQuant LIVE KV-cache lane (campaign #41 S3): the decode
// cache for GLOBAL attention layers holds TurboQuant CODES (packed Lloyd-Max
// centroid indices in a rotated basis + one f32 norm per row per head — the
// kv/turboquant Q_mse format, the same wire layout turboquant_device.go's S2
// kernels prove), and the decode SDPA reads the codes directly. Four kernel
// families:
//
//   lthn_tq_kv_store_bf16_b{2,3,4}      append: one bf16 staging row
//                                       [kvHeads × d] → per-head packed codes
//                                       + per-head γ (the recorded ICB's
//                                       per-token landing, the q8 store's twin)
//   lthn_tq_rot_rows_bf16 /             y = Π·x (and Πᵀ·x) over bf16 rows —
//   lthn_tq_unrot_rows_bf16             the once-per-step q pre-rotation and
//                                       the once-per-step output unrotation
//                                       (the O(output) fold: rotation cost
//                                       never enters the N-row scan)
//   lthn_sdpa_vector_tq_bf16_{D}        single-pass decode SDPA over codes
//   lthn_sdpa_vector_2pass_1_tq_bf16_{D} 2-pass pass 1 over codes (pass 2 is
//                                       MLX's sdpa_vector_2pass_2 UNCHANGED —
//                                       it merges f32 partials and never
//                                       touches K/V; the q8 precedent)
//
// The math (matching kv/turboquant EncodeQMSE with f32 Π/centroids): a row x
// stores γ = ‖x‖₂ and codes of y = Π·(x/γ) quantised per coordinate to the
// nearest Lloyd-Max centroid. Π is orthogonal, so scores need no unrotation:
//   q·k̃ = q·(γ Πᵀc) = γ·(Πq)·c
// — the SDPA kernels read a PRE-ROTATED q (lthn_tq_rot_rows_bf16, once per
// step per layer) and dot against centroid values in rotated space. The
// V-weighted sum accumulates in rotated space too; ONE Πᵀ per output head
// (lthn_tq_unrot_rows_bf16) lands it back. Nothing O(d²) sits inside the
// key/value scan — the house fusion rule.
//
// SDPA structure is the lthn_sdpa_vector_q8 pair verbatim where it matters:
// same loop order, same fp32 online-softmax stations, same fast::exp, same
// simd_sum points, same cross-simdgroup combine — the ONLY change is each
// lane's K/V element loads unpack BITS-wide indices and look centroids up from
// a threadgroup table (γ applied after simd_sum: the norm is row-wide, unlike
// q8's per-lane group scales). The decode form is fixed: no masks, no sinks,
// not causal, batch 1.
//
// ABI (mirroring the q8 pair with the code planes swapped in; strides are in
// BYTES of the packed-code planes — khs = ceil(d·kBits/8) per head, kss =
// kvHeads·khs per row — and the γ planes derive from them in-kernel):
//   single:  q=0 kCodes=1 vCodes=2 out=3 gqa=4 N=5 khs=6 kss=7 vhs=8 vss=9
//            scale=10 kGammas=11 vGammas=12 kCentroids=13 vCentroids=14
//   2pass_1: q=0 kCodes=1 vCodes=2 partials=3 sums=4 maxs=5 kCentroids=6
//            N=7 khs=8 kss=9 vhs=10 vss=11 scale=12 kGammas=13 vGammas=14
//            vCentroids=15   (blocks = fc 26, as MLX)
//   bits:    kBits = fc 27, vBits = fc 28 (2, 3 or 4 — the S1/S2 widths)
//
// The 2-pass table binds sit at 6 (the MLX ABI's free slot) and 15 — NOT 16:
// the recorded arch ICB is built with maxKernelBufferBindCount = 16, so bind
// indices stop at 15; an index-16 bind records as a silent no-op and the
// kernel reads garbage there (the S3 bring-up bug — encoder paths carry no
// such limit, which is why only the ICB lane broke).
//
// The per-coordinate unpack reads bit-by-bit from the packed row (LSB-first,
// cross-byte — kv/turboquant packBits, identical to lthn_tq_dequant_unrotate's
// own unpack), so no per-lane byte-alignment assumption is needed at any
// (D, BITS) pairing (D=128 b=3 puts lane spans mid-byte).

constant int lthn_tq_blocks [[function_constant(26)]];
constant int lthn_tq_kbits [[function_constant(27)]];
constant int lthn_tq_vbits [[function_constant(28)]];

// LTHN_TQ_KV_CAP is the fixed threadgroup width (and shared-memory span) of the
// row kernels below, sized to the widest head dim the dense family carries
// (gemma4 global_head_dim 512). The S2 encoder kernels cap at 256; this lane
// needs 512, hence its own store kernel rather than a reuse.
#define LTHN_TQ_KV_CAP 512

// tq_unpack_idx reads coordinate c's BITS-wide index from a packed row —
// kv/turboquant packBits layout (bit b of coordinate c at packed bit c·BITS+b).
inline int tq_unpack_idx(const device uint8_t* prow, int c, int BITS) {
  const int pos0 = c * BITS;
  int idx = 0;
  for (int b = 0; b < BITS; b++) {
    const int pos = pos0 + b;
    if ((prow[pos / 8] >> (pos % 8)) & 1) {
      idx |= (1 << b);
    }
  }
  return idx;
}

// ---------------------------------------------------------------------------------------------
// lthn_tq_kv_store_bf16<BITS> — append: one bf16 staging row [heads × d] →
// per-head packed codes + per-head γ. One threadgroup per HEAD; the codes/γ
// buffers bind at the destination CACHE ROW's byte offset (the recorded ICB
// rebinds them per token, exactly as the q8 store's outputs).
//
// ABI: row(0: bf16 [heads×d], fixed staging) pi(1: f32 [d,d] row-major)
//      centroids(2: f32 [1<<BITS] ascending) codes(3: u8, offset-bound)
//      gammas(4: f32 [heads], offset-bound) d(5).
// Dispatch: grid (heads, 1, 1), threadgroup (LTHN_TQ_KV_CAP, 1, 1).
//
// γ/rotate/quantise math matches kernels/lthn_turboquant.metal's
// lthn_tq_rotate_quant (f32 throughout, fixed power-of-two reduction
// zero-padded past d, first-minimum centroid pick) with the input widened from
// bf16; the serial lane-0 byte-major pack is kept for the same
// read-modify-write-race reason documented there.
template <int BITS>
[[kernel]] void lthn_tq_kv_store_bf16(
    const device bf16*    row       [[buffer(0)]],
    const device float*   pi        [[buffer(1)]],
    const device float*   centroids [[buffer(2)]],
    device       uint8_t* codes     [[buffer(3)]],
    device       float*   gammas    [[buffer(4)]],
    constant     int&     d         [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint c    [[thread_position_in_threadgroup]])
{
  constexpr int K = 1 << BITS;
  threadgroup float red[LTHN_TQ_KV_CAP]; // ‖x‖² reduction, then reused as u = x/γ
  threadgroup ushort idxbuf[LTHN_TQ_KV_CAP];

  const bool active = c < (uint)d;
  const device bf16* xrow = row + head * (uint)d;
  const float xc = active ? float(xrow[c]) : 0.0f;

  // γ = ‖x‖₂ over the head's row (fixed power-of-two tree, zero-padded past d)
  red[c] = xc * xc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint off = LTHN_TQ_KV_CAP / 2; off > 0; off >>= 1) {
    if (c < off) {
      red[c] += red[c + off];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float gamma = sqrt(red[0]);
  const float invGamma = (gamma > 0.0f) ? (1.0f / gamma) : 0.0f;
  if (c == 0) {
    gammas[head] = gamma;
  }

  // u = x/γ in shared memory (every thread's Π-row dot below reads ALL of u)
  red[c] = active ? (xc * invGamma) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // y = Π·u, nearest-centroid pick
  if (active) {
    float yc = 0.0f;
    const device float* pirow = pi + c * (uint)d;
    for (int j = 0; j < d; j++) {
      yc += pirow[j] * red[j];
    }
    int best = 0;
    float bestDist = fabs(yc - centroids[0]);
    for (int k = 1; k < K; k++) {
      const float dist = fabs(yc - centroids[k]);
      if (dist < bestDist) {
        bestDist = dist;
        best = k;
      }
    }
    idxbuf[c] = (ushort)best;
  } else {
    idxbuf[c] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // serial byte-major pack (lane 0) — each output byte written exactly once
  if (c == 0) {
    const int bytesPerHead = (d * BITS + 7) / 8;
    device uint8_t* prow = codes + head * (uint)bytesPerHead;
    for (int byteIdx = 0; byteIdx < bytesPerHead; byteIdx++) {
      uint8_t byteVal = 0;
      for (int bit = 0; bit < 8; bit++) {
        const int pos = byteIdx * 8 + bit;
        const int vi = pos / BITS;
        if (vi >= d) {
          break;
        }
        const int b = pos % BITS;
        if ((int(idxbuf[vi]) >> b) & 1) {
          byteVal |= (uint8_t)(1u << bit);
        }
      }
      prow[byteIdx] = byteVal;
    }
  }
}

#define LTHN_TQ_KV_STORE_INSTANTIATE(BITS)                                      \
  template [[host_name("lthn_tq_kv_store_bf16_b" #BITS)]] [[kernel]] void       \
  lthn_tq_kv_store_bf16<BITS>(                                                  \
      const device bf16*, const device float*, const device float*,             \
      device uint8_t*, device float*, constant int&, uint, uint);

LTHN_TQ_KV_STORE_INSTANTIATE(2)
LTHN_TQ_KV_STORE_INSTANTIATE(3)
LTHN_TQ_KV_STORE_INSTANTIATE(4)

// ---------------------------------------------------------------------------------------------
// lthn_tq_rot_rows_bf16<TRANSPOSE> — y = Π·x (TRANSPOSE=false) or y = Πᵀ·x
// (TRANSPOSE=true) over bf16 rows, f32 math. One threadgroup per row. The q
// pre-rotation (once per step per layer, rows = nHeads) and the output
// unrotation (once per step per layer) — both O(rows·d²), OUTSIDE the KV scan.
//
// ABI: in(0: bf16 [rows×d]) pi(1: f32 [d,d]) out(2: bf16 [rows×d]) d(3).
// Dispatch: grid (rows, 1, 1), threadgroup (LTHN_TQ_KV_CAP, 1, 1).
template <bool TRANSPOSE>
[[kernel]] void lthn_tq_rot_rows_bf16_t(
    const device bf16*  in  [[buffer(0)]],
    const device float* pi  [[buffer(1)]],
    device       bf16*  out [[buffer(2)]],
    constant     int&   d   [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint c   [[thread_position_in_threadgroup]])
{
  threadgroup float u[LTHN_TQ_KV_CAP];

  const bool active = c < (uint)d;
  const device bf16* xrow = in + row * (uint)d;
  u[c] = active ? float(xrow[c]) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (active) {
    float yc = 0.0f;
    if (TRANSPOSE) {
      // (Πᵀ·x)[c] = Σ_r Π[r][c]·x[r] — column c, strided by d
      for (int r = 0; r < d; r++) {
        yc += pi[(uint)r * (uint)d + c] * u[r];
      }
    } else {
      // (Π·x)[c] = row c of Π dot x
      const device float* pirow = pi + c * (uint)d;
      for (int j = 0; j < d; j++) {
        yc += pirow[j] * u[j];
      }
    }
    out[row * (uint)d + c] = bf16(yc);
  }
}

template [[host_name("lthn_tq_rot_rows_bf16")]] [[kernel]] void
lthn_tq_rot_rows_bf16_t<false>(
    const device bf16*, const device float*, device bf16*, constant int&, uint, uint);
template [[host_name("lthn_tq_unrot_rows_bf16")]] [[kernel]] void
lthn_tq_rot_rows_bf16_t<true>(
    const device bf16*, const device float*, device bf16*, constant int&, uint, uint);

// ---------------------------------------------------------------------------------------------
// lthn_sdpa_vector_tq<D> — single-pass decode SDPA over TurboQuant codes.
// queries are PRE-ROTATED (Πq per head — lthn_tq_rot_rows_bf16); out is the
// ROTATED-space attention output (the recorded unrotation op follows).
template <int D>
[[kernel]] void lthn_sdpa_vector_tq(
    const device bf16* queries [[buffer(0)]],
    const device uint8_t* keys [[buffer(1)]],
    const device uint8_t* values [[buffer(2)]],
    device bf16* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device float* k_gammas [[buffer(11)]],
    const device float* v_gammas [[buffer(12)]],
    const device float* k_centroids [[buffer(13)]],
    const device float* v_centroids [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  const int inner_k_stride = BN * int(k_seq_stride);
  const int inner_v_stride = BN * int(v_seq_stride);
  // γ planes: one f32 per row per head — strides derive from the byte strides
  // (kss/khs = kvHeads rows-per-row, head stride 1).
  const size_t kg_seq_stride = k_seq_stride / k_head_stride;
  const size_t vg_seq_stride = v_seq_stride / v_head_stride;
  const int inner_kg_stride = BN * int(kg_seq_stride);
  const int inner_vg_stride = BN * int(vg_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];
  threadgroup U kcent[16];
  threadgroup U vcent[16];

  // centroid tables → threadgroup memory (≤16 entries each; low threads load both)
  {
    const uint flat = simd_gid * BD + simd_lid;
    if (flat < uint(1 << lthn_tq_kbits)) {
      kcent[flat] = k_centroids[flat];
    } else if (flat < 16) {
      kcent[flat] = 0;
    }
    if (flat < uint(1 << lthn_tq_vbits)) {
      vcent[flat] = v_centroids[flat];
    } else if (flat < 16) {
      vcent[flat] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // decode form: one query row per head, batch 1
  const int head_idx = tid.x;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride;
  k_gammas += kv_head_idx + simd_gid * kg_seq_stride;
  v_gammas += kv_head_idx + simd_gid * vg_seq_stride;
  out += head_idx * D + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * static_cast<U>(queries[i]);
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -3.0e38f;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < N; i += BN) {
    // score: per-lane centroid dot in rotated space; the row γ is uniform
    // across lanes, applied after the cross-lane simd_sum.
    U partial = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      const int idx = tq_unpack_idx(keys, simd_lid * qk_per_thread + j, lthn_tq_kbits);
      partial += q[j] * kcent[idx];
    }
    U score = simd_sum(partial) * k_gammas[0];

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    const U vsf = exp_score * v_gammas[0];
    for (int j = 0; j < v_per_thread; j++) {
      const int idx = tq_unpack_idx(values, simd_lid * v_per_thread + j, lthn_tq_vbits);
      o[j] = o[j] * factor + vsf * vcent[idx];
    }

    keys += inner_k_stride;
    values += inner_v_stride;
    k_gammas += inner_kg_stride;
    v_gammas += inner_vg_stride;
  }

  // cross-simdgroup combine — MLX sdpa_vector verbatim
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<bf16>(o[i]);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// lthn_sdpa_vector_2pass_1_tq<D> — pass 1 over codes: per-block rotated-space
// partials + running sums/maxs. Pass 2 stays MLX's sdpa_vector_2pass_2
// unchanged (it merges f32 partials and never touches K/V); the merged output
// lands in rotated space and the recorded unrotation op follows it.
template <int D>
[[kernel]] void lthn_sdpa_vector_2pass_1_tq(
    const device bf16* queries [[buffer(0)]],
    const device uint8_t* keys [[buffer(1)]],
    const device uint8_t* values [[buffer(2)]],
    device bf16* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const device float* k_centroids [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device float* k_gammas [[buffer(13)]],
    const device float* v_gammas [[buffer(14)]],
    const device float* v_centroids [[buffer(15)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  const size_t kg_seq_stride = k_seq_stride / k_head_stride;
  const size_t vg_seq_stride = v_seq_stride / v_head_stride;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  threadgroup U kcent[16];
  threadgroup U vcent[16];
  {
    const uint flat = tidtg.y * BD + simd_lid;
    if (flat < uint(1 << lthn_tq_kbits)) {
      kcent[flat] = k_centroids[flat];
    } else if (flat < 16) {
      kcent[flat] = 0;
    }
    // gqa may be 1 (one 32-thread row): fold the V table into the same rows.
    if (flat < uint(1 << lthn_tq_vbits)) {
      vcent[flat] = v_centroids[flat];
    } else if (flat < 16) {
      vcent[flat] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // decode form: batch 1, q_seq_len 1 — grid (nKVHeads, 1, blocks) of
  // (32, gqa, 1), exactly the recorded arch ICB's 2-pass dispatch.
  const int kv_head_idx = tid.x;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int o_offset = q_head_idx;

  queries += o_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + block_idx * k_seq_stride;
  values += kv_head_idx * v_head_stride + block_idx * v_seq_stride;
  k_gammas += kv_head_idx + block_idx * kg_seq_stride;
  v_gammas += kv_head_idx + block_idx * vg_seq_stride;
  out += o_offset * lthn_tq_blocks * D + block_idx * D +
      simd_lid * v_per_thread;
  sums += o_offset * lthn_tq_blocks + block_idx;
  maxs += o_offset * lthn_tq_blocks + block_idx;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * static_cast<U>(queries[i]);
  }

  U max_score = -3.0e38f;
  U sum_exp_score = 0;

  for (int i = block_idx; i < N; i += lthn_tq_blocks) {
    U partial = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      const int idx = tq_unpack_idx(keys, simd_lid * qk_per_thread + j, lthn_tq_kbits);
      partial += q[j] * kcent[idx];
    }
    U score = simd_sum(partial) * k_gammas[0];

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    const U vsf = exp_score * v_gammas[0];
    for (int j = 0; j < v_per_thread; j++) {
      const int idx = tq_unpack_idx(values, simd_lid * v_per_thread + j, lthn_tq_vbits);
      o[j] = o[j] * factor + vsf * vcent[idx];
    }

    keys += lthn_tq_blocks * int(k_seq_stride);
    values += lthn_tq_blocks * int(v_seq_stride);
    k_gammas += lthn_tq_blocks * int(kg_seq_stride);
    v_gammas += lthn_tq_blocks * int(vg_seq_stride);
  }

  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<bf16>(o[i]);
  }
}

#define instantiate_lthn_sdpa_vector_tq(dim)                                \
  template [[host_name("lthn_sdpa_vector_tq_bf16_" #dim)]] [[kernel]]       \
  void lthn_sdpa_vector_tq<dim>(                                            \
      const device bf16*, const device uint8_t*, const device uint8_t*,     \
      device bf16*, const constant int&, const constant int&,               \
      const constant size_t&, const constant size_t&,                       \
      const constant size_t&, const constant size_t&,                       \
      const constant float&, const device float*, const device float*,      \
      const device float*, const device float*, uint3, uint, uint);         \
  template                                                                  \
      [[host_name("lthn_sdpa_vector_2pass_1_tq_bf16_" #dim)]] [[kernel]]    \
  void lthn_sdpa_vector_2pass_1_tq<dim>(                                    \
      const device bf16*, const device uint8_t*, const device uint8_t*,     \
      device bf16*, device float*, device float*, const device float*,     \
      const constant int&,                                                  \
      const constant size_t&, const constant size_t&,                       \
      const constant size_t&, const constant size_t&,                       \
      const constant float&, const device float*, const device float*,      \
      const device float*,                                                  \
      uint3, uint3, uint3, uint3, uint);

instantiate_lthn_sdpa_vector_tq(128)
instantiate_lthn_sdpa_vector_tq(256)
instantiate_lthn_sdpa_vector_tq(512)
