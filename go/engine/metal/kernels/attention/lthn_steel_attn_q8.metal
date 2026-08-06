// SPDX-Licence-Identifier: EUPL-1.2

// lthn_attn_q8 — the q8-READING flash attention (#375 phase 3): K/V arrive as
// the engine's int8 codes + f32 group scales (group 64, bias-free symmetric —
// the #367 q8 KV format) and dequantise INSIDE the tile load, straight into
// the bf16 threadgroup tiles the MMA body consumes. Queries stay bf16. This
// deletes the GEMM-prefix mirror machinery on the flash lane — no
// ensureQ8Mirrors (21.5GB transient on 31B@256K), no dequant dispatches, no
// mirror round-trip — and HALVES the K/V device traffic on top (int8 vs
// bf16). The MMA/softmax body is the proven steel shape, unified over
// NHALVES: 1 = the BD-256 whole-head lane (e2b geometry), 2 = the split-D
// 512 lane (each grid.z threadgroup recomputes full S from both halves and
// owns a 256-wide V/O half — see lthn_steel_attn_512.metal).
//
// q8 layout per owner row (matching decode_forward_arch_icb_q8.go):
//   codes  [row][kvDim] int8, row stride kvDim
//   scales [row][kvDim/64] float, row stride kvDim/64
//   value  = bf16(float(code) * scale)

// clang-format off
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
// clang-format on

using namespace metal;
using namespace mlx::steel;

constant constexpr const int kQ8Group = 64;

// loadQ8Tile cooperatively dequantises a BK×BDH window of a q8 K or V cache
// into a bf16 threadgroup tile. rowsLive rows are real (the sequence tail
// zero-fills), colBase picks the D-half. transposed=true lands d-major
// (the K tile layout the S MMA reads); false lands row-major (the V tile).
template <typename T, int BK, int BDH, bool transposed, int LDT>
METAL_FUNC void loadQ8Tile(
    threadgroup T* dst,
    const device char* codes,
    const device float* scales,
    int rowBase,
    int rowsLive,
    int colBase,
    int kvDim,
    int lin,
    int threads) {
  const int groupsPerRow = kvDim / kQ8Group;
  for (int idx = lin; idx < BK * BDH; idx += threads) {
    const int r = idx / BDH;
    const int c = idx % BDH;
    T val = T(0);
    if (r < rowsLive) {
      const int col = colBase + c;
      const long rowOff = long(rowBase + r);
      const float s = scales[rowOff * groupsPerRow + (col / kQ8Group)];
      const float q = float(codes[rowOff * long(kvDim) + col]);
      val = static_cast<T>(q * s);
    }
    if (transposed) {
      dst[c * LDT + r] = val;
    } else {
      dst[r * LDT + c] = val;
    }
  }
}

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BDH,     // the D-half width this threadgroup owns for V/O
    int NHALVES, // 1 = whole head (BD == BDH), 2 = split-D (BD == 2·BDH)
    int WM,
    int WN,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_q8(
    const device T* Q [[buffer(0)]],
    const device char* Kc [[buffer(1)]],
    const device char* Vc [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const device float* Ks_scales [[buffer(5)]],
    const device float* Vs_scales [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid;

  const int dHalf = NHALVES == 1 ? 0 : int(tid.z);
  const int kvDim = int(params->K_strides[2]); // q8 rows are kvDim codes wide

  Q += tid.y * params->Q_strides[1] + tid.x * BQ * params->Q_strides[2];

  const ulong kv_head_idx = uint(tid.y) / params->gqa_factor;
  // q8 code pointers: head at kvh·(NHALVES·BDH) within the row; the loader
  // adds the row and column strides itself (codes are per-ELEMENT bytes).
  const int kvHeadCol = int(kv_head_idx) * NHALVES * BDH;
  const int groupsPerRow = kvDim / kQ8Group;
  (void)groupsPerRow;

  O += tid.y * params->O_strides[1] + tid.x * BQ * params->O_strides[2] +
      dHalf * BDH;

  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short BDF = NHALVES * BDH;

  constexpr short LDQ_tgp = BDF + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BDH + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BDH);
  constexpr short tgp_mem_1 = BK * (BDH + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BDF + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BDF,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  constexpr int TQ = BQ / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TDH = BDH / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TDH, MMAFrag_acc_t> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  const int lin = int(simd_group_id) * 32 + int(simd_lane_id);
  constexpr int threadsPerTG = WM * WN * 32;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BDF, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  for (int kb = 0; kb < kb_lim; kb++) {
    const bool kTail = !align_K && kb == (params->NK_aligned);
    const int rowsLive = kTail ? params->kL_rem : BK;
    const int rowBase = kb * BK;

    Stile.clear();

    // S accumulates over the NHALVES D-halves: dequantise the K half-tile,
    // matmul the matching Q columns. One tile slot, barriers bracket fills.
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < NHALVES; h++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loadQ8Tile<T, BK, BDH, true, LDK_tgp>(
          Ks, Kc, Ks_scales, rowBase, rowsLive, kvHeadCol + h * BDH, kvDim,
          lin, threadsPerTG);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const short qColBase = h * BDH;
      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TDH; dd++) {
        simdgroup_barrier(mem_flags::mem_none);

        Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
            &Qs[Qs_offset + qColBase + dd * Qs_tile_stride]);
        Ktile.template load<T, 1, 1, LDK_tgp, 1>(
            &Ks[Ks_offset + dd * Ks_tile_stride]);

        simdgroup_barrier(mem_flags::mem_none);

        tile_matmad(Stile, Qtile, Ktile, Stile);
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    if (kTail) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    if (do_causal && kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Dequantise this half's V block
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loadQ8Tile<T, BK, BDH, false, LDV_tgp>(
        Vs, Vc, Vs_scales, rowBase, rowsLive, kvHeadCol + dHalf * BDH, kvDim,
        lin, threadsPerTG);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    Otile.template row_bin_op<MulOp>(factor);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDH; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }
  }

  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  O += (tm + sm) * params->O_strides[2] + sn;

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BDH - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}

#define instantiate_lthn_attn_q8(tname, dtype, bq, bk, bdh, nh, wm, wn)     \
  instantiate_kernel(                                                       \
      "lthn_attn_q8_" #tname "_bq" #bq "_bk" #bk "_bdh" #bdh "_nh" #nh      \
      "_wm" #wm "_wn" #wn,                                                  \
  attention_q8, dtype, bq, bk, bdh, nh, wm, wn, float)

instantiate_lthn_attn_q8(bfloat16, bfloat16_t, 32, 16, 256, 1, 4, 1);
instantiate_lthn_attn_q8(bfloat16, bfloat16_t, 16, 16, 256, 2, 2, 1);
