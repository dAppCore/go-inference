// SPDX-Licence-Identifier: EUPL-1.2

// lthn_attn_win — flash attention for gemma4's SLIDING layers (#375 phase 4):
// the majority of the family's layers (30/35 on e2b, 50/60 on 31B) window
// their attention at W (512/1024), and the deferred-ring lane serves them
// with the multiQ vector kernel — every query row re-reading its whole
// window from device (~1GB per layer per 2048-row chunk). This kernel gives
// them the flash treatment: per (head, BQ-query tile) threadgroup, the tile
// streams ONLY its own window span [max(0, qMin−W+1) .. qMax] — a fixed
// ≤ W+BQ keys regardless of depth — through threadgroup tiles shared by all
// BQ queries, with the online softmax carrying the state. K/V rows arrive
// from TWO sources: the pre-batch RING (wrapped at W rows, positions below
// basePos) and the chunk's STAGE slabs (positions basePos+i) — the same
// two-segment read the multiQ ring kernel performs, lifted to tiles. The
// S mask adds the window FLOOR to the causal cap:
//   attend iff row_pos−W < col_pos <= row_pos
// matching the engine's ring semantics (a query at pos attends the last
// min(pos+1, W) keys). Head dim 256 on both families' sliding layers — the
// proven whole-head steel shape.

// clang-format off
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
// clang-format on

using namespace metal;
using namespace mlx::steel;

// Window-lane extras beside AttnParams (which carries qL_off = basePos).
struct AttnWinParams {
  int winW; ///< sliding window (ring capacity)
  int ringLive; ///< valid pre-batch ring rows (min(basePos, winW))
};

// loadWinTile cooperatively loads a BK-row window tile whose rows are
// ABSOLUTE key positions [tileStart .. tileStart+BK): position p reads the
// ring (p % winW) below basePos, the stage (p - basePos) at or above it.
// Rows outside [winStart, qMax] zero-fill (the S mask excludes them anyway).
template <typename T, int BK, int BD, bool transposed, int LDT>
METAL_FUNC void loadWinTile(
    threadgroup T* dst,
    const device T* ring,
    const device T* stage,
    long ringRowStride,
    long stageRowStride,
    int headCol,
    int tileStart,
    int basePos,
    int winW,
    int ringLive,
    int qMax,
    int lin,
    int threads) {
  for (int idx = lin; idx < BK * BD; idx += threads) {
    const int r = idx / BD;
    const int c = idx % BD;
    const int p = tileStart + r;
    T val = T(0);
    if (p >= 0 && p <= qMax) {
      if (p >= basePos) {
        val = stage[long(p - basePos) * stageRowStride + headCol + c];
      } else if (p >= basePos - ringLive) {
        val = ring[long(p % winW) * ringRowStride + headCol + c];
      }
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
    int BD,
    int WM,
    int WN,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_win(
    const device T* Q [[buffer(0)]],
    const device T* Kring [[buffer(1)]],
    const device T* Vring [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const device T* Kstage [[buffer(5)]],
    const device T* Vstage [[buffer(6)]],
    const constant AttnWinParams* win [[buffer(7)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid;

  Q += tid.y * params->Q_strides[1] + tid.x * BQ * params->Q_strides[2];

  const ulong kv_head_idx = uint(tid.y) / params->gqa_factor;
  const int headCol = int(kv_head_idx) * BD;

  O += tid.y * params->O_strides[1] + tid.x * BQ * params->O_strides[2];

  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
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
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

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
    loader_q.load_safe(short2(BD, params->qL_rem));
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

  // This tile's key stream: absolute positions [streamStart .. qTileMax].
  const int basePos = params->qL_off;
  const int qTileMin = basePos + int(tid.x) * BQ;
  int qTileMax = basePos + params->qL - 1;
  {
    const int tileLast = qTileMin + BQ - 1;
    if (tileLast < qTileMax) {
      qTileMax = tileLast;
    }
  }
  int streamStart = qTileMin - win->winW + 1;
  if (streamStart < 0) {
    streamStart = 0;
  }
  const int nStreamTiles = (qTileMax - streamStart + BK) / BK;

  for (int kb = 0; kb < nStreamTiles; kb++) {
    const int tileStart = streamStart + kb * BK;

    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);
    loadWinTile<T, BK, BD, true, LDK_tgp>(
        Ks, Kring, Kstage, params->K_strides[2], params->K_strides[2],
        headCol, tileStart, basePos, win->winW, win->ringLive, qTileMax,
        lin, threadsPerTG);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Window + causal mask: attend iff row_pos−W < col_pos <= row_pos.
    {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            qTileMin + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = tileStart + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            const int cp = col_pos + jj;
            if (row_pos < cp || cp <= row_pos - win->winW) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    loadWinTile<T, BK, BD, false, LDV_tgp>(
        Vs, Vring, Vstage, params->V_strides[2], params->V_strides[2],
        headCol, tileStart, basePos, win->winW, win->ringLive, qTileMax,
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
      for (short id = 0; id < TD; id++) {
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
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}

#define instantiate_lthn_attn_win(tname, dtype, bq, bk, bd, wm, wn)         \
  instantiate_kernel(                                                       \
      "lthn_attn_win_" #tname "_bq" #bq "_bk" #bk "_bd" #bd                 \
      "_wm" #wm "_wn" #wn,                                                  \
  attention_win, dtype, bq, bk, bd, wm, wn, float)

instantiate_lthn_attn_win(bfloat16, bfloat16_t, 32, 16, 256, 4, 1);
