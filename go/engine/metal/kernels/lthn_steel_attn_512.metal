// SPDX-Licence-Identifier: EUPL-1.2

// lthn_attn_splitd — flash attention at head dim 512 (#375 phase 2b), the
// dimension no 32KB threadgroup budget fits whole: Q_smem alone at BD=512
// exceeds the limit at every legal warp shape. The split: TWO threadgroups
// per (query-tile, head) — grid.z picks a 256-wide D-half — and each
// RECOMPUTES the full S from both Q·K halves (the dot product needs all of
// D), then applies its P to only its V-half and writes its O-half. The
// softmax stats are recomputed identically in both halves, so the halves
// normalise self-consistently by construction; S still never touches device
// memory. The trade is QK compute ×2 for S-traffic ×0 — the side the #367
// poison receipts favour, and the half-pass memory shape is EXACTLY the
// proven BD-256 instantiation's:
//   Q_smem 16×(512+8)×2B = 16.6KB + K/V half tile 12.3KB = 28.9KB < 32KB
//   Otile 32 frags = 64 f32/lane — the same register class BD-256 runs.
// Body adapted from mlx steel_attention.h (Apple's loaders/MMATile/softmax
// fragments — MIT); tid.z is the D-half here, not the batch (B==1 always on
// this lane).

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
// clang-format on

using namespace mlx::steel;

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BDH, // the D-half this threadgroup owns for V/O (S spans 2·BDH)
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_splitd(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid;

  // Move to the block: x = query tile, y = head, z = D-HALF (batch is 1)
  const int dHalf = int(tid.z);

  Q += tid.y * params->Q_strides[1] + // Head
      tid.x * BQ * params->Q_strides[2]; // Sequence

  const ulong kv_head_idx = uint(tid.y) / params->gqa_factor;
  K += kv_head_idx * params->K_strides[1];
  V += kv_head_idx * params->V_strides[1] + dHalf * BDH; // this half's V columns

  O += tid.y * params->O_strides[1] + // Head
      tid.x * BQ * params->O_strides[2] + // Sequence
      dHalf * BDH; // this half's O columns

  // Threadgroup memory: Q resident FULL-width; K/V share one half-wide tile
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short BDF = 2 * BDH; // full head dim (S spans it)

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

  // Loaders: Q full-width once; K per half (two column-offset loaders into the
  // SAME tile slot); V only this threadgroup's half.
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BDF,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded transposed, half-width
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BDH,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BDH,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k0(
      K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  KBlockLoader loader_k1(
      K + BDH, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  // MMA tiles — identical fragment shapes to the BD-256 instantiation
  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  constexpr int TQ = BQ / (kNWarps * kFragSize); // Q seq frags per warp
  constexpr int TK = BK / kFragSize; // KV seq frags
  constexpr int TDH = BDH / kFragSize; // HALF head-dim frags

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

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load the FULL-width Q block once
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

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const bool kTail = !align_K && kb == (params->NK_aligned);

    Stile.clear();

    // S accumulates from BOTH D-halves: fill the K tile per half, matmul the
    // matching Q columns — the tile slot is reused, so a barrier brackets
    // each half's fill.
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (h == 0) {
        if (kTail) {
          loader_k0.load_safe(short2(BDH, params->kL_rem));
        } else {
          loader_k0.load_unsafe();
        }
      } else {
        if (kTail) {
          loader_k1.load_safe(short2(BDH, params->kL_rem));
        } else {
          loader_k1.load_unsafe();
        }
      }
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

    // Apply scale in float32
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Mask out the sequence tail
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

    // Causal mask
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

    // Load this half's V block
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kTail) {
      loader_v.load_safe(short2(BDH, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    // Online softmax — identical algebra to the steel body
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

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O_half += P @ V_half
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

    loader_k0.next();
    loader_k1.next();
    loader_v.next();
  }

  // Normalise and store this half's O columns
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

#define instantiate_lthn_attn_splitd(tname, dtype, bq, bk, bdh, wm, wn)     \
  instantiate_kernel(                                                       \
      "lthn_attn_splitd_" #tname "_bq" #bq "_bk" #bk "_bdh" #bdh            \
      "_wm" #wm "_wn" #wn,                                                  \
  attention_splitd, dtype, bq, bk, bdh, wm, wn, bool, float)

instantiate_lthn_attn_splitd(bfloat16, bfloat16_t, 16, 16, 256, 2, 1);
