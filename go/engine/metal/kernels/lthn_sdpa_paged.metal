// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

// Paged decode SDPA, parallel two-pass. Pass 1 runs one threadgroup per (head, page-dispatch):
// 32 lanes cooperate over the page's rows (each lane owns headDim/32 accumulator dims, the row
// dot reduces across lanes with simd_sum), writing an INDEPENDENT per-(head, page) partial —
// online-softmax max, denom, and weighted-V accumulator. No cross-page state: every page's
// dispatch is hazard-free against the others, so Metal overlaps them (the previous kernel
// carried the online softmax across pages through shared scratch, serialising the whole chain
// per layer at one scalar thread per head). Pass 2 merges the per-page partials per head with
// the standard log-sum-exp combine and writes the bf16 output.
//
// SINGLE-CELL fast path (#340): when the whole visible cache is one split window of one page
// (cellCount == 1), pass 2's log-sum-exp merge of ONE cell is an identity rescale — M is the
// cell's own max, w = exp(0) = 1, denom = the cell's own sum — so the final row is exactly
// bf16(acc[i]/S). lthn_sdpa_paged_p1_final_bf16 applies that division at store and the host
// skips pass 2 entirely (one dispatch per layer per token saved at short context). Same body,
// template-selected tail, so the arithmetic — and the output bytes — cannot drift from the
// two-pass path. A separate host_name (not a function constant) so a STALE metallib fails the
// pipeline lookup loudly and the host falls back to two passes, instead of silently running a
// kernel without the branch.

struct PagedSDPAP1Dims {
  uint nHeads;
  uint nKVHeads;
  uint headDim;
  uint pageLen;
  uint kHeadStride;
  uint kSeqStride;
  uint vHeadStride;
  uint vSeqStride;
  uint splitRows;  // rows per split window (the depth-parallelism grain)
  uint splits;     // split windows in THIS dispatch (ceil(pageLen / splitRows))
  uint cellBase;   // first partial-cell index for this dispatch's split 0
  uint cellCount;  // total partial cells across every dispatch (pass 2's merge width)
  float scale;
};

// headDim/32 accumulator dims per lane; gemma4 head dims are 256 (sliding) and 512 (full),
// so 16 covers every shipped shape (8 and 16 slices respectively).
constant constexpr uint kPagedMaxPerLane = 16;

// 8 simdgroups per threadgroup: each owns every-8th page row (its own online
// softmax over dim-sliced lanes), then simdgroup 0 merges the 8 partials with
// log-sum-exp through threadgroup memory. Cuts the sequential row chain 8x and
// runs 256 threads per head instead of 32.
constant constexpr uint kPagedSimdGroups = 8;

template <bool WRITE_FINAL>
METAL_FUNC void lthn_sdpa_paged_p1_body(
    const device bf16* q,
    const device bf16* kPage,
    const device bf16* vPage,
    device float* maxs,
    device float* sums,
    device float* acc,
    device bf16* outFinal,
    const constant PagedSDPAP1Dims& D,
    threadgroup float* tgM,
    threadgroup float* tgS,
    threadgroup float* tgO,
    uint tgid,
    uint sg,
    uint lane) {
  // grid = nHeads × splits, linearised on x (head-major): every split window is an
  // independent threadgroup so the depth parallelism GROWS with context instead of
  // serialising inside 8 simdgroups (16 TGs total was the #339 droop: 12.4 ms/token of
  // attention at position ~3500 with most of the GPU idle).
  const uint h = tgid / D.splits;
  const uint split = tgid % D.splits;
  if (h >= D.nHeads) return;
  const uint per = D.headDim / 32;
  if (per == 0 || per > kPagedMaxPerLane) return;
  const uint rowBase = split * D.splitRows;
  if (rowBase >= D.pageLen) return;
  const uint rowEnd = min(rowBase + D.splitRows, D.pageLen);

  const uint gqa = D.nHeads / D.nKVHeads;
  const uint kvh = h / gqa;
  const device bf16* qh = q + h * D.headDim + lane * per;
  const device bf16* kh = kPage + kvh * D.kHeadStride + lane * per;
  const device bf16* vh = vPage + kvh * D.vHeadStride + lane * per;

  float qv[kPagedMaxPerLane];
  for (uint i = 0; i < per; i++) {
    qv[i] = float(qh[i]);
  }

  float m = -3.0e38f;
  float s = 0.0f;
  float o[kPagedMaxPerLane];
  for (uint i = 0; i < per; i++) {
    o[i] = 0.0f;
  }

  // vectorised K/V reads: per is headDim/32 (8 or 16 on every shipped head) and the lane
  // offsets/strides are head-dim multiples, so bfloat4 loads are aligned — 4x fewer load
  // instructions on the loop that touches every cached row (the depth-scaling cost).
  const bool vec4 = (per % 4u) == 0u &&
                    ((D.kSeqStride | D.kHeadStride | D.vSeqStride | D.vHeadStride) % 4u) == 0u;
  for (uint t = rowBase + sg; t < rowEnd; t += kPagedSimdGroups) {
    const device bf16* kt = kh + t * D.kSeqStride;
    float partial = 0.0f;
    if (vec4) {
      for (uint i = 0; i < per; i += 4) {
        const bfloat4 k4 = *((const device bfloat4*)(kt + i));
        partial += qv[i] * float(k4.x) + qv[i + 1] * float(k4.y) +
                   qv[i + 2] * float(k4.z) + qv[i + 3] * float(k4.w);
      }
    } else {
      for (uint i = 0; i < per; i++) {
        partial += qv[i] * float(kt[i]);
      }
    }
    const float dot = simd_sum(partial) * D.scale;
    const float newM = max(m, dot);
    const float f = s > 0.0f ? exp(m - newM) : 0.0f;
    const float p = exp(dot - newM);
    s = s * f + p;
    const device bf16* vt = vh + t * D.vSeqStride;
    if (vec4) {
      for (uint i = 0; i < per; i += 4) {
        const bfloat4 v4 = *((const device bfloat4*)(vt + i));
        o[i] = o[i] * f + p * float(v4.x);
        o[i + 1] = o[i + 1] * f + p * float(v4.y);
        o[i + 2] = o[i + 2] * f + p * float(v4.z);
        o[i + 3] = o[i + 3] * f + p * float(v4.w);
      }
    } else {
      for (uint i = 0; i < per; i++) {
        o[i] = o[i] * f + p * float(vt[i]);
      }
    }
    m = newM;
  }

  // merge the simdgroup partials in threadgroup memory (log-sum-exp).
  if (lane == 0) {
    tgM[sg] = m;
    tgS[sg] = s;
  }
  for (uint i = 0; i < per; i++) {
    tgO[sg * D.headDim + lane * per + i] = o[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  float M = -3.0e38f;
  for (uint g = 0; g < kPagedSimdGroups; g++) {
    if (tgS[g] > 0.0f) {
      M = max(M, tgM[g]);
    }
  }
  float S = 0.0f;
  float of[kPagedMaxPerLane];
  for (uint i = 0; i < per; i++) {
    of[i] = 0.0f;
  }
  for (uint g = 0; g < kPagedSimdGroups; g++) {
    const float sgS = tgS[g];
    if (sgS <= 0.0f) {
      continue;
    }
    const float w = exp(tgM[g] - M);
    S += sgS * w;
    for (uint i = 0; i < per; i++) {
      of[i] += tgO[g * D.headDim + lane * per + i] * w;
    }
  }

  if (WRITE_FINAL) {
    // Single cell: pass 2 would compute w = exp(M - M) = 1 and denom = S, then
    // bf16(of[i]/S) — the identical operands and the identical one division, so
    // the bytes match the two-pass path exactly (of[] round-trips pass 2's float
    // scratch losslessly).
    device bf16* oh = outFinal + h * D.headDim + lane * per;
    for (uint i = 0; i < per; i++) {
      oh[i] = S > 0.0f ? bf16(of[i] / S) : bf16(0.0f);
    }
    return;
  }

  const uint cell = h * D.cellCount + D.cellBase + split;
  if (lane == 0) {
    maxs[cell] = M;
    sums[cell] = S;
  }
  device float* a = acc + cell * D.headDim + lane * per;
  for (uint i = 0; i < per; i++) {
    a[i] = of[i];
  }
}

[[kernel]] void lthn_sdpa_paged_p1_bf16(
    const device bf16* q      [[buffer(0)]],
    const device bf16* kPage  [[buffer(1)]],
    const device bf16* vPage  [[buffer(2)]],
    device float*      maxs   [[buffer(3)]],  // [nHeads * cellCount]
    device float*      sums   [[buffer(4)]],  // [nHeads * cellCount]
    device float*      acc    [[buffer(5)]],  // [nHeads * cellCount * headDim]
    const constant PagedSDPAP1Dims& D [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgM[kPagedSimdGroups];
  threadgroup float tgS[kPagedSimdGroups];
  threadgroup float tgO[kPagedSimdGroups * 512]; // headDim <= 512
  lthn_sdpa_paged_p1_body<false>(
      q, kPage, vPage, maxs, sums, acc, (device bf16*)acc, D, tgM, tgS, tgO, tgid, sg, lane);
}

[[kernel]] void lthn_sdpa_paged_p1_final_bf16(
    const device bf16* q      [[buffer(0)]],
    const device bf16* kPage  [[buffer(1)]],
    const device bf16* vPage  [[buffer(2)]],
    device bf16*       out    [[buffer(3)]],  // [nHeads * headDim]
    const constant PagedSDPAP1Dims& D [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgM[kPagedSimdGroups];
  threadgroup float tgS[kPagedSimdGroups];
  threadgroup float tgO[kPagedSimdGroups * 512]; // headDim <= 512
  lthn_sdpa_paged_p1_body<true>(
      q, kPage, vPage, (device float*)out, (device float*)out, (device float*)out, out, D,
      tgM, tgS, tgO, tgid, sg, lane);
}

// GQA-SHARED pass 1 (#356): one threadgroup per (KV head, split window) streams
// the window's K/V rows ONCE and carries TWO independent online-softmax states —
// the per-query-head kernel above reads every cached row once per QUERY head, so
// a GQA-2 model (26B/31B/12B: nHeads = 2·nKVHeads) paid 2x the bandwidth-floor
// traffic on the depth-scaling loop. Row order, accumulation order, and the
// simdgroup merge are IDENTICAL per query head, so the output bytes match the
// per-head kernel exactly. Separate host_names (not function constants): a stale
// metallib fails the pipeline lookup loudly and the host keeps the per-head
// kernels. Wider GQA ratios (E2B 8, E4B 4) keep the per-head kernel — their hot
// decode rides the recorded-ICB lane, and 8 register-resident accumulator sets
// would collapse occupancy here.
constant constexpr uint kPagedGQA2 = 2;

// GQA-2 models (26B/31B/12B) run headDim 256 on every layer, so per = headDim/32
// caps at 8 — sizing the two per-head accumulator sets at 8 keeps the register
// budget equal to the per-head kernel's single 16-wide set. The host gates the
// gqa2 pipelines on headDim <= 256; the in-kernel bound is the belt.
constant constexpr uint kPagedGQA2MaxPerLane = 8;

template <bool WRITE_FINAL>
METAL_FUNC void lthn_sdpa_paged_p1_gqa2_body(
    const device bf16* q,
    const device bf16* kPage,
    const device bf16* vPage,
    device float* maxs,
    device float* sums,
    device float* acc,
    device bf16* outFinal,
    const constant PagedSDPAP1Dims& D,
    threadgroup float* tgM,
    threadgroup float* tgS,
    threadgroup float* tgO,
    uint tgid,
    uint sg,
    uint lane) {
  const uint kvh = tgid / D.splits;
  const uint split = tgid % D.splits;
  if (kvh >= D.nKVHeads || D.nHeads != kPagedGQA2 * D.nKVHeads) return;
  const uint per = D.headDim / 32;
  if (per == 0 || per > kPagedGQA2MaxPerLane) return;
  const uint rowBase = split * D.splitRows;
  if (rowBase >= D.pageLen) return;
  const uint rowEnd = min(rowBase + D.splitRows, D.pageLen);
  const uint h0 = kvh * kPagedGQA2;

  const device bf16* kh = kPage + kvh * D.kHeadStride + lane * per;
  const device bf16* vh = vPage + kvh * D.vHeadStride + lane * per;

  float qv[kPagedGQA2][kPagedGQA2MaxPerLane];
  for (uint g = 0; g < kPagedGQA2; g++) {
    const device bf16* qh = q + (h0 + g) * D.headDim + lane * per;
    for (uint i = 0; i < per; i++) {
      qv[g][i] = float(qh[i]);
    }
  }

  float m[kPagedGQA2];
  float s[kPagedGQA2];
  float o[kPagedGQA2][kPagedGQA2MaxPerLane];
  for (uint g = 0; g < kPagedGQA2; g++) {
    m[g] = -3.0e38f;
    s[g] = 0.0f;
    for (uint i = 0; i < per; i++) {
      o[g][i] = 0.0f;
    }
  }

  const bool vec4 = (per % 4u) == 0u &&
                    ((D.kSeqStride | D.kHeadStride | D.vSeqStride | D.vHeadStride) % 4u) == 0u;
  // Two rows per iteration (t and t+8): the four K-dots and their simd_sums are
  // independent so they pipeline, and the online-softmax state folds BOTH rows
  // in one batched update — s·e^(m−M) + e^(d₁−M) + e^(d₂−M) with M the joint
  // max, mathematically the two sequential updates composed. The single-row
  // form serialised (simd_sum → exp → accumulate) per row, which left the scan
  // latency-bound at ~260 GB/s while the per-head kernel reached 432 (#356
  // anatomy bench). The exp identities round differently from the sequential
  // form, so this is a numeric TIER (the verify-fold precedent), not a
  // byte-identical swap.
  const uint stride2 = 2 * kPagedSimdGroups;
  uint t = rowBase + sg;
  for (; t + kPagedSimdGroups < rowEnd; t += stride2) {
    const uint t2 = t + kPagedSimdGroups;
    const device bf16* kt = kh + t * D.kSeqStride;
    const device bf16* kt2 = kh + t2 * D.kSeqStride;
    float pa0 = 0.0f, pa1 = 0.0f, pb0 = 0.0f, pb1 = 0.0f;
    if (vec4) {
      for (uint i = 0; i < per; i += 4) {
        const bfloat4 ka = *((const device bfloat4*)(kt + i));
        const bfloat4 kb = *((const device bfloat4*)(kt2 + i));
        const float ax = float(ka.x), ay = float(ka.y), az = float(ka.z), aw = float(ka.w);
        const float bx = float(kb.x), by = float(kb.y), bz = float(kb.z), bw = float(kb.w);
        pa0 += qv[0][i] * ax + qv[0][i + 1] * ay + qv[0][i + 2] * az + qv[0][i + 3] * aw;
        pa1 += qv[1][i] * ax + qv[1][i + 1] * ay + qv[1][i + 2] * az + qv[1][i + 3] * aw;
        pb0 += qv[0][i] * bx + qv[0][i + 1] * by + qv[0][i + 2] * bz + qv[0][i + 3] * bw;
        pb1 += qv[1][i] * bx + qv[1][i + 1] * by + qv[1][i + 2] * bz + qv[1][i + 3] * bw;
      }
    } else {
      for (uint i = 0; i < per; i++) {
        const float ax = float(kt[i]);
        const float bx = float(kt2[i]);
        pa0 += qv[0][i] * ax;
        pa1 += qv[1][i] * ax;
        pb0 += qv[0][i] * bx;
        pb1 += qv[1][i] * bx;
      }
    }
    float dA[kPagedGQA2], dB[kPagedGQA2];
    dA[0] = simd_sum(pa0) * D.scale;
    dA[1] = simd_sum(pa1) * D.scale;
    dB[0] = simd_sum(pb0) * D.scale;
    dB[1] = simd_sum(pb1) * D.scale;
    float f[kPagedGQA2], pA[kPagedGQA2], pB[kPagedGQA2];
    for (uint g = 0; g < kPagedGQA2; g++) {
      const float newM = max(m[g], max(dA[g], dB[g]));
      f[g] = s[g] > 0.0f ? exp(m[g] - newM) : 0.0f;
      pA[g] = exp(dA[g] - newM);
      pB[g] = exp(dB[g] - newM);
      s[g] = s[g] * f[g] + pA[g] + pB[g];
      m[g] = newM;
    }
    const device bf16* vt = vh + t * D.vSeqStride;
    const device bf16* vt2 = vh + t2 * D.vSeqStride;
    if (vec4) {
      for (uint i = 0; i < per; i += 4) {
        const bfloat4 va = *((const device bfloat4*)(vt + i));
        const bfloat4 vb = *((const device bfloat4*)(vt2 + i));
        o[0][i] = o[0][i] * f[0] + pA[0] * float(va.x) + pB[0] * float(vb.x);
        o[0][i + 1] = o[0][i + 1] * f[0] + pA[0] * float(va.y) + pB[0] * float(vb.y);
        o[0][i + 2] = o[0][i + 2] * f[0] + pA[0] * float(va.z) + pB[0] * float(vb.z);
        o[0][i + 3] = o[0][i + 3] * f[0] + pA[0] * float(va.w) + pB[0] * float(vb.w);
        o[1][i] = o[1][i] * f[1] + pA[1] * float(va.x) + pB[1] * float(vb.x);
        o[1][i + 1] = o[1][i + 1] * f[1] + pA[1] * float(va.y) + pB[1] * float(vb.y);
        o[1][i + 2] = o[1][i + 2] * f[1] + pA[1] * float(va.z) + pB[1] * float(vb.z);
        o[1][i + 3] = o[1][i + 3] * f[1] + pA[1] * float(va.w) + pB[1] * float(vb.w);
      }
    } else {
      for (uint i = 0; i < per; i++) {
        const float ax = float(vt[i]);
        const float bx = float(vt2[i]);
        o[0][i] = o[0][i] * f[0] + pA[0] * ax + pB[0] * bx;
        o[1][i] = o[1][i] * f[1] + pA[1] * ax + pB[1] * bx;
      }
    }
  }
  // odd tail row (window rows not a multiple of 2 simdgroup strides)
  for (; t < rowEnd; t += kPagedSimdGroups) {
    const device bf16* kt = kh + t * D.kSeqStride;
    float partial0 = 0.0f;
    float partial1 = 0.0f;
    if (vec4) {
      for (uint i = 0; i < per; i += 4) {
        const bfloat4 k4 = *((const device bfloat4*)(kt + i));
        const float x = float(k4.x), y = float(k4.y), z = float(k4.z), w = float(k4.w);
        partial0 += qv[0][i] * x + qv[0][i + 1] * y + qv[0][i + 2] * z + qv[0][i + 3] * w;
        partial1 += qv[1][i] * x + qv[1][i + 1] * y + qv[1][i + 2] * z + qv[1][i + 3] * w;
      }
    } else {
      for (uint i = 0; i < per; i++) {
        const float kx = float(kt[i]);
        partial0 += qv[0][i] * kx;
        partial1 += qv[1][i] * kx;
      }
    }
    float dot[kPagedGQA2];
    dot[0] = simd_sum(partial0) * D.scale;
    dot[1] = simd_sum(partial1) * D.scale;
    float f[kPagedGQA2];
    float p[kPagedGQA2];
    for (uint g = 0; g < kPagedGQA2; g++) {
      const float newM = max(m[g], dot[g]);
      f[g] = s[g] > 0.0f ? exp(m[g] - newM) : 0.0f;
      p[g] = exp(dot[g] - newM);
      s[g] = s[g] * f[g] + p[g];
      m[g] = newM;
    }
    const device bf16* vt = vh + t * D.vSeqStride;
    if (vec4) {
      for (uint i = 0; i < per; i += 4) {
        const bfloat4 v4 = *((const device bfloat4*)(vt + i));
        const float x = float(v4.x), y = float(v4.y), z = float(v4.z), w = float(v4.w);
        o[0][i] = o[0][i] * f[0] + p[0] * x;
        o[0][i + 1] = o[0][i + 1] * f[0] + p[0] * y;
        o[0][i + 2] = o[0][i + 2] * f[0] + p[0] * z;
        o[0][i + 3] = o[0][i + 3] * f[0] + p[0] * w;
        o[1][i] = o[1][i] * f[1] + p[1] * x;
        o[1][i + 1] = o[1][i + 1] * f[1] + p[1] * y;
        o[1][i + 2] = o[1][i + 2] * f[1] + p[1] * z;
        o[1][i + 3] = o[1][i + 3] * f[1] + p[1] * w;
      }
    } else {
      for (uint i = 0; i < per; i++) {
        const float vx = float(vt[i]);
        o[0][i] = o[0][i] * f[0] + p[0] * vx;
        o[1][i] = o[1][i] * f[1] + p[1] * vx;
      }
    }
  }

  // merge the simdgroup partials per query head, two phases through the SAME
  // threadgroup slabs (tgO at headDim 512 already fills the 32KB budget once —
  // doubling it for GQA would not fit). The merge itself is parallel over the
  // head dims: EVERY thread owns headDim/256 dims (global thread id, not the
  // lane), each folding the 8 simdgroup partials for its dims. The previous
  // form parked 7 of 8 simdgroups at the barrier while simdgroup 0 merged all
  // 256 dims — the #356 probe priced the whole row loop (loads, dots, sums,
  // exp folds) at 0.178ms/scan against 0.367 for the full kernel: HALF the
  // kernel was this merge. The 8 scalar M/S folds are recomputed per thread —
  // redundant flops in exchange for no extra barrier.
  const uint tid = sg * 32 + lane;
  const uint dimsPerThread = (D.headDim + 255) / 256;
  for (uint g = 0; g < kPagedGQA2; g++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      tgM[sg] = m[g];
      tgS[sg] = s[g];
    }
    for (uint i = 0; i < per; i++) {
      tgO[sg * D.headDim + lane * per + i] = o[g][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = -3.0e38f;
    for (uint gg = 0; gg < kPagedSimdGroups; gg++) {
      if (tgS[gg] > 0.0f) {
        M = max(M, tgM[gg]);
      }
    }
    float S = 0.0f;
    float w[kPagedSimdGroups];
    for (uint gg = 0; gg < kPagedSimdGroups; gg++) {
      const float sgS = tgS[gg];
      w[gg] = sgS > 0.0f ? exp(tgM[gg] - M) : 0.0f;
      S += sgS * w[gg];
    }
    const uint h = h0 + g;
    for (uint d = tid * dimsPerThread; d < min((tid + 1) * dimsPerThread, D.headDim); d++) {
      float of = 0.0f;
      for (uint gg = 0; gg < kPagedSimdGroups; gg++) {
        of += tgO[gg * D.headDim + d] * w[gg];
      }
      if (WRITE_FINAL) {
        outFinal[h * D.headDim + d] = S > 0.0f ? bf16(of / S) : bf16(0.0f);
      } else {
        acc[(h * D.cellCount + D.cellBase + split) * D.headDim + d] = of;
      }
    }
    if (!WRITE_FINAL && tid == 0) {
      const uint cell = h * D.cellCount + D.cellBase + split;
      maxs[cell] = M;
      sums[cell] = S;
    }
  }
}

[[kernel]] void lthn_sdpa_paged_p1_gqa2_bf16(
    const device bf16* q      [[buffer(0)]],
    const device bf16* kPage  [[buffer(1)]],
    const device bf16* vPage  [[buffer(2)]],
    device float*      maxs   [[buffer(3)]],
    device float*      sums   [[buffer(4)]],
    device float*      acc    [[buffer(5)]],
    const constant PagedSDPAP1Dims& D [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgM[kPagedSimdGroups];
  threadgroup float tgS[kPagedSimdGroups];
  threadgroup float tgO[kPagedSimdGroups * 256]; // gqa2 models: headDim <= 256
  lthn_sdpa_paged_p1_gqa2_body<false>(
      q, kPage, vPage, maxs, sums, acc, (device bf16*)acc, D, tgM, tgS, tgO, tgid, sg, lane);
}

[[kernel]] void lthn_sdpa_paged_p1_final_gqa2_bf16(
    const device bf16* q      [[buffer(0)]],
    const device bf16* kPage  [[buffer(1)]],
    const device bf16* vPage  [[buffer(2)]],
    device bf16*       out    [[buffer(3)]],
    const constant PagedSDPAP1Dims& D [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgM[kPagedSimdGroups];
  threadgroup float tgS[kPagedSimdGroups];
  threadgroup float tgO[kPagedSimdGroups * 256]; // gqa2 models: headDim <= 256
  lthn_sdpa_paged_p1_gqa2_body<true>(
      q, kPage, vPage, (device float*)out, (device float*)out, (device float*)out, out, D,
      tgM, tgS, tgO, tgid, sg, lane);
}

// ---- q8 KV pages (#357) ----------------------------------------------------
// Pages hold int8 rows quantised symmetrically per kvQ8GroupSize(=64) elements
// with f32 group scales in PARALLEL pages, same [row][kvHead][dim] order and
// ELEMENT strides as the bf16 pages. At headDim 256 a lane's per=8 slice sits
// inside one group, so each row costs the lane exactly one scale load; the
// scale multiplies the lane's partial BEFORE simd_sum (lanes carry different
// groups) and the V accumulate after the p·v product. Only the gqa2 shapes get
// q8 kernels — the host gates the q8 cache mode on gqa2-eligible geometry and
// FAILS LOUDLY rather than falling back to a bf16 kernel misreading int8 pages.

template <bool WRITE_FINAL>
METAL_FUNC void lthn_sdpa_paged_p1_gqa2_q8_body(
    const device bf16* q,
    const device char* kPage,
    const device char* vPage,
    const device float* kScales,
    const device float* vScales,
    device float* maxs,
    device float* sums,
    device float* acc,
    device bf16* outFinal,
    const constant PagedSDPAP1Dims& D,
    threadgroup float* tgM,
    threadgroup float* tgS,
    threadgroup float* tgO,
    uint tgid,
    uint sg,
    uint lane) {
  const uint kvh = tgid / D.splits;
  const uint split = tgid % D.splits;
  if (kvh >= D.nKVHeads || D.nHeads != kPagedGQA2 * D.nKVHeads) return;
  const uint per = D.headDim / 32;
  if (per == 0 || per > kPagedGQA2MaxPerLane) return;
  const uint rowBase = split * D.splitRows;
  if (rowBase >= D.pageLen) return;
  const uint rowEnd = min(rowBase + D.splitRows, D.pageLen);
  const uint h0 = kvh * kPagedGQA2;

  const device char* kh = kPage + kvh * D.kHeadStride + lane * per;
  const device char* vh = vPage + kvh * D.vHeadStride + lane * per;
  // scale addressing: one f32 per 64 elements, row-major over [row][kvh][group]
  const uint rowGroups = (D.nKVHeads * D.headDim) / 64;
  const uint headGroups = D.headDim / 64;
  const uint laneGroup = kvh * headGroups + (lane * per) / 64;

  float qv[kPagedGQA2][kPagedGQA2MaxPerLane];
  for (uint g = 0; g < kPagedGQA2; g++) {
    const device bf16* qh = q + (h0 + g) * D.headDim + lane * per;
    for (uint i = 0; i < per; i++) {
      qv[g][i] = float(qh[i]);
    }
  }

  float m[kPagedGQA2];
  float s[kPagedGQA2];
  float o[kPagedGQA2][kPagedGQA2MaxPerLane];
  for (uint g = 0; g < kPagedGQA2; g++) {
    m[g] = -3.0e38f;
    s[g] = 0.0f;
    for (uint i = 0; i < per; i++) {
      o[g][i] = 0.0f;
    }
  }

  const uint stride2 = 2 * kPagedSimdGroups;
  uint t = rowBase + sg;
  for (; t + kPagedSimdGroups < rowEnd; t += stride2) {
    const uint t2 = t + kPagedSimdGroups;
    const device char* kt = kh + t * D.kSeqStride;
    const device char* kt2 = kh + t2 * D.kSeqStride;
    const float ksA = kScales[t * rowGroups + laneGroup];
    const float ksB = kScales[t2 * rowGroups + laneGroup];
    float pa0 = 0.0f, pa1 = 0.0f, pb0 = 0.0f, pb1 = 0.0f;
    for (uint i = 0; i < per; i += 4) {
      const char4 ka = *((const device char4*)(kt + i));
      const char4 kb = *((const device char4*)(kt2 + i));
      const float ax = float(ka.x), ay = float(ka.y), az = float(ka.z), aw = float(ka.w);
      const float bx = float(kb.x), by = float(kb.y), bz = float(kb.z), bw = float(kb.w);
      pa0 += qv[0][i] * ax + qv[0][i + 1] * ay + qv[0][i + 2] * az + qv[0][i + 3] * aw;
      pa1 += qv[1][i] * ax + qv[1][i + 1] * ay + qv[1][i + 2] * az + qv[1][i + 3] * aw;
      pb0 += qv[0][i] * bx + qv[0][i + 1] * by + qv[0][i + 2] * bz + qv[0][i + 3] * bw;
      pb1 += qv[1][i] * bx + qv[1][i + 1] * by + qv[1][i + 2] * bz + qv[1][i + 3] * bw;
    }
    float dA[kPagedGQA2], dB[kPagedGQA2];
    dA[0] = simd_sum(pa0 * ksA) * D.scale;
    dA[1] = simd_sum(pa1 * ksA) * D.scale;
    dB[0] = simd_sum(pb0 * ksB) * D.scale;
    dB[1] = simd_sum(pb1 * ksB) * D.scale;
    float f[kPagedGQA2], pA[kPagedGQA2], pB[kPagedGQA2];
    for (uint g = 0; g < kPagedGQA2; g++) {
      const float newM = max(m[g], max(dA[g], dB[g]));
      f[g] = s[g] > 0.0f ? exp(m[g] - newM) : 0.0f;
      pA[g] = exp(dA[g] - newM);
      pB[g] = exp(dB[g] - newM);
      s[g] = s[g] * f[g] + pA[g] + pB[g];
      m[g] = newM;
    }
    const device char* vt = vh + t * D.vSeqStride;
    const device char* vt2 = vh + t2 * D.vSeqStride;
    const float vsA = vScales[t * rowGroups + laneGroup];
    const float vsB = vScales[t2 * rowGroups + laneGroup];
    const float pA0 = pA[0] * vsA, pB0 = pB[0] * vsB;
    const float pA1 = pA[1] * vsA, pB1 = pB[1] * vsB;
    for (uint i = 0; i < per; i += 4) {
      const char4 va = *((const device char4*)(vt + i));
      const char4 vb = *((const device char4*)(vt2 + i));
      o[0][i] = o[0][i] * f[0] + pA0 * float(va.x) + pB0 * float(vb.x);
      o[0][i + 1] = o[0][i + 1] * f[0] + pA0 * float(va.y) + pB0 * float(vb.y);
      o[0][i + 2] = o[0][i + 2] * f[0] + pA0 * float(va.z) + pB0 * float(vb.z);
      o[0][i + 3] = o[0][i + 3] * f[0] + pA0 * float(va.w) + pB0 * float(vb.w);
      o[1][i] = o[1][i] * f[1] + pA1 * float(va.x) + pB1 * float(vb.x);
      o[1][i + 1] = o[1][i + 1] * f[1] + pA1 * float(va.y) + pB1 * float(vb.y);
      o[1][i + 2] = o[1][i + 2] * f[1] + pA1 * float(va.z) + pB1 * float(vb.z);
      o[1][i + 3] = o[1][i + 3] * f[1] + pA1 * float(va.w) + pB1 * float(vb.w);
    }
  }
  for (; t < rowEnd; t += kPagedSimdGroups) {
    const device char* kt = kh + t * D.kSeqStride;
    const float ks = kScales[t * rowGroups + laneGroup];
    float partial0 = 0.0f;
    float partial1 = 0.0f;
    for (uint i = 0; i < per; i += 4) {
      const char4 k4 = *((const device char4*)(kt + i));
      const float x = float(k4.x), y = float(k4.y), z = float(k4.z), w = float(k4.w);
      partial0 += qv[0][i] * x + qv[0][i + 1] * y + qv[0][i + 2] * z + qv[0][i + 3] * w;
      partial1 += qv[1][i] * x + qv[1][i + 1] * y + qv[1][i + 2] * z + qv[1][i + 3] * w;
    }
    float dot[kPagedGQA2];
    dot[0] = simd_sum(partial0 * ks) * D.scale;
    dot[1] = simd_sum(partial1 * ks) * D.scale;
    float f[kPagedGQA2];
    float p[kPagedGQA2];
    for (uint g = 0; g < kPagedGQA2; g++) {
      const float newM = max(m[g], dot[g]);
      f[g] = s[g] > 0.0f ? exp(m[g] - newM) : 0.0f;
      p[g] = exp(dot[g] - newM);
      s[g] = s[g] * f[g] + p[g];
      m[g] = newM;
    }
    const device char* vt = vh + t * D.vSeqStride;
    const float vs = vScales[t * rowGroups + laneGroup];
    const float p0 = p[0] * vs, p1 = p[1] * vs;
    for (uint i = 0; i < per; i += 4) {
      const char4 v4 = *((const device char4*)(vt + i));
      o[0][i] = o[0][i] * f[0] + p0 * float(v4.x);
      o[0][i + 1] = o[0][i + 1] * f[0] + p0 * float(v4.y);
      o[0][i + 2] = o[0][i + 2] * f[0] + p0 * float(v4.z);
      o[0][i + 3] = o[0][i + 3] * f[0] + p0 * float(v4.w);
      o[1][i] = o[1][i] * f[1] + p1 * float(v4.x);
      o[1][i + 1] = o[1][i + 1] * f[1] + p1 * float(v4.y);
      o[1][i + 2] = o[1][i + 2] * f[1] + p1 * float(v4.z);
      o[1][i + 3] = o[1][i + 3] * f[1] + p1 * float(v4.w);
    }
  }

  // merge: identical to the bf16 gqa2 body — parallel over head dims.
  const uint tid = sg * 32 + lane;
  const uint dimsPerThread = (D.headDim + 255) / 256;
  for (uint g = 0; g < kPagedGQA2; g++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      tgM[sg] = m[g];
      tgS[sg] = s[g];
    }
    for (uint i = 0; i < per; i++) {
      tgO[sg * D.headDim + lane * per + i] = o[g][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = -3.0e38f;
    for (uint gg = 0; gg < kPagedSimdGroups; gg++) {
      if (tgS[gg] > 0.0f) {
        M = max(M, tgM[gg]);
      }
    }
    float S = 0.0f;
    float w[kPagedSimdGroups];
    for (uint gg = 0; gg < kPagedSimdGroups; gg++) {
      const float sgS = tgS[gg];
      w[gg] = sgS > 0.0f ? exp(tgM[gg] - M) : 0.0f;
      S += sgS * w[gg];
    }
    const uint h = h0 + g;
    for (uint d = tid * dimsPerThread; d < min((tid + 1) * dimsPerThread, D.headDim); d++) {
      float of = 0.0f;
      for (uint gg = 0; gg < kPagedSimdGroups; gg++) {
        of += tgO[gg * D.headDim + d] * w[gg];
      }
      if (WRITE_FINAL) {
        outFinal[h * D.headDim + d] = S > 0.0f ? bf16(of / S) : bf16(0.0f);
      } else {
        acc[(h * D.cellCount + D.cellBase + split) * D.headDim + d] = of;
      }
    }
    if (!WRITE_FINAL && tid == 0) {
      const uint cell = h * D.cellCount + D.cellBase + split;
      maxs[cell] = M;
      sums[cell] = S;
    }
  }
}

[[kernel]] void lthn_sdpa_paged_p1_gqa2_q8_bf16(
    const device bf16* q       [[buffer(0)]],
    const device char* kPage   [[buffer(1)]],
    const device char* vPage   [[buffer(2)]],
    device float*      maxs    [[buffer(3)]],
    device float*      sums    [[buffer(4)]],
    device float*      acc     [[buffer(5)]],
    const constant PagedSDPAP1Dims& D [[buffer(6)]],
    const device float* kScales [[buffer(7)]],
    const device float* vScales [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgM[kPagedSimdGroups];
  threadgroup float tgS[kPagedSimdGroups];
  threadgroup float tgO[kPagedSimdGroups * 256]; // gqa2 models: headDim <= 256
  lthn_sdpa_paged_p1_gqa2_q8_body<false>(
      q, kPage, vPage, kScales, vScales, maxs, sums, acc, (device bf16*)acc, D,
      tgM, tgS, tgO, tgid, sg, lane);
}

[[kernel]] void lthn_sdpa_paged_p1_final_gqa2_q8_bf16(
    const device bf16* q       [[buffer(0)]],
    const device char* kPage   [[buffer(1)]],
    const device char* vPage   [[buffer(2)]],
    device bf16*       out     [[buffer(3)]],
    const constant PagedSDPAP1Dims& D [[buffer(4)]],
    const device float* kScales [[buffer(7)]],
    const device float* vScales [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tgM[kPagedSimdGroups];
  threadgroup float tgS[kPagedSimdGroups];
  threadgroup float tgO[kPagedSimdGroups * 256]; // gqa2 models: headDim <= 256
  lthn_sdpa_paged_p1_gqa2_q8_body<true>(
      q, kPage, vPage, kScales, vScales, (device float*)out, (device float*)out,
      (device float*)out, out, D, tgM, tgS, tgO, tgid, sg, lane);
}

// lthn_kv_q8_store quantises one landed bf16 K or V row into its int8 page row
// + f32 group scales — the per-token landing hop that replaces the projection
// writing the page directly. Grid: one 32-lane threadgroup per 64-group; each
// lane owns two elements; the group max reduces with simd_max.
struct KVQ8StoreDims {
  uint kvDim;
};

[[kernel]] void lthn_kv_q8_store_bf16(
    const device bf16* row    [[buffer(0)]],
    device char*       out    [[buffer(1)]],
    device float*      scales [[buffer(2)]],
    const constant KVQ8StoreDims& D [[buffer(3)]],
    uint g    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint base = g * 64;
  if (base >= D.kvDim) return;
  const float a = float(row[base + lane * 2]);
  const float b = float(row[base + lane * 2 + 1]);
  const float m = simd_max(max(abs(a), abs(b)));
  const float scale = m / 127.0f;
  const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
  out[base + lane * 2] = char(clamp(rint(a * inv), -127.0f, 127.0f));
  out[base + lane * 2 + 1] = char(clamp(rint(b * inv), -127.0f, 127.0f));
  if (lane == 0) {
    scales[g] = scale;
  }
}

struct PagedSDPAP2Dims {
  uint headDim;
  uint cellCount;
};

[[kernel]] void lthn_sdpa_paged_p2_bf16(
    const device float* maxs [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* acc  [[buffer(2)]],
    device bf16*        out  [[buffer(3)]],  // [nHeads * headDim]
    const constant PagedSDPAP2Dims& D [[buffer(4)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  // parallel over head dims (one thread per dim, 256-thread groups): the
  // 32-lane form folded cellCount cells over 8 dims per lane serially, and at
  // deep-context cell counts the whole merge sat in 512 threads machine-wide
  // (#356 — the same shape the P1 merge fix addressed). The per-thread M/S
  // scalar folds are redundant across dims, in exchange for no reduction.
  const uint base = h * D.cellCount;
  float M = -3.0e38f;
  for (uint p = 0; p < D.cellCount; p++) {
    if (sums[base + p] > 0.0f) {
      M = max(M, maxs[base + p]);
    }
  }
  float denom = 0.0f;
  for (uint p = 0; p < D.cellCount; p++) {
    const float s = sums[base + p];
    if (s > 0.0f) {
      denom += s * exp(maxs[base + p] - M);
    }
  }
  for (uint d = tid; d < D.headDim; d += 256) {
    float o = 0.0f;
    for (uint p = 0; p < D.cellCount; p++) {
      const float s = sums[base + p];
      if (s <= 0.0f) {
        continue;
      }
      o += acc[(base + p) * D.headDim + d] * exp(maxs[base + p] - M);
    }
    out[h * D.headDim + d] = denom > 0.0f ? bf16(o / denom) : bf16(0.0f);
  }
}

