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
  for (uint t = rowBase + sg; t < rowEnd; t += kPagedSimdGroups) {
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
  // doubling it for GQA would not fit). Every thread participates in both
  // phases' barriers; simdgroup 0 does the merge work.
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
    if (sg != 0) {
      continue;
    }
    float M = -3.0e38f;
    for (uint gg = 0; gg < kPagedSimdGroups; gg++) {
      if (tgS[gg] > 0.0f) {
        M = max(M, tgM[gg]);
      }
    }
    float S = 0.0f;
    float of[kPagedGQA2MaxPerLane];
    for (uint i = 0; i < per; i++) {
      of[i] = 0.0f;
    }
    for (uint gg = 0; gg < kPagedSimdGroups; gg++) {
      const float sgS = tgS[gg];
      if (sgS <= 0.0f) {
        continue;
      }
      const float w = exp(tgM[gg] - M);
      S += sgS * w;
      for (uint i = 0; i < per; i++) {
        of[i] += tgO[gg * D.headDim + lane * per + i] * w;
      }
    }
    const uint h = h0 + g;
    if (WRITE_FINAL) {
      device bf16* oh = outFinal + h * D.headDim + lane * per;
      for (uint i = 0; i < per; i++) {
        oh[i] = S > 0.0f ? bf16(of[i] / S) : bf16(0.0f);
      }
      continue;
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
    uint h    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint per = D.headDim / 32;
  if (per == 0 || per > kPagedMaxPerLane) return;
  const uint base = h * D.cellCount;

  float M = -3.0e38f;
  for (uint p = 0; p < D.cellCount; p++) {
    if (sums[base + p] > 0.0f) {
      M = max(M, maxs[base + p]);
    }
  }

  float denom = 0.0f;
  float o[kPagedMaxPerLane];
  for (uint i = 0; i < per; i++) {
    o[i] = 0.0f;
  }
  for (uint p = 0; p < D.cellCount; p++) {
    const float s = sums[base + p];
    if (s <= 0.0f) {
      continue;
    }
    const float w = exp(maxs[base + p] - M);
    denom += s * w;
    const device float* a = acc + (base + p) * D.headDim + lane * per;
    for (uint i = 0; i < per; i++) {
      o[i] += a[i] * w;
    }
  }

  device bf16* oh = out + h * D.headDim + lane * per;
  for (uint i = 0; i < per; i++) {
    oh[i] = denom > 0.0f ? bf16(o[i] / denom) : bf16(0.0f);
  }
}
