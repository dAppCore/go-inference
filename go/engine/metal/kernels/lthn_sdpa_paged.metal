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

  for (uint t = rowBase + sg; t < rowEnd; t += kPagedSimdGroups) {
    const device bf16* kt = kh + t * D.kSeqStride;
    float partial = 0.0f;
    for (uint i = 0; i < per; i++) {
      partial += qv[i] * float(kt[i]);
    }
    const float dot = simd_sum(partial) * D.scale;
    const float newM = max(m, dot);
    const float f = s > 0.0f ? exp(m - newM) : 0.0f;
    const float p = exp(dot - newM);
    s = s * f + p;
    const device bf16* vt = vh + t * D.vSeqStride;
    for (uint i = 0; i < per; i++) {
      o[i] = o[i] * f + p * float(vt[i]);
    }
    m = newM;
  }

  // merge the simdgroup partials in threadgroup memory (log-sum-exp).
  threadgroup float tgM[kPagedSimdGroups];
  threadgroup float tgS[kPagedSimdGroups];
  threadgroup float tgO[kPagedSimdGroups * 512]; // headDim <= 512
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
