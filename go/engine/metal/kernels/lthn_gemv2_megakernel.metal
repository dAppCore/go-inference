// SPDX-Licence-Identifier: EUPL-1.2

// lthn_gemv2_megakernel — the FOUNDATIONAL pattern for a full-layer decode megakernel: two dependent gemvs
// (out = W2·(W1·x)) in ONE dispatch, with an in-kernel device-wide GRID BARRIER between them instead of an
// external ICB SetBarrier full-drain. Stage 1 (all threadgroups) computes mid = W1·x into device scratch;
// the grid barrier makes every threadgroup's writes visible; stage 2 computes out = W2·mid. Proves the two
// hard primitives the megakernel rests on — grid sync (≤512 TGs, verified) + cross-threadgroup coherency.
// One thread per output element; bf16 weights, f32 accumulate. The grid barrier: each TG leader arrives on
// an atomic counter (acq_rel) then spins (acquire) until all numTG arrive — bounded enough that the verified
// 512-TG co-residency ceiling holds.
#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

// One TG-leader-driven grid barrier. `phase` lets the same counter serve a single barrier per dispatch.
static inline void grid_barrier(device atomic_uint* arrive, threadgroup uint* tgScratch,
                                uint numTG, uint lid, uint maxSpin) {
  threadgroup_barrier(mem_flags::mem_device); // flush this TG's stage-1 device writes before arriving
  if (lid == 0) {
    atomic_fetch_add_explicit(arrive, 1u, memory_order_relaxed);
    for (uint i = 0; i < maxSpin; i++) {
      if (atomic_load_explicit(arrive, memory_order_relaxed) >= numTG) {
        break;
      }
    }
    *tgScratch = 1;
  }
  threadgroup_barrier(mem_flags::mem_device); // all arrived: stage-2 reads see every TG's stage-1 writes
}

[[kernel]] void lthn_gemv2_megakernel(
    const device bf16* x   [[buffer(0)]],   // [inDim]
    const device bf16* w1  [[buffer(1)]],   // [midDim × inDim] row-major
    const device bf16* w2  [[buffer(2)]],   // [outDim × midDim] row-major
    device bf16*       mid [[buffer(3)]],   // [midDim] scratch (device, cross-TG)
    device bf16*       out [[buffer(4)]],   // [outDim]
    device atomic_uint* arrive [[buffer(5)]],
    const constant uint& inDim  [[buffer(6)]],
    const constant uint& midDim [[buffer(7)]],
    const constant uint& outDim [[buffer(8)]],
    const constant uint& numTG  [[buffer(9)]],
    const constant uint& maxSpin [[buffer(10)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]) {
  threadgroup uint tgScratch;
  const uint stride = numTG * tgSize; // grid-stride so all output elements are covered by any TG count

  // Stage 1: mid = W1 · x
  for (uint i = gid; i < midDim; i += stride) {
    float acc = 0.0f;
    const device bf16* row = w1 + (uint)i * inDim;
    for (uint k = 0; k < inDim; k++) {
      acc += float(row[k]) * float(x[k]);
    }
    mid[i] = bf16(acc);
  }

  grid_barrier(arrive, &tgScratch, numTG, lid, maxSpin);

  // Stage 2: out = W2 · mid
  for (uint h = gid; h < outDim; h += stride) {
    float acc = 0.0f;
    const device bf16* row = w2 + (uint)h * midDim;
    for (uint j = 0; j < midDim; j++) {
      acc += float(row[j]) * float(mid[j]);
    }
    out[h] = bf16(acc);
  }
}
