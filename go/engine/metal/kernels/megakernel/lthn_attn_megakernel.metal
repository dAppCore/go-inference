// SPDX-Licence-Identifier: EUPL-1.2

// lthn_attn_megakernel — gemma's attention half in ONE dispatch: RMSNorm(x) → Q/K/V = W·normed → RoPE →
// write K/V to cache[pos] → SDPA over cache[0..pos] → O = Wo·attn → h = x + O. The stages are separated by
// IN-KERNEL device-scope grid barriers (macOS 26, -std=metal3.2+); every cross-threadgroup handoff buffer
// (normed, qr, attn) is device atomic_uint (one bf16 zero-extended per slot, relaxed atomic load/store —
// L2-coherent, the primitive TestCrossTGCoherencyPlainVsAtomic proves at 64/64 where plain stays stale).
// BF16 weights (dense matmul, no dequant) so it validates token-identical against AttentionStepKV. Standard
// MHA/GQA + full RoPE; the second half (FFN) is lthn_ffn_megakernel — together they stack to the full layer.
#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

static inline bf16 ld(const device atomic_uint* p, uint i) { return as_type<bf16>(ushort(atomic_load_explicit(&p[i], memory_order_relaxed))); }
static inline void st(device atomic_uint* p, uint i, bf16 v) { atomic_store_explicit(&p[i], uint(as_type<ushort>(v)), memory_order_relaxed); }

static inline void grid_barrier(device atomic_uint* arrive, uint round, uint numTG, uint lid, uint maxSpin) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device, thread_scope_device);
  if (lid == 0) {
    atomic_fetch_add_explicit(arrive, 1u, memory_order_relaxed);
    const uint target = (round + 1u) * numTG;
    for (uint i = 0; i < maxSpin; i++) {
      if (atomic_load_explicit(arrive, memory_order_relaxed) >= target) break;
    }
  }
  threadgroup_barrier(mem_flags::mem_device, thread_scope_device);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}

// bf16 dense matvec: out[o] = Σ_k W[o,k]·x[k], W row-major [outDim,inDim].
static inline float matvec(const device bf16* w, const device atomic_uint* x, uint o, uint inDim) {
  float acc = 0.0f;
  const device bf16* wr = w + (uint)o * inDim;
  for (uint k = 0; k < inDim; k++) acc += float(wr[k]) * float(ld(x, k));
  return acc;
}

[[kernel]] void lthn_attn_megakernel(
    const device bf16*  x        [[buffer(0)]],   // [dModel]
    const device bf16*  attnNormW [[buffer(1)]],  // [dModel]
    const device bf16*  wQ       [[buffer(2)]],   // [qDim, dModel]
    const device bf16*  wK       [[buffer(3)]],   // [kvDim, dModel]
    const device bf16*  wV       [[buffer(4)]],   // [kvDim, dModel]
    const device bf16*  wO       [[buffer(5)]],   // [dModel, qDim]
    device bf16*        kCache   [[buffer(6)]],   // [maxLen, kvDim]
    device bf16*        vCache   [[buffer(7)]],   // [maxLen, kvDim]
    device atomic_uint* normed   [[buffer(8)]],   // [dModel] handoff
    device atomic_uint* qr       [[buffer(9)]],   // [qDim] handoff (post-RoPE Q)
    device atomic_uint* attn     [[buffer(10)]],  // [qDim] handoff (SDPA out)
    device bf16*        out      [[buffer(11)]],  // [dModel] = x + Wo·attn
    device atomic_uint* arrive   [[buffer(12)]],
    const device float* invFreqs [[buffer(13)]],  // [headDim/2] RoPE inverse frequencies
    const constant uint& dModel  [[buffer(14)]],
    const constant uint& nHeads  [[buffer(15)]],
    const constant uint& nKVHeads [[buffer(16)]],
    const constant uint& headDim [[buffer(17)]],
    const constant uint& pos     [[buffer(18)]],  // this token's cache row (kvLen = pos+1)
    const constant float& scale  [[buffer(19)]],  // SDPA scale
    const constant float& eps    [[buffer(20)]],
    const constant uint& numTG   [[buffer(21)]],
    const constant uint& maxSpin [[buffer(22)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]) {
  const uint stride = numTG * tgSize;
  const uint qDim = nHeads * headDim, kvDim = nKVHeads * headDim;
  const uint kvLen = pos + 1u, gqa = nHeads / nKVHeads, hd2 = headDim / 2u;

  // Stage 1: normed = RMSNorm(x). Each thread redundantly computes the RMS over x (x is the resident input),
  // then writes its slice atomically. (x is read-only + already coherent, so the reduction needs no handoff.)
  float ss = 0.0f;
  for (uint k = 0; k < dModel; k++) { float v = float(x[k]); ss += v * v; }
  const float rms = rsqrt(ss / float(dModel) + eps);
  for (uint i = gid; i < dModel; i += stride) st(normed, i, bf16(float(x[i]) * rms * float(attnNormW[i])));
  grid_barrier(arrive, 0u, numTG, lid, maxSpin);

  // Stage 2: Q/K/V = W·normed; RoPE(Q)→qr, RoPE(K)→kCache[pos], V→vCache[pos]. One thread per head.
  for (uint h = gid; h < nHeads; h += stride) {
    const uint qoff = h * headDim;
    for (uint d = 0; d < hd2; d++) {              // rotate-half RoPE over the head
      const float q0 = matvec(wQ, normed, qoff + d, dModel);
      const float q1 = matvec(wQ, normed, qoff + d + hd2, dModel);
      const float ang = float(pos) * invFreqs[d];
      const float c = cos(ang), s = sin(ang);
      st(qr, qoff + d,       bf16(q0 * c - q1 * s));
      st(qr, qoff + d + hd2, bf16(q0 * s + q1 * c));
    }
  }
  for (uint hk = gid; hk < nKVHeads; hk += stride) {
    const uint koff = hk * headDim;
    const uint crow = pos * kvDim + koff;
    for (uint d = 0; d < hd2; d++) {
      const float k0 = matvec(wK, normed, koff + d, dModel);
      const float k1 = matvec(wK, normed, koff + d + hd2, dModel);
      const float ang = float(pos) * invFreqs[d];
      const float c = cos(ang), s = sin(ang);
      kCache[crow + d]       = bf16(k0 * c - k1 * s);
      kCache[crow + d + hd2] = bf16(k0 * s + k1 * c);
      vCache[crow + d]       = bf16(matvec(wV, normed, koff + d, dModel));
      vCache[crow + d + hd2] = bf16(matvec(wV, normed, koff + d + hd2, dModel));
    }
  }
  grid_barrier(arrive, 1u, numTG, lid, maxSpin);

  // Stage 3: SDPA — one thread per query head, attend over cache[0..pos]. attn[h] handoff (atomic).
  for (uint h = gid; h < nHeads; h += stride) {
    const uint qoff = h * headDim, kvh = (h / gqa) * headDim;
    float m = -3.0e38f;
    for (uint j = 0; j < kvLen; j++) {            // max score (online softmax pass 1)
      float dot = 0.0f;
      for (uint d = 0; d < headDim; d++) dot += float(ld(qr, qoff + d)) * float(kCache[j * kvDim + kvh + d]);
      dot *= scale;
      m = max(m, dot);
    }
    float denom = 0.0f;
    float acc[128];                               // headDim accumulators (one-thread-per-head proof: headDim<=128)
    for (uint d = 0; d < headDim; d++) acc[d] = 0.0f;
    for (uint j = 0; j < kvLen; j++) {
      float dot = 0.0f;
      for (uint d = 0; d < headDim; d++) dot += float(ld(qr, qoff + d)) * float(kCache[j * kvDim + kvh + d]);
      const float p = exp(dot * scale - m);
      denom += p;
      for (uint d = 0; d < headDim; d++) acc[d] += p * float(vCache[j * kvDim + kvh + d]);
    }
    for (uint d = 0; d < headDim; d++) st(attn, qoff + d, bf16(acc[d] / denom));
  }
  grid_barrier(arrive, 2u, numTG, lid, maxSpin);

  // Stage 4: out = x + Wo·attn.
  for (uint i = gid; i < dModel; i += stride) out[i] = bf16(float(x[i]) + matvec(wO, attn, i, qDim));
}
