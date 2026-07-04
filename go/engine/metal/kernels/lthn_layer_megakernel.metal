// SPDX-Licence-Identifier: EUPL-1.2

// lthn_layer_megakernel — a WHOLE gemma decode layer in ONE dispatch: the attention half
// (RMSNorm → QKV → RoPE → cache → SDPA → O → residual) chained into the FFN half (RMSNorm → gate/up →
// gelu·mul → down → residual). Six stages separated by macOS 26 device-scope grid barriers; every cross-TG
// handoff (normed, qr, attn, h, mlpNormed, gated) moves through device atomic_uint (relaxed atomic
// load/store, L2-coherent — TestCrossTGCoherencyPlainVsAtomic). bf16 dense matmul so it validates token-
// identical against the chained host reference. This is the full-layer megakernel the decode replays
// instead of ~15 barriered ICB ops — the dispatch-count collapse behind the 300+ target. -std=metal3.2+.
#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

struct LayerDims {
  uint dModel, nHeads, nKVHeads, headDim, dFF, pos, numTG, maxSpin;
  float scale, eps;
};

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

// bf16 dense matvec over an ATOMIC input: out[o] = Σ_k W[o,k]·x[k].
static inline float mv(const device bf16* w, const device atomic_uint* x, uint o, uint inDim) {
  float acc = 0.0f;
  const device bf16* wr = w + (uint)o * inDim;
  for (uint k = 0; k < inDim; k++) acc += float(wr[k]) * float(ld(x, k));
  return acc;
}

[[kernel]] void lthn_layer_megakernel(
    const device bf16*  x         [[buffer(0)]],
    const device bf16*  attnNormW [[buffer(1)]],
    const device bf16*  wQ        [[buffer(2)]],
    const device bf16*  wK        [[buffer(3)]],
    const device bf16*  wV        [[buffer(4)]],
    const device bf16*  wO        [[buffer(5)]],
    device bf16*        kCache    [[buffer(6)]],
    device bf16*        vCache    [[buffer(7)]],
    const device bf16*  mlpNormW  [[buffer(8)]],
    const device bf16*  wGate     [[buffer(9)]],
    const device bf16*  wUp       [[buffer(10)]],
    const device bf16*  wDown     [[buffer(11)]],
    device atomic_uint* normed    [[buffer(12)]],  // [dModel]
    device atomic_uint* qr        [[buffer(13)]],  // [qDim]
    device atomic_uint* attn      [[buffer(14)]],  // [qDim]
    device atomic_uint* h         [[buffer(15)]],  // [dModel] post-attention residual (attn→FFN handoff)
    device atomic_uint* mlpNormed [[buffer(16)]],  // [dModel]
    device atomic_uint* gated     [[buffer(17)]],  // [dFF]
    device bf16*        out       [[buffer(18)]],  // [dModel] = h + Wdown·gated
    device atomic_uint* arrive    [[buffer(19)]],
    const device float* invFreqs  [[buffer(20)]],  // [headDim/2]
    const constant LayerDims& D   [[buffer(21)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]) {
  const uint stride = D.numTG * tgSize;
  const uint qDim = D.nHeads * D.headDim, kvDim = D.nKVHeads * D.headDim;
  const uint kvLen = D.pos + 1u, gqa = D.nHeads / D.nKVHeads, hd2 = D.headDim / 2u;

  // Stage 1: normed = RMSNorm(x, attnNormW).
  float ss = 0.0f;
  for (uint k = 0; k < D.dModel; k++) { float v = float(x[k]); ss += v * v; }
  float rms = rsqrt(ss / float(D.dModel) + D.eps);
  for (uint i = gid; i < D.dModel; i += stride) st(normed, i, bf16(float(x[i]) * rms * float(attnNormW[i])));
  grid_barrier(arrive, 0u, D.numTG, lid, D.maxSpin);

  // Stage 2: Q/K/V = W·normed; RoPE(Q)→qr, RoPE(K)→kCache[pos], V→vCache[pos].
  for (uint hh = gid; hh < D.nHeads; hh += stride) {
    const uint qoff = hh * D.headDim;
    for (uint d = 0; d < hd2; d++) {
      const float q0 = mv(wQ, normed, qoff + d, D.dModel), q1 = mv(wQ, normed, qoff + d + hd2, D.dModel);
      const float ang = float(D.pos) * invFreqs[d], c = cos(ang), s = sin(ang);
      st(qr, qoff + d, bf16(q0 * c - q1 * s));
      st(qr, qoff + d + hd2, bf16(q0 * s + q1 * c));
    }
  }
  for (uint hk = gid; hk < D.nKVHeads; hk += stride) {
    const uint koff = hk * D.headDim, crow = D.pos * kvDim + koff;
    for (uint d = 0; d < hd2; d++) {
      const float k0 = mv(wK, normed, koff + d, D.dModel), k1 = mv(wK, normed, koff + d + hd2, D.dModel);
      const float ang = float(D.pos) * invFreqs[d], c = cos(ang), s = sin(ang);
      kCache[crow + d] = bf16(k0 * c - k1 * s);
      kCache[crow + d + hd2] = bf16(k0 * s + k1 * c);
      vCache[crow + d] = bf16(mv(wV, normed, koff + d, D.dModel));
      vCache[crow + d + hd2] = bf16(mv(wV, normed, koff + d + hd2, D.dModel));
    }
  }
  grid_barrier(arrive, 1u, D.numTG, lid, D.maxSpin);

  // Stage 3: SDPA per query head over cache[0..pos] → attn.
  for (uint hh = gid; hh < D.nHeads; hh += stride) {
    const uint qoff = hh * D.headDim, kvh = (hh / gqa) * D.headDim;
    float m = -3.0e38f;
    for (uint j = 0; j < kvLen; j++) {
      float dot = 0.0f;
      for (uint d = 0; d < D.headDim; d++) dot += float(ld(qr, qoff + d)) * float(kCache[j * kvDim + kvh + d]);
      m = max(m, dot * D.scale);
    }
    float denom = 0.0f, acc[128];
    for (uint d = 0; d < D.headDim; d++) acc[d] = 0.0f;
    for (uint j = 0; j < kvLen; j++) {
      float dot = 0.0f;
      for (uint d = 0; d < D.headDim; d++) dot += float(ld(qr, qoff + d)) * float(kCache[j * kvDim + kvh + d]);
      const float p = exp(dot * D.scale - m);
      denom += p;
      for (uint d = 0; d < D.headDim; d++) acc[d] += p * float(vCache[j * kvDim + kvh + d]);
    }
    for (uint d = 0; d < D.headDim; d++) st(attn, qoff + d, bf16(acc[d] / denom));
  }
  grid_barrier(arrive, 2u, D.numTG, lid, D.maxSpin);

  // Stage 4: h = x + Wo·attn  (post-attention residual, atomic handoff into the FFN half).
  for (uint i = gid; i < D.dModel; i += stride) st(h, i, bf16(float(x[i]) + mv(wO, attn, i, qDim)));
  grid_barrier(arrive, 3u, D.numTG, lid, D.maxSpin);

  // Stage 5: mlpNormed = RMSNorm(h, mlpNormW).
  float ssh = 0.0f;
  for (uint k = 0; k < D.dModel; k++) { float v = float(ld(h, k)); ssh += v * v; }
  float rmsh = rsqrt(ssh / float(D.dModel) + D.eps);
  for (uint i = gid; i < D.dModel; i += stride) st(mlpNormed, i, bf16(float(ld(h, i)) * rmsh * float(mlpNormW[i])));
  grid_barrier(arrive, 4u, D.numTG, lid, D.maxSpin);

  // Stage 6: gated = gelu(Wgate·mlpNormed) · (Wup·mlpNormed)  (gemma tanh gelu, bf16-rounded gate/up).
  for (uint i = gid; i < D.dFF; i += stride) {
    const float g = float(bf16(mv(wGate, mlpNormed, i, D.dModel)));
    const float u = float(bf16(mv(wUp, mlpNormed, i, D.dModel)));
    const float inner = g + 0.044715f * (g * g * g);
    const float t = precise::tanh(0.7978845608028654f * inner);
    st(gated, i, bf16(0.5f * g * (1.0f + t) * u));
  }
  grid_barrier(arrive, 5u, D.numTG, lid, D.maxSpin);

  // Stage 7: out = h + Wdown·gated  (FFN residual).
  for (uint i = gid; i < D.dModel; i += stride) out[i] = bf16(float(ld(h, i)) + mv(wDown, gated, i, D.dFF));
}
