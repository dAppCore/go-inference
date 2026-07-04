// SPDX-Licence-Identifier: EUPL-1.2

// lthn_ffn_megakernel — gemma's whole SwiGLU MLP in ONE dispatch: gate=qgemv(Wg,x), up=qgemv(Wu,x),
// gated=gelu(gate)·up, [grid barrier], down=qgemv(Wd,gated). Replaces the decode's gate/up + gelu·up + down
// (three barriered ICB ops) with a single kernel whose stages are separated by an IN-KERNEL device-wide grid
// barrier instead of external SetBarrier full-drains. The cross-TG handoff (gated, produced by all TGs in
// stage 1, read by stage 2) moves through RELAXED ATOMICS (L2-coherent) across a macOS 26 DEVICE-SCOPE
// barrier — the combination TestCrossTGCoherencyPlainVsAtomic proves coherent 64/64 (plain stays stale in
// per-TG L1). Needs -std=metal3.2+ for thread_scope_device / seq_cst. The gemvs inline the verified 4-bit
// affine dequant (token-identical to the steel qmv); the gelu matches lthn_gelu_gate_mul_bf16 (fp32 tanh).
// This is the first real decode-stage megakernel — the pattern the full layer stacks onto.
#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

// 4-bit affine dequant gemv for ONE output row over a PLAIN bf16 input. One 32-lane simdgroup owns
// the row; each lane accumulates k=lane,lane+32,... and simd_sum combines the reduction.
static inline float qgemv_row_simd(const device uint8_t* prow, const device bf16* srow, const device bf16* brow,
                                   const device bf16* x, uint inDim, uint groupSize, uint lane) {
  float partial = 0.0f;
  for (uint k = lane; k < inDim; k += 32u) {
    const uint g = k / groupSize;
    const float s = float(srow[g]);
    const float b = float(brow[g]);
    const uint8_t pb = prow[k >> 1];
    const float code = (k & 1u) == 0u ? float(pb & 0x0F) : float(pb >> 4);
    partial += (s * code + b) * float(x[k]);
  }
  return simd_sum(partial);
}

// Same gemv but reading the input through ATOMIC load (stage 2: x = gated, written cross-TG in stage 1).
// Each gated slot holds one bf16 zero-extended into a uint; the relaxed atomic load is L2-coherent so a
// distant TG's stage-1 write is seen after the device-scope grid barrier.
static inline float qgemv_row_atomic_x_simd(const device uint8_t* prow, const device bf16* srow, const device bf16* brow,
                                            const device atomic_uint* x, uint inDim, uint groupSize, uint lane) {
  float partial = 0.0f;
  for (uint k = lane; k < inDim; k += 32u) {
    const uint g = k / groupSize;
    const float s = float(srow[g]);
    const float b = float(brow[g]);
    const uint8_t pb = prow[k >> 1];
    const float code = (k & 1u) == 0u ? float(pb & 0x0F) : float(pb >> 4);
    const bf16 xv = as_type<bf16>(ushort(atomic_load_explicit(&x[k], memory_order_relaxed)));
    partial += (s * code + b) * float(xv);
  }
  return simd_sum(partial);
}

// Device-wide grid barrier: seq_cst device fence releases this TG's stage-1 writes device-wide, the
// device-scope threadgroup_barrier + atomic arrival counter sync all TGs, the trailing fence acquires the
// other TGs' writes. macOS 26 (thread_scope_device / memory_order_seq_cst), -std=metal3.2+.
static inline void grid_barrier(device atomic_uint* arrive, uint numTG, uint lid, uint maxSpin) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device, thread_scope_device);
  if (lid == 0) {
    atomic_fetch_add_explicit(arrive, 1u, memory_order_relaxed);
    for (uint i = 0; i < maxSpin; i++) {
      if (atomic_load_explicit(arrive, memory_order_relaxed) >= numTG) {
        break;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device, thread_scope_device);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}

[[kernel]] void lthn_ffn_megakernel(
    const device bf16*    x      [[buffer(0)]],   // [hidden]
    const device uint8_t* gateP  [[buffer(1)]],
    const device bf16*    gateS  [[buffer(2)]],
    const device bf16*    gateB  [[buffer(3)]],
    const device uint8_t* upP    [[buffer(4)]],
    const device bf16*    upS    [[buffer(5)]],
    const device bf16*    upB    [[buffer(6)]],
    const device uint8_t* downP  [[buffer(7)]],
    const device bf16*    downS  [[buffer(8)]],
    const device bf16*    downB  [[buffer(9)]],
    device atomic_uint*   gated  [[buffer(10)]],  // [ff] cross-TG handoff: one bf16 (zero-extended) per slot,
                                                  // accessed atomically (L2-coherent) so stage 2 sees stage 1
                                                  // across distant TGs over the device-scope grid barrier.
    device bf16*          out    [[buffer(11)]],  // [hidden]
    device atomic_uint*   arrive [[buffer(12)]],
    const constant uint& hidden  [[buffer(13)]],
    const constant uint& ff      [[buffer(14)]],
    const constant uint& groupSize [[buffer(15)]],
    const constant uint& numTG   [[buffer(16)]],
    const constant uint& maxSpin [[buffer(17)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint tg_pos [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]]) {
  const uint simdgroupsPerTG = tgSize / 32u;
  const uint row0 = tg_pos * simdgroupsPerTG + simd_gid;
  const uint rowStride = numTG * simdgroupsPerTG;
  const uint rowPackedH = hidden / 2;       // gate/up rows reduce over hidden
  const uint rowSBH = hidden / groupSize;
  const uint rowPackedF = ff / 2;           // down rows reduce over ff
  const uint rowSBF = ff / groupSize;

  // Stage 1: gated[i] = gelu(qgemv(Wg,x)_i) · qgemv(Wu,x)_i  (written atomically for the cross-TG handoff)
  for (uint row = row0; row < ff; row += rowStride) {
    // round gate/up to bf16 BEFORE the gelu — the separate-op path writes them as bf16 (qmv output) and the
    // gelu kernel reads bf16, so matching the rounding point keeps the fusion token-identical.
    const float g = float(bf16(qgemv_row_simd(gateP + row * rowPackedH, gateS + row * rowSBH, gateB + row * rowSBH, x, hidden, groupSize, lane)));
    const float u = float(bf16(qgemv_row_simd(upP + row * rowPackedH, upS + row * rowSBH, upB + row * rowSBH, x, hidden, groupSize, lane)));
    if (lane == 0u) {
      const float inner = g + 0.044715f * (g * g * g);
      const float t = precise::tanh(0.7978845608028654f * inner);
      const float gelu = 0.5f * g * (1.0f + t);
      atomic_store_explicit(&gated[row], uint(as_type<ushort>(bf16(gelu * u))), memory_order_relaxed);
    }
  }

  grid_barrier(arrive, numTG, lid, maxSpin);

  // Stage 2: out[h] = qgemv(Wd, gated)_h  (gated read atomically — coherent across the device-scope barrier)
  for (uint row = row0; row < hidden; row += rowStride) {
    const float y = qgemv_row_atomic_x_simd(downP + row * rowPackedF, downS + row * rowSBF, downB + row * rowSBF, gated, ff, groupSize, lane);
    if (lane == 0u) {
      out[row] = bf16(y);
    }
  }
}
