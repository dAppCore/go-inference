// SPDX-Licence-Identifier: EUPL-1.2
//
// lthn_gated_delta.metal — the gated delta-rule recurrence with register-resident state, the S1
// kernel of the hybrid-recurrence campaign (docs/design-hybrid-recurrence.md). One simdgroup owns
// one (value-head, dv-row) of the state: lane L holds DK/32 contiguous state floats in registers,
// the whole T-loop runs inside the kernel (T=1 decode and T=chunk prefill are the same dispatch),
// and the state buffer is read once at entry / written once per snapshot — the mlx-lm
// gated_delta_step shape (grid (32, Dv, Hv), tg (32,4,1)) with two llama.cpp steals: the 1/sqrt(Dk)
// scale applied to the OUTPUT (not to q), and K trailing per-token state snapshots written in-pass
// (slot s = the state s tokens back; slot 0 = most recent) so a speculative verify can roll the
// recurrence back to any accept boundary by slot copy instead of recompute.
//
// Contract (all f32; B=1 — the composed lane is single-stream, a batch axis is a later slice):
//   q, k   [T, Hk, Dk]  pre-normalised (the block ℓ2-norms k and rms/ℓ2-norms q upstream; no norm
//                       and no q-scale in here — y is scaled by 1/sqrt(Dk) at write instead)
//   v      [T, Hv, Dv]
//   g,beta [T, Hv]      decay in (0,1] (already exp'd) and write strength (already sigmoid'd)
//   state  [kSlots, Hv, Dv, Dk]  in-place; slot 0 is the live state on entry
//   y      [T, Hv, Dv]
// GQA is index arithmetic (hk = hv/(Hv/Hk)), never materialised. Per step:
//   S *= g; kv = simd_sum(S·k); delta = (v − kv)·beta; S += k·delta; y = simd_sum(S·q)·scale.
// With kSlots > 1 the running state is written to slot (T−1−t) whenever that lands in
// [0, kSlots): the final state IS slot 0, older slots hold trailing snapshots, and slots beyond
// T−1 keep their prior (caller-owned) contents. kSlots <= 1 writes slot 0 once at exit.
// Each simdgroup's state row is disjoint, so the in-place update has no cross-threadgroup hazard.

#include <metal_stdlib>
using namespace metal;

template <short DK>
inline void lthn_gated_delta_step_impl(
    device const float * q,
    device const float * k,
    device const float * v,
    device const float * g,
    device const float * beta,
    device       float * state,
    device       float * y,
    const uint  T,
    const uint  kSlots,
    const uint  Hk,
    const uint  Hv,
    const uint  Dv,
    const uint3 tpig,
    const ushort lane)
{
    constexpr short n_per_t = DK / 32;
    const uint dv = tpig.y;
    const uint hv = tpig.z;
    if (dv >= Dv) { // uniform per simdgroup (all 32 lanes share dv), no divergent simd_sum
        return;
    }
    const uint hk = hv / (Hv / Hk);
    const float scale = rsqrt((float)DK);

    const uint stateRow  = (hv * Dv + dv) * DK; // within a slot
    const uint stateSlot = Hv * Dv * DK;        // one slot's stride
    device float * s0 = state + stateRow;       // slot 0 = the live state

    float st[n_per_t];
    #pragma unroll
    for (short i = 0; i < n_per_t; ++i) {
        st[i] = s0[n_per_t * lane + i];
    }

    device const float * q_ = q + hk * DK;
    device const float * k_ = k + hk * DK;
    device const float * v_ = v + hv * Dv + dv;
    device       float * y_ = y + hv * Dv + dv;

    for (uint t = 0; t < T; ++t) {
        const float gt = g[t * Hv + hv];
        const float bt = beta[t * Hv + hv];

        float kv = 0.0f;
        #pragma unroll
        for (short i = 0; i < n_per_t; ++i) {
            const short s = n_per_t * lane + i;
            st[i] = st[i] * gt;
            kv += st[i] * k_[s];
        }
        kv = simd_sum(kv);

        const float delta = (v_[0] - kv) * bt;

        float out = 0.0f;
        #pragma unroll
        for (short i = 0; i < n_per_t; ++i) {
            const short s = n_per_t * lane + i;
            st[i] = st[i] + k_[s] * delta;
            out += st[i] * q_[s];
        }
        out = simd_sum(out);
        if (lane == 0) {
            y_[0] = out * scale;
        }

        if (kSlots > 1) {
            const int slot = (int)T - 1 - (int)t;
            if (slot >= 0 && slot < (int)kSlots) {
                device float * snap = state + (uint)slot * stateSlot + stateRow;
                #pragma unroll
                for (short i = 0; i < n_per_t; ++i) {
                    snap[n_per_t * lane + i] = st[i];
                }
            }
        }

        q_ += Hk * DK;
        k_ += Hk * DK;
        v_ += Hv * Dv;
        y_ += Hv * Dv;
    }

    if (kSlots <= 1) {
        #pragma unroll
        for (short i = 0; i < n_per_t; ++i) {
            s0[n_per_t * lane + i] = st[i];
        }
    }
}

// Dk is a compile-time instantiation (it sizes the per-lane register array); every Qwen 3.5/3.6
// hybrid ships Dk = 128, and the 64-wide variant serves the small fixture shapes.
#define LTHN_GATED_DELTA_KERNEL(DKV)                                                    \
kernel void lthn_gated_delta_step_f32_dk##DKV(                                          \
    device const float * q      [[buffer(0)]],                                          \
    device const float * k      [[buffer(1)]],                                          \
    device const float * v      [[buffer(2)]],                                          \
    device const float * g      [[buffer(3)]],                                          \
    device const float * beta   [[buffer(4)]],                                          \
    device       float * state  [[buffer(5)]],                                          \
    device       float * y      [[buffer(6)]],                                          \
    constant     int   & T      [[buffer(7)]],                                          \
    constant     int   & kSlots [[buffer(8)]],                                          \
    constant     int   & Hk     [[buffer(9)]],                                          \
    constant     int   & Hv     [[buffer(10)]],                                         \
    constant     int   & Dv     [[buffer(11)]],                                         \
    uint3  tpig [[thread_position_in_grid]],                                            \
    ushort lane [[thread_index_in_simdgroup]])                                          \
{                                                                                       \
    lthn_gated_delta_step_impl<DKV>(q, k, v, g, beta, state, y,                         \
        (uint)T, (uint)kSlots, (uint)Hk, (uint)Hv, (uint)Dv, tpig, lane);               \
}

LTHN_GATED_DELTA_KERNEL(128)
LTHN_GATED_DELTA_KERNEL(64)
