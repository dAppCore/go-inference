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

// ---------------------------------------------------------------------------------------------
// S2 stage kernels — the pre/post stages around the recurrence, so the whole gated-delta block
// rides one command buffer (docs/design-hybrid-recurrence.md S2). All f32, layouts as the host
// stages in qwen3.GatedDeltaForwardScratchFromInputF32.

// lthn_gd_conv_silu_split_norm — causal depthwise conv over the ring-padded qkv rows, SiLU, split
// into q|k|v head rows, and ℓ2-normalisation of the q/k rows (eps 1e-6, the host formula), in one
// dispatch. One simdgroup per (row, t): rows 0..Hk-1 = q heads, Hk..2Hk-1 = k heads,
// 2Hk..2Hk+Hv-1 = v heads; lane L owns DK/32 contiguous channels of that head row. The padded
// sequence is [ring (K-1 rows); qkv (T rows)]: padded[t+k] with t+k < K-1 reads the ring, else
// qkv row (t+k-(K-1)) — bit-exact with mamba2.CausalConv1dF32's window. Dk == Dv by family truth
// (the Go guard enforces it); DK is the compile-time instantiation.
template <short DK>
inline void lthn_gd_conv_silu_split_norm_impl(
    device const float * ring,   // [(K-1), convDim]
    device const float * qkv,    // [T, convDim]
    device const float * convW,  // [convDim, K]
    device const float * convB,  // [convDim] (hasBias == 0 -> ignored)
    device       float * qOut,   // [T, Hk, DK] ℓ2-normalised
    device       float * kOut,   // [T, Hk, DK] ℓ2-normalised
    device       float * vOut,   // [T, Hv, DK]
    const uint T, const uint K, const uint Hk, const uint Hv, const uint hasBias,
    const uint3 tpig, const ushort lane)
{
    constexpr short n_per = DK / 32;
    const uint row = tpig.y; // 0..2Hk+Hv-1
    const uint t   = tpig.z;
    const uint convDim = (2u * Hk + Hv) * DK;
    const uint qDim = Hk * DK;

    uint ch0;            // this head row's first channel in conv space
    device float * dst;  // this head row's output base
    bool normed;
    if (row < Hk) {
        ch0 = row * DK;
        dst = qOut + (t * Hk + row) * DK;
        normed = true;
    } else if (row < 2u * Hk) {
        ch0 = qDim + (row - Hk) * DK;
        dst = kOut + (t * Hk + (row - Hk)) * DK;
        normed = true;
    } else {
        ch0 = 2u * qDim + (row - 2u * Hk) * DK;
        dst = vOut + (t * Hv + (row - 2u * Hk)) * DK;
        normed = false;
    }

    float val[n_per];
    float ss = 0.0f;
    #pragma unroll
    for (short i = 0; i < n_per; ++i) {
        const uint ch = ch0 + (uint)(n_per * lane + i);
        float acc = (hasBias != 0u) ? convB[ch] : 0.0f;
        for (uint k = 0; k < K; ++k) {
            const uint pr = t + k; // padded row
            const float x = (pr < K - 1u) ? ring[pr * convDim + ch]
                                          : qkv[(pr - (K - 1u)) * convDim + ch];
            acc += convW[ch * K + k] * x;
        }
        val[i] = acc / (1.0f + exp(-acc)); // SiLU
        ss += val[i] * val[i];
    }
    if (normed) {
        ss = simd_sum(ss);
        const float inv = 1.0f / sqrt(ss + 1e-6f);
        #pragma unroll
        for (short i = 0; i < n_per; ++i) {
            dst[n_per * lane + i] = val[i] * inv;
        }
    } else {
        #pragma unroll
        for (short i = 0; i < n_per; ++i) {
            dst[n_per * lane + i] = val[i];
        }
    }
}

#define LTHN_GD_CONV_KERNEL(DKV)                                                        \
kernel void lthn_gd_conv_silu_split_norm_dk##DKV(                                       \
    device const float * ring   [[buffer(0)]],                                          \
    device const float * qkv    [[buffer(1)]],                                          \
    device const float * convW  [[buffer(2)]],                                          \
    device const float * convB  [[buffer(3)]],                                          \
    device       float * qOut   [[buffer(4)]],                                          \
    device       float * kOut   [[buffer(5)]],                                          \
    device       float * vOut   [[buffer(6)]],                                          \
    constant     int   & T      [[buffer(7)]],                                          \
    constant     int   & K      [[buffer(8)]],                                          \
    constant     int   & Hk     [[buffer(9)]],                                          \
    constant     int   & Hv     [[buffer(10)]],                                         \
    constant     int   & hasBias[[buffer(11)]],                                         \
    uint3  tpig [[thread_position_in_grid]],                                            \
    ushort lane [[thread_index_in_simdgroup]])                                          \
{                                                                                       \
    lthn_gd_conv_silu_split_norm_impl<DKV>(ring, qkv, convW, convB, qOut, kOut, vOut,   \
        (uint)T, (uint)K, (uint)Hk, (uint)Hv, (uint)hasBias, tpig, lane);               \
}

LTHN_GD_CONV_KERNEL(128)
LTHN_GD_CONV_KERNEL(64)

// lthn_gd_ring_advance — the conv ring update: ring' = the last K-1 rows of [ring; qkv]. One
// thread per channel owns its whole column: it reads all K-1 source values into registers FIRST,
// then writes — no cross-thread hazard even though ring updates in place.
kernel void lthn_gd_ring_advance(
    device       float * ring [[buffer(0)]], // [(K-1), convDim], updated in place
    device const float * qkv  [[buffer(1)]], // [T, convDim]
    constant     int   & T    [[buffer(2)]],
    constant     int   & K    [[buffer(3)]],
    constant     int   & convDim [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)convDim) {
        return;
    }
    const uint pad = (uint)K - 1u;
    float keep[8]; // K <= 9 by construction (family K = 4)
    for (uint j = 0; j < pad; ++j) {
        const uint src = (uint)T + j; // padded row index
        keep[j] = (src < pad) ? ring[src * (uint)convDim + gid]
                              : qkv[(src - pad) * (uint)convDim + gid];
    }
    for (uint j = 0; j < pad; ++j) {
        ring[j * (uint)convDim + gid] = keep[j];
    }
}

// lthn_gd_gates — the α/β gate transform, elementwise over [T, Hv]:
// α = exp(−exp(A_log[h])·softplus(a + dt_bias[h])), β = sigmoid(b) — the host formulas verbatim
// (softplus guards v > 20 exactly as qwen3.gdSoftplus does).
kernel void lthn_gd_gates(
    device const float * a      [[buffer(0)]], // [T, Hv]
    device const float * b      [[buffer(1)]], // [T, Hv]
    device const float * aLog   [[buffer(2)]], // [Hv]
    device const float * dtBias [[buffer(3)]], // [Hv]
    device       float * gOut   [[buffer(4)]], // [T, Hv]
    device       float * bOut   [[buffer(5)]], // [T, Hv]
    constant     int   & total  [[buffer(6)]], // T*Hv
    constant     int   & Hv     [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)total) {
        return;
    }
    const uint h = gid % (uint)Hv;
    const float v = a[gid] + dtBias[h];
    const float sp = (v > 20.0f) ? v : log(1.0f + exp(v));
    gOut[gid] = exp(-exp(aLog[h]) * sp);
    bOut[gid] = 1.0f / (1.0f + exp(-b[gid]));
}

// lthn_gd_gated_rmsnorm_silu — the block's output gate: per (t, value-head) row over Dv,
// out = RMSNorm(o)·w · SiLU(z). One simdgroup per row, lane-blocked like the recurrence.
template <short DV>
inline void lthn_gd_gated_rmsnorm_silu_impl(
    device const float * o,    // [rows, DV]
    device const float * z,    // [rows, DV]
    device const float * w,    // [DV]
    device       float * out,  // [rows, DV]
    const uint rows, const float eps,
    const uint3 tpig, const ushort lane)
{
    constexpr short n_per = DV / 32;
    const uint row = tpig.y;
    if (row >= rows) {
        return;
    }
    device const float * o_ = o + row * DV;
    device const float * z_ = z + row * DV;
    device       float * y_ = out + row * DV;
    float ov[n_per];
    float ss = 0.0f;
    #pragma unroll
    for (short i = 0; i < n_per; ++i) {
        ov[i] = o_[n_per * lane + i];
        ss += ov[i] * ov[i];
    }
    ss = simd_sum(ss);
    const float rms = sqrt(ss / (float)DV + eps);
    #pragma unroll
    for (short i = 0; i < n_per; ++i) {
        const uint idx = (uint)(n_per * lane + i);
        const float zv = z_[idx];
        y_[idx] = (ov[i] / rms) * w[idx] * (zv / (1.0f + exp(-zv)));
    }
}

#define LTHN_GD_GATED_NORM_KERNEL(DVV)                                                  \
kernel void lthn_gd_gated_rmsnorm_silu_dv##DVV(                                         \
    device const float * o    [[buffer(0)]],                                            \
    device const float * z    [[buffer(1)]],                                            \
    device const float * w    [[buffer(2)]],                                            \
    device       float * out  [[buffer(3)]],                                            \
    constant     int   & rows [[buffer(4)]],                                            \
    constant     float & eps  [[buffer(5)]],                                            \
    uint3  tpig [[thread_position_in_grid]],                                            \
    ushort lane [[thread_index_in_simdgroup]])                                          \
{                                                                                       \
    lthn_gd_gated_rmsnorm_silu_impl<DVV>(o, z, w, out, (uint)rows, eps, tpig, lane);    \
}

LTHN_GD_GATED_NORM_KERNEL(128)
LTHN_GD_GATED_NORM_KERNEL(64)
