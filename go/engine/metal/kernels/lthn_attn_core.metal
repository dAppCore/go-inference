// SPDX-Licence-Identifier: EUPL-1.2
//
// lthn_attn_core.metal — the composed lane's attention core on device (#26 / #18 S6a): the four
// kernels that replace continueFromQKV's host loops between the fold CBs, so an attention layer
// runs as ONE command buffer over a device-resident KV cache. Ported line-for-line from
// go/model/composed/attention.go (qprep/kprep mirror the de-interleave + per-head norm +
// applyRotaryHalf block; sdpa mirrors the causal max-subtract softmax loop, window clamp
// included; gate mirrors the σ-gate multiply). All f32 — the composed lane's activation tier.
//
// Norm modes (qkNorm constant): 0 = none (the real Qwen 3.5/3.6 attention — no q/k norm tensors),
// 1 = RMS with weight (the fixture arches), 2 = ℓ2. LayerNorm QK-norm declines to the host path.
// RoPE is the HALF convention: pair (i, i+RD/2), angle = pos·theta^(−2i/RD), dims ≥ RD unchanged.

#include <metal_stdlib>
using namespace metal;

// Shared per-row transform: norm (mode) then partial rotary, over one head row staged in
// threadgroup memory (the rope pairing crosses lanes). row is tg memory [HD]; each simdgroup
// owns one row. w may be nil-equivalent (bound to row itself) when qkNorm != 1.
inline void lthn_attn_norm_rope_row(
    threadgroup float * row,
    threadgroup float * orig,
    device const float * w,
    const uint HD, const uint RD, const uint qkNorm,
    const float eps, const float theta, const uint pos,
    const ushort lane)
{
    const uint nPer = (HD + 31) / 32;
    // norm: reduce Σx² across the row.
    float ss = 0.0f;
    for (uint i = 0; i < nPer; ++i) {
        const uint idx = lane * nPer + i;
        if (idx < HD) {
            ss += row[idx] * row[idx];
        }
    }
    ss = simd_sum(ss);
    if (qkNorm == 1u) { // RMS with weight
        const float inv = 1.0f / sqrt(ss / (float)HD + eps);
        for (uint i = 0; i < nPer; ++i) {
            const uint idx = lane * nPer + i;
            if (idx < HD) {
                row[idx] = row[idx] * inv * w[idx];
            }
        }
    } else if (qkNorm == 2u) { // ℓ2
        const float inv = 1.0f / sqrt(ss + eps);
        for (uint i = 0; i < nPer; ++i) {
            const uint idx = lane * nPer + i;
            if (idx < HD) {
                row[idx] = row[idx] * inv;
            }
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // partial rotary, half pairing: snapshot the normed row, then each lane rewrites only its own
    // elements from the snapshot — no cross-lane write hazards.
    for (uint i = 0; i < nPer; ++i) {
        const uint idx = lane * nPer + i;
        if (idx < HD) {
            orig[idx] = row[idx];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint halfRD = RD / 2;
    for (uint i = 0; i < nPer; ++i) {
        const uint idx = lane * nPer + i;
        if (idx >= RD) {
            continue;
        }
        const uint lo = (idx < halfRD) ? idx : idx - halfRD;
        const float freq = exp2(((float)(2 * lo) / (float)RD) * -log2(theta));
        const float ang = (float)pos * freq;
        const float c = cos(ang), sn = sin(ang);
        const float a = orig[lo], b = orig[lo + halfRD];
        row[idx] = (idx < halfRD) ? (a * c - b * sn) : (b * c + a * sn);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// lthn_attn_qprep_f32 — de-interleave the output gate (when gated: qRaw row = per-head
// [q_h(HD); gate_h(HD)]), then norm + rope each q head row. One simdgroup per (head, t):
// grid (32, H, L), tg (32,1,1). Writes q [L,H,HD] and (gated) gate [L,H,HD].
kernel void lthn_attn_qprep_f32(
    device const float * qRaw  [[buffer(0)]], // [L, qCols]
    device const float * w     [[buffer(1)]], // [HD] (qkNorm==1) else ignored
    device       float * q     [[buffer(2)]], // [L, H, HD]
    device       float * gate  [[buffer(3)]], // [L, H, HD] (gated) else ignored
    constant     int   & H      [[buffer(4)]],
    constant     int   & HD     [[buffer(5)]],
    constant     int   & RD     [[buffer(6)]],
    constant     int   & gated  [[buffer(7)]],
    constant     int   & qkNorm [[buffer(8)]],
    constant     float & eps    [[buffer(9)]],
    constant     float & theta  [[buffer(10)]],
    constant     int   & pos0   [[buffer(11)]],
    uint3  tpig [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float row[256];
    threadgroup float orig[256];
    const uint hd = tpig.y;
    const uint t = tpig.z;
    const uint uHD = (uint)HD;
    const uint qCols = (gated != 0) ? (uint)(2 * H * HD) : (uint)(H * HD);
    device const float * src = qRaw + t * qCols + ((gated != 0) ? hd * 2u * uHD : hd * uHD);
    const uint nPer = (uHD + 31) / 32;
    for (uint i = 0; i < nPer; ++i) {
        const uint idx = lane * nPer + i;
        if (idx < uHD) {
            row[idx] = src[idx];
            if (gated != 0) {
                gate[(t * (uint)H + hd) * uHD + idx] = src[uHD + idx];
            }
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    lthn_attn_norm_rope_row(row, orig, w, uHD, (uint)RD, (uint)qkNorm, eps, theta, (uint)pos0 + t, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device float * dst = q + (t * (uint)H + hd) * uHD;
    for (uint i = 0; i < nPer; ++i) {
        const uint idx = lane * nPer + i;
        if (idx < uHD) {
            dst[idx] = row[idx];
        }
    }
}

// lthn_attn_kprep_f32 — norm + rope each k head row and write it into the KV cache slot at
// position pos0+t. One simdgroup per (kv head, t): grid (32, KVH, L). k is the raw projection
// [L, KVH*HD]; cacheK is [cap, KVH, HD] with rows pos0..pos0+L−1 written.
kernel void lthn_attn_kprep_f32(
    device const float * k      [[buffer(0)]],
    device const float * w      [[buffer(1)]],
    device       float * cacheK [[buffer(2)]],
    constant     int   & KVH    [[buffer(3)]],
    constant     int   & HD     [[buffer(4)]],
    constant     int   & RD     [[buffer(5)]],
    constant     int   & qkNorm [[buffer(6)]],
    constant     float & eps    [[buffer(7)]],
    constant     float & theta  [[buffer(8)]],
    constant     int   & pos0   [[buffer(9)]],
    uint3  tpig [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float row[256];
    threadgroup float orig[256];
    const uint hd = tpig.y;
    const uint t = tpig.z;
    const uint uHD = (uint)HD;
    device const float * src = k + (t * (uint)KVH + hd) * uHD;
    const uint nPer = (uHD + 31) / 32;
    for (uint i = 0; i < nPer; ++i) {
        const uint idx = lane * nPer + i;
        if (idx < uHD) {
            row[idx] = src[idx];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    lthn_attn_norm_rope_row(row, orig, w, uHD, (uint)RD, (uint)qkNorm, eps, theta, (uint)pos0 + t, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device float * dst = cacheK + (((uint)pos0 + t) * (uint)KVH + hd) * uHD;
    for (uint i = 0; i < nPer; ++i) {
        const uint idx = lane * nPer + i;
        if (idx < uHD) {
            dst[idx] = row[idx];
        }
    }
}

// lthn_attn_sdpa_f32 — causal GQA attention over the device cache: query t (position pos0+t)
// attends keys first..pos0+t (first = the sliding-window clamp), online max-rescaled softmax,
// weighted V sum. One simdgroup per (head, t): grid (32, H, L). out [L,H,HD].
kernel void lthn_attn_sdpa_f32(
    device const float * q      [[buffer(0)]],  // [L, H, HD] (prepped)
    device const float * cacheK [[buffer(1)]],  // [cap, KVH, HD]
    device const float * cacheV [[buffer(2)]],  // [cap, KVH, HD]
    device       float * out    [[buffer(3)]],  // [L, H, HD]
    constant     int   & H      [[buffer(4)]],
    constant     int   & KVH    [[buffer(5)]],
    constant     int   & HD     [[buffer(6)]],
    constant     int   & pos0   [[buffer(7)]],
    constant     int   & window [[buffer(8)]],  // 0 = full causal
    uint3  tpig [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint hd = tpig.y;
    const uint t = tpig.z;
    const uint uHD = (uint)HD;
    const uint rep = (uint)H / (uint)KVH;
    const uint kvh = hd / rep;
    const float scale = 1.0f / sqrt((float)HD);
    const uint last = (uint)pos0 + t; // inclusive
    uint first = 0;
    if (window > 0 && last + 1 > (uint)window) {
        first = last + 1 - (uint)window;
    }
    const uint nPer = (uHD + 31) / 32;
    float qv[8];
    device const float * qrow = q + (t * (uint)H + hd) * uHD;
    for (uint i = 0; i < nPer; ++i) {
        const uint idx = lane * nPer + i;
        qv[i] = (idx < uHD) ? qrow[idx] * scale : 0.0f;
    }
    float acc[8];
    for (uint i = 0; i < nPer; ++i) {
        acc[i] = 0.0f;
    }
    float m = -INFINITY;
    float sum = 0.0f;
    for (uint j = first; j <= last; ++j) {
        device const float * krow = cacheK + (j * (uint)KVH + kvh) * uHD;
        float dot = 0.0f;
        for (uint i = 0; i < nPer; ++i) {
            const uint idx = lane * nPer + i;
            if (idx < uHD) {
                dot += qv[i] * krow[idx];
            }
        }
        dot = simd_sum(dot);
        const float mNew = max(m, dot);
        const float rescale = (m == -INFINITY) ? 0.0f : exp(m - mNew);
        const float p = exp(dot - mNew);
        sum = sum * rescale + p;
        device const float * vrow = cacheV + (j * (uint)KVH + kvh) * uHD;
        for (uint i = 0; i < nPer; ++i) {
            const uint idx = lane * nPer + i;
            if (idx < uHD) {
                acc[i] = acc[i] * rescale + p * vrow[idx];
            }
        }
        m = mNew;
    }
    device float * orow = out + (t * (uint)H + hd) * uHD;
    const float inv = 1.0f / sum;
    for (uint i = 0; i < nPer; ++i) {
        const uint idx = lane * nPer + i;
        if (idx < uHD) {
            orow[idx] = acc[i] * inv;
        }
    }
}

// lthn_attn_gate_sigmoid_f32 — the σ-gate: out[i] *= sigmoid(gate[i]), elementwise over
// [L·H·HD]. The transformers qwen3_5 reference hardcodes SIGMOID here (output_gate_type is not
// consumed) — attention.go's host gate does the same; an earlier silu reading of the σ was wrong
// on both sides of the original parity test.
kernel void lthn_attn_gate_sigmoid_f32(
    device       float * out   [[buffer(0)]],
    device const float * gate  [[buffer(1)]],
    constant     int   & total [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)total) {
        return;
    }
    out[gid] *= 1.0f / (1.0f + exp(-gate[gid]));
}
