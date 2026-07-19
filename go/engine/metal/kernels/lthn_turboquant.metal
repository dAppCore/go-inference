// SPDX-Licence-Identifier: EUPL-1.2
//
// lthn_turboquant.metal — device-side TurboQuant KV-cache quantisation (arxiv.org/abs/2504.19874,
// RFC #41 slice S2), row-parallel encoder-level primitives proven byte/index-parity against the S1
// host reference (kv/turboquant, go-inference's Householder-QR Haar rotation + Lloyd-Max centroid
// solve — NOT the unrelated FWHT+uniform scheme engine/metal's snapshot-restore path already carries
// under the "TurboQuantKV" name; that is a different, pre-existing codec for a different job). This
// slice builds and proves the KERNELS only — wiring into the live KV cache/SDPA decode path is S3.
//
// Contract (all f32; one threadgroup per ROW, d ≤ 256 — the S1 doc's "64-256 in practice" range):
//   lthn_tq_rotate_quant<BITS>:     y = Π·(x/γ), γ = ||x||₂ per row, each y-coordinate quantised to
//                                   its nearest Lloyd-Max centroid (BITS bits), packed LSB-first
//                                   exactly as kv/turboquant's packBits (bit b of coordinate i sits
//                                   at packed bit i*BITS+b) — so a device/host packed-index diff
//                                   reduces to a plain per-coordinate int compare, no bit-order
//                                   ambiguity.
//   lthn_tq_dequant_unrotate<BITS>: indices -> centroids -> Πᵀ·ỹ·γ back to f32 rows.
// Π is a resident [d,d] row-major buffer (Π[i][j] at i*d+j — reused UNCHANGED for both kernels; the
// dequant kernel reads it column-wise (Πᵀ), exactly mirroring kv/turboquant's matrix.mulVec /
// mulVecT pair, which also share one underlying store). Centroids are a small resident [1<<BITS]
// buffer, ascending, matching kv/turboquant's centroidsFor. γ is computed in-kernel (rotate_quant) or
// supplied (dequant_unrotate, from rotate_quant's own output).
//
// Threadgroup shape is FIXED at 256 threads regardless of the row's actual d (queried per-thread as
// a runtime scalar, not the dispatch width) — d is "64-256 in practice" but is NOT guaranteed a power
// of 2 (e.g. head_dim 80/96 exist in the wild), and the classic halving tree reduction silently drops
// a dangling element on an odd count at some fold step. Padding every reduction to a fixed
// power-of-two span (256) with zero contributions from threads ≥ d sidesteps that class of bug
// entirely, at the cost of a few idle lanes on smaller rows — cheap next to the O(d²) rotation
// already dominating the threadgroup's work.
//
// The b-bit pack is single-threaded BY DESIGN (thread 0, after a barrier, over the threadgroup's
// already-computed index array): sub-byte fields packed by several threads writing into the same
// output byte is a read-modify-write race unless every byte is assembled from a byte-major loop
// (iterate output BYTES, not input values, so each byte is written exactly once with a value that
// does not depend on the destination buffer's prior contents) — which is what this does. That matters
// specifically because the Go wrapper's scratch buffers are POOLED and reused across calls; a naive
// per-value OR-into-device-memory pack would leak stale bits from a wider previous row into a
// narrower later one. d ≤ 256 keeps the serial pass cheap (a fixed, small upper bound, not a
// bottleneck) — a parallel bit-packing scheme is a possible follow-up, not required for correctness.

#include <metal_stdlib>
using namespace metal;

// LTHN_TQ_CAP is the fixed threadgroup width (and per-row shared-memory span) every instantiation
// dispatches, independent of the row's actual dimension d ≤ LTHN_TQ_CAP.
#define LTHN_TQ_CAP 256

// ---------------------------------------------------------------------------------------------
// lthn_tq_rotate_quant<BITS> — encode: x -> (gamma, packed indices).
//
// ABI: x(0) pi(1) centroids(2) gammaOut(3) packedOut(4) d(5).
//   x          [numRows, d]                 f32, row-major
//   pi         [d, d]                       f32, row-major Π[i][j] = pi[i*d+j]
//   centroids  [1<<BITS]                    f32, ascending (kv/turboquant's centroidsFor)
//   gammaOut   [numRows]                    f32, written by lane 0
//   packedOut  [numRows, ceil(d*BITS/8)]    u8,  packBits layout (LSB-first, cross-byte)
//   d          row dimension (≤ LTHN_TQ_CAP)
// Dispatch: grid (numRows, 1, 1), threadgroup (LTHN_TQ_CAP, 1, 1).
template <int BITS>
[[kernel]] void lthn_tq_rotate_quant(
    const device float*  x         [[buffer(0)]],
    const device float*  pi        [[buffer(1)]],
    const device float*  centroids [[buffer(2)]],
    device       float*  gammaOut  [[buffer(3)]],
    device       uint8_t* packedOut [[buffer(4)]],
    constant     int&    d         [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint c   [[thread_position_in_threadgroup]])
{
    constexpr int K = 1 << BITS;
    threadgroup float red[LTHN_TQ_CAP]; // sum-of-squares reduction, then reused as u = x/γ
    threadgroup ushort idxbuf[LTHN_TQ_CAP];

    const bool active = c < (uint)d;
    const device float* xrow = x + (uint)row * (uint)d;
    const float xc = active ? xrow[c] : 0.0f;

    // ---- γ = ||x||₂ over the row (fixed power-of-two tree, zero-padded past d) ----
    red[c] = xc * xc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = LTHN_TQ_CAP / 2; off > 0; off >>= 1) {
        if (c < off) {
            red[c] += red[c + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float gamma = sqrt(red[0]);
    const float invGamma = (gamma > 0.0f) ? (1.0f / gamma) : 0.0f;
    if (c == 0) {
        gammaOut[row] = gamma;
    }

    // ---- u = x/γ, materialised in shared memory (every thread's dot product below needs ALL
    // of u, not just its own coordinate) ----
    red[c] = active ? (xc * invGamma) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- y = Π·u (row i of Π dot u), nearest-centroid pack ----
    if (active) {
        float yc = 0.0f;
        const device float* pirow = pi + c * (uint)d;
        for (int j = 0; j < d; j++) {
            yc += pirow[j] * red[j];
        }
        int best = 0;
        float bestDist = fabs(yc - centroids[0]);
        for (int k = 1; k < K; k++) {
            const float dist = fabs(yc - centroids[k]);
            if (dist < bestDist) {
                bestDist = dist;
                best = k;
            }
        }
        idxbuf[c] = (ushort)best;
    } else {
        idxbuf[c] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- serial byte-major pack (lane 0 only — see the file header for why) ----
    if (c == 0) {
        const int bytesPerRow = (d * BITS + 7) / 8;
        device uint8_t* prow = packedOut + (uint)row * (uint)bytesPerRow;
        for (int byteIdx = 0; byteIdx < bytesPerRow; byteIdx++) {
            uint8_t byteVal = 0;
            for (int bit = 0; bit < 8; bit++) {
                const int pos = byteIdx * 8 + bit;
                const int vi = pos / BITS;
                if (vi >= d) {
                    break;
                }
                const int b = pos % BITS;
                const int v = int(idxbuf[vi]);
                if ((v >> b) & 1) {
                    byteVal |= (uint8_t)(1u << bit);
                }
            }
            prow[byteIdx] = byteVal;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// lthn_tq_dequant_unrotate<BITS> — decode: (packed indices, gamma) -> f32 rows.
//
// ABI: packed(0) pi(1) centroids(2) gamma(3) out(4) d(5). Same shapes as the encoder's outputs;
// pi is the SAME [d,d] buffer, read transposed (Πᵀ) via swapped index arithmetic — mirroring
// kv/turboquant's mulVecT, which reuses the one matrix store rather than materialising a transpose.
// Dispatch: grid (numRows, 1, 1), threadgroup (LTHN_TQ_CAP, 1, 1).
template <int BITS>
[[kernel]] void lthn_tq_dequant_unrotate(
    const device uint8_t* packed    [[buffer(0)]],
    const device float*   pi        [[buffer(1)]],
    const device float*   centroids [[buffer(2)]],
    const device float*   gamma     [[buffer(3)]],
    device       float*   out       [[buffer(4)]],
    constant     int&     d         [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint o   [[thread_position_in_threadgroup]])
{
    threadgroup float ys[LTHN_TQ_CAP]; // centroid value per ROTATED coordinate r

    const bool active = o < (uint)d;
    const int bytesPerRow = (d * BITS + 7) / 8;
    const device uint8_t* prow = packed + (uint)row * (uint)bytesPerRow;

    if (active) {
        // unpack this thread's own coordinate index — pure reads, safe even where a value's bits
        // straddle a byte another thread also reads.
        const int pos0 = int(o) * BITS;
        int idx = 0;
        for (int b = 0; b < BITS; b++) {
            const int pos = pos0 + b;
            const int byteIdx = pos / 8;
            const int bitIdx = pos % 8;
            if ((prow[byteIdx] >> bitIdx) & 1) {
                idx |= (1 << b);
            }
        }
        ys[o] = centroids[idx];
    } else {
        ys[o] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        // u[o] = Σ_r Π[r][o]·y[r] = (Πᵀ·y)[o] — column o of Π, strided by d.
        float uo = 0.0f;
        for (int r = 0; r < d; r++) {
            uo += pi[(uint)r * (uint)d + o] * ys[r];
        }
        out[(uint)row * (uint)d + o] = uo * gamma[row];
    }
}

// ---------------------------------------------------------------------------------------------
#define LTHN_TQ_INSTANTIATE(BITS)                                                              \
  template [[host_name("lthn_tq_rotate_quant_b" #BITS)]] [[kernel]] void                       \
  lthn_tq_rotate_quant<BITS>(                                                                  \
      const device float*, const device float*, const device float*,                           \
      device float*, device uint8_t*, constant int&, uint, uint);                              \
  template [[host_name("lthn_tq_dequant_unrotate_b" #BITS)]] [[kernel]] void                   \
  lthn_tq_dequant_unrotate<BITS>(                                                               \
      const device uint8_t*, const device float*, const device float*,                         \
      const device float*, device float*, constant int&, uint, uint);

LTHN_TQ_INSTANTIATE(2)
LTHN_TQ_INSTANTIATE(3)
LTHN_TQ_INSTANTIATE(4)
