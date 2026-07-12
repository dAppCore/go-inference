// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"math"

	core "dappco.re/go"
)

// block.go assembles the RWKV-7 time-mix block from WKV7F32 + the projections, mirroring metal's
// mixer.Forward exactly (so native can replace it): project r/k/v/a/b and the log-decay w = -exp(WProj)
// from the hidden state, run the WKV7 recurrence, project the read-out back. Simpler than the mamba2
// block — a single [H,K,V] state, no causal conv, no gated norm. The in/out projections go through the
// ProjMatMul hook (host matNT by default; native injects a device GEMM, see backend.go).

// BlockWeights is one RWKV-7 layer's f32 projection weights (the loader widens the bf16 checkpoint).
// R/W/K/A/B project the hidden state [D] to [H*K]; V to [H*V]; OutProj maps the read-out [H*V] back to
// [D]. Each is row-major [outDim, D] (OutProj is [D, H*V]) — the y = x·Wᵀ Linear convention.
type BlockWeights struct {
	RProj   []float32
	WProj   []float32
	KProj   []float32
	VProj   []float32
	AProj   []float32
	BProj   []float32
	OutProj []float32
}

// BlockConfig is the per-layer WKV7 geometry.
type BlockConfig struct {
	NumHeads, KeyDim, ValueDim int // H, K, V
}

func (c BlockConfig) hk() int { return c.NumHeads * c.KeyDim }
func (c BlockConfig) hv() int { return c.NumHeads * c.ValueDim }

// matNT computes out[M,N] = in[M,K] @ w[N,K]ᵀ (the Linear y = x·Wᵀ), f32 host.
func matNT(in, w []float32, M, K, N int) []float32 {
	return matNTInto(nil, in, w, M, K, N)
}

// matNTInto is matNT writing into out, reusing it when cap(out) ≥ M·N (else it allocates a fresh M·N
// slab). Identical f64 accumulation + write order to the fresh-buffer form — only WHERE the result lands
// changes, so the output is bit-identical.
func matNTInto(out, in, w []float32, M, K, N int) []float32 {
	if cap(out) < M*N {
		out = make([]float32, M*N)
	} else {
		out = out[:M*N]
	}
	for m := range M {
		for n := range N {
			var acc float64
			for k := range K {
				acc += float64(in[m*K+k]) * float64(w[n*K+k])
			}
			out[m*N+n] = float32(acc)
		}
	}
	return out
}

// BlockScratch holds the reusable projection-output buffers for BlockForwardScratchF32 — the six input
// projections (r/w/k/v/a/b) plus the out-proj. BlockForwardScratchNoProjF32 uses the same Scratch but
// leaves the out-proj buffer untouched (it stops one GEMM short). A caller stepping one sequence (a decode
// session — single-goroutine) keeps one Scratch and passes it every token so the projection GEMMs write
// into resident buffers instead of allocating. NEVER share a Scratch across concurrently-stepped sessions:
// the buffers are mutable and unsynchronised (they mirror the recurrent [H,K,V] state's ownership —
// per-session, threaded, never on the shared weights). Buffers grow to fit and are reused thereafter.
type BlockScratch struct {
	r, wDecay, kp, vp, ap, bp, out []float32
}

// BlockForwardF32 is BlockForwardScratchF32 with a fresh (nil) scratch — every projection allocates, the
// behaviour before the write-into seam. Kept for existing callers and the engine backend parity tests;
// bit-identical to the scratch path.
func BlockForwardF32(x []float32, w *BlockWeights, cfg BlockConfig, prior []float32, L, D int) (out, state []float32, err error) {
	return BlockForwardScratchF32(x, w, cfg, prior, L, D, nil)
}

// BlockForwardScratchF32 runs one chunk of the RWKV-7 time-mix block over x [L, D], threading the single
// [H,K,V] state (prior nil ⇒ fresh). Returns out [L, D] and the advanced state. When sc is non-nil the
// seven projection outputs write into its buffers (reused across calls); nil ⇒ each allocates fresh (the
// BlockForwardF32 path). The recurrent state is always freshly allocated — carried information, not
// scratch. It is BlockForwardScratchNoProjF32 plus the out_proj GEMM — the projection is split out so the
// composed session can instead fold it into the FFN-tail command buffer (see composed.projMixer /
// ResidualNormMLPProjDevice).
func BlockForwardScratchF32(x []float32, w *BlockWeights, cfg BlockConfig, prior []float32, L, D int, sc *BlockScratch) (out, state []float32, err error) {
	if sc == nil {
		sc = &BlockScratch{} // throwaway: nil buffers ⇒ every projection allocates, the legacy path
	}
	o, hv, state, err := BlockForwardScratchNoProjF32(x, w, cfg, prior, L, D, sc)
	if err != nil {
		return nil, nil, err
	}
	if len(w.OutProj) != D*hv {
		return nil, nil, core.NewError("rwkv7.BlockForwardF32: OutProj must be [D,H*V]")
	}
	out, err = projMatMulInto(sc.out, o, w.OutProj, L, hv, D)
	if err != nil {
		return nil, nil, err
	}
	sc.out = out
	return out, state, nil
}

// BlockForwardScratchNoProjF32 is BlockForwardScratchF32 up to but NOT including out_proj: it returns the
// recurrence read-out o [L, H*V] (the WKV7 output — RWKV-7's time-mix has no gate/norm between the
// recurrence and out_proj, unlike mamba2/gated-delta's gated RMSNorm), hv (=H*V), and the advanced [H,K,V]
// state. The composed session uses it to fold out_proj into the FFN-tail command buffer
// (composed.projMixer); BlockForwardScratchF32 wraps it with the out_proj GEMM. The state advances
// identically — only the final projection is deferred to the caller. sc's six input-projection buffers
// are reused as in the wrapper; nil ⇒ each allocates fresh.
func BlockForwardScratchNoProjF32(x []float32, w *BlockWeights, cfg BlockConfig, prior []float32, L, D int, sc *BlockScratch) (o []float32, hv int, state []float32, err error) {
	if sc == nil {
		sc = &BlockScratch{} // throwaway: nil buffers ⇒ every projection allocates, the legacy path
	}
	if w == nil {
		return nil, 0, nil, core.NewError("rwkv7.BlockForwardF32: nil weights")
	}
	H, K, V := cfg.NumHeads, cfg.KeyDim, cfg.ValueDim
	hk, hv := cfg.hk(), cfg.hv()
	if H <= 0 || K <= 0 || V <= 0 || len(x) != L*D {
		return nil, 0, nil, core.NewError("rwkv7.BlockForwardF32: bad geometry or x size")
	}
	if len(w.RProj) != hk*D || len(w.WProj) != hk*D || len(w.KProj) != hk*D || len(w.AProj) != hk*D || len(w.BProj) != hk*D {
		return nil, 0, nil, core.NewError("rwkv7.BlockForwardF32: R/W/K/A/B must each be [H*K, D]")
	}
	if len(w.VProj) != hv*D {
		return nil, 0, nil, core.NewError("rwkv7.BlockForwardF32: V must be [H*V,D]")
	}

	r, err := projMatMulInto(sc.r, x, w.RProj, L, D, hk)
	if err != nil {
		return nil, 0, nil, err
	}
	sc.r = r
	wDecay, err := projMatMulInto(sc.wDecay, x, w.WProj, L, D, hk)
	if err != nil {
		return nil, 0, nil, err
	}
	sc.wDecay = wDecay
	kp, err := projMatMulInto(sc.kp, x, w.KProj, L, D, hk)
	if err != nil {
		return nil, 0, nil, err
	}
	sc.kp = kp
	vp, err := projMatMulInto(sc.vp, x, w.VProj, L, D, hv)
	if err != nil {
		return nil, 0, nil, err
	}
	sc.vp = vp
	ap, err := projMatMulInto(sc.ap, x, w.AProj, L, D, hk)
	if err != nil {
		return nil, 0, nil, err
	}
	sc.ap = ap
	bp, err := projMatMulInto(sc.bp, x, w.BProj, L, D, hk)
	if err != nil {
		return nil, 0, nil, err
	}
	sc.bp = bp
	return BlockForwardScratchFromInputF32(r, wDecay, kp, vp, ap, bp, w, cfg, prior, L, D, sc)
}

// BlockForwardScratchFromInputF32 resumes RWKV-7 from its six ALREADY-COMPUTED input projections.
// wDecay is the raw WProj result and is transformed to -exp(w) here, matching the ordinary path. This
// lets a predecessor tail compute the input RMSNorm and all six GEMMs in its command buffer without
// changing the recurrence or deferred out_proj result.
func BlockForwardScratchFromInputF32(r, wDecay, kp, vp, ap, bp []float32, w *BlockWeights, cfg BlockConfig, prior []float32, L, D int, sc *BlockScratch) (o []float32, hv int, state []float32, err error) {
	if sc == nil {
		sc = &BlockScratch{}
	}
	if w == nil {
		return nil, 0, nil, core.NewError("rwkv7.BlockForwardF32: nil weights")
	}
	H, K, V := cfg.NumHeads, cfg.KeyDim, cfg.ValueDim
	hk, hv := cfg.hk(), cfg.hv()
	if H <= 0 || K <= 0 || V <= 0 || len(r) != L*hk || len(wDecay) != L*hk || len(kp) != L*hk ||
		len(vp) != L*hv || len(ap) != L*hk || len(bp) != L*hk {
		return nil, 0, nil, core.NewError("rwkv7.BlockForwardF32: projected input size mismatch")
	}
	for i := range wDecay {
		wDecay[i] = float32(-math.Exp(float64(wDecay[i])))
	}

	o, state, err = WKV7F32(r, wDecay, kp, vp, ap, bp, prior, L, H, K, V) // o [L, H*V]
	if err != nil {
		return nil, 0, nil, err
	}
	return o, hv, state, nil
}
