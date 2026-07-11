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
	out := make([]float32, M*N)
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

// BlockForwardF32 runs one chunk of the RWKV-7 time-mix block over x [L, D], threading the single
// [H,K,V] state (prior nil ⇒ fresh). Returns out [L, D] and the advanced state.
func BlockForwardF32(x []float32, w *BlockWeights, cfg BlockConfig, prior []float32, L, D int) (out, state []float32, err error) {
	if w == nil {
		return nil, nil, core.NewError("rwkv7.BlockForwardF32: nil weights")
	}
	H, K, V := cfg.NumHeads, cfg.KeyDim, cfg.ValueDim
	hk, hv := cfg.hk(), cfg.hv()
	if H <= 0 || K <= 0 || V <= 0 || len(x) != L*D {
		return nil, nil, core.NewError("rwkv7.BlockForwardF32: bad geometry or x size")
	}
	if len(w.RProj) != hk*D || len(w.WProj) != hk*D || len(w.KProj) != hk*D || len(w.AProj) != hk*D || len(w.BProj) != hk*D {
		return nil, nil, core.NewError("rwkv7.BlockForwardF32: R/W/K/A/B must each be [H*K, D]")
	}
	if len(w.VProj) != hv*D || len(w.OutProj) != D*hv {
		return nil, nil, core.NewError("rwkv7.BlockForwardF32: V must be [H*V,D] and OutProj [D,H*V]")
	}

	r, err := projMatMul(x, w.RProj, L, D, hk)
	if err != nil {
		return nil, nil, err
	}
	wDecay, err := projMatMul(x, w.WProj, L, D, hk)
	if err != nil {
		return nil, nil, err
	}
	// RWKV-7 log-decay: w = -exp(WProj) ≤ 0. The projection output is dead after this transform, so
	// map it in place rather than allocating a second buffer — bit-identical, one fewer alloc/token.
	for i := range wDecay {
		wDecay[i] = float32(-math.Exp(float64(wDecay[i])))
	}
	kp, err := projMatMul(x, w.KProj, L, D, hk)
	if err != nil {
		return nil, nil, err
	}
	vp, err := projMatMul(x, w.VProj, L, D, hv)
	if err != nil {
		return nil, nil, err
	}
	ap, err := projMatMul(x, w.AProj, L, D, hk)
	if err != nil {
		return nil, nil, err
	}
	bp, err := projMatMul(x, w.BProj, L, D, hk)
	if err != nil {
		return nil, nil, err
	}

	o, state, err := WKV7F32(r, wDecay, kp, vp, ap, bp, prior, L, H, K, V) // o [L, H*V]
	if err != nil {
		return nil, nil, err
	}
	out, err = projMatMul(o, w.OutProj, L, hv, D)
	if err != nil {
		return nil, nil, err
	}
	return out, state, nil
}
