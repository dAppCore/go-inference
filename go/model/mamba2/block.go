// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"math"

	core "dappco.re/go"
)

// block.go assembles the full Mamba-2 block from the two core ops (SSDScanF32, CausalConv1dF32) plus the
// standard pieces, mirroring metal's mixer.Forward exactly (so native can replace it transparently):
//
//	in-proj → split(z | xBC | dt) → causal conv(xBC) → SiLU → split(x | B | C)
//	→ group-expand B/C → dt=softplus(dt+dt_bias) → A=-exp(A_log) → SSD scan
//	→ gated RMSNorm: RMSNorm(y)·SiLU(z) → out-proj
//
// Pure Go over f32 host slices; the conv-state ring + the SSM state thread through for streaming decode.

// BlockWeights is one Mamba-2 layer's f32 weights (the loader widens the bf16 checkpoint into these).
// InProj is [projDim, D] (projDim = 2*dInner + 2*G*N + H); OutProj is [D, dInner]; ConvWeight is
// [convDim, K] (convDim = dInner + 2*G*N); the rest are per-head/per-channel vectors (nil = absent).
type BlockWeights struct {
	InProj     []float32
	ConvWeight []float32
	ConvBias   []float32
	ALog       []float32
	D          []float32
	DtBias     []float32
	Norm       []float32
	OutProj    []float32
}

// BlockConfig is the per-layer SSD geometry.
type BlockConfig struct {
	NumHeads, HeadDim, StateDim, NumGroups, ConvKernel int
	Eps                                                float32
}

func (c BlockConfig) dInner() int  { return c.NumHeads * c.HeadDim }
func (c BlockConfig) convDim() int { return c.dInner() + 2*c.NumGroups*c.StateDim }
func (c BlockConfig) projDim() int { return 2*c.dInner() + 2*c.NumGroups*c.StateDim + c.NumHeads }

func silu(v float64) float64 { return v / (1 + math.Exp(-v)) }
func softplus(v float64) float64 { // log(1+e^v), numerically stable
	if v > 20 {
		return v
	}
	return math.Log1p(math.Exp(v))
}

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

// BlockForwardF32 runs one chunk of the Mamba-2 block over x [L, D], threading the conv-state ring
// (priorConv [(K-1),convDim]) and the SSM state (priorSSM [H,P,N]); both nil for a fresh sequence.
// Returns out [L, D] and the advanced (newConv, newSSM) for the next chunk.
func BlockForwardF32(x []float32, w *BlockWeights, cfg BlockConfig, priorConv, priorSSM []float32, L, D int) (out, newConv, newSSM []float32, err error) {
	if w == nil {
		return nil, nil, nil, core.NewError("mamba2.BlockForwardF32: nil weights")
	}
	H, P, N, G, K := cfg.NumHeads, cfg.HeadDim, cfg.StateDim, cfg.NumGroups, cfg.ConvKernel
	dInner, convDim, projDim := cfg.dInner(), cfg.convDim(), cfg.projDim()
	if len(x) != L*D || len(w.InProj) != projDim*D || len(w.OutProj) != D*dInner {
		return nil, nil, nil, core.NewError("mamba2.BlockForwardF32: x/InProj/OutProj size mismatch")
	}
	if H%G != 0 {
		return nil, nil, nil, core.NewError("mamba2.BlockForwardF32: num_heads must be a multiple of num_groups")
	}

	proj, err := projMatMul(x, w.InProj, L, D, projDim) // [L, projDim] (device GEMM when a backend is wired)
	if err != nil {
		return nil, nil, nil, err
	}
	// split z | xBC | dt along the channel axis.
	z := make([]float32, L*dInner)
	xBC := make([]float32, L*convDim)
	dtRaw := make([]float32, L*H)
	for t := range L {
		row := proj[t*projDim:]
		copy(z[t*dInner:(t+1)*dInner], row[0:dInner])
		copy(xBC[t*convDim:(t+1)*convDim], row[dInner:dInner+convDim])
		copy(dtRaw[t*H:(t+1)*H], row[dInner+convDim:dInner+convDim+H])
	}

	convOut, newConv, err := CausalConv1dF32(xBC, w.ConvWeight, w.ConvBias, priorConv, L, convDim, K)
	if err != nil {
		return nil, nil, nil, err
	}
	for i := range convOut { // SiLU activation after the conv
		convOut[i] = float32(silu(float64(convOut[i])))
	}

	// split conv output x_inner | B | C, expand B/C groups to heads.
	groupDim := G * N
	xHeads := make([]float32, L*H*P) // [L,H,P]
	bHeads := make([]float32, L*H*N) // [L,H,N]
	cHeads := make([]float32, L*H*N)
	headsPerGroup := H / G
	for t := range L {
		crow := convOut[t*convDim:]
		copy(xHeads[t*dInner:(t+1)*dInner], crow[0:dInner]) // dInner == H*P, same layout
		for h := range H {
			g := h / headsPerGroup
			copy(bHeads[(t*H+h)*N:(t*H+h+1)*N], crow[dInner+g*N:dInner+g*N+N])
			copy(cHeads[(t*H+h)*N:(t*H+h+1)*N], crow[dInner+groupDim+g*N:dInner+groupDim+g*N+N])
		}
	}

	// dt = softplus(dt + dt_bias) ; A = -exp(A_log)
	dt := make([]float32, L*H)
	for t := range L {
		for h := range H {
			v := float64(dtRaw[t*H+h])
			if w.DtBias != nil {
				v += float64(w.DtBias[h])
			}
			dt[t*H+h] = float32(softplus(v))
		}
	}
	a := make([]float32, H)
	for h := range H {
		a[h] = float32(-math.Exp(float64(w.ALog[h])))
	}

	y, newSSM, err := SSDScanF32(xHeads, dt, a, bHeads, cHeads, w.D, priorSSM, L, H, P, N)
	if err != nil {
		return nil, nil, nil, err
	}

	// gated RMSNorm (HF/state-spaces MambaRMSNormGated): the gate is applied BEFORE the norm —
	// g = y·SiLU(z), then normalise g and scale by the weight. This is NOT RMSNorm(y)·SiLU(z) (gate
	// after), the form metal's shared flakernel uses: on a real mamba2 checkpoint that gate-after form
	// inflates the activations ~5× and corrupts the logit distribution (confirmed against the HF smoke).
	gated := make([]float32, L*dInner)
	g := make([]float64, dInner)
	for t := range L {
		yr := y[t*dInner : (t+1)*dInner]
		zr := z[t*dInner : (t+1)*dInner]
		var ss float64
		for i := range dInner {
			gi := float64(yr[i]) * silu(float64(zr[i]))
			g[i] = gi
			ss += gi * gi
		}
		rms := math.Sqrt(ss/float64(dInner) + float64(cfg.Eps))
		for i := range dInner {
			normed := g[i] / rms
			if w.Norm != nil {
				normed *= float64(w.Norm[i])
			}
			gated[t*dInner+i] = float32(normed)
		}
	}
	out, err = projMatMul(gated, w.OutProj, L, dInner, D)
	if err != nil {
		return nil, nil, nil, err
	}
	return out, newConv, newSSM, nil
}
