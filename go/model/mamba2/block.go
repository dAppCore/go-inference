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

// BlockScratch holds the reusable projection-output buffers for BlockForwardScratchF32 — the in-proj
// [L,projDim] and the out-proj [L,D]. BlockForwardScratchNoProjF32 uses the same Scratch but leaves the
// out-proj buffer untouched (it stops one GEMM short). A caller stepping one sequence (a decode session —
// single-goroutine) keeps one Scratch and passes it every token so the projection GEMMs write into
// resident buffers instead of allocating. NEVER share a Scratch across concurrently-stepped sessions: the
// buffers are mutable and unsynchronised (they mirror the recurrent conv/SSM state's ownership —
// per-session, threaded, never on the shared weights). Buffers grow to fit and are reused thereafter.
type BlockScratch struct {
	proj, out []float32
}

// BlockForwardF32 is BlockForwardScratchF32 with a fresh (nil) scratch — both projections allocate, the
// behaviour before the write-into seam. Kept for existing callers and the engine backend parity tests;
// bit-identical to the scratch path.
func BlockForwardF32(x []float32, w *BlockWeights, cfg BlockConfig, priorConv, priorSSM []float32, L, D int) (out, newConv, newSSM []float32, err error) {
	return BlockForwardScratchF32(x, w, cfg, priorConv, priorSSM, L, D, nil)
}

// BlockForwardScratchF32 runs one chunk of the Mamba-2 block over x [L, D], threading the conv-state ring
// (priorConv [(K-1),convDim]) and the SSM state (priorSSM [H,P,N]); both nil for a fresh sequence.
// Returns out [L, D] and the advanced (newConv, newSSM) for the next chunk. When sc is non-nil the in/out
// projection outputs write into its buffers (reused across calls); nil ⇒ each allocates fresh (the
// BlockForwardF32 path). The recurrent state (newConv/newSSM) is always freshly allocated — carried
// information, not scratch. It is BlockForwardScratchNoProjF32 plus the out_proj GEMM — the projection is
// split out so the composed session can instead fold it into the FFN-tail command buffer (see
// composed.projMixer / ResidualNormMLPProjDevice).
func BlockForwardScratchF32(x []float32, w *BlockWeights, cfg BlockConfig, priorConv, priorSSM []float32, L, D int, sc *BlockScratch) (out, newConv, newSSM []float32, err error) {
	if sc == nil {
		sc = &BlockScratch{} // throwaway: nil buffers ⇒ both projections allocate, the legacy path
	}
	gated, dInner, newConv, newSSM, err := BlockForwardScratchNoProjF32(x, w, cfg, priorConv, priorSSM, L, D, sc)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(w.OutProj) != D*dInner {
		return nil, nil, nil, core.NewError("mamba2.BlockForwardF32: x/InProj/OutProj size mismatch")
	}
	out, err = projMatMulInto(sc.out, gated, w.OutProj, L, dInner, D)
	if err != nil {
		return nil, nil, nil, err
	}
	sc.out = out
	return out, newConv, newSSM, nil
}

// BlockForwardScratchNoProjF32 is BlockForwardScratchF32 up to but NOT including out_proj: it returns the
// gated pre-projection hidden [L, dInner] (per-token gate-before-norm RMSNorm — see the doc note below),
// dInner itself, and the advanced (newConv, newSSM). The composed session uses it to fold out_proj into
// the FFN-tail command buffer (composed.projMixer); BlockForwardScratchF32 wraps it with the out_proj
// GEMM. The state advances identically — only the final projection is deferred to the caller. sc's
// in-proj buffer is reused as in the wrapper; nil ⇒ it allocates fresh.
func BlockForwardScratchNoProjF32(x []float32, w *BlockWeights, cfg BlockConfig, priorConv, priorSSM []float32, L, D int, sc *BlockScratch) (gated []float32, dInner int, newConv, newSSM []float32, err error) {
	if sc == nil {
		sc = &BlockScratch{} // throwaway: nil buffers ⇒ the in-proj allocates, the legacy path
	}
	if w == nil {
		return nil, 0, nil, nil, core.NewError("mamba2.BlockForwardF32: nil weights")
	}
	H, G := cfg.NumHeads, cfg.NumGroups
	projDim := cfg.projDim()
	if len(x) != L*D || len(w.InProj) != projDim*D {
		return nil, 0, nil, nil, core.NewError("mamba2.BlockForwardF32: x/InProj size mismatch")
	}
	if H%G != 0 {
		return nil, 0, nil, nil, core.NewError("mamba2.BlockForwardF32: num_heads must be a multiple of num_groups")
	}

	proj, err := projMatMulInto(sc.proj, x, w.InProj, L, D, projDim) // [L, projDim] (device GEMM when a backend is wired)
	if err != nil {
		return nil, 0, nil, nil, err
	}
	sc.proj = proj
	return BlockForwardScratchFromInputF32(proj, w, cfg, priorConv, priorSSM, L, D, sc)
}

// BlockForwardScratchFromInputF32 resumes a Mamba-2 block from an ALREADY-COMPUTED in_proj output
// [L,projDim]. It is the recurrence half of BlockForwardScratchNoProjF32: callers that fold the input
// RMSNorm and InProj GEMM into a predecessor's command buffer can skip that standalone projection while
// preserving the causal-conv/SSM state transition and the gated pre-out_proj result byte for byte.
func BlockForwardScratchFromInputF32(proj []float32, w *BlockWeights, cfg BlockConfig, priorConv, priorSSM []float32, L, D int, sc *BlockScratch) (gated []float32, dInner int, newConv, newSSM []float32, err error) {
	if sc == nil {
		sc = &BlockScratch{}
	}
	if w == nil {
		return nil, 0, nil, nil, core.NewError("mamba2.BlockForwardF32: nil weights")
	}
	H, P, N, G, K := cfg.NumHeads, cfg.HeadDim, cfg.StateDim, cfg.NumGroups, cfg.ConvKernel
	dInner, convDim, projDim := cfg.dInner(), cfg.convDim(), cfg.projDim()
	if len(proj) != L*projDim {
		return nil, 0, nil, nil, core.NewError("mamba2.BlockForwardF32: projected input size mismatch")
	}
	if H%G != 0 {
		return nil, 0, nil, nil, core.NewError("mamba2.BlockForwardF32: num_heads must be a multiple of num_groups")
	}
	// split z | xBC | dt along the channel axis. One backing slab, three non-overlapping capped
	// windows: each is filled once then read-only (xBC feeds the conv, dtRaw the dt map, z the gate),
	// so the slab is bit-identical to three makes and saves 2 allocs per block per token.
	zN, xbcN, dtN := L*dInner, L*convDim, L*H
	splitBuf := make([]float32, zN+xbcN+dtN)
	z := splitBuf[0:zN:zN]
	xBC := splitBuf[zN : zN+xbcN : zN+xbcN]
	dtRaw := splitBuf[zN+xbcN : zN+xbcN+dtN : zN+xbcN+dtN]
	for t := range L {
		row := proj[t*projDim:]
		copy(z[t*dInner:(t+1)*dInner], row[0:dInner])
		copy(xBC[t*convDim:(t+1)*convDim], row[dInner:dInner+convDim])
		copy(dtRaw[t*H:(t+1)*H], row[dInner+convDim:dInner+convDim+H])
	}

	convOut, newConv, err := CausalConv1dF32(xBC, w.ConvWeight, w.ConvBias, priorConv, L, convDim, K)
	if err != nil {
		return nil, 0, nil, nil, err
	}
	for i := range convOut { // SiLU activation after the conv
		convOut[i] = float32(silu(float64(convOut[i])))
	}

	// split conv output x_inner | B | C, expand B/C groups to heads. One backing slab, three capped
	// windows (xHeads [L,H,P] | bHeads [L,H,N] | cHeads [L,H,N]) — all read-only inputs to the scan.
	groupDim := G * N
	xhN, bhN, chN := L*H*P, L*H*N, L*H*N
	headBuf := make([]float32, xhN+bhN+chN)
	xHeads := headBuf[0:xhN:xhN]
	bHeads := headBuf[xhN : xhN+bhN : xhN+bhN]
	cHeads := headBuf[xhN+bhN : xhN+bhN+chN : xhN+bhN+chN]
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

	// dt = softplus(dt + dt_bias) ; A = -exp(A_log). One slab, two capped windows (dt [L,H] | a [H]) —
	// both read-only inputs to the scan.
	dtaBuf := make([]float32, L*H+H)
	dt := dtaBuf[0 : L*H : L*H]
	a := dtaBuf[L*H : L*H+H : L*H+H]
	for t := range L {
		for h := range H {
			v := float64(dtRaw[t*H+h])
			if w.DtBias != nil {
				v += float64(w.DtBias[h])
			}
			dt[t*H+h] = float32(softplus(v))
		}
	}
	for h := range H {
		a[h] = float32(-math.Exp(float64(w.ALog[h])))
	}

	y, newSSM, err := SSDScanF32(xHeads, dt, a, bHeads, cHeads, w.D, priorSSM, L, H, P, N)
	if err != nil {
		return nil, 0, nil, nil, err
	}

	// gated RMSNorm (HF/state-spaces MambaRMSNormGated): the gate is applied BEFORE the norm —
	// g = y·SiLU(z), then normalise g and scale by the weight. This is NOT RMSNorm(y)·SiLU(z) (gate
	// after), the form metal's shared flakernel uses: on a real mamba2 checkpoint that gate-after form
	// inflates the activations ~5× and corrupts the logit distribution (confirmed against the HF smoke).
	// The scan output y is written in place as the gated result: within each row all of y's row is read
	// into g (f64) before that row is overwritten, and y is dead after this stage, so reusing it is
	// bit-identical and drops the gated buffer.
	gated = y
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
	return gated, dInner, newConv, newSSM, nil
}
