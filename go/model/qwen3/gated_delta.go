// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	"math"
	"runtime"

	core "dappco.re/go"
	"dappco.re/go/inference/model/deltanet"
	"dappco.re/go/inference/model/mamba2"
)

// gated_delta.go is the Qwen 3.6 GatedDeltaNet linear-attention block (the "linear_attention" layers of
// the hybrid schedule), mirroring metal's GatedDeltaMixer.Forward exactly so native can serve it:
//
//	in_proj_qkv → causal depthwise conv (ring) → SiLU → split q|k|v → GQA-repeat(q,k: key→value heads)
//	→ l2norm(q) → α=exp(−exp(A_log)·softplus(a+dt_bias)), β=sigmoid(b)
//	→ GatedDeltaRule → gated RMSNorm: RMSNorm(o)·SiLU(z) → out_proj
//
// The causal conv is reused from mamba2 (the shared FLA primitive metal keeps in flakernel), the
// recurrence from pkg/model/deltanet. Pure Go host f32; the conv-state ring + the delta state thread for
// decode. Projections go through ProjMatMul (host matNT default; native injects a device GEMM).

// GatedDeltaConfig is the per-layer geometry. The delta state is square, so KeyHeadDim == ValueHeadDim ==
// HeadDim; q/k use KeyHeads (GQA-repeated up to ValueHeads), v uses ValueHeads.
type GatedDeltaConfig struct {
	KeyHeads, ValueHeads, HeadDim, ConvKernel int
	Eps                                       float32
}

func (c GatedDeltaConfig) qDim() int    { return c.KeyHeads * c.HeadDim }
func (c GatedDeltaConfig) vDim() int    { return c.ValueHeads * c.HeadDim }
func (c GatedDeltaConfig) convDim() int { return 2*c.qDim() + c.vDim() } // q | k | v

// GatedDeltaWeights is one layer's f32 weights (the loader widens the bf16 checkpoint). InProjQKV is
// [convDim, D]; ConvWeight [convDim, K]; InProjA/InProjB [ValueHeads, D]; ALog/DtBias [ValueHeads];
// InProjZ [vDim, D]; Norm [HeadDim] (per-value-head gated RMSNorm); OutProj [D, vDim].
type GatedDeltaWeights struct {
	InProjQKV  []float32
	ConvWeight []float32
	ConvBias   []float32
	InProjA    []float32
	ALog       []float32
	DtBias     []float32
	InProjB    []float32
	InProjZ    []float32
	Norm       []float32
	OutProj    []float32
}

// ProjMatMul is the device-GEMM seam for the gated-delta projections (host matNT default; native injects
// its steel GEMM). AX-8: the lib declares the hook, the backend sets it.
var ProjMatMul func(x, w []float32, M, K, N int) ([]float32, error)

// ProjMatMulInto is the OPTIONAL write-into sibling of ProjMatMul: a backend that can target a
// caller-owned output buffer sets this so the projection GEMM skips its per-call output alloc (the
// dominant per-token decode cost). nil ⇒ not injected — the caller falls back to ProjMatMul, then the
// host matNTInto. AX-8: the lib declares the hook, the backend sets it. Into is preferred when set and
// the legacy ProjMatMul stays the fallback, so a backend that wired only the old hook keeps working.
var ProjMatMulInto func(out, x, w []float32, M, K, N int) ([]float32, error)

// deviceMinWork is the M·K·N floor below which the projections ignore the device hooks — a tiny
// GEMV (the gated-delta in_proj_a/b are [ValueHeads, D] = a few KMACs) pays a full command-buffer
// round-trip for microseconds of compute, so sub-MMAC shapes stay on the host path. Mirrors
// composed.deviceMinWork.
const deviceMinWork = 1 << 20

func projMatMul(x, w []float32, M, K, N int) ([]float32, error) {
	if ProjMatMul != nil && M*K*N >= deviceMinWork {
		return ProjMatMul(x, w, M, K, N)
	}
	return matNT(x, w, M, K, N), nil
}

// projMatMulInto runs y = x[M,K] @ w[N,K]ᵀ into out (reused when cap(out) ≥ M·N, else a fresh slab).
// It prefers the write-into backend hook, then the legacy fresh-buffer hook (out is ignored there —
// correctness kept, no reuse), then the host matNTInto. The RETURNED slice is authoritative (it may be a
// freshly grown/allocated buffer); callers store it back into their scratch to retain the growth.
func projMatMulInto(out, x, w []float32, M, K, N int) ([]float32, error) {
	if M*K*N >= deviceMinWork {
		if ProjMatMulInto != nil {
			return ProjMatMulInto(out, x, w, M, K, N)
		}
		if ProjMatMul != nil {
			return ProjMatMul(x, w, M, K, N)
		}
	}
	return matNTInto(out, x, w, M, K, N), nil
}

// matNT computes out[M,N] = in[M,K] @ w[N,K]ᵀ (the Linear y = x·Wᵀ), f32 host.
func matNT(in, w []float32, M, K, N int) []float32 {
	return matNTInto(nil, in, w, M, K, N)
}

// matNTInto is matNT writing into out, reusing it when cap(out) ≥ M·N (else it allocates a fresh M·N
// slab). Identical f64 accumulation + write order to the fresh-buffer form — only WHERE the result lands
// changes, so the output is bit-identical.
//
// Large shapes shard the OUTPUT COLUMNS across CPU cores (mirrors composed's matNTInto): each
// out[m·N+n] keeps exactly the serial per-element k-accumulation order, so the sharded form stays
// bit-identical; small shapes stay serial below the fan-out floor.
func matNTInto(out, in, w []float32, M, K, N int) []float32 {
	if cap(out) < M*N {
		out = make([]float32, M*N)
	} else {
		out = out[:M*N]
	}
	if M*K*N < matNTParMinWork {
		matNTCols(out, in, w, M, K, N, 0, N)
		return out
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > N {
		workers = N
	}
	span := (N + workers - 1) / workers
	var wg core.WaitGroup
	for lo := 0; lo < N; lo += span {
		hi := lo + span
		if hi > N {
			hi = N
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			matNTCols(out, in, w, M, K, N, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return out
}

// matNTParMinWork is the M·K·N floor below which matNTInto stays serial — under ~1 MMAC the
// fan-out/join overhead exceeds the compute it spreads.
const matNTParMinWork = 1 << 20

// matNTCols is the serial kernel over output columns [n0,n1) — the one accumulation-order-defining
// loop both the serial and sharded paths run.
func matNTCols(out, in, w []float32, M, K, N, n0, n1 int) {
	for m := range M {
		for n := n0; n < n1; n++ {
			var acc float64
			for k := range K {
				acc += float64(in[m*K+k]) * float64(w[n*K+k])
			}
			out[m*N+n] = float32(acc)
		}
	}
}

func gdSilu(v float64) float64 { return v / (1 + math.Exp(-v)) }

func gdSoftplus(v float64) float64 {
	if v > 20 {
		return v
	}
	return math.Log1p(math.Exp(v))
}

// GatedDeltaScratch holds the reusable projection-output buffers for GatedDeltaForwardScratchF32. A
// caller that steps one sequence (a decode session — single-goroutine) keeps one Scratch and passes it
// every token so the five projection GEMMs (qkv, a, b, z, out) write into resident buffers instead of
// allocating — the dominant per-token cost. NEVER share a Scratch across concurrently-stepped sessions:
// the buffers are mutable and unsynchronised (they mirror the recurrent conv/delta state's ownership —
// per-session, threaded, never on the shared weights). Buffers grow to fit and are reused thereafter.
type GatedDeltaScratch struct {
	qkv, aProj, bProj, zProj, out []float32
}

// GatedDeltaForwardF32 is GatedDeltaForwardScratchF32 with a fresh (nil) scratch — every projection
// allocates, the behaviour before the write-into seam. Kept for existing callers and the engine backend
// parity tests; bit-identical to the scratch path.
func GatedDeltaForwardF32(x []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, priorConv, priorDelta []float32, L, D int) (out, newConv, newDelta []float32, err error) {
	return GatedDeltaForwardScratchF32(x, w, cfg, priorConv, priorDelta, L, D, nil)
}

// GatedDeltaForwardScratchF32 runs one chunk of the Qwen 3.6 gated-delta block over x [L, D], threading
// the [conv-state ring, delta state] across calls (priorConv [(K-1),convDim], priorDelta [ValueHeads,
// HeadDim,HeadDim]; both nil ⇒ fresh). Returns out [L, D] and the advanced (newConv, newDelta). When sc
// is non-nil the five projection outputs write into its buffers (reused across calls); nil ⇒ each
// projection allocates fresh (the GatedDeltaForwardF32 path). The recurrent state (newConv/newDelta) is
// always freshly allocated — it is carried information, not scratch.
func GatedDeltaForwardScratchF32(x []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, priorConv, priorDelta []float32, L, D int, sc *GatedDeltaScratch) (out, newConv, newDelta []float32, err error) {
	if sc == nil {
		sc = &GatedDeltaScratch{} // throwaway: nil buffers ⇒ every projection allocates, the legacy path
	}
	if w == nil {
		return nil, nil, nil, core.NewError("qwen3.GatedDeltaForwardF32: nil weights")
	}
	KH, VH, HD, K := cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.ConvKernel
	if KH <= 0 || VH <= 0 || HD <= 0 || VH%KH != 0 || len(x) != L*D {
		return nil, nil, nil, core.NewError("qwen3.GatedDeltaForwardF32: bad geometry or x size")
	}
	qDim, vDim, convDim := cfg.qDim(), cfg.vDim(), cfg.convDim()
	rep := VH / KH
	scale := float32(1.0 / math.Sqrt(float64(HD)))

	qkv, err := projMatMulInto(sc.qkv, x, w.InProjQKV, L, D, convDim)
	if err != nil {
		return nil, nil, nil, err
	}
	sc.qkv = qkv
	convOut, newConv, err := mamba2.CausalConv1dF32(qkv, w.ConvWeight, w.ConvBias, priorConv, L, convDim, K)
	if err != nil {
		return nil, nil, nil, err
	}
	for i := range convOut {
		convOut[i] = float32(gdSilu(float64(convOut[i])))
	}

	// split q|k|v, GQA-repeat q,k (value head vh reads key head vh/rep), l2-normalise q.
	// q,k,v are read-only inputs to the recurrence (q is l2-normalised in place over its own window),
	// so one backing slab carved into three non-overlapping capped windows is bit-identical to three
	// makes and saves 2 allocs/token.
	qkvN := L * VH * HD
	qkvBuf := make([]float32, 3*qkvN)
	q := qkvBuf[0:qkvN:qkvN]
	k := qkvBuf[qkvN : 2*qkvN : 2*qkvN]
	v := qkvBuf[2*qkvN : 3*qkvN : 3*qkvN]
	for t := range L {
		base := t * convDim
		for vh := range VH {
			kh := vh / rep
			copy(q[(t*VH+vh)*HD:(t*VH+vh+1)*HD], convOut[base+kh*HD:base+kh*HD+HD])
			copy(k[(t*VH+vh)*HD:(t*VH+vh+1)*HD], convOut[base+qDim+kh*HD:base+qDim+kh*HD+HD])
		}
		copy(v[t*VH*HD:(t+1)*VH*HD], convOut[base+2*qDim:base+2*qDim+vDim])
	}
	for row := 0; row < L*VH; row++ { // l2-normalise q over HD (kernel l2-norms k itself)
		var ss float64
		for i := range HD {
			qv := float64(q[row*HD+i])
			ss += qv * qv
		}
		inv := float32(1.0 / math.Sqrt(ss+1e-6))
		for i := range HD {
			q[row*HD+i] *= inv
		}
	}

	// α = exp(−exp(A_log)·softplus(a+dt_bias)) ∈ (0,1] ; β = sigmoid(b). Per (token, value-head). The
	// two projection outputs are each read once and then dead, and α/β are the same [L,VH] shape, so
	// map α over aProj and β over bProj in place — the element-wise transform (output i depends only
	// on input i) makes this bit-identical and needs no separate α/β buffer.
	alpha, err := projMatMulInto(sc.aProj, x, w.InProjA, L, D, VH)
	if err != nil {
		return nil, nil, nil, err
	}
	sc.aProj = alpha
	beta, err := projMatMulInto(sc.bProj, x, w.InProjB, L, D, VH)
	if err != nil {
		return nil, nil, nil, err
	}
	sc.bProj = beta
	for i := 0; i < L*VH; i++ {
		h := i % VH
		dt := gdSoftplus(float64(alpha[i]) + float64(w.DtBias[h]))
		aDecay := -math.Exp(float64(w.ALog[h]))
		beta[i] = float32(1.0 / (1.0 + math.Exp(-float64(beta[i]))))
		alpha[i] = float32(math.Exp(aDecay * dt))
	}

	o, newDelta, err := deltanet.GatedDeltaRuleF32(q, k, v, beta, alpha, priorDelta, L, VH, HD, scale, 0)
	if err != nil {
		return nil, nil, nil, err
	}

	// gated RMSNorm: per (token, value-head) RMSNorm(o over HD)·SiLU(z), then out-proj. o is [L,VH,HD]
	// = [L·vDim], the gated shape, and is dead after this stage; each row's o is fully read (ss) then
	// each element is read once more immediately before its own write, so the gated result is written
	// in place over o — bit-identical, one fewer alloc per token.
	zProj, err := projMatMulInto(sc.zProj, x, w.InProjZ, L, D, vDim)
	if err != nil {
		return nil, nil, nil, err
	}
	sc.zProj = zProj
	gated := o
	for row := 0; row < L*VH; row++ {
		var ss float64
		for i := range HD {
			ov := float64(o[row*HD+i])
			ss += ov * ov
		}
		rms := math.Sqrt(ss/float64(HD) + float64(cfg.Eps))
		for i := range HD {
			normed := float64(o[row*HD+i]) / rms * float64(w.Norm[i])
			gated[row*HD+i] = float32(normed * gdSilu(float64(zProj[row*HD+i])))
		}
	}
	out, err = projMatMulInto(sc.out, gated, w.OutProj, L, vDim, D)
	if err != nil {
		return nil, nil, nil, err
	}
	sc.out = out
	return out, newConv, newDelta, nil
}
