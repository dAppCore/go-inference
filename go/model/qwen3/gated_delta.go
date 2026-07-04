// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	"math"

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

func projMatMul(x, w []float32, M, K, N int) ([]float32, error) {
	if ProjMatMul != nil {
		return ProjMatMul(x, w, M, K, N)
	}
	return matNT(x, w, M, K, N), nil
}

func matNT(in, w []float32, M, K, N int) []float32 {
	out := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var acc float64
			for k := 0; k < K; k++ {
				acc += float64(in[m*K+k]) * float64(w[n*K+k])
			}
			out[m*N+n] = float32(acc)
		}
	}
	return out
}

func gdSilu(v float64) float64 { return v / (1 + math.Exp(-v)) }

func gdSoftplus(v float64) float64 {
	if v > 20 {
		return v
	}
	return math.Log1p(math.Exp(v))
}

// GatedDeltaForwardF32 runs one chunk of the Qwen 3.6 gated-delta block over x [L, D], threading the
// [conv-state ring, delta state] across calls (priorConv [(K-1),convDim], priorDelta [ValueHeads,HeadDim,
// HeadDim]; both nil ⇒ fresh). Returns out [L, D] and the advanced (newConv, newDelta).
func GatedDeltaForwardF32(x []float32, w *GatedDeltaWeights, cfg GatedDeltaConfig, priorConv, priorDelta []float32, L, D int) (out, newConv, newDelta []float32, err error) {
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

	qkv, err := projMatMul(x, w.InProjQKV, L, D, convDim)
	if err != nil {
		return nil, nil, nil, err
	}
	convOut, newConv, err := mamba2.CausalConv1dF32(qkv, w.ConvWeight, w.ConvBias, priorConv, L, convDim, K)
	if err != nil {
		return nil, nil, nil, err
	}
	for i := range convOut {
		convOut[i] = float32(gdSilu(float64(convOut[i])))
	}

	// split q|k|v, GQA-repeat q,k (value head vh reads key head vh/rep), l2-normalise q.
	q := make([]float32, L*VH*HD)
	k := make([]float32, L*VH*HD)
	v := make([]float32, L*VH*HD)
	for t := 0; t < L; t++ {
		base := t * convDim
		for vh := 0; vh < VH; vh++ {
			kh := vh / rep
			copy(q[(t*VH+vh)*HD:(t*VH+vh+1)*HD], convOut[base+kh*HD:base+kh*HD+HD])
			copy(k[(t*VH+vh)*HD:(t*VH+vh+1)*HD], convOut[base+qDim+kh*HD:base+qDim+kh*HD+HD])
		}
		copy(v[t*VH*HD:(t+1)*VH*HD], convOut[base+2*qDim:base+2*qDim+vDim])
	}
	for row := 0; row < L*VH; row++ { // l2-normalise q over HD (kernel l2-norms k itself)
		var ss float64
		for i := 0; i < HD; i++ {
			qv := float64(q[row*HD+i])
			ss += qv * qv
		}
		inv := float32(1.0 / math.Sqrt(ss+1e-6))
		for i := 0; i < HD; i++ {
			q[row*HD+i] *= inv
		}
	}

	// α = exp(−exp(A_log)·softplus(a+dt_bias)) ∈ (0,1] ; β = sigmoid(b). Per (token, value-head).
	aProj, err := projMatMul(x, w.InProjA, L, D, VH)
	if err != nil {
		return nil, nil, nil, err
	}
	bProj, err := projMatMul(x, w.InProjB, L, D, VH)
	if err != nil {
		return nil, nil, nil, err
	}
	alpha := make([]float32, L*VH)
	beta := make([]float32, L*VH)
	for i := 0; i < L*VH; i++ {
		h := i % VH
		dt := gdSoftplus(float64(aProj[i]) + float64(w.DtBias[h]))
		aDecay := -math.Exp(float64(w.ALog[h]))
		alpha[i] = float32(math.Exp(aDecay * dt))
		beta[i] = float32(1.0 / (1.0 + math.Exp(-float64(bProj[i]))))
	}

	o, newDelta, err := deltanet.GatedDeltaRuleF32(q, k, v, beta, alpha, priorDelta, L, VH, HD, scale, 0)
	if err != nil {
		return nil, nil, nil, err
	}

	// gated RMSNorm: per (token, value-head) RMSNorm(o over HD)·SiLU(z), then out-proj.
	zProj, err := projMatMul(x, w.InProjZ, L, D, vDim)
	if err != nil {
		return nil, nil, nil, err
	}
	gated := make([]float32, L*vDim)
	for row := 0; row < L*VH; row++ {
		var ss float64
		for i := 0; i < HD; i++ {
			ov := float64(o[row*HD+i])
			ss += ov * ov
		}
		rms := math.Sqrt(ss/float64(HD) + float64(cfg.Eps))
		for i := 0; i < HD; i++ {
			normed := float64(o[row*HD+i]) / rms * float64(w.Norm[i])
			gated[row*HD+i] = float32(normed * gdSilu(float64(zProj[row*HD+i])))
		}
	}
	out, err = projMatMul(gated, w.OutProj, L, vDim, D)
	if err != nil {
		return nil, nil, nil, err
	}
	return out, newConv, newDelta, nil
}
