// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"math"

	core "dappco.re/go"
)

// timemix.go is the REAL RWKV-7 "Goose" time-mix chain — fla.layers.rwkv7.RWKV7Attention.forward, ported
// equation-for-equation from the reference (github.com/fla-org/flash-linear-attention, the same upstream
// recurrence.go's WKV7F32 already mirrors) and cross-checked against the real checkpoint's tensor names
// (RWKV/RWKV7-Goose-World2.8-0.1B-HF). It is DELIBERATELY SEPARATE from block.go's BlockWeights/
// BlockForward*: those model a simplified stand-in (six full-rank projections straight from x, no
// token-shift, no LoRA decay/gate, no kk/k-update, no GroupNorm, no r_k bonus) kept for the composed-hybrid
// Mixer adapter (retired with the composed engine, #50) and engine/metal's device-GEMM seam — extending it in place would
// have broken its golden-bit-pattern tests AND those two consumers. This file reuses the one piece of
// existing code that IS the real architecture already: recurrence.go's WKV7F32 (its r/w/k/v/a/b kernel-
// level convention matches fla's chunk_rwkv7/fused_recurrent_rwkv7 exactly, verified line-by-line against
// fla/ops/rwkv7/fused_recurrent.py's fused_recurrent_rwkv7_fwd_kernel — the state update, the a=-kk/
// b=kk*iclr argument mapping, and the o=S_new^T.r readout are bit-for-bit the same maths).
//
// The chain, per fla/layers/rwkv7.py (config: fuse_norm=false, the RWKV7-Goose-World2.8-0.1B-HF setting):
//
//	delta          = tokenShift(x, priorShift)                          // shifted(x) - x
//	xr,xw,xk,xv,xa,xg = x + delta*{XR,XW,XK,XV,XA,XG}                     // addcmul, per-channel mix
//	r              = xr @ RProj^T
//	w              = -exp(-0.5) * sigmoid(WLora(xw))                     // final LOG-decay, ready for WKV7F32
//	k_raw          = xk @ KProj^T
//	v_raw          = xv @ VProj^T
//	v              = v_raw                            (layer 0; v_first := v_raw)
//	                 lerp(v_raw, v_first, sigmoid(VLora(xv)))  (layer >0)
//	iclr (a)       = sigmoid(ALora(xa))
//	g              = GLora(xg)                          // inner sigmoid only, no bias, no outer transform
//	kk             = L2normalize_per_head(k_raw * KK)                    // PyTorch F.normalize semantics
//	k              = k_raw * (1 + (iclr-1)*KA)                           // fed to the kernel AND the bonus
//	o, state       = WKV7F32(r, w, k, v, a=-kk, b=kk*iclr, priorState)
//	oNorm          = GroupNorm_perHead(o, H groups, eps=K*normEps)
//	bonus[t,h]     = Σ_i r[t,h,i]*k[t,h,i]*RK[h,i]
//	oGated         = (oNorm + bonus*v) * g
//	out            = oGated @ OProj^T
//
// v_first is a SAME-FORWARD-CALL cross-layer pipe (fla.models.rwkv7.modeling_rwkv7.RWKV7Model.forward's
// local `v_first`), not decode state: it is recomputed from layer 0's CURRENT token(s) every call and
// threaded through layers 1..N-1 of THAT call, unlike wkvState/shift which carry across calls.

// timeMixWeights holds one RWKV-7 layer's real time-mix parameters (fla.layers.rwkv7.RWKV7Attention),
// loaded verbatim from the checkpoint's r_proj/k_proj/v_proj/o_proj/x_*/k_k/k_a/r_k/g_norm tensors plus
// the w_lora/a_lora/v_lora/g_lora LoRA-MLPs (loader.go's loadLora).
type timeMixWeights struct {
	XR, XW, XK, XV, XA, XG     []float32 // [D] token-shift mix vectors
	RProj, KProj, VProj, OProj []float32 // [D,D], [D,D], [Dv,D], [D,Dv] (Linear y=x.W^T layout)
	WLora, ALora, GLora        lora
	VLora                      *lora     // nil for layer 0 (no value-residual mixing at the first layer)
	KK, KA                     []float32 // [D]
	RK                         []float32 // [H,K] flat — the output "bonus" correction weight
	GroupNormW, GroupNormB     []float32 // [Dv]; GroupNormB nil ⇒ affine bias absent
}

// timeMixState is one layer's carried state across decode calls: the recurrent WKV7 [H,K,V] matrix
// (recurrence.go's WKV7F32 state) plus this layer's OWN token-shift register [D] (distinct from the
// channel-mix's shift register — they observe different input streams, see model.go). Both nil ⇒ fresh.
type timeMixState struct {
	WKV   []float32
	Shift []float32
}

// negExpHalf = -exp(-0.5), fla's literal decay scale (RWKV7Attention.forward: "-0.6065306597126334").
const negExpHalf = -0.6065306597126334

// timeMixForward runs one layer's RWKV-7 time-mix over x [L,D] (already attn_norm'd). layerIdx selects the
// layer-0 v_first-defining branch vs every other layer's value-residual lerp; vFirst is layer 0's v for
// THIS call (ignored at layer 0). normEps is the checkpoint's norm_eps (GroupNorm's eps is K*normEps,
// fla's `eps=self.head_dim*norm_eps`). Returns the out_proj'd [L,D] output, the vFirst to thread into the
// NEXT layer of this call (layer 0 sets it; every other layer passes its input through unchanged), and the
// advanced state.
func timeMixForward(x []float32, w *timeMixWeights, cfg BlockConfig, layerIdx int, vFirst []float32, st timeMixState, L, D int, normEps float32) (out, vFirstOut []float32, newState timeMixState, err error) {
	if w == nil {
		return nil, nil, timeMixState{}, core.NewError("rwkv7.timeMixForward: nil weights")
	}
	H, K, V := cfg.NumHeads, cfg.KeyDim, cfg.ValueDim
	if H <= 0 || K <= 0 || V <= 0 || H*K != D || len(x) != L*D {
		return nil, nil, timeMixState{}, core.NewError("rwkv7.timeMixForward: bad geometry or x size")
	}

	delta, newShift := tokenShift(x, st.Shift, L, D)
	xr := addcmulRows(x, delta, w.XR, L, D)
	xw := addcmulRows(x, delta, w.XW, L, D)
	xk := addcmulRows(x, delta, w.XK, L, D)
	xv := addcmulRows(x, delta, w.XV, L, D)
	xa := addcmulRows(x, delta, w.XA, L, D)
	xg := addcmulRows(x, delta, w.XG, L, D)

	r, err := projMatMul(xr, w.RProj, L, D, D)
	if err != nil {
		return nil, nil, timeMixState{}, err
	}

	wRaw, err := w.WLora.forward(xw, L, tanhF32)
	if err != nil {
		return nil, nil, timeMixState{}, err
	}
	wDecay := make([]float32, len(wRaw))
	for i, v := range wRaw {
		wDecay[i] = float32(negExpHalf * float64(sigmoidF32(v)))
	}

	kRaw, err := projMatMul(xk, w.KProj, L, D, D)
	if err != nil {
		return nil, nil, timeMixState{}, err
	}
	vRaw, err := projMatMul(xv, w.VProj, L, D, H*V)
	if err != nil {
		return nil, nil, timeMixState{}, err
	}

	var v []float32
	if layerIdx == 0 {
		v = vRaw
		vFirstOut = append([]float32(nil), vRaw...)
	} else {
		if w.VLora == nil {
			return nil, nil, timeMixState{}, core.NewError("rwkv7.timeMixForward: layer >0 missing v_lora")
		}
		vGateRaw, verr := w.VLora.forward(xv, L, nil)
		if verr != nil {
			return nil, nil, timeMixState{}, verr
		}
		if len(vFirst) != len(vRaw) {
			return nil, nil, timeMixState{}, core.NewError("rwkv7.timeMixForward: vFirst size mismatch")
		}
		v = make([]float32, len(vRaw))
		for i := range v {
			g := float64(sigmoidF32(vGateRaw[i]))
			v[i] = float32((1-g)*float64(vRaw[i]) + g*float64(vFirst[i])) // lerp(v_raw, v_first, g)
		}
		vFirstOut = vFirst
	}

	aRaw, err := w.ALora.forward(xa, L, nil)
	if err != nil {
		return nil, nil, timeMixState{}, err
	}
	a := make([]float32, len(aRaw))
	for i, x := range aRaw {
		a[i] = sigmoidF32(x)
	}

	g, err := w.GLora.forward(xg, L, sigmoidF32) // inner sigmoid IS the only transform; no bias, no outer
	if err != nil {
		return nil, nil, timeMixState{}, err
	}

	// kk = L2-normalise(k_raw*KK) per head — PyTorch F.normalize(dim=-1,p=2): x / max(||x||_2, 1e-12).
	kk := make([]float32, L*D)
	for t := range L {
		tb := t * D
		for h := range H {
			hb := tb + h*K
			tmp := make([]float64, K)
			var ss float64
			for i := range K {
				val := float64(kRaw[hb+i]) * float64(w.KK[h*K+i])
				tmp[i] = val
				ss += val * val
			}
			denom := math.Sqrt(ss)
			if denom < 1e-12 {
				denom = 1e-12
			}
			for i := range K {
				kk[hb+i] = float32(tmp[i] / denom)
			}
		}
	}

	// k = k_raw * (1 + (a-1)*KA) — fla.ops.rwkv7.fused_k_update.k_update_ref.
	k := make([]float32, L*D)
	for t := range L {
		tb := t * D
		for c := range D {
			k[tb+c] = float32(float64(kRaw[tb+c]) * (1 + (float64(a[tb+c])-1)*float64(w.KA[c])))
		}
	}

	// Kernel args: a=-kk, b=kk*iclr — recurrence.go's WKV7F32 convention, matching fla's kernel exactly.
	aKernel := make([]float32, L*D)
	bKernel := make([]float32, L*D)
	for i := range aKernel {
		aKernel[i] = -kk[i]
		bKernel[i] = float32(float64(kk[i]) * float64(a[i]))
	}

	o, newWKV, err := WKV7F32(r, wDecay, k, v, aKernel, bKernel, st.WKV, L, H, K, V)
	if err != nil {
		return nil, nil, timeMixState{}, err
	}

	oNorm := groupNormHeads(o, w.GroupNormW, w.GroupNormB, L, H, V, float32(K)*normEps)

	// bonus[t,h] = Σ_i r[t,h,i]*k[t,h,i]*RK[h,i]; oGated = (oNorm + bonus*v) * g — gate_output_correction.
	oGated := make([]float32, L*H*V)
	for t := range L {
		rb := t * D
		vb := t * H * V
		for h := range H {
			hkb := rb + h*K
			var corr float64
			rkb := h * K
			for i := range K {
				corr += float64(r[hkb+i]) * float64(k[hkb+i]) * float64(w.RK[rkb+i])
			}
			hvb := vb + h*V
			for j := range V {
				val := (float64(oNorm[hvb+j]) + corr*float64(v[hvb+j])) * float64(g[hvb+j])
				oGated[hvb+j] = float32(val)
			}
		}
	}

	out, err = projMatMul(oGated, w.OProj, L, H*V, D)
	if err != nil {
		return nil, nil, timeMixState{}, err
	}
	return out, vFirstOut, timeMixState{WKV: newWKV, Shift: newShift}, nil
}
