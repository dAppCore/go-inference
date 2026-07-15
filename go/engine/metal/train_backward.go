// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
)

// train_backward.go opens native training (12-14): the no-cgo path has only a forward, so unlike
// pkg/metal — which calls mlx's C autodiff (grad.go: mlx_closure / VJP) — native must build its own
// reverse-mode gradients, op by op, and chain them in reverse of the forward. These are the load-bearing
// VJPs the rest compose on: the linear layer (every projection + the LoRA A/B factors) and RMSNorm
// (every block's normalisation). Gradients are f32 (the precision metal's optimiser accumulates in) and
// the matmuls run through the steel GEMM (MatMulF32), so they match metal numerically. Each is verified
// by central finite differences (train_backward_test.go).

// LinearBackwardF32 is the vector-Jacobian product of the linear y = x · Wᵀ, where x is [M,K], W is
// [N,K] row-major (the way every projection weight is stored — out_features × in_features), and the
// forward output y is [M,N]. Given the upstream gradient dy [M,N] it returns:
//
//	dx [M,K] = dy · W        (∂L/∂x — flows to the previous layer)
//	dW [N,K] = dyᵀ · x       (∂L/∂W — the weight's gradient the optimiser steps)
//
// Both are computed in f32 via the fused steel GEMM, so they are byte-for-byte what metal's autodiff
// would produce for the same matmul. This is the backward half of a trainable Linear; a LoRA adapter
// composes two of these (the down-projection A and the up-projection B).
func LinearBackwardF32(dy, x, w []float32, M, K, N int) (dx, dw []float32, err error) {
	if len(dy) != M*N || len(x) != M*K || len(w) != N*K {
		return nil, nil, core.NewError("native.LinearBackwardF32: dy[M,N]/x[M,K]/w[N,K] size mismatch")
	}
	// dx = dy[M,N] · W[N,K]  → [M,K]  (nn GEMM: contract over N)
	dx, err = MatMulF32(dy, w, M, N, K)
	if err != nil {
		return nil, nil, err
	}
	// dW = dyᵀ[N,M] · x[M,K] → [N,K]  (nn GEMM: contract over M). Transpose dy host-side first.
	dyT := make([]float32, N*M)
	for m := range M {
		row := dy[m*N : (m+1)*N]
		for n := range N {
			dyT[n*M+m] = row[n]
		}
	}
	dw, err = MatMulF32(dyT, x, N, M, K)
	if err != nil {
		return nil, nil, err
	}
	return dx, dw, nil
}

// RMSNormBackwardF32 is the VJP of the (plain, no +1) RMSNorm over the last axis: for each of the rows
// rows×n, y_i = g_i · x_i / r with r = sqrt(mean(x²) + eps). Given dy it returns dx and the weight
// gradient dg (summed across rows, the shape of g). Per row:
//
//	dx_i = (g_i·dy_i)/r − x_i·(Σ_k g_k·dy_k·x_k)/(n·r³)
//	dg_i += dy_i·x_i/r
//
// f32. This is the normalisation backward every transformer block needs; it composes with the linear
// VJP into a full MLP/attention-block backward.
func RMSNormBackwardF32(dy, x, g []float32, rows, n int, eps float32) (dx, dg []float32, err error) {
	if len(dy) != rows*n || len(x) != rows*n || len(g) != n {
		return nil, nil, core.NewError("native.RMSNormBackwardF32: dy/x must be [rows,n] and g [n]")
	}
	dx = make([]float32, rows*n)
	dg = make([]float32, n)
	for r := range rows {
		xr, dyr, dxr := x[r*n:(r+1)*n], dy[r*n:(r+1)*n], dx[r*n:(r+1)*n]
		var ss float64
		for i := range n {
			ss += float64(xr[i]) * float64(xr[i])
		}
		rms := math.Sqrt(ss/float64(n) + float64(eps))
		var dot float64 // Σ_k g_k·dy_k·x_k
		for k := range n {
			dot += float64(g[k]) * float64(dyr[k]) * float64(xr[k])
		}
		coef := dot / (float64(n) * rms * rms * rms)
		for i := range n {
			dxr[i] = float32(float64(g[i])*float64(dyr[i])/rms - float64(xr[i])*coef)
			dg[i] += float32(float64(dyr[i]) * float64(xr[i]) / rms)
		}
	}
	return dx, dg, nil
}

const (
	geluC = 0.7978845608028654 // sqrt(2/π)
	geluA = 0.044715
)

// geluTanh is the tanh-approx GELU gemma's MLP uses: 0.5·z·(1+tanh(c·(z+a·z³))).
func geluTanh(z float64) float64 {
	u := geluC * (z + geluA*z*z*z)
	return 0.5 * z * (1 + math.Tanh(u))
}

// GeluGateMulBackwardF32 is the VJP of the gemma MLP activation gated = gelu_tanh(gate) · up (the
// elementwise product of the GELU'd gate branch with the up branch). Given the upstream gradient
// dgated it returns dgate and dup:
//
//	dup_i   = dgated_i · gelu(gate_i)
//	dgate_i = dgated_i · up_i · gelu'(gate_i)
//
// with gelu'(z) = 0.5(1+tanh u) + 0.5·z·(1−tanh²u)·c·(1+3a·z²), u = c(z+a·z³). f32. With the linear and
// RMSNorm VJPs this completes a full MLP-block backward (rms → gate/up linears → this → down linear).
func GeluGateMulBackwardF32(dgated, gate, up []float32, n int) (dgate, dup []float32, err error) {
	if len(dgated) != n || len(gate) != n || len(up) != n {
		return nil, nil, core.NewError("native.GeluGateMulBackwardF32: dgated/gate/up must be length n")
	}
	dgate = make([]float32, n)
	dup = make([]float32, n)
	for i := range n {
		z := float64(gate[i])
		u := geluC * (z + geluA*z*z*z)
		th := math.Tanh(u)
		gz := 0.5 * z * (1 + th)
		dgelu := 0.5*(1+th) + 0.5*z*(1-th*th)*geluC*(1+3*geluA*z*z)
		dup[i] = float32(float64(dgated[i]) * gz)
		dgate[i] = float32(float64(dgated[i]) * float64(up[i]) * dgelu)
	}
	return dgate, dup, nil
}

// rmsNormForwardF32 is the plain (no +1) RMSNorm forward over rows of width n, returning the normed
// rows (the backward recomputes this to feed the projection VJPs).
func rmsNormForwardF32(h, g []float32, rows, n int, eps float32) []float32 {
	out := make([]float32, rows*n)
	for r := range rows {
		hr, or := h[r*n:(r+1)*n], out[r*n:(r+1)*n]
		var ss float64
		for i := range n {
			ss += float64(hr[i]) * float64(hr[i])
		}
		rms := math.Sqrt(ss/float64(n) + float64(eps))
		for i := range n {
			or[i] = float32(float64(g[i]) * float64(hr[i]) / rms)
		}
	}
	return out
}

// MLPBlockForwardF32 is the forward of a gemma MLP block — out = h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h)))
// — the forward whose VJP is MLPBlockBackwardF32. A stacked SFT runs this per layer (saving each layer's
// input h) so the backward can chain across the stack. f32.
func MLPBlockForwardF32(h, normW, wGate, wUp, wDown []float32, M, dModel, dFF int, eps float32) ([]float32, error) {
	if len(h) != M*dModel || len(normW) != dModel {
		return nil, core.NewError("native.MLPBlockForwardF32: h[M,dModel]/normW[dModel] size mismatch")
	}
	normed := rmsNormForwardF32(h, normW, M, dModel, eps)
	gate, err := MatMulF32NT(normed, wGate, M, dModel, dFF)
	if err != nil {
		return nil, err
	}
	up, err := MatMulF32NT(normed, wUp, M, dModel, dFF)
	if err != nil {
		return nil, err
	}
	gated := make([]float32, M*dFF)
	for i := range gated {
		gated[i] = float32(geluTanh(float64(gate[i])) * float64(up[i]))
	}
	down, err := MatMulF32NT(gated, wDown, M, dFF, dModel)
	if err != nil {
		return nil, err
	}
	out := make([]float32, M*dModel)
	for i := range out {
		out[i] = h[i] + down[i]
	}
	return out, nil
}

// MLPBlockGrads holds the parameter gradients of one gemma MLP block (the norm weight + the three
// projection weights), plus dh — the gradient w.r.t. the block input that flows to the previous layer.
type MLPBlockGrads struct {
	DH           []float32 // [M,dModel] gradient to the previous op (includes the residual)
	DNormW       []float32 // [dModel]
	DWGate, DWUp []float32 // [dFF,dModel]
	DWDown       []float32 // [dModel,dFF]
}

// MLPBlockBackwardF32 is the VJP of a full gemma MLP block — out = h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h)))
// — composed from the linear, RMSNorm and gelu·up VJPs, proving they chain. Given dout [M,dModel] it
// recomputes the forward (normed, gate, up, gated) and backpropagates: through the down projection, the
// gelu·up activation, the gate/up projections (summing the two gradients into rms's output since rms
// feeds both branches), the RMSNorm, and the residual (dh = dout + dh_through_norm). All f32. This is a
// real multi-op backward graph on the no-cgo path, gradient-checked end to end.
func MLPBlockBackwardF32(dout, h, normW, wGate, wUp, wDown []float32, M, dModel, dFF int, eps float32) (*MLPBlockGrads, error) {
	if len(dout) != M*dModel || len(h) != M*dModel || len(normW) != dModel {
		return nil, core.NewError("native.MLPBlockBackwardF32: dout/h must be [M,dModel] and normW [dModel]")
	}
	if len(wGate) != dFF*dModel || len(wUp) != dFF*dModel || len(wDown) != dModel*dFF {
		return nil, core.NewError("native.MLPBlockBackwardF32: projection weight size mismatch")
	}
	// recompute forward intermediates needed by the backward.
	normed := rmsNormForwardF32(h, normW, M, dModel, eps)
	gate, err := MatMulF32NT(normed, wGate, M, dModel, dFF)
	if err != nil {
		return nil, err
	}
	up, err := MatMulF32NT(normed, wUp, M, dModel, dFF)
	if err != nil {
		return nil, err
	}
	gated := make([]float32, M*dFF)
	for i := range gated {
		gated[i] = float32(geluTanh(float64(gate[i])) * float64(up[i]))
	}
	// backward: down projection (gated @ wDownᵀ → down; out = h + down).
	dGated, dWDown, err := LinearBackwardF32(dout, gated, wDown, M, dFF, dModel)
	if err != nil {
		return nil, err
	}
	// activation gelu(gate)·up (elementwise over all M·dFF).
	dGate, dUp, err := GeluGateMulBackwardF32(dGated, gate, up, M*dFF)
	if err != nil {
		return nil, err
	}
	// gate/up projections (normed @ Wᵀ); rms's output feeds BOTH, so sum the two input gradients.
	dNormedG, dWGate, err := LinearBackwardF32(dGate, normed, wGate, M, dModel, dFF)
	if err != nil {
		return nil, err
	}
	dNormedU, dWUp, err := LinearBackwardF32(dUp, normed, wUp, M, dModel, dFF)
	if err != nil {
		return nil, err
	}
	dNormed := make([]float32, M*dModel)
	for i := range dNormed {
		dNormed[i] = dNormedG[i] + dNormedU[i]
	}
	// RMSNorm, then the residual: dh = dout + (gradient through the norm path).
	dHNorm, dNormW, err := RMSNormBackwardF32(dNormed, h, normW, M, dModel, eps)
	if err != nil {
		return nil, err
	}
	dH := make([]float32, M*dModel)
	for i := range dH {
		dH[i] = dout[i] + dHNorm[i]
	}
	return &MLPBlockGrads{DH: dH, DNormW: dNormW, DWGate: dWGate, DWUp: dWUp, DWDown: dWDown}, nil
}

// SoftmaxBackwardF32 is the VJP of a row-wise softmax y = softmax(x) over the last axis (rows×n) — the
// attention backward's key new op (the QKᵀ and ·V steps are matmuls, already covered). Given dy and the
// softmax OUTPUT y (cheaper to pass than recomputing), per row:
//
//	dx_i = y_i · (dy_i − Σ_j y_j·dy_j)
//
// f32. Composed with the matmul VJP (for QKᵀ and probs·V) and the RoPE VJP this gives the attention
// block backward; the softmax is the only non-matmul/non-elementwise piece, so it is the gate to it.
func SoftmaxBackwardF32(dy, y []float32, rows, n int) (dx []float32, err error) {
	if len(dy) != rows*n || len(y) != rows*n {
		return nil, core.NewError("native.SoftmaxBackwardF32: dy and y must be [rows,n]")
	}
	dx = make([]float32, rows*n)
	for r := range rows {
		yr, dyr, dxr := y[r*n:(r+1)*n], dy[r*n:(r+1)*n], dx[r*n:(r+1)*n]
		var dot float64 // Σ_j y_j·dy_j
		for j := range n {
			dot += float64(yr[j]) * float64(dyr[j])
		}
		for i := range n {
			dxr[i] = float32(float64(yr[i]) * (float64(dyr[i]) - dot))
		}
	}
	return dx, nil
}

// RoPEBackwardF32 is the VJP of rotary position embedding on a head-major [nHeads, headDim] vector at
// position pos (half-split convention: pair j with j+rotaryDim/2; dims ≥ rotaryDim pass through for
// partial rotary). RoPE is an orthogonal rotation by angle θ_j = pos·base^(−2j/rotaryDim), so its
// Jacobian is that rotation and the VJP is the INVERSE rotation (by −θ_j):
//
//	dx[j]      =  dy[j]·cos + dy[j+h]·sin
//	dx[j+h]    = −dy[j]·sin + dy[j+h]·cos      (h = rotaryDim/2)
//
// f32. The q/k projections are RoPE'd before attention, so this sits between their linear VJP and the
// QKᵀ VJP in the attention-block backward.
func RoPEBackwardF32(dy []float32, pos, nHeads, headDim, rotaryDim int, base float32) ([]float32, error) {
	if len(dy) != nHeads*headDim || rotaryDim > headDim || rotaryDim%2 != 0 {
		return nil, core.NewError("native.RoPEBackwardF32: dy must be [nHeads,headDim] and rotaryDim even ≤ headDim")
	}
	dx := make([]float32, len(dy))
	copy(dx, dy) // dims ≥ rotaryDim pass through unchanged (partial rotary)
	h := rotaryDim / 2
	for head := range nHeads {
		off := head * headDim
		for j := range h {
			invFreq := math.Pow(float64(base), -2*float64(j)/float64(rotaryDim))
			ang := float64(pos) * invFreq
			c, s := math.Cos(ang), math.Sin(ang)
			a, b := float64(dy[off+j]), float64(dy[off+j+h])
			dx[off+j] = float32(a*c + b*s)
			dx[off+j+h] = float32(-a*s + b*c)
		}
	}
	return dx, nil
}

// AttnSingleHeadBackwardF32 is the VJP of single-head scaled-dot-product attention — O = softmax(Q·Kᵀ·scale
// [+ causal mask])·V, with Q,K,V each [L, d] — composed from the softmax and matmul VJPs (the other half
// of a transformer layer's backward, the MLP block being the first). Given dOut [L,d] it recomputes the
// scores/probs and backpropagates:
//
//	dV = Pᵀ·dOut ; dP = dOut·Vᵀ ; dS = softmaxVJP(dP, P) ; dQ = dS·K·scale ; dK = dSᵀ·Q·scale
//
// Causal masking sets future scores to −inf so their P (and thus dS) are 0 — handled by the recompute.
// f32. Multi-head + GQA is this per (kv-shared) head with the per-head projection VJPs around it.
func AttnSingleHeadBackwardF32(dOut, q, k, v []float32, L, d int, scale float32, causal bool) (dQ, dK, dV []float32, err error) {
	if len(dOut) != L*d || len(q) != L*d || len(k) != L*d || len(v) != L*d {
		return nil, nil, nil, core.NewError("native.AttnSingleHeadBackwardF32: dOut/q/k/v must be [L,d]")
	}
	// recompute S = q·kᵀ·scale (causal-masked) and P = rowwise softmax(S).
	s, err := MatMulF32NT(q, k, L, d, L) // [L,L]
	if err != nil {
		return nil, nil, nil, err
	}
	p := make([]float32, L*L)
	for i := range L {
		row := s[i*L : (i+1)*L]
		mx := float32(math.Inf(-1))
		lim := L - 1
		if causal {
			lim = i
		}
		for j := 0; j <= lim; j++ {
			row[j] *= scale
			if row[j] > mx {
				mx = row[j]
			}
		}
		var sum float64
		for j := range L {
			if j > lim {
				p[i*L+j] = 0
				continue
			}
			e := math.Exp(float64(row[j] - mx))
			p[i*L+j] = float32(e)
			sum += e
		}
		for j := 0; j <= lim; j++ {
			p[i*L+j] = float32(float64(p[i*L+j]) / sum)
		}
	}
	// dV = Pᵀ·dOut ; dP = dOut·Vᵀ
	dV, err = MatMulF32(transposeF32(p, L, L), dOut, L, L, d)
	if err != nil {
		return nil, nil, nil, err
	}
	dP, err := MatMulF32NT(dOut, v, L, d, L)
	if err != nil {
		return nil, nil, nil, err
	}
	// dS = softmax VJP (row-wise), then dQ = dS·K·scale, dK = dSᵀ·Q·scale
	dS, err := SoftmaxBackwardF32(dP, p, L, L)
	if err != nil {
		return nil, nil, nil, err
	}
	dQ, err = MatMulF32(dS, k, L, L, d)
	if err != nil {
		return nil, nil, nil, err
	}
	dK, err = MatMulF32(transposeF32(dS, L, L), q, L, L, d)
	if err != nil {
		return nil, nil, nil, err
	}
	for i := range dQ {
		dQ[i] *= scale
	}
	for i := range dK {
		dK[i] *= scale
	}
	return dQ, dK, dV, nil
}

// ropeForwardF32 rotates one head-major [nHeads,headDim] vector at position pos (half-split, partial
// rotary), the forward the block backward recomputes for q/k.
func ropeForwardF32(x []float32, pos, nHeads, headDim, rotaryDim int, base float32) []float32 {
	y := make([]float32, len(x))
	copy(y, x)
	h := rotaryDim / 2
	for head := range nHeads {
		off := head * headDim
		for j := range h {
			invFreq := math.Pow(float64(base), -2*float64(j)/float64(rotaryDim))
			ang := float64(pos) * invFreq
			c, s := math.Cos(ang), math.Sin(ang)
			a, b := float64(x[off+j]), float64(x[off+j+h])
			y[off+j] = float32(a*c - b*s)
			y[off+j+h] = float32(a*s + b*c)
		}
	}
	return y
}

// sdpaForwardSingleHeadF32 recomputes O = softmax(q·kᵀ·scale [+causal])·v for one head [L,d] — the
// attention output the block backward needs for the output-projection VJP.
func sdpaForwardSingleHeadF32(q, k, v []float32, L, d int, scale float32, causal bool) ([]float32, error) {
	s, err := MatMulF32NT(q, k, L, d, L)
	if err != nil {
		return nil, err
	}
	p := make([]float32, L*L)
	for i := range L {
		lim := L - 1
		if causal {
			lim = i
		}
		mx := float32(math.Inf(-1))
		for j := 0; j <= lim; j++ {
			s[i*L+j] *= scale
			if s[i*L+j] > mx {
				mx = s[i*L+j]
			}
		}
		var sum float64
		for j := 0; j <= lim; j++ {
			e := math.Exp(float64(s[i*L+j] - mx))
			p[i*L+j] = float32(e)
			sum += e
		}
		for j := 0; j <= lim; j++ {
			p[i*L+j] = float32(float64(p[i*L+j]) / sum)
		}
	}
	return MatMulF32(p, v, L, L, d)
}

// QKNormBackwardF32 is the VJP of gemma4's per-head query/key RMSNorm: each of the H heads' d-vectors
// (in the head-major [L, H·d] tensor) is RMSNorm'd by a SHARED [d] weight before RoPE. Because the
// head-major layout makes every head's d-vector contiguous, this is exactly RMSNormBackwardF32 over
// L·H rows of width d — the dNormW gradient sums across all head-rows (the weight is shared). Returns
// dx [L,H·d] and dNormW [d]. f32. The gemma4-specific decoration that sits between the q/k projection
// VJP and the RoPE VJP in a real gemma4 attention-block backward.
func QKNormBackwardF32(dy, x, normW []float32, L, H, d int, eps float32) (dx, dNormW []float32, err error) {
	if len(dy) != L*H*d || len(x) != L*H*d || len(normW) != d {
		return nil, nil, core.NewError("native.QKNormBackwardF32: dy/x must be [L,H·d] and normW [d]")
	}
	return RMSNormBackwardF32(dy, x, normW, L*H, d, eps)
}

// gatherHeadF32 extracts head h (width d) from a head-major [L, nHeads·d] tensor into [L, d].
func gatherHeadF32(x []float32, L, nHeads, d, h int) []float32 {
	out := make([]float32, L*d)
	for i := range L {
		copy(out[i*d:(i+1)*d], x[i*nHeads*d+h*d:i*nHeads*d+(h+1)*d])
	}
	return out
}

// scatterAddHeadF32 adds a per-head [L,d] gradient back into head h of a [L, nHeads·d] tensor (ADD, so
// GQA's several query heads accumulate into their shared kv head).
func scatterAddHeadF32(dst, src []float32, L, nHeads, d, h int) {
	for i := range L {
		for j := range d {
			dst[i*nHeads*d+h*d+j] += src[i*d+j]
		}
	}
}

// MultiHeadAttnBackwardF32 is the VJP of multi-head GQA scaled-dot-product attention: H query heads,
// Hkv key/value heads (H % Hkv == 0; query head h reads kv head h/(H/Hkv)), each head [L,d]. Q is
// [L,H·d], K and V are [L,Hkv·d], dOut [L,H·d]. It runs the single-head VJP per query head and SUMS the
// dK/dV of all query heads sharing a kv head into that kv head (the GQA reduction). Returns dQ [L,H·d],
// dK/dV [L,Hkv·d]. f32. This is the head structure of a real gemma4 attention layer.
func MultiHeadAttnBackwardF32(dOut, q, k, v []float32, L, H, Hkv, d int, scale float32, causal bool) (dQ, dK, dV []float32, err error) {
	if H%Hkv != 0 {
		return nil, nil, nil, core.NewError("native.MultiHeadAttnBackwardF32: H must be a multiple of Hkv")
	}
	if len(dOut) != L*H*d || len(q) != L*H*d || len(k) != L*Hkv*d || len(v) != L*Hkv*d {
		return nil, nil, nil, core.NewError("native.MultiHeadAttnBackwardF32: q/dOut [L,H·d], k/v [L,Hkv·d] size mismatch")
	}
	gqa := H / Hkv
	dQ = make([]float32, L*H*d)
	dK = make([]float32, L*Hkv*d)
	dV = make([]float32, L*Hkv*d)
	for h := range H {
		hk := h / gqa
		qh := gatherHeadF32(q, L, H, d, h)
		kh := gatherHeadF32(k, L, Hkv, d, hk)
		vh := gatherHeadF32(v, L, Hkv, d, hk)
		doh := gatherHeadF32(dOut, L, H, d, h)
		dqh, dkh, dvh, e := AttnSingleHeadBackwardF32(doh, qh, kh, vh, L, d, scale, causal)
		if e != nil {
			return nil, nil, nil, e
		}
		scatterAddHeadF32(dQ, dqh, L, H, d, h)    // each query head writes its own slot
		scatterAddHeadF32(dK, dkh, L, Hkv, d, hk) // GQA: heads sharing hk accumulate
		scatterAddHeadF32(dV, dvh, L, Hkv, d, hk)
	}
	return dQ, dK, dV, nil
}

// multiHeadSDPAForwardF32 recomputes O [L,H·d] = per-head SDPA(q_h, k_{h/gqa}, v_{h/gqa}) — the multi-head
// GQA attention output the block backward needs for the output-projection VJP.
func multiHeadSDPAForwardF32(q, k, v []float32, L, H, Hkv, d int, scale float32, causal bool) ([]float32, error) {
	gqa := H / Hkv
	out := make([]float32, L*H*d)
	for h := range H {
		hk := h / gqa
		oh, err := sdpaForwardSingleHeadF32(gatherHeadF32(q, L, H, d, h), gatherHeadF32(k, L, Hkv, d, hk), gatherHeadF32(v, L, Hkv, d, hk), L, d, scale, causal)
		if err != nil {
			return nil, err
		}
		scatterAddHeadF32(out, oh, L, H, d, h)
	}
	return out, nil
}

// MultiHeadAttnBlockForwardF32 is the forward of the multi-head GQA attention block whose VJP is
// MultiHeadAttnBlockBackwardF32 — out = h + Wo·MHSDPA(RoPE(Wq·rms(h)), RoPE(Wk·rms(h)), Wv·rms(h)).
// Used to verify the host backward's recompute matches the engine's real forward (the capstone's
// forward-match check) and, with MLPBlockForwardF32, to run a host layer forward for training.
func MultiHeadAttnBlockForwardF32(h, normW, wQ, wK, wV, wO []float32, L, dModel, H, Hkv, d, rotaryDim int, base, scale, eps float32, causal bool) ([]float32, error) {
	qDim, kvDim := H*d, Hkv*d
	normed := rmsNormForwardF32(h, normW, L, dModel, eps)
	q, err := MatMulF32NT(normed, wQ, L, dModel, qDim)
	if err != nil {
		return nil, err
	}
	k, err := MatMulF32NT(normed, wK, L, dModel, kvDim)
	if err != nil {
		return nil, err
	}
	v, err := MatMulF32NT(normed, wV, L, dModel, kvDim)
	if err != nil {
		return nil, err
	}
	qr, kr := make([]float32, L*qDim), make([]float32, L*kvDim)
	for i := range L {
		copy(qr[i*qDim:(i+1)*qDim], ropeForwardF32(q[i*qDim:(i+1)*qDim], i, H, d, rotaryDim, base))
		copy(kr[i*kvDim:(i+1)*kvDim], ropeForwardF32(k[i*kvDim:(i+1)*kvDim], i, Hkv, d, rotaryDim, base))
	}
	o, err := multiHeadSDPAForwardF32(qr, kr, v, L, H, Hkv, d, scale, causal)
	if err != nil {
		return nil, err
	}
	attnOut, err := MatMulF32NT(o, wO, L, qDim, dModel)
	if err != nil {
		return nil, err
	}
	out := make([]float32, L*dModel)
	for i := range out {
		out[i] = h[i] + attnOut[i]
	}
	return out, nil
}

// MultiHeadAttnBlockBackwardF32 is the VJP of a full MULTI-HEAD GQA attention block (the real gemma4 head
// structure, no QK-norm variant) — out = h + Wo·MHSDPA(RoPE(Wq·rms(h)), RoPE(Wk·rms(h)), Wv·rms(h)) —
// composing the multi-head GQA SDPA VJP, the q/k/v/o projection VJPs (q is [L,H·d], k/v are [L,Hkv·d]),
// per-position-per-head RoPE, RMSNorm and the residual. This is the attention block of a real gemma4
// layer; with the MLP block it is a full multi-head layer backward. f32, gradient-checked end to end.
func MultiHeadAttnBlockBackwardF32(dout, h, normW, wQ, wK, wV, wO []float32, L, dModel, H, Hkv, d, rotaryDim int, base, scale, eps float32, causal bool) (*AttnBlockGrads, error) {
	qDim, kvDim := H*d, Hkv*d
	if len(dout) != L*dModel || len(h) != L*dModel || len(normW) != dModel {
		return nil, core.NewError("native.MultiHeadAttnBlockBackwardF32: dout/h [L,dModel], normW [dModel]")
	}
	if len(wQ) != qDim*dModel || len(wK) != kvDim*dModel || len(wV) != kvDim*dModel || len(wO) != dModel*qDim {
		return nil, core.NewError("native.MultiHeadAttnBlockBackwardF32: projection weight size mismatch")
	}
	// forward recompute.
	normed := rmsNormForwardF32(h, normW, L, dModel, eps)
	q, err := MatMulF32NT(normed, wQ, L, dModel, qDim)
	if err != nil {
		return nil, err
	}
	k, err := MatMulF32NT(normed, wK, L, dModel, kvDim)
	if err != nil {
		return nil, err
	}
	v, err := MatMulF32NT(normed, wV, L, dModel, kvDim)
	if err != nil {
		return nil, err
	}
	qr, kr := make([]float32, L*qDim), make([]float32, L*kvDim)
	for i := range L { // per-position RoPE, all heads in the row at once
		copy(qr[i*qDim:(i+1)*qDim], ropeForwardF32(q[i*qDim:(i+1)*qDim], i, H, d, rotaryDim, base))
		copy(kr[i*kvDim:(i+1)*kvDim], ropeForwardF32(k[i*kvDim:(i+1)*kvDim], i, Hkv, d, rotaryDim, base))
	}
	o, err := multiHeadSDPAForwardF32(qr, kr, v, L, H, Hkv, d, scale, causal)
	if err != nil {
		return nil, err
	}
	// backward: output projection → MH SDPA core → RoPE → q/k/v projections → norm → residual.
	dO, dWO, err := LinearBackwardF32(dout, o, wO, L, qDim, dModel)
	if err != nil {
		return nil, err
	}
	dqr, dkr, dv, err := MultiHeadAttnBackwardF32(dO, qr, kr, v, L, H, Hkv, d, scale, causal)
	if err != nil {
		return nil, err
	}
	dq, dk := make([]float32, L*qDim), make([]float32, L*kvDim)
	for i := range L {
		drq, e1 := RoPEBackwardF32(dqr[i*qDim:(i+1)*qDim], i, H, d, rotaryDim, base)
		if e1 != nil {
			return nil, e1
		}
		drk, e2 := RoPEBackwardF32(dkr[i*kvDim:(i+1)*kvDim], i, Hkv, d, rotaryDim, base)
		if e2 != nil {
			return nil, e2
		}
		copy(dq[i*qDim:(i+1)*qDim], drq)
		copy(dk[i*kvDim:(i+1)*kvDim], drk)
	}
	dnQ, dWQ, err := LinearBackwardF32(dq, normed, wQ, L, dModel, qDim)
	if err != nil {
		return nil, err
	}
	dnK, dWK, err := LinearBackwardF32(dk, normed, wK, L, dModel, kvDim)
	if err != nil {
		return nil, err
	}
	dnV, dWV, err := LinearBackwardF32(dv, normed, wV, L, dModel, kvDim)
	if err != nil {
		return nil, err
	}
	dNormed := make([]float32, L*dModel)
	for i := range dNormed {
		dNormed[i] = dnQ[i] + dnK[i] + dnV[i]
	}
	dHNorm, dNormW, err := RMSNormBackwardF32(dNormed, h, normW, L, dModel, eps)
	if err != nil {
		return nil, err
	}
	dH := make([]float32, L*dModel)
	for i := range dH {
		dH[i] = dout[i] + dHNorm[i]
	}
	return &AttnBlockGrads{DH: dH, DNormW: dNormW, DWQ: dWQ, DWK: dWK, DWV: dWV, DWO: dWO}, nil
}

// AttnBlockGrads holds one attention block's parameter gradients (norm + q/k/v/o projections) and dH.
type AttnBlockGrads struct {
	DH            []float32 // [L,dModel]
	DNormW        []float32 // [dModel]
	DWQ, DWK, DWV []float32 // [d,dModel]
	DWO           []float32 // [dModel,d]
}

// AttnBlockBackwardF32 is the VJP of a full single-head attention block —
// out = h + Wo·SDPA(RoPE(Wq·rms(h)), RoPE(Wk·rms(h)), Wv·rms(h)) — composing the RMSNorm, linear (q/k/v/o),
// RoPE and SDPA-core VJPs, with each of the L rows RoPE'd by its own position and the residual. This is
// the attention counterpart to MLPBlockBackwardF32; together they are a complete transformer layer's
// backward, the unit a full-stack SFT chains across layers. f32, gradient-checked end to end.
func AttnBlockBackwardF32(dout, h, normW, wQ, wK, wV, wO []float32, L, dModel, d, rotaryDim int, base, scale, eps float32, causal bool) (*AttnBlockGrads, error) {
	if len(dout) != L*dModel || len(h) != L*dModel || len(normW) != dModel {
		return nil, core.NewError("native.AttnBlockBackwardF32: dout/h must be [L,dModel] and normW [dModel]")
	}
	if len(wQ) != d*dModel || len(wK) != d*dModel || len(wV) != d*dModel || len(wO) != dModel*d {
		return nil, core.NewError("native.AttnBlockBackwardF32: projection weight size mismatch")
	}
	// forward recompute: normed → q/k/v → rope(q,k) per row → O → attnOut.
	normed := rmsNormForwardF32(h, normW, L, dModel, eps)
	q, err := MatMulF32NT(normed, wQ, L, dModel, d)
	if err != nil {
		return nil, err
	}
	k, err := MatMulF32NT(normed, wK, L, dModel, d)
	if err != nil {
		return nil, err
	}
	v, err := MatMulF32NT(normed, wV, L, dModel, d)
	if err != nil {
		return nil, err
	}
	qr := make([]float32, L*d)
	kr := make([]float32, L*d)
	for i := range L {
		copy(qr[i*d:(i+1)*d], ropeForwardF32(q[i*d:(i+1)*d], i, 1, d, rotaryDim, base))
		copy(kr[i*d:(i+1)*d], ropeForwardF32(k[i*d:(i+1)*d], i, 1, d, rotaryDim, base))
	}
	o, err := sdpaForwardSingleHeadF32(qr, kr, v, L, d, scale, causal)
	if err != nil {
		return nil, err
	}
	// backward: output projection (attnOut = O·Woᵀ; out = h + attnOut).
	dO, dWO, err := LinearBackwardF32(dout, o, wO, L, d, dModel)
	if err != nil {
		return nil, err
	}
	// SDPA core: dO → dqr, dkr, dv (uses the rope'd q,k).
	dqr, dkr, dv, err := AttnSingleHeadBackwardF32(dO, qr, kr, v, L, d, scale, causal)
	if err != nil {
		return nil, err
	}
	// RoPE backward per row → gradient w.r.t. the pre-rope q,k.
	dq := make([]float32, L*d)
	dk := make([]float32, L*d)
	for i := range L {
		drq, e1 := RoPEBackwardF32(dqr[i*d:(i+1)*d], i, 1, d, rotaryDim, base)
		if e1 != nil {
			return nil, e1
		}
		drk, e2 := RoPEBackwardF32(dkr[i*d:(i+1)*d], i, 1, d, rotaryDim, base)
		if e2 != nil {
			return nil, e2
		}
		copy(dq[i*d:(i+1)*d], drq)
		copy(dk[i*d:(i+1)*d], drk)
	}
	// q/k/v projections (normed·Wᵀ); normed feeds all three → sum the input gradients.
	dnQ, dWQ, err := LinearBackwardF32(dq, normed, wQ, L, dModel, d)
	if err != nil {
		return nil, err
	}
	dnK, dWK, err := LinearBackwardF32(dk, normed, wK, L, dModel, d)
	if err != nil {
		return nil, err
	}
	dnV, dWV, err := LinearBackwardF32(dv, normed, wV, L, dModel, d)
	if err != nil {
		return nil, err
	}
	dNormed := make([]float32, L*dModel)
	for i := range dNormed {
		dNormed[i] = dnQ[i] + dnK[i] + dnV[i]
	}
	// RMSNorm + residual: dh = dout + grad-through-norm.
	dHNorm, dNormW, err := RMSNormBackwardF32(dNormed, h, normW, L, dModel, eps)
	if err != nil {
		return nil, err
	}
	dH := make([]float32, L*dModel)
	for i := range dH {
		dH[i] = dout[i] + dHNorm[i]
	}
	return &AttnBlockGrads{DH: dH, DNormW: dNormW, DWQ: dWQ, DWK: dWK, DWV: dWV, DWO: dWO}, nil
}
