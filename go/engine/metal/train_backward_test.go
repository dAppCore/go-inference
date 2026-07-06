// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// TestLinearBackwardF32 verifies the linear VJP against finite differences — the standard correctness
// bar for a gradient. With a fixed random cotangent dy, the scalar loss is L = Σ y·dy, so ∂L/∂x and
// ∂L/∂W are exactly what LinearBackwardF32 returns; we check a sample of entries against the central
// finite difference (L(θ+ε) − L(θ−ε)) / 2ε of the real forward y = x·Wᵀ.
func TestLinearBackwardF32(t *testing.T) {
	requireNativeRuntime(t)
	const M, K, N = 3, 4, 5
	x := syntheticFloat32(M*K, 1)
	w := syntheticFloat32(N*K, 2)
	dy := syntheticFloat32(M*N, 3)

	forward := func(x, w []float32) []float32 {
		y, err := MatMulF32NT(x, w, M, K, N) // x[M,K] · w[N,K]ᵀ = y[M,N]
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		return y
	}
	loss := func(x, w []float32) float64 { // L = Σ y·dy
		y := forward(x, w)
		var s float64
		for i := range y {
			s += float64(y[i]) * float64(dy[i])
		}
		return s
	}

	dx, dw, err := LinearBackwardF32(dy, x, w, M, K, N)
	if err != nil {
		t.Fatalf("LinearBackwardF32: %v", err)
	}

	const eps = 1.0 / 256 // bf16-free f32 forward; a coarse step keeps finite-diff noise low
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps
			lp := loss(x, w)
			params[i] = orig - eps
			lm := loss(x, w)
			params[i] = orig
			fd := (lp - lm) / (2 * eps)
			if math.Abs(fd-float64(grad[i])) > 1e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dx", x, dx)
	check("dw", w, dw)
	t.Logf("linear VJP matches finite differences: dx[%d] dw[%d] all within tol", len(dx), len(dw))
}

// TestRMSNormBackwardF32 verifies the RMSNorm VJP against central finite differences of the plain
// (no +1) RMSNorm forward y_i = g_i·x_i/sqrt(mean(x²)+eps), with L = Σ y·dy.
func TestRMSNormBackwardF32(t *testing.T) {
	const rows, n = 3, 8
	eps := float32(1e-5)
	x := syntheticFloat32(rows*n, 1)
	g := syntheticFloat32(n, 2)
	dy := syntheticFloat32(rows*n, 3)

	forward := func(x, g []float32) []float32 {
		y := make([]float32, rows*n)
		for r := range rows {
			var ss float64
			for i := range n {
				ss += float64(x[r*n+i]) * float64(x[r*n+i])
			}
			rms := math.Sqrt(ss/float64(n) + float64(eps))
			for i := range n {
				y[r*n+i] = float32(float64(g[i]) * float64(x[r*n+i]) / rms)
			}
		}
		return y
	}
	loss := func(x, g []float32) float64 {
		y := forward(x, g)
		var s float64
		for i := range y {
			s += float64(y[i]) * float64(dy[i])
		}
		return s
	}

	dx, dg, err := RMSNormBackwardF32(dy, x, g, rows, n, eps)
	if err != nil {
		t.Fatalf("RMSNormBackwardF32: %v", err)
	}
	const eps2 = 1.0 / 512
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps2
			lp := loss(x, g)
			params[i] = orig - eps2
			lm := loss(x, g)
			params[i] = orig
			fd := (lp - lm) / (2 * eps2)
			if math.Abs(fd-float64(grad[i])) > 1e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dx", x, dx)
	check("dg", g, dg)
	t.Logf("RMSNorm VJP matches finite differences: dx[%d] dg[%d] all within tol", len(dx), len(dg))
}

// TestGeluGateMulBackwardF32 verifies the MLP activation VJP against finite differences of the forward
// gated_i = gelu_tanh(gate_i)·up_i, with L = Σ gated·dgated.
func TestGeluGateMulBackwardF32(t *testing.T) {
	const n = 12
	gate := syntheticFloat32(n, 1)
	up := syntheticFloat32(n, 2)
	dgated := syntheticFloat32(n, 3)

	loss := func(gate, up []float32) float64 {
		var s float64
		for i := range n {
			s += geluTanh(float64(gate[i])) * float64(up[i]) * float64(dgated[i])
		}
		return s
	}
	dgate, dup, err := GeluGateMulBackwardF32(dgated, gate, up, n)
	if err != nil {
		t.Fatalf("GeluGateMulBackwardF32: %v", err)
	}
	const eps = 1.0 / 1024
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps
			lp := loss(gate, up)
			params[i] = orig - eps
			lm := loss(gate, up)
			params[i] = orig
			fd := (lp - lm) / (2 * eps)
			if math.Abs(fd-float64(grad[i])) > 1e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dgate", gate, dgate)
	check("dup", up, dup)
	t.Logf("gelu·up VJP matches finite differences: dgate[%d] dup[%d] all within tol", len(dgate), len(dup))
}

// TestMLPBlockBackwardF32 gradient-checks the COMPOSED MLP-block backward end to end — proving the
// linear, gelu·up and RMSNorm VJPs chain correctly (including the rms→gate+up branch sum and the
// residual) — against finite differences of the full block forward out = h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h))).
func TestMLPBlockBackwardF32(t *testing.T) {
	requireNativeRuntime(t)
	const M, dModel, dFF = 2, 8, 16
	eps := float32(1e-5)
	h := syntheticFloat32(M*dModel, 1)
	normW := syntheticFloat32(dModel, 2)
	wGate := syntheticFloat32(dFF*dModel, 3)
	wUp := syntheticFloat32(dFF*dModel, 4)
	wDown := syntheticFloat32(dModel*dFF, 5)
	dout := syntheticFloat32(M*dModel, 6)

	forward := func() []float32 {
		normed := rmsNormForwardF32(h, normW, M, dModel, eps)
		gate, err := MatMulF32NT(normed, wGate, M, dModel, dFF)
		if err != nil {
			t.Fatal(err)
		}
		up, err := MatMulF32NT(normed, wUp, M, dModel, dFF)
		if err != nil {
			t.Fatal(err)
		}
		gated := make([]float32, M*dFF)
		for i := range gated {
			gated[i] = float32(geluTanh(float64(gate[i])) * float64(up[i]))
		}
		down, err := MatMulF32NT(gated, wDown, M, dFF, dModel)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, M*dModel)
		for i := range out {
			out[i] = h[i] + down[i]
		}
		return out
	}
	loss := func() float64 {
		out := forward()
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dout[i])
		}
		return s
	}

	g, err := MLPBlockBackwardF32(dout, h, normW, wGate, wUp, wDown, M, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MLPBlockBackwardF32: %v", err)
	}
	const eps2 = 1.0 / 512
	// check a strided sample of each gradient (full finite-diff over every weight is needlessly slow).
	check := func(name string, params, grad []float32) {
		step := 1
		if len(params) > 12 {
			step = len(params) / 12
		}
		for i := 0; i < len(params); i += step {
			orig := params[i]
			params[i] = orig + eps2
			lp := loss()
			params[i] = orig - eps2
			lm := loss()
			params[i] = orig
			fd := (lp - lm) / (2 * eps2)
			if math.Abs(fd-float64(grad[i])) > 2e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dH", h, g.DH)
	check("dNormW", normW, g.DNormW)
	check("dWGate", wGate, g.DWGate)
	check("dWUp", wUp, g.DWUp)
	check("dWDown", wDown, g.DWDown)
	t.Logf("MLP-block backward chains correctly: dH/dNormW/dWGate/dWUp/dWDown all match finite differences")
}

// TestSoftmaxBackwardF32 verifies the softmax VJP against finite differences of the row-wise softmax
// forward, with L = Σ y·dy.
func TestSoftmaxBackwardF32(t *testing.T) {
	const rows, n = 3, 7
	x := syntheticFloat32(rows*n, 1)
	dy := syntheticFloat32(rows*n, 2)

	softmax := func(x []float32) []float32 {
		y := make([]float32, rows*n)
		for r := range rows {
			xr, yr := x[r*n:(r+1)*n], y[r*n:(r+1)*n]
			mx := xr[0]
			for _, v := range xr {
				if v > mx {
					mx = v
				}
			}
			var sum float64
			for i, v := range xr {
				e := math.Exp(float64(v - mx))
				yr[i] = float32(e)
				sum += e
			}
			for i := range yr {
				yr[i] = float32(float64(yr[i]) / sum)
			}
		}
		return y
	}
	loss := func(x []float32) float64 {
		y := softmax(x)
		var s float64
		for i := range y {
			s += float64(y[i]) * float64(dy[i])
		}
		return s
	}
	y := softmax(x)
	dx, err := SoftmaxBackwardF32(dy, y, rows, n)
	if err != nil {
		t.Fatalf("SoftmaxBackwardF32: %v", err)
	}
	const eps = 1.0 / 1024
	for i := range x {
		orig := x[i]
		x[i] = orig + eps
		lp := loss(x)
		x[i] = orig - eps
		lm := loss(x)
		x[i] = orig
		fd := (lp - lm) / (2 * eps)
		if math.Abs(fd-float64(dx[i])) > 1e-2*(1+math.Abs(fd)) {
			t.Errorf("dx[%d]: analytic %.5f vs finite-diff %.5f", i, dx[i], fd)
		}
	}
	t.Logf("softmax VJP matches finite differences: dx[%d] within tol", len(dx))
}

// TestRoPEBackwardF32 verifies the RoPE VJP against finite differences of the rotation forward
// (half-split, partial rotary: rotaryDim < headDim), with L = Σ y·dy.
func TestRoPEBackwardF32(t *testing.T) {
	const nHeads, headDim, rotaryDim, pos = 2, 8, 4, 5
	base := float32(10000)
	x := syntheticFloat32(nHeads*headDim, 1)
	dy := syntheticFloat32(nHeads*headDim, 2)
	h := rotaryDim / 2

	forward := func(x []float32) []float32 {
		y := make([]float32, len(x))
		copy(y, x)
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
	loss := func(x []float32) float64 {
		y := forward(x)
		var s float64
		for i := range y {
			s += float64(y[i]) * float64(dy[i])
		}
		return s
	}
	dx, err := RoPEBackwardF32(dy, pos, nHeads, headDim, rotaryDim, base)
	if err != nil {
		t.Fatalf("RoPEBackwardF32: %v", err)
	}
	const eps = 1.0 / 1024
	for i := range x {
		orig := x[i]
		x[i] = orig + eps
		lp := loss(x)
		x[i] = orig - eps
		lm := loss(x)
		x[i] = orig
		fd := (lp - lm) / (2 * eps)
		if math.Abs(fd-float64(dx[i])) > 1e-2*(1+math.Abs(fd)) {
			t.Errorf("dx[%d]: analytic %.5f vs finite-diff %.5f", i, dx[i], fd)
		}
	}
	t.Logf("RoPE VJP matches finite differences: dx[%d] within tol (incl partial-rotary passthrough)", len(dx))
}

// TestAttnSingleHeadBackwardF32 gradient-checks the composed single-head causal attention backward
// against finite differences of O = softmax(Q·Kᵀ·scale + causal)·V, with L = Σ O·dOut — proving the
// softmax + matmul VJPs chain into an attention backward (the layer's other half).
func TestAttnSingleHeadBackwardF32(t *testing.T) {
	requireNativeRuntime(t)
	const L, d = 4, 6
	scale := float32(1.0 / math.Sqrt(d))
	q := syntheticFloat32(L*d, 1)
	k := syntheticFloat32(L*d, 2)
	v := syntheticFloat32(L*d, 3)
	dOut := syntheticFloat32(L*d, 4)

	forward := func() []float32 {
		s, err := MatMulF32NT(q, k, L, d, L)
		if err != nil {
			t.Fatal(err)
		}
		p := make([]float32, L*L)
		for i := range L {
			mx := float32(math.Inf(-1))
			for j := 0; j <= i; j++ {
				s[i*L+j] *= scale
				if s[i*L+j] > mx {
					mx = s[i*L+j]
				}
			}
			var sum float64
			for j := 0; j <= i; j++ {
				e := math.Exp(float64(s[i*L+j] - mx))
				p[i*L+j] = float32(e)
				sum += e
			}
			for j := 0; j <= i; j++ {
				p[i*L+j] = float32(float64(p[i*L+j]) / sum)
			}
		}
		o, err := MatMulF32(p, v, L, L, d)
		if err != nil {
			t.Fatal(err)
		}
		return o
	}
	loss := func() float64 {
		o := forward()
		var s float64
		for i := range o {
			s += float64(o[i]) * float64(dOut[i])
		}
		return s
	}

	dQ, dK, dV, err := AttnSingleHeadBackwardF32(dOut, q, k, v, L, d, scale, true)
	if err != nil {
		t.Fatalf("AttnSingleHeadBackwardF32: %v", err)
	}
	const eps = 1.0 / 1024
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps
			lp := loss()
			params[i] = orig - eps
			lm := loss()
			params[i] = orig
			fd := (lp - lm) / (2 * eps)
			if math.Abs(fd-float64(grad[i])) > 2e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dQ", q, dQ)
	check("dK", k, dK)
	check("dV", v, dV)
	t.Logf("attention backward chains correctly: dQ/dK/dV all match finite differences (causal)")
}

// TestAttnBlockBackwardF32 gradient-checks the COMPOSED full attention block (norm → q/k/v proj → RoPE →
// SDPA → o proj → residual) against finite differences of its forward — proving the RMSNorm, linear,
// RoPE and SDPA-core VJPs chain into a complete attention block (the transformer layer's other half).
func TestAttnBlockBackwardF32(t *testing.T) {
	requireNativeRuntime(t)
	const L, dModel, d, rotaryDim = 3, 8, 8, 4
	base, eps := float32(10000), float32(1e-5)
	scale := float32(1.0 / math.Sqrt(d))
	h := syntheticFloat32(L*dModel, 1)
	normW := syntheticFloat32(dModel, 2)
	wQ := syntheticFloat32(d*dModel, 3)
	wK := syntheticFloat32(d*dModel, 4)
	wV := syntheticFloat32(d*dModel, 5)
	wO := syntheticFloat32(dModel*d, 6)
	dout := syntheticFloat32(L*dModel, 7)

	forward := func() []float32 {
		normed := rmsNormForwardF32(h, normW, L, dModel, eps)
		q, err := MatMulF32NT(normed, wQ, L, dModel, d)
		if err != nil {
			t.Fatal(err)
		}
		k, err := MatMulF32NT(normed, wK, L, dModel, d)
		if err != nil {
			t.Fatal(err)
		}
		v, err := MatMulF32NT(normed, wV, L, dModel, d)
		if err != nil {
			t.Fatal(err)
		}
		qr := make([]float32, L*d)
		kr := make([]float32, L*d)
		for i := range L {
			copy(qr[i*d:(i+1)*d], ropeForwardF32(q[i*d:(i+1)*d], i, 1, d, rotaryDim, base))
			copy(kr[i*d:(i+1)*d], ropeForwardF32(k[i*d:(i+1)*d], i, 1, d, rotaryDim, base))
		}
		o, err := sdpaForwardSingleHeadF32(qr, kr, v, L, d, scale, true)
		if err != nil {
			t.Fatal(err)
		}
		attnOut, err := MatMulF32NT(o, wO, L, d, dModel)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, L*dModel)
		for i := range out {
			out[i] = h[i] + attnOut[i]
		}
		return out
	}
	loss := func() float64 {
		out := forward()
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dout[i])
		}
		return s
	}

	g, err := AttnBlockBackwardF32(dout, h, normW, wQ, wK, wV, wO, L, dModel, d, rotaryDim, base, scale, eps, true)
	if err != nil {
		t.Fatalf("AttnBlockBackwardF32: %v", err)
	}
	const eps2 = 1.0 / 1024
	check := func(name string, params, grad []float32) {
		step := 1
		if len(params) > 10 {
			step = len(params) / 10
		}
		for i := 0; i < len(params); i += step {
			orig := params[i]
			params[i] = orig + eps2
			lp := loss()
			params[i] = orig - eps2
			lm := loss()
			params[i] = orig
			fd := (lp - lm) / (2 * eps2)
			if math.Abs(fd-float64(grad[i])) > 2e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dH", h, g.DH)
	check("dNormW", normW, g.DNormW)
	check("dWQ", wQ, g.DWQ)
	check("dWK", wK, g.DWK)
	check("dWV", wV, g.DWV)
	check("dWO", wO, g.DWO)
	t.Logf("attention-BLOCK backward chains correctly: dH/dNormW/dWQ/dWK/dWV/dWO all match finite differences")
}

// TestMultiHeadAttnBackwardF32 gradient-checks the multi-head GQA attention backward (H=4 query heads,
// Hkv=2 kv heads, causal) against finite differences of the per-head SDPA forward — proving the GQA
// reduction (query heads sharing a kv head sum their dK/dV) is correct.
func TestMultiHeadAttnBackwardF32(t *testing.T) {
	requireNativeRuntime(t)
	const L, H, Hkv, d = 4, 4, 2, 6
	scale := float32(1.0 / math.Sqrt(d))
	gqa := H / Hkv
	q := syntheticFloat32(L*H*d, 1)
	k := syntheticFloat32(L*Hkv*d, 2)
	v := syntheticFloat32(L*Hkv*d, 3)
	dOut := syntheticFloat32(L*H*d, 4)

	// per-head causal SDPA forward, assembled into [L,H·d].
	headSDPA := func(qh, kh, vh []float32) []float32 {
		o, err := sdpaForwardSingleHeadF32(qh, kh, vh, L, d, scale, true)
		if err != nil {
			t.Fatal(err)
		}
		return o
	}
	loss := func() float64 {
		var s float64
		for h := range H {
			hk := h / gqa
			o := headSDPA(gatherHeadF32(q, L, H, d, h), gatherHeadF32(k, L, Hkv, d, hk), gatherHeadF32(v, L, Hkv, d, hk))
			doh := gatherHeadF32(dOut, L, H, d, h)
			for i := range o {
				s += float64(o[i]) * float64(doh[i])
			}
		}
		return s
	}
	dQ, dK, dV, err := MultiHeadAttnBackwardF32(dOut, q, k, v, L, H, Hkv, d, scale, true)
	if err != nil {
		t.Fatalf("MultiHeadAttnBackwardF32: %v", err)
	}
	const eps = 1.0 / 1024
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps
			lp := loss()
			params[i] = orig - eps
			lm := loss()
			params[i] = orig
			fd := (lp - lm) / (2 * eps)
			if math.Abs(fd-float64(grad[i])) > 2e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dQ", q, dQ)
	check("dK", k, dK) // GQA: each kv head's grad is the sum over its 2 query heads
	check("dV", v, dV)
	t.Logf("multi-head GQA attention backward correct: dQ[%d] dK[%d] dV[%d] match finite differences (H=%d Hkv=%d)", len(dQ), len(dK), len(dV), H, Hkv)
}

// TestQKNormBackwardF32 verifies gemma4's per-head q/k RMSNorm VJP against finite differences of the
// per-head RMSNorm forward (each head's d-vector normed by a shared [d] weight), with L = Σ y·dy.
func TestQKNormBackwardF32(t *testing.T) {
	const L, H, d = 2, 2, 4
	eps := float32(1e-5)
	x := syntheticFloat32(L*H*d, 1)
	normW := syntheticFloat32(d, 2)
	dy := syntheticFloat32(L*H*d, 3)

	forward := func(x, normW []float32) []float32 {
		y := make([]float32, L*H*d)
		for hr := range L * H { // each head-row is one d-vector
			xr := x[hr*d : (hr+1)*d]
			var ss float64
			for i := range d {
				ss += float64(xr[i]) * float64(xr[i])
			}
			rms := math.Sqrt(ss/float64(d) + float64(eps))
			for i := range d {
				y[hr*d+i] = float32(float64(normW[i]) * float64(xr[i]) / rms)
			}
		}
		return y
	}
	loss := func(x, normW []float32) float64 {
		y := forward(x, normW)
		var s float64
		for i := range y {
			s += float64(y[i]) * float64(dy[i])
		}
		return s
	}
	dx, dNormW, err := QKNormBackwardF32(dy, x, normW, L, H, d, eps)
	if err != nil {
		t.Fatalf("QKNormBackwardF32: %v", err)
	}
	const eps2 = 1.0 / 1024
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps2
			lp := loss(x, normW)
			params[i] = orig - eps2
			lm := loss(x, normW)
			params[i] = orig
			fd := (lp - lm) / (2 * eps2)
			if math.Abs(fd-float64(grad[i])) > 1e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dx", x, dx)
	check("dNormW", normW, dNormW)
	t.Logf("gemma4 QK-norm VJP matches finite differences: dx[%d] dNormW[%d] within tol", len(dx), len(dNormW))
}

// TestMultiHeadAttnBlockBackwardF32 gradient-checks the full MULTI-HEAD GQA attention block (the real
// gemma4 head structure) end-to-end against finite differences of its forward.
func TestMultiHeadAttnBlockBackwardF32(t *testing.T) {
	requireNativeRuntime(t)
	const L, dModel, H, Hkv, d, rotaryDim = 3, 16, 4, 2, 4, 4
	base, eps := float32(10000), float32(1e-5)
	scale := float32(1.0 / math.Sqrt(d))
	qDim, kvDim := H*d, Hkv*d
	hh := syntheticFloat32(L*dModel, 1)
	normW := syntheticFloat32(dModel, 2)
	wQ := syntheticFloat32(qDim*dModel, 3)
	wK := syntheticFloat32(kvDim*dModel, 4)
	wV := syntheticFloat32(kvDim*dModel, 5)
	wO := syntheticFloat32(dModel*qDim, 6)
	dout := syntheticFloat32(L*dModel, 7)

	forward := func() []float32 {
		normed := rmsNormForwardF32(hh, normW, L, dModel, eps)
		q, err := MatMulF32NT(normed, wQ, L, dModel, qDim)
		if err != nil {
			t.Fatal(err)
		}
		k, err := MatMulF32NT(normed, wK, L, dModel, kvDim)
		if err != nil {
			t.Fatal(err)
		}
		v, err := MatMulF32NT(normed, wV, L, dModel, kvDim)
		if err != nil {
			t.Fatal(err)
		}
		qr, kr := make([]float32, L*qDim), make([]float32, L*kvDim)
		for i := range L {
			copy(qr[i*qDim:(i+1)*qDim], ropeForwardF32(q[i*qDim:(i+1)*qDim], i, H, d, rotaryDim, base))
			copy(kr[i*kvDim:(i+1)*kvDim], ropeForwardF32(k[i*kvDim:(i+1)*kvDim], i, Hkv, d, rotaryDim, base))
		}
		o, err := multiHeadSDPAForwardF32(qr, kr, v, L, H, Hkv, d, scale, true)
		if err != nil {
			t.Fatal(err)
		}
		attnOut, err := MatMulF32NT(o, wO, L, qDim, dModel)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, L*dModel)
		for i := range out {
			out[i] = hh[i] + attnOut[i]
		}
		return out
	}
	loss := func() float64 {
		out := forward()
		var s float64
		for i := range out {
			s += float64(out[i]) * float64(dout[i])
		}
		return s
	}
	g, err := MultiHeadAttnBlockBackwardF32(dout, hh, normW, wQ, wK, wV, wO, L, dModel, H, Hkv, d, rotaryDim, base, scale, eps, true)
	if err != nil {
		t.Fatalf("MultiHeadAttnBlockBackwardF32: %v", err)
	}
	const eps2 = 1.0 / 1024
	check := func(name string, params, grad []float32) {
		step := 1
		if len(params) > 10 {
			step = len(params) / 10
		}
		for i := 0; i < len(params); i += step {
			orig := params[i]
			params[i] = orig + eps2
			lp := loss()
			params[i] = orig - eps2
			lm := loss()
			params[i] = orig
			fd := (lp - lm) / (2 * eps2)
			if math.Abs(fd-float64(grad[i])) > 2e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dH", hh, g.DH)
	check("dNormW", normW, g.DNormW)
	check("dWQ", wQ, g.DWQ)
	check("dWK", wK, g.DWK)
	check("dWV", wV, g.DWV)
	check("dWO", wO, g.DWO)
	t.Logf("MULTI-HEAD GQA attention block backward chains correctly: dH/dNormW/dWQ/dWK/dWV/dWO match finite differences (H=%d Hkv=%d)", H, Hkv)
}
