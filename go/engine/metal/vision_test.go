// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// vision_test.go validates the native vision tower against SELF-CONTAINED pure-Go fp32 references —
// it imports NO pkg/metal, so the tests survive that package's deletion (the whole point of the
// decoupling). The references transcribe metal's ACTUAL algorithm (vision_forward.go) with its real
// parameters (the attention scale is 1.0 — buildGemma4VisionModel's hardcoded value, not 1/√headDim).
// Each native op runs its Metal kernels; the reference runs Go loops, so a match validates the device
// path. bf16-in/fp32-accum vs the fp32 reference is the only expected deviation, reported as rel-L2.

// --- pure-Go fp32 references of metal's actual vision ops ---

// refMatmul: out[L,N] = in[L,K] @ Wᵀ, W row-major [N,K].
func refMatmul(in, w []float32, L, N, K int) []float32 {
	out := make([]float32, L*N)
	for r := range L {
		for n := range N {
			var acc float32
			for k := range K {
				acc += in[r*K+k] * w[n*K+k]
			}
			out[r*N+n] = acc
		}
	}
	return out
}

// refRMS RMS-normalises v (x·rsqrt(mean(x²)+eps)), scaling by w when non-nil — the plain gemma RMSNorm.
func refRMS(v, w []float32, eps float32) []float32 {
	out := make([]float32, len(v))
	var ss float32
	for _, x := range v {
		ss += x * x
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(v))+eps)))
	for i := range v {
		out[i] = v[i] * inv
		if w != nil {
			out[i] *= w[i]
		}
	}
	return out
}

func refRMSRows(m, w []float32, rows, axis int, eps float32) []float32 {
	o := make([]float32, len(m))
	for r := range rows {
		copy(o[r*axis:r*axis+axis], refRMS(m[r*axis:r*axis+axis], w, eps))
	}
	return o
}

func refGeluTanh(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Tanh(float64(0.7978845608028654*(x+0.044715*x*x*x)))))
}

// refRoPE2D transcribes metal's gemma4VisionApply2DRoPE: [L,N,d] → head-major [N,L,d].
func refRoPE2D(x []float32, L, N, headDim, gridW int, base float32) []float32 {
	rp := 2 * (headDim / 4)
	half := rp / 2
	inv := make([]float64, half)
	for i := range half {
		inv[i] = 1.0 / math.Pow(float64(base), float64(2*i)/float64(rp))
	}
	o := make([]float32, N*L*headDim)
	part := func(out, in []float32, coord float64) {
		for d := range rp {
			a := coord * inv[d%half]
			c, s := float32(math.Cos(a)), float32(math.Sin(a))
			var rot float32
			if d < half {
				rot = -in[half+d]
			} else {
				rot = in[d-half]
			}
			out[d] = in[d]*c + rot*s
		}
	}
	for pos := range L {
		cx, cy := float64(pos%gridW), float64(pos/gridW)
		for h := range N {
			in := x[(pos*N+h)*headDim : (pos*N+h)*headDim+headDim]
			out := o[(h*L+pos)*headDim : (h*L+pos)*headDim+headDim]
			part(out[0:rp], in[0:rp], cx)
			part(out[rp:2*rp], in[rp:2*rp], cy)
			for d := 2 * rp; d < headDim; d++ {
				out[d] = in[d]
			}
		}
	}
	return o
}

// refAttention: full non-causal attention, q/k/v head-major [N,L,d], scale applied to scores.
func refAttention(q, k, v []float32, N, L, headDim int, scale float32) []float32 {
	out := make([]float32, N*L*headDim)
	for h := range N {
		qh, kh, vh := q[h*L*headDim:], k[h*L*headDim:], v[h*L*headDim:]
		for i := range L {
			sc := make([]float32, L)
			mx := float32(math.Inf(-1))
			for j := range L {
				var s float32
				for d := range headDim {
					s += qh[i*headDim+d] * kh[j*headDim+d]
				}
				sc[j] = s * scale
				if sc[j] > mx {
					mx = sc[j]
				}
			}
			var sum float32
			for j := range sc {
				sc[j] = float32(math.Exp(float64(sc[j] - mx)))
				sum += sc[j]
			}
			for d := range headDim {
				var acc float32
				for j := range L {
					acc += sc[j] / sum * vh[j*headDim+d]
				}
				out[(h*L+i)*headDim+d] = acc
			}
		}
	}
	return out
}

func bf16Round(f []float32) []float32 {
	out := make([]float32, len(f))
	for i, v := range f {
		out[i] = bf16ToF32(byte(f32ToBF16(v)), byte(f32ToBF16(v)>>8))
	}
	return out
}

func relL2Cos(got, want []float32) (float64, float64) {
	var sumSq, refSq, dot, na, nb float64
	for i := range want {
		d := float64(got[i] - want[i])
		sumSq += d * d
		refSq += float64(want[i]) * float64(want[i])
		dot += float64(got[i]) * float64(want[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(sumSq / (refSq + 1e-12)), dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
}

// TestMatRowsBF16 validates the multi-row projection (looped gemv) against a pure-Go fp32 matmul.
// (Recorded separately: MatRowsBF16 is byte-IDENTICAL to metal.Matmul across the gemv and steel-GEMM
// regimes — see the 1/n commit; here we keep the durable check self-contained.)
func matRowsBF16Fixture(L, outDim, inDim int) ([]byte, []byte) {
	in := toBF16Bytes(syntheticFloat32(L*inDim, inDim+5))
	w := toBF16Bytes(syntheticFloat32(outDim*inDim, outDim+7))
	return w, in
}

func matRowsBF16LoopedMatVecReference(tb testing.TB, w, in []byte, L, outDim, inDim int) []byte {
	tb.Helper()
	out := make([]byte, L*outDim*bf16Size)
	for r := range L {
		row, err := MatVecBF16(w, in[r*inDim*bf16Size:(r+1)*inDim*bf16Size], outDim, inDim)
		if err != nil {
			tb.Fatalf("MatVecBF16 row %d: %v", r, err)
		}
		copy(out[r*outDim*bf16Size:(r+1)*outDim*bf16Size], row)
	}
	return out
}

func TestMatRowsBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const L, outDim, inDim = 4, 128, 256
	w, in := matRowsBF16Fixture(L, outDim, inDim)
	if _, err := MatRowsBF16(w, in, L, outDim, inDim); err != nil {
		t.Fatalf("MatRowsBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MatRowsBF16(w, in, L, outDim, inDim); err != nil {
			t.Fatalf("MatRowsBF16: %v", err)
		}
	})
	if allocs > 109 {
		t.Fatalf("MatRowsBF16 allocations = %.0f, want <= 109", allocs)
	}
}

func TestMatRowsBF16MatchesLoopedMatVecReference(t *testing.T) {
	requireNativeRuntime(t)

	const L, outDim, inDim = 5, 96, 256
	w, in := matRowsBF16Fixture(L, outDim, inDim)
	got, err := MatRowsBF16(w, in, L, outDim, inDim)
	if err != nil {
		t.Fatalf("MatRowsBF16: %v", err)
	}
	want := matRowsBF16LoopedMatVecReference(t, w, in, L, outDim, inDim)
	eqBytes(t, "MatRowsBF16 vs looped MatVecBF16", got, want)
}

func TestMatRowsF32AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const L, outDim, inDim = 4, 128, 256
	w := syntheticFloat32(outDim*inDim, outDim+7)
	in := syntheticFloat32(L*inDim, inDim+5)
	if _, err := matRowsF32(w, in, L, outDim, inDim); err != nil {
		t.Fatalf("matRowsF32 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := matRowsF32(w, in, L, outDim, inDim); err != nil {
			t.Fatalf("matRowsF32: %v", err)
		}
	})
	if allocs > 223 {
		t.Fatalf("matRowsF32 allocations = %.0f, want <= 223", allocs)
	}
}

func TestMatRowsF32MatchesReference(t *testing.T) {
	requireNativeRuntime(t)

	const L, outDim, inDim = 4, 128, 256
	w := syntheticFloat32(outDim*inDim, outDim+7)
	in := syntheticFloat32(L*inDim, inDim+5)
	got, err := matRowsF32(w, in, L, outDim, inDim)
	if err != nil {
		t.Fatalf("matRowsF32: %v", err)
	}
	want := refMatmul(in, w, L, outDim, inDim)
	relL2, cos := relL2Cos(got, want)
	if relL2 > 1e-5 || cos < 0.999999 {
		t.Fatalf("matRowsF32 vs ref: rel-L2=%.3e cosine=%.6f", relL2, cos)
	}
}

func TestMatRowsBF16(t *testing.T) {
	requireNativeRuntime(t)
	const L, K, N = 64, 768, 768
	in, w := bf16Round(syntheticFloat32(L*K, 3)), bf16Round(syntheticFloat32(N*K, 7))
	got, err := MatRowsBF16(toBF16Bytes(w), toBF16Bytes(in), L, N, K)
	if err != nil {
		t.Fatalf("MatRowsBF16: %v", err)
	}
	relL2, cos := relL2Cos(bf16Floats(got), refMatmul(in, w, L, N, K))
	t.Logf("MatRowsBF16 vs fp32 matmul [L=%d K=%d N=%d]: rel-L2=%.3e cosine=%.6f", L, K, N, relL2, cos)
	if relL2 > 5e-3 {
		t.Fatalf("MatRowsBF16 rel-L2 %.3e > 5e-3", relL2)
	}
}

// TestVisionPatchEmbed validates patch-embed (scale (x-0.5)·2 → project → +posEmb) vs a pure-Go ref.
func TestVisionPatchEmbed(t *testing.T) {
	requireNativeRuntime(t)
	const L, patchDim, hidden = 64, 768, 768
	px, w, pe := bf16Round(syntheticFloat32(L*patchDim, 5)), bf16Round(syntheticFloat32(hidden*patchDim, 9)), bf16Round(syntheticFloat32(L*hidden, 13))
	got, err := VisionPatchEmbed(toBF16Bytes(px), toBF16Bytes(w), toBF16Bytes(pe), L, patchDim, hidden)
	if err != nil {
		t.Fatalf("VisionPatchEmbed: %v", err)
	}
	scaled := make([]float32, len(px))
	for i, v := range px {
		scaled[i] = (v - 0.5) * 2
	}
	want := refMatmul(scaled, w, L, hidden, patchDim)
	for i := range want {
		want[i] += pe[i]
	}
	relL2, cos := relL2Cos(bf16Floats(got), want)
	t.Logf("VisionPatchEmbed vs fp32 reference: rel-L2=%.3e cosine=%.6f", relL2, cos)
	if relL2 > 5e-3 {
		t.Fatalf("VisionPatchEmbed rel-L2 %.3e > 5e-3 — wiring bug", relL2)
	}
	noPos, err := VisionPatchEmbed(toBF16Bytes(px), toBF16Bytes(w), nil, L, patchDim, hidden)
	if err != nil || len(noPos) != L*hidden*bf16Size {
		t.Fatalf("nil-posEmb path: err=%v len=%d", err, len(noPos))
	}
}

// TestVisionSDPA validates the decomposed full attention vs a pure-Go fp32 attention reference.
func TestVisionSDPA(t *testing.T) {
	requireNativeRuntime(t)
	const L, nHeads, headDim = 64, 4, 64
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q, k, v := bf16Round(syntheticFloat32(nHeads*L*headDim, 3)), bf16Round(syntheticFloat32(nHeads*L*headDim, 5)), bf16Round(syntheticFloat32(nHeads*L*headDim, 7))
	got, err := VisionSDPA(toBF16Bytes(q), toBF16Bytes(k), toBF16Bytes(v), L, nHeads, nHeads, headDim, scale)
	if err != nil {
		t.Fatalf("VisionSDPA: %v", err)
	}
	relL2, cos := relL2Cos(bf16Floats(got), refAttention(q, k, v, nHeads, L, headDim, scale))
	t.Logf("VisionSDPA vs fp32 attention reference [L=%d heads=%d d=%d]: rel-L2=%.3e cosine=%.6f", L, nHeads, headDim, relL2, cos)
	if relL2 > 1e-2 {
		t.Fatalf("VisionSDPA rel-L2 %.3e > 1e-2", relL2)
	}
}

func visionSDPAWithKernelSoftmax(t *testing.T, q, k, v []byte, L, nHeads, nKVHeads, headDim int, scale float32) []byte {
	t.Helper()
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		t.Fatalf("bad head geometry")
	}
	grp := nHeads / nKVHeads
	out := make([]byte, nHeads*L*headDim*bf16Size)
	for h := range nHeads {
		kvh := h / grp
		qh := bf16HeadF32(q, h, L, headDim)
		kh := bf16HeadF32(k, kvh, L, headDim)
		vh := bf16HeadF32(v, kvh, L, headDim)

		scores, err := matRowsF32(kh, qh, L, L, headDim)
		if err != nil {
			t.Fatalf("matRowsF32 scores: %v", err)
		}
		for i := range scores {
			scores[i] *= scale
		}
		probs, err := SoftmaxF32(scores, L)
		if err != nil {
			t.Fatalf("SoftmaxF32: %v", err)
		}
		oh, err := matRowsF32(transposeF32(vh, L, headDim), probs, L, headDim, L)
		if err != nil {
			t.Fatalf("matRowsF32 output: %v", err)
		}
		base := h * L * headDim * bf16Size
		for i, val := range oh {
			hh := f32ToBF16(val)
			out[base+i*bf16Size], out[base+i*bf16Size+1] = byte(hh), byte(hh>>8)
		}
	}
	return out
}

func TestVisionSDPAUsesKernelSoftmax(t *testing.T) {
	requireNativeRuntime(t)
	const L, nHeads, nKVHeads, headDim = 97, 4, 2, 64
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := toBF16Bytes(bf16Round(syntheticFloat32(nHeads*L*headDim, 17)))
	k := toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*L*headDim, 19)))
	v := toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*L*headDim, 23)))

	got, err := VisionSDPA(q, k, v, L, nHeads, nKVHeads, headDim, scale)
	if err != nil {
		t.Fatalf("VisionSDPA: %v", err)
	}
	want := visionSDPAWithKernelSoftmax(t, q, k, v, L, nHeads, nKVHeads, headDim, scale)
	eqBytes(t, "VisionSDPA kernel-softmax route", got, want)
}

func TestVisionSDPAScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small := getVisionSDPAScratch(32, 4, 2, 64)
	putVisionSDPAScratch(small)
	large := getVisionSDPAScratch(64, 4, 2, 64)
	putVisionSDPAScratch(large)

	gotSmall := getVisionSDPAScratch(32, 4, 2, 64)
	defer putVisionSDPAScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("VisionSDPA scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge := getVisionSDPAScratch(64, 4, 2, 64)
	defer putVisionSDPAScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("VisionSDPA scratch pool evicted the large scratch after reusing the small scratch")
	}
}

func TestVisionSDPAAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const L, nHeads, nKVHeads, headDim = 64, 4, 2, 64
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	q := toBF16Bytes(bf16Round(syntheticFloat32(nHeads*L*headDim, 31)))
	k := toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*L*headDim, 37)))
	v := toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*L*headDim, 41)))
	if _, err := VisionSDPA(q, k, v, L, nHeads, nKVHeads, headDim, scale); err != nil {
		t.Fatalf("VisionSDPA warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(3, func() {
		if _, err := VisionSDPA(q, k, v, L, nHeads, nKVHeads, headDim, scale); err != nil {
			t.Fatalf("VisionSDPA: %v", err)
		}
	})
	if allocs > 1300 {
		t.Fatalf("VisionSDPA allocations = %.0f, want <= 1300", allocs)
	}
}

// TestVisionEncoderLayer validates the full encoder layer vs a pure-Go fp32 reference of metal's
// actual layer — at the REAL attention scale 1.0 (not the 1/√headDim that was wrongly assumed).
func TestVisionEncoderLayer(t *testing.T) {
	requireNativeRuntime(t)
	const hidden, nHeads, headDim, gridH, gridW, ffDim = 256, 4, 64, 4, 4, 512
	const L = gridH * gridW
	qDim := nHeads * headDim
	eps, base := float32(1e-6), float32(100)
	w := func(salt, n int) []float32 { return bf16Round(syntheticFloat32(n, salt)) }

	nw := &VisionLayerWeights{
		InputNorm: toBF16Bytes(w(1, hidden)), PostAttnNorm: toBF16Bytes(w(2, hidden)), PreFFNorm: toBF16Bytes(w(3, hidden)), PostFFNorm: toBF16Bytes(w(4, hidden)),
		WQ: toBF16Bytes(w(5, qDim*hidden)), WK: toBF16Bytes(w(6, qDim*hidden)), WV: toBF16Bytes(w(7, qDim*hidden)), WO: toBF16Bytes(w(8, hidden*qDim)),
		QNorm: toBF16Bytes(w(9, headDim)), KNorm: toBF16Bytes(w(10, headDim)),
		WGate: toBF16Bytes(w(11, ffDim*hidden)), WUp: toBF16Bytes(w(12, ffDim*hidden)), WDown: toBF16Bytes(w(13, hidden*ffDim)),
	}
	cfg := VisionConfig{Hidden: hidden, NumLayers: 1, NumHeads: nHeads, NumKVHeads: nHeads, HeadDim: headDim, GridH: gridH, GridW: gridW, RopeBase: base, RMSNormEps: eps}
	x := bf16Round(syntheticFloat32(L*hidden, 20))
	got, err := VisionEncoderLayer(toBF16Bytes(x), nw, cfg)
	if err != nil {
		t.Fatalf("VisionEncoderLayer: %v", err)
	}

	// reference: pre-norm block, attention scale 1.0
	add := func(a, b []float32) []float32 {
		o := make([]float32, len(a))
		for i := range a {
			o[i] = a[i] + b[i]
		}
		return o
	}
	headRMS := func(m, wt []float32) []float32 { return refRMSRows(m, wt, L*nHeads, headDim, eps) }
	transHead := func(m []float32) []float32 { // [L,N,d] → [N,L,d]
		o := make([]float32, len(m))
		for pos := range L {
			for h := range nHeads {
				copy(o[(h*L+pos)*headDim:(h*L+pos)*headDim+headDim], m[(pos*nHeads+h)*headDim:(pos*nHeads+h)*headDim+headDim])
			}
		}
		return o
	}
	normed := refRMSRows(x, w(1, hidden), L, hidden, eps)
	q := refRoPE2D(headRMS(refMatmul(normed, w(5, qDim*hidden), L, qDim, hidden), w(9, headDim)), L, nHeads, headDim, gridW, base)
	k := refRoPE2D(headRMS(refMatmul(normed, w(6, qDim*hidden), L, qDim, hidden), w(10, headDim)), L, nHeads, headDim, gridW, base)
	v := transHead(headRMS(refMatmul(normed, w(7, qDim*hidden), L, qDim, hidden), nil))
	attn := refAttention(q, k, v, nHeads, L, headDim, 1.0) // scale 1.0 — the actual value
	tok := make([]float32, L*qDim)
	for h := range nHeads {
		for i := range L {
			copy(tok[(i*nHeads+h)*headDim:(i*nHeads+h)*headDim+headDim], attn[(h*L+i)*headDim:(h*L+i)*headDim+headDim])
		}
	}
	attnOut := refMatmul(tok, w(8, hidden*qDim), L, hidden, qDim)
	h := add(x, refRMSRows(attnOut, w(2, hidden), L, hidden, eps))
	ffIn := refRMSRows(h, w(3, hidden), L, hidden, eps)
	gate, up := refMatmul(ffIn, w(11, ffDim*hidden), L, ffDim, hidden), refMatmul(ffIn, w(12, ffDim*hidden), L, ffDim, hidden)
	gated := make([]float32, len(gate))
	for i := range gate {
		gated[i] = refGeluTanh(gate[i]) * up[i]
	}
	ff := refMatmul(gated, w(13, hidden*ffDim), L, hidden, ffDim)
	want := add(h, refRMSRows(ff, w(4, hidden), L, hidden, eps))

	relL2, cos := relL2Cos(bf16Floats(got), want)
	t.Logf("VisionEncoderLayer vs fp32 reference (scale 1.0) [hidden=%d heads=%d d=%d L=%d]: rel-L2=%.3e cosine=%.6f", hidden, nHeads, headDim, L, relL2, cos)
	if cos < 0.999 || relL2 > 3e-2 {
		t.Fatalf("VisionEncoderLayer rel-L2 %.3e cosine %.6f — beyond bf16 tolerance (wiring/scale bug)", relL2, cos)
	}
}

// refEncoderLayer is the pure-Go fp32 reference of metal's actual encoder layer (scale 1.0), shared by
// the layer and tower tests. ws keys mirror the synthetic salts (1-4 norms, 5-8 QKVO, 9-10 QK-norm,
// 11-13 gate/up/down).
func refEncoderLayer(x []float32, ws map[int][]float32, L, hidden, nHeads, headDim, gridW, ffDim int, base, eps float32) []float32 {
	qDim := nHeads * headDim
	add := func(a, b []float32) []float32 {
		o := make([]float32, len(a))
		for i := range a {
			o[i] = a[i] + b[i]
		}
		return o
	}
	headRMS := func(m, wt []float32) []float32 { return refRMSRows(m, wt, L*nHeads, headDim, eps) }
	transHead := func(m []float32) []float32 {
		o := make([]float32, len(m))
		for pos := range L {
			for h := range nHeads {
				copy(o[(h*L+pos)*headDim:(h*L+pos)*headDim+headDim], m[(pos*nHeads+h)*headDim:(pos*nHeads+h)*headDim+headDim])
			}
		}
		return o
	}
	normed := refRMSRows(x, ws[1], L, hidden, eps)
	q := refRoPE2D(headRMS(refMatmul(normed, ws[5], L, qDim, hidden), ws[9]), L, nHeads, headDim, gridW, base)
	k := refRoPE2D(headRMS(refMatmul(normed, ws[6], L, qDim, hidden), ws[10]), L, nHeads, headDim, gridW, base)
	v := transHead(headRMS(refMatmul(normed, ws[7], L, qDim, hidden), nil))
	attn := refAttention(q, k, v, nHeads, L, headDim, 1.0)
	tok := make([]float32, L*qDim)
	for h := range nHeads {
		for i := range L {
			copy(tok[(i*nHeads+h)*headDim:(i*nHeads+h)*headDim+headDim], attn[(h*L+i)*headDim:(h*L+i)*headDim+headDim])
		}
	}
	hh := add(x, refRMSRows(refMatmul(tok, ws[8], L, hidden, qDim), ws[2], L, hidden, eps))
	ffIn := refRMSRows(hh, ws[3], L, hidden, eps)
	gate, up := refMatmul(ffIn, ws[11], L, ffDim, hidden), refMatmul(ffIn, ws[12], L, ffDim, hidden)
	gated := make([]float32, len(gate))
	for i := range gate {
		gated[i] = refGeluTanh(gate[i]) * up[i]
	}
	return add(hh, refRMSRows(refMatmul(gated, ws[13], L, hidden, ffDim), ws[4], L, hidden, eps))
}

// TestVisionTower validates the whole tower (grid → patch-embed → encoder layer → post-norm → grid
// pooler → standardize → projector) against a self-contained pure-Go reference of metal's actual
// Gemma4VisionModel.Forward — at the real scale 1.0, with the √Hidden pooler scale and the spatial
// poolByGrid path (poolKernel 2 over a 4×4 grid → 4 soft tokens).
func TestVisionTower(t *testing.T) {
	requireNativeRuntime(t)
	const hidden, nHeads, headDim, patchDim, gridH, gridW, ffDim, poolK, textHid = 256, 4, 64, 128, 4, 4, 512, 2, 128
	const L = gridH * gridW
	qDim := nHeads * headDim
	eps, base := float32(1e-6), float32(100)
	w := func(salt, n int) []float32 { return bf16Round(syntheticFloat32(n, salt)) }
	ws := map[int][]float32{1: w(1, hidden), 2: w(2, hidden), 3: w(3, hidden), 4: w(4, hidden),
		5: w(5, qDim*hidden), 6: w(6, qDim*hidden), 7: w(7, qDim*hidden), 8: w(8, hidden*qDim),
		9: w(9, headDim), 10: w(10, headDim), 11: w(11, ffDim*hidden), 12: w(12, ffDim*hidden), 13: w(13, hidden*ffDim)}
	patchW, postLN, stdBias, stdScale, projW := w(30, hidden*patchDim), w(31, hidden), w(32, hidden), w(33, hidden), w(34, textHid*hidden)

	nw := &VisionWeights{
		PatchEmbedding: toBF16Bytes(patchW), PostLayernorm: toBF16Bytes(postLN), StdBias: toBF16Bytes(stdBias), StdScale: toBF16Bytes(stdScale),
		Layers: []VisionLayerWeights{{
			InputNorm: toBF16Bytes(ws[1]), PostAttnNorm: toBF16Bytes(ws[2]), PreFFNorm: toBF16Bytes(ws[3]), PostFFNorm: toBF16Bytes(ws[4]),
			WQ: toBF16Bytes(ws[5]), WK: toBF16Bytes(ws[6]), WV: toBF16Bytes(ws[7]), WO: toBF16Bytes(ws[8]), QNorm: toBF16Bytes(ws[9]), KNorm: toBF16Bytes(ws[10]),
			WGate: toBF16Bytes(ws[11]), WUp: toBF16Bytes(ws[12]), WDown: toBF16Bytes(ws[13]),
		}},
		Projector: VisionProjectorWeights{Projection: VisionProjectorLinear{Weight: toBF16Bytes(projW)}, Eps: eps},
	}
	cfg := VisionConfig{Hidden: hidden, PatchDim: patchDim, NumLayers: 1, NumHeads: nHeads, NumKVHeads: nHeads, HeadDim: headDim, RopeBase: base, RMSNormEps: eps, PoolKernel: poolK}
	px := bf16Round(syntheticFloat32(L*patchDim, 20))
	got, err := VisionTower(toBF16Bytes(px), nw, cfg)
	if err != nil {
		t.Fatalf("VisionTower: %v", err)
	}

	// reference tower
	scaled := make([]float32, len(px))
	for i, v := range px {
		scaled[i] = (v - 0.5) * 2
	}
	h := refEncoderLayer(refMatmul(scaled, patchW, L, hidden, patchDim), ws, L, hidden, nHeads, headDim, gridW, ffDim, base, eps)
	h = refRMSRows(h, postLN, L, hidden, eps)
	rows, cols := gridH/poolK, gridW/poolK
	embScale := float32(math.Sqrt(float64(hidden)))
	np := rows * cols
	pooled := make([]float32, np*hidden)
	for y := range rows {
		for x := range cols {
			for hh := range hidden {
				var acc float32
				for dy := range poolK {
					for dx := range poolK {
						acc += h[((y*poolK+dy)*gridW+(x*poolK+dx))*hidden+hh]
					}
				}
				pooled[(y*cols+x)*hidden+hh] = (acc/float32(poolK*poolK)*embScale - stdBias[hh]) * stdScale[hh]
			}
		}
	}
	want := refMatmul(refRMSRows(pooled, nil, np, hidden, eps), projW, np, textHid, hidden)

	relL2, cos := relL2Cos(bf16Floats(got), want)
	t.Logf("VisionTower vs fp32 reference [L=%d pooled=%d textHidden=%d]: rel-L2=%.3e cosine=%.6f", L, np, textHid, relL2, cos)
	if len(got) != np*textHid*bf16Size {
		t.Fatalf("VisionTower output length %d, want %d", len(got), np*textHid*bf16Size)
	}
	if cos < 0.999 || relL2 > 1e-2 {
		t.Fatalf("VisionTower rel-L2 %.3e cosine %.6f — tower assembly/pooler/projector bug", relL2, cos)
	}
}

// TestVisionInjectFeatures pins the image-placeholder splice: each image-token position takes the next
// vision feature row in order, the rest pass through, and a slot/feature count mismatch errors. Pure
// host logic — no device needed.
func TestVisionInjectFeatures(t *testing.T) {
	const H = 8
	const imgTok = int32(99)
	tokenIDs := []int32{10, imgTok, 11, imgTok, 12} // image tokens at positions 1 and 3
	emb := toBF16Bytes(syntheticFloat32(5*H, 3))
	feat := toBF16Bytes(syntheticFloat32(2*H, 7))
	got, err := VisionInjectFeatures(emb, tokenIDs, feat, imgTok, H)
	if err != nil {
		t.Fatalf("VisionInjectFeatures: %v", err)
	}
	g, e, f := bf16Floats(got), bf16Floats(emb), bf16Floats(feat)
	eq := func(a, b []float32, name string) {
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%s mismatch at %d: %v vs %v", name, i, a[i], b[i])
			}
		}
	}
	eq(g[1*H:2*H], f[0:H], "pos1=feature0")     // first image slot → feature 0
	eq(g[3*H:4*H], f[1*H:2*H], "pos3=feature1") // second image slot → feature 1
	eq(g[0:H], e[0:H], "pos0 unchanged")
	eq(g[2*H:3*H], e[2*H:3*H], "pos2 unchanged")
	eq(g[4*H:5*H], e[4*H:5*H], "pos4 unchanged")

	// slot/feature count mismatch must error (1 feature for 2 slots).
	if _, err := VisionInjectFeatures(emb, tokenIDs, toBF16Bytes(syntheticFloat32(H, 1)), imgTok, H); err == nil {
		t.Fatal("expected an error on feature/slot count mismatch")
	}
}

// TestAudioInjectFeatures pins the audio-placeholder splice against the same
// host contract as vision: audio rows replace audio-token embeddings in order,
// ordinary token embeddings pass through, and slot/row mismatches fail.
func TestAudioInjectFeatures(t *testing.T) {
	const H = 8
	const audioTok = int32(77)
	tokenIDs := []int32{10, audioTok, audioTok, 11, 12}
	emb := toBF16Bytes(syntheticFloat32(5*H, 5))
	feat := toBF16Bytes(syntheticFloat32(2*H, 9))
	got, err := AudioInjectFeatures(emb, tokenIDs, feat, audioTok, H)
	if err != nil {
		t.Fatalf("AudioInjectFeatures: %v", err)
	}
	g, e, f := bf16Floats(got), bf16Floats(emb), bf16Floats(feat)
	eq := func(a, b []float32, name string) {
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%s mismatch at %d: %v vs %v", name, i, a[i], b[i])
			}
		}
	}
	eq(g[1*H:2*H], f[0:H], "pos1=feature0")
	eq(g[2*H:3*H], f[1*H:2*H], "pos2=feature1")
	eq(g[0:H], e[0:H], "pos0 unchanged")
	eq(g[3*H:4*H], e[3*H:4*H], "pos3 unchanged")
	eq(g[4*H:5*H], e[4*H:5*H], "pos4 unchanged")

	if _, err := AudioInjectFeatures(emb, tokenIDs, toBF16Bytes(syntheticFloat32(H, 1)), audioTok, H); err == nil {
		t.Fatal("expected an error on audio feature/slot count mismatch")
	}
}
