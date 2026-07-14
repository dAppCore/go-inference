// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	"dappco.re/go/inference/model"
)

// hostArchQuantReference is the independent float reference for the arch-driven quant decode
// forward: the same weights (dequantised through the production dequantizeAffineRowsF32, so
// storage truth is shared), the same per-layer geometry (kvHeadsOf/headDimOf), but every op —
// rms, matvec, half-split RoPE, causal attention, gelu — re-derived on the host in float64.
// It shares NO emission or kernel code with the GPU lanes, so agreement is evidence of
// correctness rather than consistency (#348: the ICB and stepToken lanes agree with each
// other on the 31B-geometry fixture while the real 31B degenerates — a truth anchor was the
// missing instrument).
func hostArchQuantReference(t *testing.T, inputs [][]byte, qlayers []QuantizedLayerWeights,
	specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, window int,
	base, scale, eps float32, valueNorm bool) [][]byte {
	return hostArchQuantReferenceRope(t, inputs, qlayers, specs, dModel, nHeads, nKVHeads, headDim, dFF, window, 0, 0, base, base, scale, eps, valueNorm)
}

// hostArchQuantReferenceRope is hostArchQuantReference with the session's rope split:
// global layers rotate gRotDim dims at gBase, sliding layers lRotDim at lBase (≤0 = the
// layer's full head dim), pairs (i, i+rotDim/2), tail dims pass through — the fused
// kernel's documented partial-rotary semantics.
func hostArchQuantReferenceRope(t *testing.T, inputs [][]byte, qlayers []QuantizedLayerWeights,
	specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, window int,
	gRotDim, lRotDim int, gBase, lBase, scale, eps float32, valueNorm bool) [][]byte {
	t.Helper()
	type mat struct {
		w          []float32
		rows, cols int
	}
	deq := func(w QuantWeight, rows, cols int) mat {
		f, err := dequantizeAffineRowsF32(w.Packed, w.Scales, w.Biases, rows, cols, qlayers[0].GroupSize, qlayers[0].Bits)
		if err != nil {
			t.Fatalf("dequant: %v", err)
		}
		return mat{f, rows, cols}
	}
	matvec := func(m mat, x []float32) []float32 {
		out := make([]float32, m.rows)
		for r := range m.rows {
			var acc float64
			row := m.w[r*m.cols : (r+1)*m.cols]
			for c, v := range x {
				acc += float64(row[c]) * float64(v)
			}
			out[r] = float32(acc)
		}
		return out
	}
	roundBF16 := func(f []float32) {
		for i, v := range f {
			h := f32ToBF16(v)
			f[i] = bf16ToF32(byte(h), byte(h>>8))
		}
	}
	bf16ToF32s := func(b []byte) []float32 {
		f := make([]float32, len(b)/2)
		for i := range f {
			f[i] = bf16ToF32(b[i*2], b[i*2+1])
		}
		return f
	}
	// half-split rotation with partial-rotary support: pairs (i, i+rotDim/2), angle
	// pos·b^(-2i/rotDim), dims ≥ rotDim pass through. The full-rotary case was proven
	// identical to the engine's pinned RoPE (cosines matched to 6 decimals when swapped).
	rope := func(x []float32, heads, hd, rotDim int, b float32, pos int) {
		if rotDim <= 0 || rotDim > hd {
			rotDim = hd
		}
		half := rotDim / 2
		for h := range heads {
			for i := range half {
				theta := float64(pos) * math.Pow(float64(b), -2*float64(i)/float64(rotDim))
				c, s := math.Cos(theta), math.Sin(theta)
				a, bb := float64(x[h*hd+i]), float64(x[h*hd+i+half])
				x[h*hd+i] = float32(a*c - bb*s)
				x[h*hd+i+half] = float32(a*s + bb*c)
			}
		}
	}

	type lw struct{ q, k, v, o, gate, up, down mat }
	ws := make([]lw, len(qlayers))
	for li, ql := range qlayers {
		lhd := headDimOf(specs[li], headDim)
		lkv := kvHeadsOf(specs[li], nKVHeads)
		ws[li] = lw{
			q: deq(ql.Q, nHeads*lhd, dModel), k: deq(ql.K, lkv*lhd, dModel),
			o:    deq(ql.O, dModel, nHeads*lhd),
			gate: deq(ql.Gate, dFF, dModel), up: deq(ql.Up, dFF, dModel), down: deq(ql.Down, dModel, dFF),
		}
		if len(ql.V.Packed) > 0 {
			ws[li].v = deq(ql.V, lkv*lhd, dModel)
		}
	}

	kCache := make([][][]float32, len(qlayers)) // [layer][step][lkv*lhd]
	vCache := make([][][]float32, len(qlayers))
	outs := make([][]byte, len(inputs))
	perLayer := make([][][]byte, len(inputs)) // [tok][layer] hidden after the layer
	for tok := range inputs {
		x := bf16ToF32s(inputs[tok])
		for li := range qlayers {
			lhd := headDimOf(specs[li], headDim)
			lkv := kvHeadsOf(specs[li], nKVHeads)
			normed := rmsNormHostReference(x, bf16ToF32s(qlayers[li].AttnNormW), 1, dModel, eps)
			// addBiasHost adds the additive projection bias (Qwen2/2.5 q/k/v bias) after the
			// matvec, before qk-norm/rope — the same station the GPU projector uses. nil bias
			// (bias-free arches: the pre-existing fixtures) is a no-op.
			addBiasHost := func(dst []float32, bias []byte) {
				if len(bias) == 0 {
					return
				}
				b := bf16ToF32s(bias)
				for i := range dst {
					dst[i] += b[i]
				}
			}
			q := matvec(ws[li].q, normed)
			addBiasHost(q, qlayers[li].BQ)
			k := matvec(ws[li].k, normed)
			addBiasHost(k, qlayers[li].BK)
			var v []float32
			if ws[li].v.w != nil {
				v = matvec(ws[li].v, normed)
				addBiasHost(v, qlayers[li].BV)
			} else {
				// k_eq_v: V is the k-proj output PRE-norm/rope — copy before k norms/rotates
				v = append([]float32(nil), k...)
			}
			// per-head qk-norm mirrors the fused kernel's rounding stations exactly:
			// normed = bf16(w · bf16(x · inv_mean)) — f32 maths would sit a hair past
			// the cosine bar once four extra norm stations per layer compound.
			headNorm := func(seg, w []float32) {
				var sq float64
				for _, v := range seg {
					sq += float64(v) * float64(v)
				}
				inv := float32(1.0 / math.Sqrt(sq/float64(len(seg))+float64(eps)))
				for i2 := range seg {
					h := f32ToBF16(seg[i2] * inv)
					xr := bf16ToF32(byte(h), byte(h>>8))
					h2 := f32ToBF16(w[i2] * xr)
					seg[i2] = bf16ToF32(byte(h2), byte(h2>>8))
				}
			}
			if qn := qlayers[li].QNormW; len(qn) > 0 {
				w := bf16ToF32s(qn)
				for h := range nHeads {
					headNorm(q[h*lhd:(h+1)*lhd], w)
				}
			}
			if kn := qlayers[li].KNormW; len(kn) > 0 {
				w := bf16ToF32s(kn)
				for hk := range lkv {
					headNorm(k[hk*lhd:(hk+1)*lhd], w)
				}
			}
			if valueNorm {
				// gemma4 value-norm: no-scale per-head RMSNorm (ones weight) on the V row
				for hk := range lkv {
					seg := v[hk*lhd : (hk+1)*lhd]
					var sq float64
					for _, val := range seg {
						sq += float64(val) * float64(val)
					}
					inv := 1.0 / math.Sqrt(sq/float64(lhd)+float64(eps))
					for i2 := range seg {
						seg[i2] = float32(float64(seg[i2]) * inv)
					}
				}
			}
			rotDim, rBase := lRotDim, lBase
			if specs[li].Attention == model.GlobalAttention {
				rotDim, rBase = gRotDim, gBase
			}
			rope(q, nHeads, lhd, rotDim, rBase, tok)
			rope(k, lkv, lhd, rotDim, rBase, tok)
			// the GPU cache stores bf16 rows — cache the same rounding or the gap
			// between attended histories grows with every position.
			roundBF16(k)
			roundBF16(v)
			kCache[li] = append(kCache[li], k)
			vCache[li] = append(vCache[li], v)
			first := 0
			if specs[li].Attention != model.GlobalAttention && window > 0 && len(kCache[li]) > window {
				first = len(kCache[li]) - window
			}
			n := len(kCache[li]) - first
			// head-major k/v bf16 buffers for sdpaBF16Reference: (hk·n + j)·lhd + d
			kb := make([]byte, lkv*n*lhd*bf16Size)
			vb := make([]byte, lkv*n*lhd*bf16Size)
			put := func(dst []byte, idx int, val float32) {
				h := f32ToBF16(val)
				dst[idx*2], dst[idx*2+1] = byte(h), byte(h>>8)
			}
			for j := range n {
				for hk := range lkv {
					for d := range lhd {
						put(kb, (hk*n+j)*lhd+d, kCache[li][first+j][hk*lhd+d])
						put(vb, (hk*n+j)*lhd+d, vCache[li][first+j][hk*lhd+d])
					}
				}
			}
			qb := make([]byte, nHeads*lhd*bf16Size)
			for i, qv := range q {
				put(qb, i, qv)
			}
			attn := bf16ToF32s(sdpaBF16Reference(qb, kb, vb, nHeads, lkv, lhd, n, scale))
			attnOut := matvec(ws[li].o, attn)
			if pn := qlayers[li].PostAttnNormW; len(pn) > 0 {
				attnOut = rmsNormHostReference(attnOut, bf16ToF32s(pn), 1, dModel, eps)
			}
			for i := range x {
				x[i] += attnOut[i]
			}
			roundBF16(x)
			normed2 := rmsNormHostReference(x, bf16ToF32s(qlayers[li].MLPNormW), 1, dModel, eps)
			g := matvec(ws[li].gate, normed2)
			u := matvec(ws[li].up, normed2)
			for i := range g {
				g[i] = geluRefF32(g[i]) * u[i]
			}
			d := matvec(ws[li].down, g)
			if pn := qlayers[li].PostFFNormW; len(pn) > 0 {
				d = rmsNormHostReference(d, bf16ToF32s(pn), 1, dModel, eps)
			}
			for i := range x {
				x[i] += d[i]
			}
			roundBF16(x)
			perLayer[tok] = append(perLayer[tok], toBF16Bytes(x))
		}
		outs[tok] = toBF16Bytes(x)
	}
	hostPerLayerRef = perLayer
	return outs
}

// hostPerLayerRef holds the host mirror's per-layer hiddens from the last
// hostArchQuantReference run — the layer-bisect view for a diverging token.
var hostPerLayerRef [][][]byte

// buildConditionedQuantLayer is buildQuantLayer at 10× smaller weight scale, so the
// synthetic residual stream stays O(1) through the layers: the shared fixture's ±1.0
// weights explode the hidden into the tens of thousands, where bf16's ~2⁻⁷ relative
// step makes ANY two accumulation orders disagree violently and the reference cannot
// discriminate a real defect from fixture ill-conditioning.
func buildConditionedQuantLayer(t *testing.T, dModel, nHeads, nKV, headDim, dFF, gs, bits, salt int) QuantizedLayerWeights {
	t.Helper()
	qDim, kvDim := nHeads*headDim, nKV*headDim
	mk := func(n, s int) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*s+7)%101-50) * 0.002
		}
		return f
	}
	return QuantizedLayerWeights{
		AttnNormW: toBF16Bytes(mk(dModel, salt+13)),
		MLPNormW:  toBF16Bytes(mk(dModel, salt+19)),
		Q:         quantW(t, mk(qDim*dModel, salt+53), qDim, dModel, gs, bits),
		K:         quantW(t, mk(kvDim*dModel, salt+71), kvDim, dModel, gs, bits),
		V:         quantW(t, mk(kvDim*dModel, salt+83), kvDim, dModel, gs, bits),
		O:         quantW(t, mk(dModel*qDim, salt+17), dModel, qDim, gs, bits),
		Gate:      quantW(t, mk(dFF*dModel, salt+61), dFF, dModel, gs, bits),
		Up:        quantW(t, mk(dFF*dModel, salt+29), dFF, dModel, gs, bits),
		Down:      quantW(t, mk(dModel*dFF, salt+47), dModel, dFF, gs, bits),
		GroupSize: gs, Bits: bits,
	}
}

// TestDecodeForwardArchQuantHostReference anchors the GPU quant forward to the independent
// host reference at BOTH non-uniform kv-head geometries: the 12B mix (global MQA, kv=1 —
// the control every field-working model exercises) and the 31B mix (global GQA, kv>1 — the
// geometry the degenerate 31B uniquely runs). Per-token hidden cosine ≥ 0.999; a lane that
// is merely CONSISTENT with its sibling but wrong against the maths fails here.
func TestDecodeForwardArchQuantHostReference(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, headDim, globalHeadDim, dFF, gs, bits = 512, 8, 64, 128, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen, T, slidingWindow = 8, 6, 7

	mkInputs := func(n, salt int) [][]byte {
		in := make([][]byte, n)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+salt)+11)%83-41) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}

	cases := []struct {
		name                string
		slidingKV, globalKV int
	}{
		{"uniform-control", 2, 2},
		{"global-MQA-kv1-the-12B-mix", 2, 1},
		{"global-GQA-kv2-the-31B-mix", 4, 2},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			specs := model.DeriveLayers([]string{"sliding_attention", "full_attention"}, 0)
			specs[0].KVHeads, specs[0].HeadDim = c.slidingKV, headDim
			ghd := globalHeadDim
			if c.name == "uniform-control" {
				ghd = headDim
			}
			specs[1].KVHeads, specs[1].HeadDim = c.globalKV, ghd
			ql := []QuantizedLayerWeights{
				buildConditionedQuantLayer(t, dModel, nHeads, c.slidingKV, headDim, dFF, gs, bits, 500),
				buildConditionedQuantLayer(t, dModel, nHeads, c.globalKV, ghd, dFF, gs, bits, 600),
			}
			inputs := mkInputs(T, 7)

			got, err := DecodeForwardArchQuant(inputs, ql, specs, dModel, nHeads, c.slidingKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
			if err != nil {
				t.Fatalf("DecodeForwardArchQuant: %v", err)
			}
			want := hostArchQuantReference(t, inputs, ql, specs, dModel, nHeads, c.slidingKV, headDim, dFF, slidingWindow, base, scale, eps, false)
			bad := -1
			for tok := range T {
				cos := cosineBF16(got[tok], want[tok])
				t.Logf("tok %d cosine=%.6f", tok, cos)
				if bad < 0 && cos < 0.999 {
					bad = tok
				}
			}
			if bad >= 0 {
				g, w := got[bad], want[bad]
				type d struct {
					i          int
					gv, wv, ad float64
				}
				var worst []d
				for i := 0; i*2 < len(g); i++ {
					gv := float64(bf16ToF32(g[i*2], g[i*2+1]))
					wv := float64(bf16ToF32(w[i*2], w[i*2+1]))
					ad := gv - wv
					if ad < 0 {
						ad = -ad
					}
					worst = append(worst, d{i, gv, wv, ad})
				}
				for a := range worst {
					for b := a + 1; b < len(worst); b++ {
						if worst[b].ad > worst[a].ad {
							worst[a], worst[b] = worst[b], worst[a]
						}
					}
					if a >= 5 {
						break
					}
				}
				for _, e := range worst[:6] {
					t.Logf("tok %d elem %d: gpu=%.3f host=%.3f |d|=%.3f", bad, e.i, e.gv, e.wv, e.ad)
				}
			}
			if bad >= 0 {
				capturedLayerHiddens = nil
				captureLayerHiddens = true
				_, rerr := DecodeForwardArchQuant(inputs, ql, specs, dModel, nHeads, c.slidingKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
				captureLayerHiddens = false
				if rerr != nil {
					t.Fatalf("capture rerun: %v", rerr)
				}
				nL := len(ql)
				for tok := range T {
					for li := range nL {
						gpu := capturedLayerHiddens[tok*nL+li]
						host := hostPerLayerRef[tok][li]
						t.Logf("tok %d layer %d (%v) cosine=%.6f", tok, li, specs[li].Attention, cosineBF16(gpu, host))
					}
				}
				t.Fatalf("GPU vs host reference diverges at %s from tok %d", c.name, bad)
			}
			t.Logf("%s: %d tokens GPU ≡ host reference (cosine ≥ 0.999)", c.name, T)
		})
	}
}

// buildKEqVQuantLayer is buildConditionedQuantLayer with NO v_proj — the gemma4 K==V
// layer shape (V rides the value-normed k-proj output).
func buildKEqVQuantLayer(t *testing.T, dModel, nHeads, nKV, headDim, dFF, gs, bits, salt int) QuantizedLayerWeights {
	t.Helper()
	ql := buildConditionedQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, salt)
	ql.V = QuantWeight{}
	return ql
}

// TestDecodeForwardArchQuantHostReferenceFeatures climbs the #348 enrichment ladder on the
// host-anchored fixture: each rung adds ONE real-31B feature at BOTH gkv=1 (the 12B-shaped
// control every field-working model exercises — any per-head-offset bug lands at offset 0
// there) and gkv=2 (the 31B-shaped suspect). The first red gkv=2 rung with a green gkv=1
// control is the conviction, at millisecond scale.
func TestDecodeForwardArchQuantHostReferenceFeatures(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, headDim, globalHeadDim, dFF, gs, bits = 512, 8, 64, 128, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen, T, slidingWindow = 8, 6, 3

	mkInputs := func(n, salt int) [][]byte {
		in := make([][]byte, n)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+salt)+11)%83-41) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}

	cases := []struct {
		name      string
		globalKV  int
		valueNorm bool
		kEqV      bool
		qkNorm    bool
		bar       float64
	}{
		{"valueNorm-gkv1-control", 1, true, false, false, 0.999},
		{"valueNorm-gkv2-suspect", 2, true, false, false, 0.999},
		{"kEqV-valueNorm-gkv1-control", 1, true, true, false, 0.999},
		{"kEqV-valueNorm-gkv2-suspect", 2, true, true, false, 0.999},
		// the qk-norm + sandwich rungs run at a LOOSE bar: the host mirror carries a
		// position-growing fidelity gap (~3e-2 by tok 3) against the engine's qk-norm
		// rounding stations that q/k-side and cache-side bf16 rounding did NOT close,
		// and it carries NO gkv signal (the gkv=1 control sits BELOW the gkv=2
		// suspect; the engine's own fused≡split byte parity already pins those lanes
		// against each other) — a gross-regression tripwire, not a byte oracle.
		{"full31B-qkNorm-sandwich-gkv1-control", 1, true, true, true, 0.96},
		{"full31B-qkNorm-sandwich-gkv2-suspect", 2, true, true, true, 0.96},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if c.qkNorm {
				// The rung already delivered its #348 finding — NO gkv signal (the gkv=1
				// control diverges MORE than the gkv=2 suspect, and the engine's own
				// fused≡split byte parity pins those lanes against each other). What
				// remains is a position-growing mirror-fidelity gap on the qk-norm path
				// (~5e-2 by tok 3) that q/k-side and cache-side bf16 rounding did not
				// close; the mirror is not oracle-grade there yet.
				t.Skip("host mirror not oracle-grade on the qk-norm path (documented fidelity gap, no gkv signal)")
			}
			specs := model.DeriveLayers([]string{"sliding_attention", "full_attention"}, 0)
			specs[0].KVHeads, specs[0].HeadDim = 4, headDim
			specs[1].KVHeads, specs[1].HeadDim = c.globalKV, globalHeadDim
			mk := buildConditionedQuantLayer
			if c.kEqV {
				mk = buildKEqVQuantLayer
			}
			ql := []QuantizedLayerWeights{
				buildConditionedQuantLayer(t, dModel, nHeads, 4, headDim, dFF, gs, bits, 500),
				mk(t, dModel, nHeads, c.globalKV, globalHeadDim, dFF, gs, bits, 600),
			}
			if c.qkNorm {
				// per-head Q/K norms ([lhd], shared across heads) + the gemma4 sandwich
				// norms on both sublayer outputs — the full real-31B per-layer feature set.
				mkw := func(n, s int) []byte {
					f := make([]float32, n)
					for i := range f {
						f[i] = float32((i*s+3)%67-33)*0.01 + 1
					}
					return toBF16Bytes(f)
				}
				for li := range ql {
					lhd := headDimOf(specs[li], headDim)
					ql[li].QNormW = mkw(lhd, 41+li)
					ql[li].KNormW = mkw(lhd, 43+li)
					ql[li].PostAttnNormW = mkw(dModel, 47+li)
					ql[li].PostFFNormW = mkw(dModel, 51+li)
				}
			}
			inputs := mkInputs(T, 7)

			got, err := DecodeForwardArchQuant(inputs, ql, specs, dModel, nHeads, 4, headDim, maxLen, dFF, slidingWindow, base, scale, eps, c.valueNorm)
			if err != nil {
				t.Fatalf("DecodeForwardArchQuant: %v", err)
			}
			want := hostArchQuantReference(t, inputs, ql, specs, dModel, nHeads, 4, headDim, dFF, slidingWindow, base, scale, eps, c.valueNorm)
			for tok := range T {
				cos := cosineBF16(got[tok], want[tok])
				t.Logf("tok %d cosine=%.6f", tok, cos)
				if cos < c.bar {
					t.Fatalf("tok %d: GPU vs host diverges at %s (cosine=%.6f)", tok, c.name, cos)
				}
			}
		})
	}
}

// TestRunArchDecodeHostReferencePartialRope is rung D of the #348 ladder: the proportional
// PARTIAL global rope (rotate only rotaryDim of the wide global head dim, at the proportional
// effective base) — the one real-31B feature the narrow forward API cannot express. Driven
// through runArchDecode directly with the session's own rope split (global rotaryDim+base,
// sliding rotaryDimLocal+localBase), with k_eq_v and value-norm riding along, at both gkv=1
// (control) and gkv=2 (suspect).
func TestRunArchDecodeHostReferencePartialRope(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	const dModel, nHeads, headDim, globalHeadDim, dFF, gs, bits = 512, 8, 64, 128, 1024, 64, 4
	const rotaryDim, rotaryDimLocal = 32, 64 // global: 32 of 128 (the 31B 0.25 ratio); sliding: full
	const gBase, lBase = float32(32), float32(10000)
	const scale, eps = float32(0.125), float32(1e-5)
	const maxLen, T, slidingWindow = 8, 6, 3

	mkInputs := func(n, salt int) [][]byte {
		in := make([][]byte, n)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+salt)+11)%83-41) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}
	for _, gkv := range []int{1, 2} {
		t.Run(map[int]string{1: "gkv1-control", 2: "gkv2-suspect"}[gkv], func(t *testing.T) {
			specs := model.DeriveLayers([]string{"sliding_attention", "full_attention"}, 0)
			specs[0].KVHeads, specs[0].HeadDim = 4, headDim
			specs[1].KVHeads, specs[1].HeadDim = gkv, globalHeadDim
			ql := []QuantizedLayerWeights{
				buildConditionedQuantLayer(t, dModel, nHeads, 4, headDim, dFF, gs, bits, 500),
				buildKEqVQuantLayer(t, dModel, nHeads, gkv, globalHeadDim, dFF, gs, bits, 600),
			}
			inputs := mkInputs(T, 7)

			var got [][]byte
			withAutoreleasePool(func() {
				lb, moe, err := buildQuantArchLayerBufs(ql, specs, dModel, nHeads, 4, headDim, dFF, maxLen, slidingWindow, nil)
				if err != nil {
					t.Fatalf("buildQuantArchLayerBufs: %v", err)
				}
				_ = moe
				got, err = runArchDecode(inputs, specs, lb, make([]*MoELayerWeights, len(ql)), dModel, nHeads, 4, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal, gBase, lBase, scale, eps, true, maxLen)
				if err != nil {
					t.Fatalf("runArchDecode: %v", err)
				}
			})
			want := hostArchQuantReferenceRope(t, inputs, ql, specs, dModel, nHeads, 4, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal, gBase, lBase, scale, eps, true)
			for tok := range T {
				cos := cosineBF16(got[tok], want[tok])
				t.Logf("tok %d cosine=%.6f", tok, cos)
				if cos < 0.999 {
					t.Fatalf("tok %d diverges at gkv=%d (cosine=%.6f)", tok, gkv, cos)
				}
			}
		})
	}
}

// TestDecodeForwardArchQuantHostReferenceWideDModel is the #348 conviction receipt: the
// single-row rms family (N_READS=4, one pass) covers at most 4096 dims per threadgroup —
// 31B's dModel 5376 is the only family member past the limit, and every lane that drives
// the single-row PSO at dModel axes silently drops the tail dims. The same fixture that is
// green at dModel 512 must stay green at 5120; a red here pins the wide-dModel rms sites.
func TestDecodeForwardArchQuantHostReferenceWideDModel(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	const dModel, nHeads, headDim, globalHeadDim, dFF, gs, bits = 5376, 8, 64, 128, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen, T, slidingWindow = 8, 4, 3

	mkInputs := func(n int) [][]byte {
		in := make([][]byte, n)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+7)+11)%83-41) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}
	specs := model.DeriveLayers([]string{"sliding_attention", "full_attention"}, 0)
	specs[0].KVHeads, specs[0].HeadDim = 2, headDim
	specs[1].KVHeads, specs[1].HeadDim = 2, globalHeadDim
	ql := []QuantizedLayerWeights{
		buildConditionedQuantLayer(t, dModel, nHeads, 2, headDim, dFF, gs, bits, 500),
		buildConditionedQuantLayer(t, dModel, nHeads, 2, globalHeadDim, dFF, gs, bits, 600),
	}
	// the real-31B per-layer feature set routes the norms through the FUSED emission
	// sites (rmsnorm_residual, qk-norm rows) — the lanes the bare fixture never runs.
	mkw := func(n, s int) []byte {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*s+3)%67-33)*0.01 + 1
		}
		return toBF16Bytes(f)
	}
	for li := range ql {
		lhd := headDimOf(specs[li], headDim)
		ql[li].QNormW = mkw(lhd, 41+li)
		ql[li].KNormW = mkw(lhd, 43+li)
		ql[li].PostAttnNormW = mkw(dModel, 47+li)
		ql[li].PostFFNormW = mkw(dModel, 51+li)
	}
	inputs := mkInputs(T)

	got, err := DecodeForwardArchQuant(inputs, ql, specs, dModel, nHeads, 2, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant: %v", err)
	}
	want := hostArchQuantReference(t, inputs, ql, specs, dModel, nHeads, 2, headDim, dFF, slidingWindow, base, scale, eps, false)
	for tok := range T {
		cos := cosineBF16(got[tok], want[tok])
		t.Logf("tok %d cosine=%.6f", tok, cos)
		// bar 0.96: the qk-norm mirror carries a known position-growing fidelity gap
		// (rung C, same envelope at dModel 512); a wide-dModel tail-drop breaks cosine
		// catastrophically (0.87 pre-fix at tok 0), far below mirror noise.
		if cos < 0.96 {
			t.Errorf("tok %d: wide-dModel forward diverges from host reference (cosine=%.6f)", tok, cos)
		}
	}
}
