// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
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
	base, scale, eps float32) [][]byte {
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
	// rotation via the engine's own host-callable RoPE (its convention is pinned by
	// rope_test's invariants) — the mirror's independence claim covers the LANE
	// composition (projection/cache/attention/residual order), not the rotation maths.
	rope := func(x []float32, heads, hd, pos int) {
		out, err := RoPE(x, 1, heads, hd, base, 1, pos, false)
		if err != nil {
			t.Fatalf("host RoPE: %v", err)
		}
		copy(x, out)
	}

	type lw struct{ q, k, v, o, gate, up, down mat }
	ws := make([]lw, len(qlayers))
	for li, ql := range qlayers {
		lhd := headDimOf(specs[li], headDim)
		lkv := kvHeadsOf(specs[li], nKVHeads)
		ws[li] = lw{
			q: deq(ql.Q, nHeads*lhd, dModel), k: deq(ql.K, lkv*lhd, dModel), v: deq(ql.V, lkv*lhd, dModel),
			o:    deq(ql.O, dModel, nHeads*lhd),
			gate: deq(ql.Gate, dFF, dModel), up: deq(ql.Up, dFF, dModel), down: deq(ql.Down, dModel, dFF),
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
			q := matvec(ws[li].q, normed)
			k := matvec(ws[li].k, normed)
			v := matvec(ws[li].v, normed)
			rope(q, nHeads, lhd, tok)
			rope(k, lkv, lhd, tok)
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
			want := hostArchQuantReference(t, inputs, ql, specs, dModel, nHeads, c.slidingKV, headDim, dFF, slidingWindow, base, scale, eps)
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
