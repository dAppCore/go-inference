// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestRealModelOpDiff is #348's per-op conviction instrument: GEMMA4_SNAP names the real
// checkpoint, GEMMA4_OPS a directory of mlx-side op dumps (mlx_op_dump.py — each op's
// last-position row as raw f32, plus the full post-rope K/V caches). Every op runs HERE
// on the SAME weights with MLX'S OWN INPUT tensor, so divergence cannot compound and the
// first disagreeing op is the defect (or the convention difference to reconcile).
func TestRealModelOpDiff(t *testing.T) {
	snap, ops := os.Getenv("GEMMA4_SNAP"), os.Getenv("GEMMA4_OPS")
	if snap == "" || ops == "" {
		t.Skip("GEMMA4_SNAP / GEMMA4_OPS not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("metal init: %v", err)
	}
	m, dm, err := model.Load(snap)
	if err != nil {
		t.Fatalf("model.Load: %v", err)
	}
	defer func() { _ = dm.Close() }()

	readF32 := func(name string) []float32 {
		r := core.ReadFile(ops + "/" + name + ".bin")
		if !r.OK {
			t.Fatalf("read %s: %v", name, r.Value)
		}
		b := r.Value.([]byte)
		f := make([]float32, len(b)/4)
		for i := range f {
			bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
			f[i] = math.Float32frombits(bits)
		}
		return f
	}
	deq := func(l *model.Linear) []float32 {
		if l.Quantised() {
			f, derr := dequantizeAffineRowsF32(l.Weight, l.Scales, l.Biases, l.OutDim, l.InDim, l.GroupSize, l.Bits)
			if derr != nil {
				t.Fatalf("dequant: %v", derr)
			}
			return f
		}
		f := make([]float32, l.OutDim*l.InDim)
		for i := range f {
			f[i] = bf16ToF32(l.Weight[i*2], l.Weight[i*2+1])
		}
		return f
	}
	matvec := func(w []float32, rows, cols int, x []float32) []float32 {
		out := make([]float32, rows)
		for r := range rows {
			var acc float64
			row := w[r*cols : (r+1)*cols]
			for c, v := range x {
				acc += float64(row[c]) * float64(v)
			}
			out[r] = float32(acc)
		}
		return out
	}
	bf16w := func(b []byte) []float32 {
		f := make([]float32, len(b)/2)
		for i := range f {
			f[i] = bf16ToF32(b[i*2], b[i*2+1])
		}
		return f
	}
	rms := func(x, w []float32, eps float32) []float32 {
		var sq float64
		for _, v := range x {
			sq += float64(v) * float64(v)
		}
		inv := 1.0 / math.Sqrt(sq/float64(len(x))+float64(eps))
		out := make([]float32, len(x))
		for i := range x {
			out[i] = float32(float64(x[i]) * inv * float64(w[i]))
		}
		return out
	}
	headRMS := func(x []float32, heads, hd int, w []float32, eps float32) []float32 {
		out := make([]float32, len(x))
		for h := range heads {
			seg := x[h*hd : (h+1)*hd]
			var sq float64
			for _, v := range seg {
				sq += float64(v) * float64(v)
			}
			inv := 1.0 / math.Sqrt(sq/float64(hd)+float64(eps))
			for i := range seg {
				wv := float64(1)
				if w != nil {
					wv = float64(w[i])
				}
				out[h*hd+i] = float32(float64(seg[i]) * inv * wv)
			}
		}
		return out
	}
	// mlx's ProportionalRoPE law (rope_utils.py): pairs (i, i + hd/2) over the FULL head
	// dim, only the first rotDim/2 pairs live (inf period beyond), frequencies normalised
	// by the FULL dim at the ORIGINAL base: θ_i = pos · base^(-2i/hd).
	rope := func(x []float32, heads, hd, rotDim int, base float32, pos int) []float32 {
		out := append([]float32(nil), x...)
		half := hd / 2
		live := rotDim / 2
		for h := range heads {
			for i := range live {
				theta := float64(pos) * math.Pow(float64(base), -2*float64(i)/float64(hd))
				c, s := math.Cos(theta), math.Sin(theta)
				a, b := float64(out[h*hd+i]), float64(out[h*hd+i+half])
				out[h*hd+i] = float32(a*c - b*s)
				out[h*hd+i+half] = float32(a*s + b*c)
			}
		}
		return out
	}
	toBF := func(f []float32) []byte { return toBF16Bytes(f) }
	report := func(name string, got, want []float32) {
		var dot, ng, nw, amax float64
		for i := range got {
			g, w := float64(got[i]), float64(want[i])
			dot += g * w
			ng += g * g
			nw += w * w
			if d := math.Abs(g - w); d > amax {
				amax = d
			}
		}
		cos := dot / (math.Sqrt(ng)*math.Sqrt(nw) + 1e-30)
		flag := ""
		if cos < 0.999 {
			flag = "  <<< DIVERGES"
		}
		t.Logf("%-24s cos=%.6f maxdiff=%.4f l2(ours)=%.3f l2(mlx)=%.3f%s", name, cos, amax, math.Sqrt(ng), math.Sqrt(nw), flag)
	}

	const eps = 1e-6
	const T = 28
	nHeads := 32
	if v := os.Getenv("GEMMA4_HEADS"); v != "" {
		if r := core.Atoi(v); r.OK {
			nHeads = r.Value.(int)
		}
	}
	layers := []int{0, 5}
	globalLayer := 5
	if v := os.Getenv("GEMMA4_GLOBAL"); v != "" {
		if r := core.Atoi(v); r.OK {
			globalLayer = r.Value.(int)
			layers = []int{0, globalLayer}
		}
	}
	pos := T - 1
	for _, li := range layers {
		L := m.Layers[li]
		pre := core.Sprintf("L%02d.", li)
		hd := L.Q.OutDim / nHeads
		lkv := L.K.OutDim / hd
		rotDim, base := hd, float32(10000) // sliding: full rotary at the local base
		if li == globalLayer {
			rotDim, base = hd/4, 1000000 // global: partial 0.25, freqs normalised over FULL hd at the ORIGINAL theta
		}
		t.Logf("=== layer %d: hd=%d kv=%d rotDim=%d base=%.0f ===", li, hd, lkv, rotDim, base)

		hIn := readF32(pre + "h_in")
		// 1. input rms
		normedOurs := rms(hIn, bf16w(L.AttnNorm), eps)
		report(pre+"normed", normedOurs, readF32(pre+"normed"))
		normed := readF32(pre + "normed") // feed MLX's from here on

		// 2. projections off MLX's normed — host dequant maths AND the live qmv KERNEL
		report(pre+"q_proj", matvec(deq(L.Q), L.Q.OutDim, L.Q.InDim, normed), readF32(pre+"q_proj"))
		report(pre+"k_proj", matvec(deq(L.K), L.K.OutDim, L.K.InDim, normed), readF32(pre+"k_proj"))
		if L.Q.Quantised() {
			bfToF := func(b []byte) []float32 {
				f := make([]float32, len(b)/2)
				for i := range f {
					f[i] = bf16ToF32(b[i*2], b[i*2+1])
				}
				return f
			}
			normedBF := toBF(normed)
			qk, kerr := QMVBF16(normedBF, L.Q.Weight, L.Q.Scales, L.Q.Biases, L.Q.OutDim, L.Q.InDim, L.Q.GroupSize, L.Q.Bits)
			if kerr != nil {
				t.Fatalf("QMVBF16 q: %v", kerr)
			}
			report(pre+"q_proj(KERNEL)", bfToF(qk), readF32(pre+"q_proj"))
			kk, kerr2 := QMVBF16(normedBF, L.K.Weight, L.K.Scales, L.K.Biases, L.K.OutDim, L.K.InDim, L.K.GroupSize, L.K.Bits)
			if kerr2 != nil {
				t.Fatalf("QMVBF16 k: %v", kerr2)
			}
			report(pre+"k_proj(KERNEL)", bfToF(kk), readF32(pre+"k_proj"))
		}
		{
			rk, rerr := RMSNorm(hIn, bf16w(L.AttnNorm), 1, len(hIn), eps)
			if rerr != nil {
				t.Fatalf("RMSNorm kernel: %v", rerr)
			}
			report(pre+"normed(KERNEL)", rk, readF32(pre+"normed"))
		}

		// 3. q norm + rope off MLX's q_proj
		qMLX := readF32(pre + "q_proj")
		qnr := rope(headRMS(qMLX, nHeads, hd, bf16w(L.QNorm), eps), nHeads, hd, rotDim, base, pos)
		report(pre+"q_normed_roped", qnr, readF32(pre+"q_normed_roped"))

		// 4. k norm + rope off MLX's k_proj (vs the cache's LAST row)
		kMLX := readF32(pre + "k_proj")
		knr := rope(headRMS(kMLX, lkv, hd, bf16w(L.KNorm), eps), lkv, hd, rotDim, base, pos)
		kCache := readF32(pre + "k_cache_full") // [kvh, T, hd] head-major
		kLast := make([]float32, lkv*hd)
		for hk := range lkv {
			copy(kLast[hk*hd:(hk+1)*hd], kCache[(hk*T+pos)*hd:(hk*T+pos+1)*hd])
		}
		report(pre+"k_normed_roped", knr, kLast)

		// 5. v (k_eq_v on the global: value-normed pre-norm k-proj; else v_proj) vs cache last row
		var vOurs []float32
		if L.V != nil {
			vOurs = headRMS(matvec(deq(L.V), L.V.OutDim, L.V.InDim, normed), lkv, hd, nil, eps)
		} else {
			vOurs = headRMS(kMLX, lkv, hd, nil, eps)
		}
		vCache := readF32(pre + "v_cache_full")
		vLast := make([]float32, lkv*hd)
		for hk := range lkv {
			copy(vLast[hk*hd:(hk+1)*hd], vCache[(hk*T+pos)*hd:(hk*T+pos+1)*hd])
		}
		report(pre+"v_normed", vOurs, vLast)

		// 6. SDPA through the ENGINE's kernel on MLX's exact q + caches (scale 1.0)
		qb := toBF(readF32(pre + "q_normed_roped"))
		out, serr := SDPA(qb, toBF(kCache), toBF(vCache), 1, nHeads, lkv, hd, T, 1.0)
		if serr != nil {
			t.Fatalf("SDPA: %v", serr)
		}
		sdpaOurs := make([]float32, nHeads*hd)
		for i := range sdpaOurs {
			sdpaOurs[i] = bf16ToF32(out[i*2], out[i*2+1])
		}
		report(pre+"sdpa(ENGINE)", sdpaOurs, readF32(pre+"sdpa"))

		// 7. o proj + tail off MLX's sdpa — host AND kernel
		sdpaMLX := readF32(pre + "sdpa")
		report(pre+"o_proj", matvec(deq(L.O), L.O.OutDim, L.O.InDim, sdpaMLX), readF32(pre+"o_proj"))
		if L.O.Quantised() {
			bfToF := func(b []byte) []float32 {
				f := make([]float32, len(b)/2)
				for i := range f {
					f[i] = bf16ToF32(b[i*2], b[i*2+1])
				}
				return f
			}
			ok2, oerr := QMVBF16(toBF(sdpaMLX), L.O.Weight, L.O.Scales, L.O.Biases, L.O.OutDim, L.O.InDim, L.O.GroupSize, L.O.Bits)
			if oerr != nil {
				t.Fatalf("QMVBF16 o: %v", oerr)
			}
			report(pre+"o_proj(KERNEL)", bfToF(ok2), readF32(pre+"o_proj"))
		}
		oMLX := readF32(pre + "o_proj")
		pa := rms(oMLX, bf16w(L.PostAttnNorm), eps)
		report(pre+"post_attn_norm", pa, readF32(pre+"post_attn_norm"))
		h1 := make([]float32, len(hIn))
		paMLX := readF32(pre + "post_attn_norm")
		for i := range h1 {
			h1[i] = hIn[i] + paMLX[i]
		}
		report(pre+"resid_attn", h1, readF32(pre+"resid_attn"))

		// 8. MLP off MLX's resid
		h1MLX := readF32(pre + "resid_attn")
		pf := rms(h1MLX, bf16w(L.MLPNorm), eps)
		g := matvec(deq(L.Gate), L.Gate.OutDim, L.Gate.InDim, pf)
		u := matvec(deq(L.Up), L.Up.OutDim, L.Up.InDim, pf)
		for i := range g {
			g[i] = geluRefF32(g[i]) * u[i]
		}
		report(pre+"mlp_down", matvec(deq(L.Down), L.Down.OutDim, L.Down.InDim, g), readF32(pre+"mlp_down"))

		// 9. post-ff norm + residual + layer scalar
		dMLX := readF32(pre + "mlp_down")
		pff := rms(dMLX, bf16w(L.PostFFNorm), eps)
		report(pre+"post_ff_norm", pff, readF32(pre+"post_ff_norm"))
		scalar := float32(1)
		if len(L.LayerScalar) >= 2 {
			scalar = bf16ToF32(L.LayerScalar[0], L.LayerScalar[1])
		}
		hOut := make([]float32, len(h1MLX))
		pffMLX := readF32(pre + "post_ff_norm")
		for i := range hOut {
			hOut[i] = (h1MLX[i] + pffMLX[i]) * scalar
		}
		report(pre+"h_out_scaled", hOut, readF32(pre+"h_out_scaled"))
	}
}
