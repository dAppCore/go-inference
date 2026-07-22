// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

// attn_core_backend_test.go gates the device attention core against a test-local
// reference implementing composed's continueFromQKV math exactly (attention.go:398-520: the
// per-head [q;gate] de-interleave, RMS-with-weight or no-op QK-norm, applyRotaryHalf partial
// rotary at pos0+t, causal max-subtract softmax with the sliding-window first-clamp, the weighted
// V sum, and the σ-gate multiply). f64 accumulation host-side; the kernels are f32 — scaled
// tolerance.

func attnCoreHostRef(qRaw, k, v, qNormW, kNormW, cacheK, cacheV, out []float32, L, H, KVH, HD, RD, pos0, window, gated, qkNorm int, eps, theta float32) {
	rep := H / KVH
	scale := 1.0 / math.Sqrt(float64(HD))
	rope := func(row []float32, pos int) {
		half := RD / 2
		for i := 0; i < half; i++ {
			freq := 1.0 / math.Pow(float64(theta), float64(2*i)/float64(RD))
			ang := float64(pos) * freq
			c, s := math.Cos(ang), math.Sin(ang)
			a, b := float64(row[i]), float64(row[i+half])
			row[i] = float32(a*c - b*s)
			row[i+half] = float32(b*c + a*s)
		}
	}
	norm := func(row, w []float32) {
		if qkNorm == 0 {
			return
		}
		var ss float64
		for _, x := range row {
			ss += float64(x) * float64(x)
		}
		if qkNorm == 1 {
			inv := 1.0 / math.Sqrt(ss/float64(HD)+float64(eps))
			for i := range row {
				row[i] = float32(float64(row[i]) * inv * float64(w[i]))
			}
		} else {
			inv := 1.0 / math.Sqrt(ss+float64(eps))
			for i := range row {
				row[i] = float32(float64(row[i]) * inv)
			}
		}
	}
	qCols := H * HD
	if gated != 0 {
		qCols = 2 * H * HD
	}
	q := make([]float32, L*H*HD)
	gate := make([]float32, L*H*HD)
	for t := 0; t < L; t++ {
		for hd := 0; hd < H; hd++ {
			if gated != 0 {
				src := qRaw[t*qCols+hd*2*HD:]
				copy(q[(t*H+hd)*HD:(t*H+hd+1)*HD], src[:HD])
				copy(gate[(t*H+hd)*HD:(t*H+hd+1)*HD], src[HD:2*HD])
			} else {
				copy(q[(t*H+hd)*HD:(t*H+hd+1)*HD], qRaw[t*qCols+hd*HD:t*qCols+(hd+1)*HD])
			}
			row := q[(t*H+hd)*HD : (t*H+hd+1)*HD]
			norm(row, qNormW)
			rope(row, pos0+t)
		}
		for hd := 0; hd < KVH; hd++ {
			row := make([]float32, HD)
			copy(row, k[(t*KVH+hd)*HD:(t*KVH+hd+1)*HD])
			norm(row, kNormW)
			rope(row, pos0+t)
			copy(cacheK[((pos0+t)*KVH+hd)*HD:((pos0+t)*KVH+hd+1)*HD], row)
			copy(cacheV[((pos0+t)*KVH+hd)*HD:((pos0+t)*KVH+hd+1)*HD], v[(t*KVH+hd)*HD:(t*KVH+hd+1)*HD])
		}
	}
	scores := make([]float64, pos0+L)
	for t := 0; t < L; t++ {
		last := pos0 + t
		first := 0
		if window > 0 && last+1 > window {
			first = last + 1 - window
		}
		for hd := 0; hd < H; hd++ {
			kvh := hd / rep
			qrow := q[(t*H+hd)*HD:]
			maxS := math.Inf(-1)
			for j := first; j <= last; j++ {
				var dot float64
				for d := 0; d < HD; d++ {
					dot += float64(qrow[d]) * float64(cacheK[(j*KVH+kvh)*HD+d])
				}
				dot *= scale
				scores[j] = dot
				if dot > maxS {
					maxS = dot
				}
			}
			var sum float64
			for j := first; j <= last; j++ {
				scores[j] = math.Exp(scores[j] - maxS)
				sum += scores[j]
			}
			orow := out[(t*H+hd)*HD:]
			for d := 0; d < HD; d++ {
				var acc float64
				for j := first; j <= last; j++ {
					acc += scores[j] * float64(cacheV[(j*KVH+kvh)*HD+d])
				}
				val := acc / sum
				if gated != 0 { // sigmoid, NOT silu — the transformers qwen3_5 hardcoded convention
					g := float64(gate[(t*H+hd)*HD+d])
					val *= 1 / (1 + math.Exp(-g))
				}
				orow[d] = float32(val)
			}
		}
	}
}

// TestAttnCoreDevice_Good gates the device core against the host reference at the real Qwen3.5-2B
// attention shape (H=8 KVH=2 HD=256 RD=64 gated, no QK-norm) and the fixture shape (H=4 KVH=2
// HD=128 RD=128 ungated, RMS QK-norm), both over a carried cache (pos0=5) and a windowed variant.
func TestAttnCoreDevice_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — attention core")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable: %v", err)
	}
	for _, tc := range []struct {
		name                                   string
		L, H, KVH, HD, RD, pos0, window, gated int
		qkNorm                                 int
	}{
		{"real2B-decode", 1, 8, 2, 256, 64, 5, 0, 1, 0},
		{"real2B-chunk", 4, 8, 2, 256, 64, 3, 0, 1, 0},
		{"fixture-qknorm", 2, 4, 2, 128, 128, 5, 0, 0, 1},
		{"windowed", 3, 4, 2, 128, 64, 6, 4, 1, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !attnCoreUsable(tc.H, tc.KVH, tc.HD, tc.RD) {
				t.Skip("attention core kernels unavailable")
			}
			rng := rand.New(rand.NewSource(9))
			fill := func(n int) []float32 {
				o := make([]float32, n)
				for i := range o {
					o[i] = -1 + 2*rng.Float32()
				}
				return o
			}
			qCols := tc.H * tc.HD
			if tc.gated != 0 {
				qCols = 2 * tc.H * tc.HD
			}
			capRows := tc.pos0 + tc.L + 2
			qRaw := fill(tc.L * qCols)
			k := fill(tc.L * tc.KVH * tc.HD)
			v := fill(tc.L * tc.KVH * tc.HD)
			var qw, kw []float32
			if tc.qkNorm == 1 {
				qw = fill(tc.HD)
				kw = fill(tc.HD)
			}
			ckDev := fill(capRows * tc.KVH * tc.HD) // rows < pos0 = live history; rows >= pos0 overwritten
			cvDev := fill(capRows * tc.KVH * tc.HD)
			ckHost := append([]float32(nil), ckDev...)
			cvHost := append([]float32(nil), cvDev...)
			outDev := make([]float32, tc.L*tc.H*tc.HD)
			outHost := make([]float32, tc.L*tc.H*tc.HD)

			attnCoreHostRef(qRaw, k, v, qw, kw, ckHost, cvHost, outHost, tc.L, tc.H, tc.KVH, tc.HD, tc.RD, tc.pos0, tc.window, tc.gated, tc.qkNorm, 1e-6, 1e6)
			if err := AttnCoreDeviceRun(qRaw, k, v, qw, kw, ckDev, cvDev, outDev, tc.L, tc.H, tc.KVH, tc.HD, tc.RD, tc.pos0, tc.window, tc.gated, tc.qkNorm, 1e-6, 1e6); err != nil {
				t.Fatalf("AttnCoreDeviceRun: %v", err)
			}
			outRel := gdScaledDiff(t, "out", outDev, outHost)
			ckRel := gdScaledDiff(t, "cacheK", ckDev, ckHost)
			cvRel := gdScaledDiff(t, "cacheV", cvDev, cvHost)
			t.Logf("scaled max diff: out=%.3e cacheK=%.3e cacheV=%.3e", outRel, ckRel, cvRel)
			if outRel > 5e-4 || ckRel > 5e-4 || cvRel > 5e-4 {
				t.Fatalf("drift beyond f32 rounding: out=%.3e k=%.3e v=%.3e", outRel, ckRel, cvRel)
			}
		})
	}
}

// TestAttnCoreDevice_Bad pins the rejections: unservable geometry and size mismatches.
func TestAttnCoreDevice_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — attention core")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable: %v", err)
	}
	if attnCoreUsable(4, 2, 512, 64) {
		t.Fatal("HD>256 must not be usable")
	}
	if attnCoreUsable(3, 2, 128, 64) {
		t.Fatal("H%KVH!=0 must not be usable")
	}
	out := make([]float32, 4*128)
	ck := make([]float32, 8*2*128)
	if err := AttnCoreDeviceRun(make([]float32, 3), nil, nil, nil, nil, ck, ck, out, 1, 4, 2, 128, 64, 0, 0, 0, 0, 1e-6, 1e6); err == nil {
		t.Fatal("qRaw size mismatch must error")
	}
}
