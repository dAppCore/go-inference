// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/arch/Qwen/qwen35"
)

// qwen_vision_encoder_test.go proves the ported qwen tower forward on synthetic weights: shape,
// bit-determinism, position-sensitivity (the 2-D rope + learned-table addition), and numerical parity
// against an INDEPENDENT naive f64 re-implementation of the reference equations written directly in
// this file — the porting-slip guard (a transposed weight, a wrong fused band, a missed bias all
// diverge loudly). The tiny shapes stay under the device-GEMM work floor, so these tests pin the host
// reference tier; the device tier above the floor is the engine's standing steel-GEMM contract.

// qtSeq fills a deterministic small-valued f32 slice (|v| ≤ 0.5) — small enough that softmax and the
// activations stay in their well-conditioned range.
func qtSeq(n int, seed uint32) []float32 {
	out := make([]float32, n)
	s := seed*2654435761 + 7
	for i := range out {
		s = s*1664525 + 1013904223
		out[i] = float32(int32(s>>9)%1000)/2000 - 0.25
	}
	return out
}

// qtTowerDims: hidden 8, 2 heads × headDim 4, patchDim 6, FF 12, merge 2, text hidden 10, a 2×2
// learned position table (side 2 — interpolated onto the 2×4 test grid).
const (
	qtHidden  = 8
	qtHeadDim = 4
	qtPatch   = 6
	qtFF      = 12
	qtText    = 10
)

func qtLinear(out, in int, seed uint32, bias bool) qwen35.VisionLinear {
	l := qwen35.VisionLinear{W: qtSeq(out*in, seed), Out: out, In: in}
	if bias {
		l.B = qtSeq(out, seed+1)
	}
	return l
}

// qtRealTower builds the REAL-layout tiny tower: GELU MLP, biases everywhere, no q/k norms, plain
// MHA, a learned position table.
func qtRealTower(blocks int) *qwen35.VisionTower {
	tower := &qwen35.VisionTower{
		Patch:    qtLinear(qtHidden, qtPatch, 1, true),
		PosEmbed: qtSeq(4*qtHidden, 3), // side-2 square table
		Merger: qwen35.VisionMerger{
			NormW: qtSeq(qtHidden, 5), NormB: qtSeq(qtHidden, 6),
			L1: qtLinear(qtHidden*4, qtHidden*4, 7, true),
			L2: qtLinear(qtText, qtHidden*4, 9, true),
		},
		Cfg: qwen35.VisionTowerConfig{
			Hidden: qtHidden, PatchDim: qtPatch,
			NumHeads: 2, NumKVHeads: 2, HeadDim: qtHeadDim,
			PatchSize: 1, InChannels: 3, TemporalPatchSize: 2,
			MergeSize: 2, TextHidden: qtText,
			RopeTheta: 10000, Eps: 1e-6,
			LearnedPositions: true, ImageTokenID: 777,
		},
	}
	for i := range blocks {
		seed := uint32(100 * (i + 1))
		tower.Blocks = append(tower.Blocks, qwen35.VisionBlock{
			Norm1W: qtSeq(qtHidden, seed+1), Norm1B: qtSeq(qtHidden, seed+2),
			Norm2W: qtSeq(qtHidden, seed+3), Norm2B: qtSeq(qtHidden, seed+4),
			Attn: qwen35.VisionAttnWeights{
				Q: qtLinear(qtHidden, qtHidden, seed+5, true),
				K: qtLinear(qtHidden, qtHidden, seed+7, true),
				V: qtLinear(qtHidden, qtHidden, seed+9, true),
				O: qtLinear(qtHidden, qtHidden, seed+11, true),
			},
			MLP: qwen35.VisionMLPWeights{
				FC1:  qtLinear(qtFF, qtHidden, seed+13, true),
				FC2:  qtLinear(qtHidden, qtFF, seed+15, true),
				GELU: true,
			},
		})
	}
	return tower
}

// qtGuessedTower builds the GUESSED-layout tiny tower: SwiGLU MLP, GQA (2 query heads, 1 kv head),
// per-head q/k RMS norms, no learned positions, no biases.
func qtGuessedTower() *qwen35.VisionTower {
	return &qwen35.VisionTower{
		Patch: qtLinear(qtHidden, qtPatch, 31, false),
		Merger: qwen35.VisionMerger{
			NormW: qtSeq(qtHidden, 33), NormB: nil,
			L1: qtLinear(qtHidden*4, qtHidden*4, 35, false),
			L2: qtLinear(qtText, qtHidden*4, 37, false),
		},
		Blocks: []qwen35.VisionBlock{{
			Norm1W: qtSeq(qtHidden, 41), Norm2W: qtSeq(qtHidden, 42),
			Attn: qwen35.VisionAttnWeights{
				Q:     qtLinear(qtHidden, qtHidden, 43, false),
				K:     qtLinear(qtHeadDim, qtHidden, 45, false), // 1 kv head
				V:     qtLinear(qtHeadDim, qtHidden, 47, false),
				O:     qtLinear(qtHidden, qtHidden, 49, false),
				QNorm: qtSeq(qtHeadDim, 51),
				KNorm: qtSeq(qtHeadDim, 52),
			},
			MLP: qwen35.VisionMLPWeights{
				Gate: qtLinear(qtFF, qtHidden, 53, false),
				Up:   qtLinear(qtFF, qtHidden, 55, false),
				Down: qtLinear(qtHidden, qtFF, 57, false),
			},
		}},
		Cfg: qwen35.VisionTowerConfig{
			Hidden: qtHidden, PatchDim: qtPatch,
			NumHeads: 2, NumKVHeads: 1, HeadDim: qtHeadDim,
			MergeSize: 2, TextHidden: qtText,
			RopeTheta: 10000, Eps: 1e-6,
		},
	}
}

// --- the independent naive oracle (pure f64 within each stage, f32-rounded between stages — the
// same inter-stage storage tier as the implementation, so parity is tight) ---

func oLinear(x []float32, w qwen35.VisionLinear, L int) []float32 {
	out := make([]float32, L*w.Out)
	for t := range L {
		for o := range w.Out {
			acc := 0.0
			for k := range w.In {
				acc += float64(x[t*w.In+k]) * float64(w.W[o*w.In+k])
			}
			if w.B != nil {
				acc += float64(w.B[o])
			}
			out[t*w.Out+o] = float32(acc)
		}
	}
	return out
}

func oLayerNorm(x, w, b []float32, rows, d int, eps float32) []float32 {
	out := make([]float32, rows*d)
	for r := range rows {
		mean := 0.0
		for i := range d {
			mean += float64(x[r*d+i])
		}
		mean /= float64(d)
		va := 0.0
		for i := range d {
			dv := float64(x[r*d+i]) - mean
			va += dv * dv
		}
		inv := 1 / math.Sqrt(va/float64(d)+float64(eps))
		for i := range d {
			v := (float64(x[r*d+i]) - mean) * inv * float64(w[i])
			if b != nil {
				v += float64(b[i])
			}
			out[r*d+i] = float32(v)
		}
	}
	return out
}

func oRope(vec []float32, row, col, hd int, theta float64) {
	half, quarter := hd/2, hd/4
	inv := make([]float64, quarter)
	for i := range quarter {
		inv[i] = 1 / math.Pow(theta, float64(2*i)/float64(half))
	}
	out := make([]float32, hd)
	for d := range half {
		var a float64
		if d < quarter {
			a = float64(row) * inv[d]
		} else {
			a = float64(col) * inv[d-quarter]
		}
		c, s := float32(math.Cos(a)), float32(math.Sin(a))
		out[d] = vec[d]*c - vec[d+half]*s
		out[d+half] = vec[d+half]*c + vec[d]*s
	}
	copy(vec, out)
}

func oRMSHead(x, w []float32, eps float32) {
	if len(w) == 0 {
		return
	}
	ss := 0.0
	for _, e := range x {
		ss += float64(e) * float64(e)
	}
	r := math.Sqrt(ss/float64(len(x)) + float64(eps))
	for i := range x {
		x[i] = float32(float64(x[i]) / r * float64(w[i]))
	}
}

func oAttention(x []float32, w qwen35.VisionAttnWeights, L, gridW int, cfg qwen35.VisionTowerConfig) []float32 {
	H, KVH, HD := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	q, k, v := oLinear(x, w.Q, L), oLinear(x, w.K, L), oLinear(x, w.V, L)
	theta := float64(cfg.RopeTheta)
	for t := range L {
		row, col := t/gridW, t%gridW
		for h := range H {
			hq := q[t*H*HD+h*HD : t*H*HD+(h+1)*HD]
			oRMSHead(hq, w.QNorm, cfg.Eps)
			oRope(hq, row, col, HD, theta)
		}
		for h := range KVH {
			hk := k[t*KVH*HD+h*HD : t*KVH*HD+(h+1)*HD]
			oRMSHead(hk, w.KNorm, cfg.Eps)
			oRope(hk, row, col, HD, theta)
		}
	}
	rep := H / KVH
	out := make([]float32, L*H*HD)
	for t := range L {
		for h := range H {
			kvh := h / rep
			sc := make([]float64, L)
			maxS := math.Inf(-1)
			for j := range L {
				dot := 0.0
				for d := range HD {
					dot += float64(q[t*H*HD+h*HD+d]) * float64(k[j*KVH*HD+kvh*HD+d])
				}
				sc[j] = dot / math.Sqrt(float64(HD))
				if sc[j] > maxS {
					maxS = sc[j]
				}
			}
			sum := 0.0
			for j := range L {
				sc[j] = math.Exp(sc[j] - maxS)
				sum += sc[j]
			}
			for d := range HD {
				acc := 0.0
				for j := range L {
					acc += sc[j] * float64(v[j*KVH*HD+kvh*HD+d])
				}
				out[t*H*HD+h*HD+d] = float32(acc / sum)
			}
		}
	}
	return oLinear(out, w.O, L)
}

func oGelu(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Tanh(float64(0.7978845608028654*(x+0.044715*x*x*x)))))
}

func oMLP(x []float32, w qwen35.VisionMLPWeights, L int) []float32 {
	if w.GELU {
		h := oLinear(x, w.FC1, L)
		for i := range h {
			h[i] = oGelu(h[i])
		}
		return oLinear(h, w.FC2, L)
	}
	g, u := oLinear(x, w.Gate, L), oLinear(x, w.Up, L)
	h := make([]float32, len(g))
	for i := range h {
		gv := float64(g[i])
		h[i] = float32(gv / (1 + math.Exp(-gv)) * float64(u[i]))
	}
	return oLinear(h, w.Down, L)
}

func oAdd(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// oForward is the whole naive tower forward — the oracle.
func oForward(t *testing.T, tower *qwen35.VisionTower, patches []float32, gridH, gridW int) []float32 {
	t.Helper()
	cfg := tower.Cfg
	L := gridH * gridW
	h := oLinear(patches, tower.Patch, L)
	if len(tower.PosEmbed) > 0 {
		pos := tower.PosEmbed
		if len(pos) != L*cfg.Hidden {
			var err error
			pos, err = qwen35.InterpolatePosEmbed(tower.PosEmbed, cfg.Hidden, gridH, gridW)
			if err != nil {
				t.Fatalf("oracle pos interp: %v", err)
			}
		}
		h = oAdd(h, pos)
	}
	for i := range tower.Blocks {
		b := &tower.Blocks[i]
		n1 := oLayerNorm(h, b.Norm1W, b.Norm1B, L, cfg.Hidden, cfg.Eps)
		h = oAdd(h, oAttention(n1, b.Attn, L, gridW, cfg))
		n2 := oLayerNorm(h, b.Norm2W, b.Norm2B, L, cfg.Hidden, cfg.Eps)
		h = oAdd(h, oMLP(n2, b.MLP, L))
	}
	M := cfg.MergeSize
	normed := oLayerNorm(h, tower.Merger.NormW, tower.Merger.NormB, L, cfg.Hidden, cfg.Eps)
	outRows, outCols := (gridH/M)*(gridW/M), cfg.Hidden*M*M
	merged := make([]float32, outRows*outCols)
	idx := 0
	for by := 0; by < gridH; by += M {
		for bx := 0; bx < gridW; bx += M {
			col := 0
			for dy := range M {
				for dx := range M {
					copy(merged[idx*outCols+col:idx*outCols+col+cfg.Hidden], normed[((by+dy)*gridW+(bx+dx))*cfg.Hidden:((by+dy)*gridW+(bx+dx))*cfg.Hidden+cfg.Hidden])
					col += cfg.Hidden
				}
			}
			idx++
		}
	}
	h1 := oLinear(merged, tower.Merger.L1, outRows)
	for i := range h1 {
		h1[i] = oGelu(h1[i])
	}
	return oLinear(h1, tower.Merger.L2, outRows)
}

func TestQwenVisionTowerForward_Good(t *testing.T) {
	tower := qtRealTower(2)
	const gridH, gridW = 2, 4
	patches := qtSeq(gridH*gridW*qtPatch, 999)
	features, softTokens, err := QwenVisionTowerForward(patches, gridH, gridW, tower)
	if err != nil {
		t.Fatalf("QwenVisionTowerForward: %v", err)
	}
	if want := (gridH / 2) * (gridW / 2); softTokens != want {
		t.Fatalf("softTokens = %d, want %d ((gridH/M)·(gridW/M))", softTokens, want)
	}
	if len(features) != softTokens*qtText {
		t.Fatalf("features len = %d, want %d (softTokens·TextHidden)", len(features), softTokens*qtText)
	}
	// Bit-determinism: an identical second run produces identical bytes.
	again, _, err := QwenVisionTowerForward(patches, gridH, gridW, tower)
	if err != nil {
		t.Fatalf("second run: %v", err)
	}
	for i := range features {
		if math.Float32bits(features[i]) != math.Float32bits(again[i]) {
			t.Fatalf("forward is not bit-deterministic at %d: %v vs %v", i, features[i], again[i])
		}
	}
}

func TestQwenVisionTowerForward_OracleParity_Good(t *testing.T) {
	for _, tc := range []struct {
		name  string
		tower *qwen35.VisionTower
	}{
		{"real_gelu_mha_learnedpos", qtRealTower(2)},
		{"guessed_swiglu_gqa_qknorm", qtGuessedTower()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const gridH, gridW = 2, 4
			patches := qtSeq(gridH*gridW*qtPatch, 777)
			got, softTokens, err := QwenVisionTowerForward(patches, gridH, gridW, tc.tower)
			if err != nil {
				t.Fatalf("QwenVisionTowerForward: %v", err)
			}
			want := oForward(t, tc.tower, patches, gridH, gridW)
			if len(got) != len(want) || softTokens*qtText != len(want) {
				t.Fatalf("shape mismatch: got %d, oracle %d", len(got), len(want))
			}
			var maxAbs float64
			for i := range got {
				if d := math.Abs(float64(got[i] - want[i])); d > maxAbs {
					maxAbs = d
				}
			}
			// The implementation and the oracle share the inter-stage f32 tier and the ascending-k
			// f64 accumulation, so parity is at f32 rounding noise; 1e-5 catches any structural slip
			// (transposed weight, wrong fused band, missed bias) by orders of magnitude.
			if maxAbs > 1e-5 {
				t.Fatalf("oracle divergence: max abs diff %v > 1e-5", maxAbs)
			}
		})
	}
}

// TestQwenVisionTowerForward_PositionSensitivity_Good proves both position mechanisms are ACTIVE:
// swapping two patch rows changes the output (2-D rope + learned table make position load-bearing;
// a tower that ignored position would emit permutation-covariant merged rows for a uniform grid).
func TestQwenVisionTowerForward_PositionSensitivity_Good(t *testing.T) {
	tower := qtRealTower(1)
	const gridH, gridW = 2, 4
	patches := qtSeq(gridH*gridW*qtPatch, 555)
	base, _, err := QwenVisionTowerForward(patches, gridH, gridW, tower)
	if err != nil {
		t.Fatalf("base run: %v", err)
	}
	swapped := append([]float32(nil), patches...)
	copy(swapped[0:qtPatch], patches[qtPatch:2*qtPatch])
	copy(swapped[qtPatch:2*qtPatch], patches[0:qtPatch])
	moved, _, err := QwenVisionTowerForward(swapped, gridH, gridW, tower)
	if err != nil {
		t.Fatalf("swapped run: %v", err)
	}
	same := true
	for i := range base {
		if base[i] != moved[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("swapping two patches left every soft token unchanged — position encoding is not engaging")
	}
}

func TestQwenVisionTowerForward_BadPatchBuffer_Bad(t *testing.T) {
	tower := qtRealTower(1)
	if _, _, err := QwenVisionTowerForward(qtSeq(5, 1), 2, 4, tower); err == nil {
		t.Fatal("a patch buffer not matching L·PatchDim must fail loudly")
	}
	if _, _, err := QwenVisionTowerForward(nil, 0, 0, tower); err == nil {
		t.Fatal("an empty grid must fail loudly")
	}
}

func TestQwenVisionTowerForward_MergeMismatch_Bad(t *testing.T) {
	tower := qtRealTower(1)
	// grid 3×4: rows not divisible by merge size 2 — the merger must refuse, not mis-merge.
	if _, _, err := QwenVisionTowerForward(qtSeq(3*4*qtPatch, 2), 3, 4, tower); err == nil {
		t.Fatal("a grid not divisible by the merge size must fail loudly")
	}
}
