// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"math"
)

// primitives.go is the host-f32 math shared by vision.go and textdecoder.go: linear
// projection, RMSNorm (GlmOcrRMSNorm — T5LayerNorm-style, no bias), LayerNorm (the vision
// merger's post_projection_norm only), SiLU/GELU activations, and the scaled-dot-product
// attention core both towers share (vision: non-causal, heads==kvHeads; text: causal, GQA).
// f64 accumulation throughout for precision, matching this tree's house convention (e.g.
// arch/openai/whisper/attention.go's linearForward/mhaCore).

func addVec(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// linearForward computes y[T,Out] = x[T,In]·Wᵀ + b (b nil ⇒ no bias).
func linearForward(x []float32, w LinearWeights, T int) []float32 {
	out := make([]float32, T*w.Out)
	for t := range T {
		xi := x[t*w.In : (t+1)*w.In]
		oi := out[t*w.Out : (t+1)*w.Out]
		for o := range w.Out {
			var acc float64
			wr := w.Weight[o*w.In : (o+1)*w.In]
			for k := range w.In {
				acc += float64(xi[k]) * float64(wr[k])
			}
			if w.Bias != nil {
				acc += float64(w.Bias[o])
			}
			oi[o] = float32(acc)
		}
	}
	return out
}

// rmsNormForward applies GlmOcrRMSNorm over the last dimension D of x[T,D]: y = x·rsqrt(mean(x²)+eps)·weight.
func rmsNormForward(x []float32, w RMSNormWeights, T, D int, eps float32) []float32 {
	out := make([]float32, T*D)
	for t := range T {
		row := x[t*D : (t+1)*D]
		var sumSq float64
		for _, v := range row {
			sumSq += float64(v) * float64(v)
		}
		variance := sumSq / float64(D)
		inv := 1.0 / math.Sqrt(variance+float64(eps))
		orow := out[t*D : (t+1)*D]
		for i, v := range row {
			orow[i] = float32(float64(v)*inv) * w.Weight[i]
		}
	}
	return out
}

// layerNormForward applies nn.LayerNorm (default eps=1e-5) over the last dimension D of x[T,D]
// — used only by the vision merger's post_projection_norm.
func layerNormForward(x []float32, w LayerNormWeights, T, D int) []float32 {
	const eps = 1e-5
	out := make([]float32, T*D)
	for t := range T {
		row := x[t*D : (t+1)*D]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(D)
		var vsum float64
		for _, v := range row {
			d := float64(v) - mean
			vsum += d * d
		}
		variance := vsum / float64(D)
		inv := 1.0 / math.Sqrt(variance+eps)
		orow := out[t*D : (t+1)*D]
		for i, v := range row {
			normed := (float64(v) - mean) * inv
			orow[i] = float32(normed*float64(w.Weight[i]) + float64(w.Bias[i]))
		}
	}
	return out
}

// silu is x·sigmoid(x) — GLM-OCR's hidden_act for both the vision and text MLPs.
func silu(x float32) float32 {
	xf := float64(x)
	return float32(xf / (1 + math.Exp(-xf)))
}

// swiGLUForward computes down(silu(gate(x))·up(x)) over x[T,In] — the vision MLP, text MLP, and
// the vision merger's second stage all share this exact shape (see their respective doc
// comments for which weights carry bias).
func swiGLUForward(x []float32, gate, up, down LinearWeights, T int) []float32 {
	g := linearForward(x, gate, T)
	u := linearForward(x, up, T)
	act := make([]float32, len(g))
	for i := range g {
		act[i] = silu(g[i]) * u[i]
	}
	return linearForward(act, down, T)
}

// geluExact is the erf-based GELU (nn.GELU()'s default, NOT the tanh approximation) — the
// vision merger's act1, applied once between proj and the SwiGLU stage.
func geluExact(x float32) float32 {
	xf := float64(x)
	return float32(0.5 * xf * (1 + math.Erf(xf/math.Sqrt2)))
}

// mhaCore runs scaled dot-product attention over already-projected (and, where applicable,
// already-rope'd) q[T,heads*headDim] / k[T,kvHeads*headDim] / v[T,kvHeads*headDim]. Query head h
// reads KV head h/(heads/kvHeads) — repeat_kv's grouping (kvHeads==heads, groupSize==1, is
// plain MHA — the vision tower's shape). causal restricts query row i to keys [0,i] (the text
// decoder); vision attention is never causal (a single image is one full, non-causal segment —
// this package never batches more than one image per call, so cu_seqlens is always the trivial
// single-segment case and never needs representing explicitly). Returns [T,heads*headDim] with
// heads concatenated in channel order, ready for the output projection.
func mhaCore(q, k, v []float32, T, heads, kvHeads, headDim int, causal bool) []float32 {
	groupSize := heads / kvHeads
	kvDim := kvHeads * headDim
	qDim := heads * headDim
	scale := 1.0 / math.Sqrt(float64(headDim))
	out := make([]float32, T*qDim)
	scores := make([]float64, T)
	for h := range heads {
		kvh := h / groupSize
		qOff := h * headDim
		kOff := kvh * headDim
		for i := range T {
			limit := T
			if causal {
				limit = i + 1
			}
			qi := q[i*qDim+qOff : i*qDim+qOff+headDim]
			maxScore := math.Inf(-1)
			for j := range limit {
				kj := k[j*kvDim+kOff : j*kvDim+kOff+headDim]
				var dot float64
				for c := range headDim {
					dot += float64(qi[c]) * float64(kj[c])
				}
				dot *= scale
				scores[j] = dot
				if dot > maxScore {
					maxScore = dot
				}
			}
			var sum float64
			for j := range limit {
				e := math.Exp(scores[j] - maxScore)
				scores[j] = e
				sum += e
			}
			oi := out[i*qDim+qOff : i*qDim+qOff+headDim]
			for c := range headDim {
				var acc float64
				for j := range limit {
					acc += scores[j] * float64(v[j*kvDim+kOff+c])
				}
				oi[c] = float32(acc / sum)
			}
		}
	}
	return out
}
