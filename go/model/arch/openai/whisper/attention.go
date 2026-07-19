// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"math"

	core "dappco.re/go"
)

// attention.go is WhisperAttention ported host-side: standard multi-head softmax(QKᵀ)V, scaling folded
// into Q right after its projection (matching modeling_whisper.py's eager_attention_forward — the
// attention core itself runs with scaling=1.0 because the reference already scaled Q). Three shapes
// share the one mhaCore: encoder self-attention (Tq==Tk, non-causal), decoder self-attention (Tq==Tk,
// causal — the FULL growing sequence is recomputed every decode step; see decoder.go's doc comment for
// why that is the deliberate v1 simplification), and decoder cross-attention (Tq = decoder length,
// Tk = 1500 fixed encoder positions, K/V PRECOMPUTED once per request — the design's explicit ask).

// linearForward computes y[T,Out] = x[T,In]·Wᵀ + b (b nil ⇒ no bias — Whisper's k_proj). f64
// accumulation for precision, matching mamba2.matNT's house convention (arch/mamba2/block.go).
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

// layerNormForward applies LayerNorm over the last dimension D of x[T,D], eps=1e-5 (nn.LayerNorm's
// default — no Whisper layer overrides it).
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

// gelu is the exact erf-based GELU (transformers' ACT2FN["gelu"] — nn.functional.gelu, NOT the tanh
// approximation "gelu_new"/"gelu_pytorch_tanh" uses; Whisper's activation_function is plain "gelu" in
// every published config).
func gelu(x float32) float32 {
	xf := float64(x)
	return float32(0.5 * xf * (1 + math.Erf(xf/math.Sqrt2)))
}

func geluRow(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = gelu(v)
	}
	return out
}

// mhaCore runs scaled dot-product attention given ALREADY-PROJECTED q[Tq,D]/k[Tk,D]/v[Tk,D] split into H
// heads of headDim=D/H, Q pre-scaled by 1/√headDim (callers apply the scale right after q_proj — see
// projectQScaled). causal restricts query position i to keys [0,i] (decoder self-attention only;
// encoder self-attention and cross-attention always attend to every key). Returns [Tq,D] with heads
// concatenated back in channel order, ready for out_proj.
func mhaCore(q, k, v []float32, Tq, Tk, H, headDim int, causal bool) []float32 {
	D := H * headDim
	out := make([]float32, Tq*D)
	scores := make([]float64, Tk)
	for h := range H {
		off := h * headDim
		for i := range Tq {
			limit := Tk
			if causal {
				limit = i + 1
				if limit > Tk {
					limit = Tk
				}
			}
			qi := q[i*D+off : i*D+off+headDim]
			var maxScore float64 = math.Inf(-1)
			for j := range limit {
				kj := k[j*D+off : j*D+off+headDim]
				var dot float64
				for c := range headDim {
					dot += float64(qi[c]) * float64(kj[c])
				}
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
			oi := out[i*D+off : i*D+off+headDim]
			for c := range headDim {
				var acc float64
				for j := range limit {
					acc += scores[j] * float64(v[j*D+off+c])
				}
				oi[c] = float32(acc / sum)
			}
		}
	}
	return out
}

// projectQScaled projects x through w.Q then scales by 1/√headDim — WhisperAttention applies the
// scaling to Q immediately after its projection (before head-splitting), so mhaCore itself runs with an
// implicit scale of 1.0 (matching eager_attention_forward's scaling=1.0 call, since the reference has
// already scaled Q by then).
func projectQScaled(x []float32, w LinearWeights, T, headDim int) []float32 {
	q := linearForward(x, w, T)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for i := range q {
		q[i] *= scale
	}
	return q
}

// selfAttentionForward is one WhisperAttention self-attention pass: project q/k/v from x, run mhaCore
// (causal for the decoder, non-causal for the encoder), out_proj.
func selfAttentionForward(x []float32, T, D, H int, causal bool, w AttnWeights) ([]float32, error) {
	if D%H != 0 {
		return nil, core.NewError("whisper.selfAttentionForward: d_model not divisible by heads")
	}
	headDim := D / H
	q := projectQScaled(x, w.Q, T, headDim)
	k := linearForward(x, w.K, T)
	v := linearForward(x, w.V, T)
	attn := mhaCore(q, k, v, T, T, H, headDim, causal)
	return linearForward(attn, w.Out, T), nil
}

// precomputeCrossKV projects the (fixed, already-encoded) encoder output through one decoder layer's
// cross-attention K/V — the standard Whisper serving trick the design calls out explicitly: cross K/V
// depend only on the encoder output, so this runs ONCE per request, not once per decode step.
func precomputeCrossKV(encOut []float32, Tenc int, w AttnWeights) (k, v []float32) {
	return linearForward(encOut, w.K, Tenc), linearForward(encOut, w.V, Tenc)
}

// crossAttentionForward is one WhisperAttention cross-attention pass over PRECOMPUTED encoder K/V:
// project q from the decoder hidden xq only, run mhaCore (never causal — the decoder attends to the
// whole encoder output every step), out_proj.
func crossAttentionForward(xq []float32, Tq, D, H int, w AttnWeights, encK, encV []float32, Tenc int) ([]float32, error) {
	if D%H != 0 {
		return nil, core.NewError("whisper.crossAttentionForward: d_model not divisible by heads")
	}
	headDim := D / H
	q := projectQScaled(xq, w.Q, Tq, headDim)
	attn := mhaCore(q, encK, encV, Tq, Tenc, H, headDim, false)
	return linearForward(attn, w.Out, Tq), nil
}
