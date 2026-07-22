// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import "math"

// math.go is the host-f32 primitive layer vision.go/decoder.go share: dense projections,
// RMSNorm/LayerNorm, SiLU/GELU activations, and the rotate-half RoPE application both the vision
// tower's 2D (h/w) rotary embedding and the text decoder's 1D rotary embedding reduce to — see
// applyRotaryHalf's doc comment. f64 accumulation throughout for precision, matching
// whisper.linearForward/mamba2's house convention.

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

// rmsNormForward applies RMSNorm over the last dimension D of x[T,D]: y = x/rms(x) * weight, no
// bias, no mean-subtraction (distinct from layerNormForward below).
func rmsNormForward(x []float32, w RMSNormWeights, T, D int, eps float32) []float32 {
	out := make([]float32, T*D)
	for t := range T {
		row := x[t*D : (t+1)*D]
		var sumSq float64
		for _, v := range row {
			sumSq += float64(v) * float64(v)
		}
		inv := 1.0 / math.Sqrt(sumSq/float64(D)+float64(eps))
		orow := out[t*D : (t+1)*D]
		for i, v := range row {
			orow[i] = float32(float64(v) * inv * float64(w.Weight[i]))
		}
	}
	return out
}

// layerNormForward applies a standard mean/variance LayerNorm over the last dimension D of
// x[T,D]. The only user is the vision PatchMerger's ln_q (hard-coded eps=1e-6 upstream — every
// other norm in this architecture is RMSNorm).
func layerNormForward(x []float32, w LayerNormWeights, T, D int, eps float32) []float32 {
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
		inv := 1.0 / math.Sqrt(vsum/float64(D)+float64(eps))
		orow := out[t*D : (t+1)*D]
		for i, v := range row {
			normed := (float64(v) - mean) * inv
			orow[i] = float32(normed*float64(w.Weight[i]) + float64(w.Bias[i]))
		}
	}
	return out
}

// silu is the SiLU/Swish activation x·sigmoid(x) — DotsSwiGLUFFN's and Qwen2MLP's gate.
func silu(x float32) float32 {
	xf := float64(x)
	return float32(xf / (1 + math.Exp(-xf)))
}

func siluRow(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = silu(v)
	}
	return out
}

// gelu is the exact erf-based GELU (torch.nn.GELU()'s default approximate="none") — the vision
// PatchMerger's activation.
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

// swiGLU computes down(silu(gate(x)) * up(x)) — the shared MLP shape for both the vision tower's
// DotsSwiGLUFFN and the text decoder's Qwen2MLP (same maths, different weight sets/dims).
func swiGLU(x []float32, gate, up, down LinearWeights, T int) []float32 {
	g := siluRow(linearForward(x, gate, T))
	u := linearForward(x, up, T)
	for i := range g {
		g[i] *= u[i]
	}
	return linearForward(g, down, T)
}

func addRows(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// applyRotaryHalf applies the standard "rotate-half" RoPE formula to ONE head's vector
// vec[headDim] in place, given the UN-duplicated half-length frequency table cosHalf/sinHalf
// (len(vec) == 2*len(cosHalf)):
//
//	out[c]        = vec[c]*cosHalf[c]      - vec[c+half]*sinHalf[c]
//	out[c+half]   = vec[c+half]*cosHalf[c] + vec[c]*sinHalf[c]
//
// This is the SAME mechanic both rotary embeddings in this architecture reduce to — the text
// decoder's 1D RoPE (cosHalf[c] = cos(pos·invFreq[c]), one frequency axis) and the vision tower's
// 2D RoPE (cosHalf built by concatenating an h-position half-table and a w-position half-table —
// see visionRotaryTable in vision.go); apply_rotary_pos_emb/apply_rotary_pos_emb_vision in the
// reference both duplicate their half-length cos/sin into the full headDim via
// `cat(freqs,freqs)`/`repeat(1,1,2)` BEFORE this same rotate-half multiply, which is exactly what
// reading cosHalf/sinHalf twice (once directly, once at the +half offset) reproduces without
// materialising the duplicated table.
func applyRotaryHalf(vec []float32, cosHalf, sinHalf []float32) {
	half := len(cosHalf)
	for c := range half {
		x1, x2 := vec[c], vec[c+half]
		vec[c] = x1*cosHalf[c] - x2*sinHalf[c]
		vec[c+half] = x2*cosHalf[c] + x1*sinHalf[c]
	}
}
