// SPDX-Licence-Identifier: EUPL-1.2

package audio

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
)

// primitives.go is the pure-host float32 numeric kernel set the Conformer tower composes: GEMM, conv2d,
// layer/RMS norm, softmax, the element-wise activations and the clamps. On the engine/metal reference
// these dispatch to GPU kernels; here they are plain host arithmetic so the tower runs with no GPU
// dependency. The math mirrors the reference op-for-op (torch LayerNorm variance, mlx tanh-GELU, the
// im2col conv), so the shared HF goldens gate the composition at cosine >= 0.999.

// GEMM is the engine-neutral acceleration seam for tower matrix products. Implementations return
// ok=false when a product cannot be run on their device; the tower then computes that product on the
// host. Inputs and outputs are row-major float32. transposeB selects A*B^T rather than A*B.
type GEMM interface {
	MatMul(a, b []float32, m, k, n int, transposeB bool) (out []float32, ok bool)
}

func dispatchMatMul(gemm GEMM, a, b []float32, m, k, n int, transposeB bool) ([]float32, bool) {
	if gemm == nil || m <= 0 || k <= 0 || n <= 0 {
		return nil, false
	}
	out, ok := gemm.MatMul(a, b, m, k, n, transposeB)
	return out, ok && len(out) == m*n
}

// matMulNT computes out[M,N] = a[M,K] · w[N,K]ᵀ — the linear/attention "NT" form (weight stored
// [outDim,inDim]). f32 accumulation.
func matMulNT(a, w []float32, m, k, n int) []float32 {
	return matMulNTWith(nil, a, w, m, k, n)
}

func matMulNTWith(gemm GEMM, a, w []float32, m, k, n int) []float32 {
	if out, ok := dispatchMatMul(gemm, a, w, m, k, n, true); ok {
		return out
	}
	out := make([]float32, m*n)
	for i := range m {
		arow := a[i*k : i*k+k]
		orow := out[i*n : i*n+n]
		for j := range n {
			wrow := w[j*k : j*k+k]
			var acc float32
			for p := range k {
				acc += arow[p] * wrow[p]
			}
			orow[j] = acc
		}
	}
	return out
}

// matMulNN computes out[M,N] = a[M,K] · b[K,N] — the attention "NN" form (b already [K,N]). f32
// accumulation.
func matMulNN(a, b []float32, m, k, n int) []float32 {
	return matMulNNWith(nil, a, b, m, k, n)
}

func matMulNNWith(gemm GEMM, a, b []float32, m, k, n int) []float32 {
	if out, ok := dispatchMatMul(gemm, a, b, m, k, n, false); ok {
		return out
	}
	out := make([]float32, m*n)
	for i := range m {
		arow := a[i*k : i*k+k]
		orow := out[i*n : i*n+n]
		for p := range k {
			av := arow[p]
			brow := b[p*n : p*n+n]
			for j := range n {
				orow[j] += av * brow[j]
			}
		}
	}
	return out
}

// matMulMixedNT widens the bf16 weight [outDim,inDim] to f32 and runs matMulNT — the linear forward
// (in [L,inDim] f32, weight bf16, out [L,outDim] f32), matching the engine's promote-and-GEMM.
func matMulMixedNT(in []float32, weight []byte, l, inDim, outDim int) []float32 {
	return matMulMixedNTWith(nil, in, weight, l, inDim, outDim)
}

func matMulMixedNTWith(gemm GEMM, in []float32, weight []byte, l, inDim, outDim int) []float32 {
	return matMulNTWith(gemm, in, bf16ToF32Slice(weight), l, inDim, outDim)
}

// transpose returns the [cols,rows] transpose of a [rows,cols] matrix.
func transpose(x []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := range rows {
		for c := range cols {
			out[c*rows+r] = x[r*cols+c]
		}
	}
	return out
}

// clampF32 clamps a copy of x to [min,max] (the ±gradient-clipping select); min==max ⇒ pass-through.
func clampF32(x []float32, min, max float32) []float32 {
	if min == max {
		return x
	}
	out := append([]float32(nil), x...)
	for i, v := range out {
		if v < min {
			out[i] = min
		} else if v > max {
			out[i] = max
		}
	}
	return out
}

// applyClip clamps x to a per-linear activation bound when present (absent ⇒ pass-through).
func applyClip(x []float32, c model.LoadedAudioClipBound) []float32 {
	if !c.Present {
		return x
	}
	return clampF32(x, c.Min, c.Max)
}

// linear runs a clippable linear on f32 [L,inDim]: clip input → bf16-widen matmul → clip output →
// [L,outDim]. Mirrors the engine's ClippableLinear.Forward in f32.
func linear(in []float32, w model.LoadedAudioLinear, l, inDim, outDim int) []float32 {
	return linearWith(nil, in, w, l, inDim, outDim)
}

func linearWith(gemm GEMM, in []float32, w model.LoadedAudioLinear, l, inDim, outDim int) []float32 {
	clipped := applyClip(in, w.Clip.In)
	out := matMulMixedNTWith(gemm, clipped, w.Weight, l, inDim, outDim)
	return applyClip(out, w.Clip.Out)
}

// projectorReference is the engine-neutral embed_audio oracle: no-scale RMS norm followed by the
// projector's row-major linear transform. Quantised projectors are widened with the canonical MLX
// affine dequantiser before the host float32 multiply.
func projectorReference(in []float32, rows int, projector model.LoadedAudioLinear, eps float32) ([]float32, error) {
	if rows <= 0 || projector.InDim <= 0 || projector.OutDim <= 0 || len(in) != rows*projector.InDim {
		return nil, core.NewError("audio.projectorReference: invalid projector geometry")
	}
	if eps < 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return nil, core.NewError("audio.projectorReference: epsilon must be non-negative and finite")
	}
	normed := append([]float32(nil), in...)
	for row := range rows {
		values := normed[row*projector.InDim : (row+1)*projector.InDim]
		var squares float32
		for _, value := range values {
			squares += value * value
		}
		invRMS := float32(1 / math.Sqrt(float64(squares/float32(projector.InDim)+eps)))
		for index := range values {
			values[index] *= invRMS
		}
	}
	var weights []float32
	var err error
	if len(projector.Scales) > 0 || len(projector.Biases) > 0 {
		if projector.Kind != mlxaffine.Mode || len(projector.Scales) == 0 || len(projector.Biases) == 0 {
			return nil, core.NewError("audio.projectorReference: incomplete MLX affine projector")
		}
		weights, err = mlxaffine.DequantizeTensor(projector.Weight, projector.Scales, projector.Biases,
			projector.OutDim, projector.InDim, projector.Bits, projector.GroupSize)
		if err != nil {
			return nil, core.E("audio.projectorReference", "dequantise projector", err)
		}
	} else {
		if len(projector.Weight) != projector.OutDim*projector.InDim*bf16Size {
			return nil, core.NewError("audio.projectorReference: BF16 projector size mismatch")
		}
		weights = bf16ToF32Slice(projector.Weight)
	}
	return matMulNT(normed, weights, rows, projector.InDim, projector.OutDim), nil
}

// mulScalar scales every element of x by s (fresh slice).
func mulScalar(x []float32, s float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * s
	}
	return out
}

// add returns a+b element-wise (fresh slice).
func add(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// mul returns a·b element-wise (fresh slice).
func mul(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out
}

// sigmoid computes 1/(1+e^-x) element-wise (fresh slice).
func sigmoid(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = 1 / (1 + float32(math.Exp(float64(-v))))
	}
	return out
}

// silu computes x·σ(x) element-wise (fresh slice).
func silu(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
	return out
}

// gelu computes the tanh-approximation GELU element-wise, matching mlx's gelu_approx (the engine's
// composed Gelu): 0.5·x·(1+tanh(0.7978845608028654·(x+0.044715·x³))).
func gelu(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		inner := float64(v) + 0.044715*float64(v)*float64(v)*float64(v)
		t := math.Tanh(0.7978845608028654 * inner)
		out[i] = float32(0.5 * float64(v) * (1 + t))
	}
	return out
}

// relu computes max(x,0) element-wise (fresh slice) — the subsampler's fp32-promoting activation.
func relu(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

// activate applies the Conformer activation on f32 (silu default, relu, or tanh-GELU).
func activate(x []float32, act string) []float32 {
	switch act {
	case "relu":
		return relu(x)
	case "gelu", "gelu_pytorch_tanh":
		return gelu(x)
	default: // silu / swish / ""
		return silu(x)
	}
}

// rmsNorm RMS-normalises each [axis] row of [rows,axis] f32 (x·rsqrt(mean(x²)+eps)), scaling by the
// bf16 weight — the plain gemma RMSNorm (no +1 bias). Returns a fresh slice.
func rmsNorm(x []float32, weight []byte, rows, axis int, eps float32) []float32 {
	w := bf16ToF32Slice(weight)
	out := make([]float32, len(x))
	for r := range rows {
		row := x[r*axis : r*axis+axis]
		orow := out[r*axis : r*axis+axis]
		var ss float32
		for _, v := range row {
			ss += v * v
		}
		inv := float32(1.0 / math.Sqrt(float64(ss/float32(axis)+eps)))
		for i, v := range row {
			orow[i] = v * inv * w[i]
		}
	}
	return out
}

// layerNorm normalises each [axis] row of [rows,axis] f32 with torch LayerNorm semantics —
// (x-mean)·rsqrt(var+eps)·weight + bias, population variance. weight/bias are bf16 (bias may be zeros).
// Returns a fresh slice.
func layerNorm(x []float32, weight, bias []byte, rows, axis int, eps float32) []float32 {
	w := bf16ToF32Slice(weight)
	var b []float32
	if bias != nil {
		b = bf16ToF32Slice(bias)
	}
	out := make([]float32, len(x))
	for r := range rows {
		row := x[r*axis : r*axis+axis]
		orow := out[r*axis : r*axis+axis]
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(axis)
		var variance float32
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(axis)
		inv := float32(1.0 / math.Sqrt(float64(variance+eps)))
		for i, v := range row {
			o := (v - mean) * inv * w[i]
			if b != nil {
				o += b[i]
			}
			orow[i] = o
		}
	}
	return out
}

// softmax normalises each [axis] row of [rows,axis] f32 in place-returning (max-subtract, exp, sum) —
// the row-wise softmax over the last axis. Returns a fresh slice.
func softmax(x []float32, rows, axis int) []float32 {
	out := make([]float32, len(x))
	for r := range rows {
		row := x[r*axis : r*axis+axis]
		orow := out[r*axis : r*axis+axis]
		mx := row[0]
		for _, v := range row {
			if v > mx {
				mx = v
			}
		}
		var sum float32
		for i, v := range row {
			e := float32(math.Exp(float64(v - mx)))
			orow[i] = e
			sum += e
		}
		inv := 1 / sum
		for i := range orow {
			orow[i] *= inv
		}
	}
	return out
}

// tanh computes tanh element-wise into out (fresh slice) — the attention soft-cap.
func tanh(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(math.Tanh(float64(v)))
	}
	return out
}

// roundBf16 rounds x through bfloat16 and back to f32 — mirrors the engine's bf16 sub-block boundaries
// (the subsampler runs layer0's conv + LayerNorm in bf16 before the fp32-promoting ReLU).
func roundBf16(x []float32) []float32 {
	return bf16ToF32Slice(f32ToBf16Slice(x))
}

// convOut is the subsample conv output length for a stride-2, pad-1, kernel-3 axis: (in-1)/2 + 1.
func convOut(in int) int { return (in+2-3)/2 + 1 }

// conv2dF32 runs a 2-D convolution (NHWC input, OHWI weight [outC,kh,kw,inC]) via im2col + matMulNT —
// the same unfold-then-GEMM the engine conv dispatches. Zero-pads out-of-range taps.
func conv2dF32(in, weight []float32, batch, h, w, inC, outC, kh, kw, strideH, strideW, padH, padW int) []float32 {
	outH := (h+2*padH-kh)/strideH + 1
	outW := (w+2*padW-kw)/strideW + 1
	k := kh * kw * inC
	out := make([]float32, batch*outH*outW*outC)
	for n := range batch {
		unfolded := make([]float32, outH*outW*k)
		for oh := range outH {
			for ow := range outW {
				m := oh*outW + ow
				for r := range kh {
					ih := oh*strideH - padH + r
					if ih < 0 || ih >= h {
						continue
					}
					for c := range kw {
						iw := ow*strideW - padW + c
						if iw < 0 || iw >= w {
							continue
						}
						inBase := ((n*h+ih)*w + iw) * inC
						kBase := (r*kw + c) * inC
						copy(unfolded[m*k+kBase:m*k+kBase+inC], in[inBase:inBase+inC])
					}
				}
			}
		}
		o := matMulNT(unfolded, weight, outH*outW, k, outC)
		copy(out[n*outH*outW*outC:(n+1)*outH*outW*outC], o)
	}
	return out
}
