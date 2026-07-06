// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// audio_f32.go is the fp32 audio-block path. The gemma4 audio tower's gradient-clipping clamp scalars
// are f32 (metal.FromValue), so metal.Clip(bf16, f32, f32) PROMOTES the activation to f32 — from the
// first clamp the whole tower runs in fp32 (RMSNorm→f32, Matmul(f32,bf16)→f32, …). The bf16 blocks
// are therefore only data-dependently byte-identical; these fp32 blocks match metal's real Forward
// for any data. The bf16 weights stay bf16 on disk and are widened per matmul (an exact cast, exactly
// what mlx does promoting a mixed-dtype matmul).

// clampF32 is metal.Clip on fp32 — a select to [min,max] (byte-identical; min==max ⇒ pass-through).
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

// mulScalarF32 is metal.MulScalar on fp32 — a single f32 multiply per element (byte-identical).
func mulScalarF32(x []float32, s float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * s
	}
	return out
}

// applyF32 clamps an fp32 activation when the bound is present (the per-linear input/output clamp on a
// fp32 activation; metal.Clip(f32, …) stays f32).
func (c ClipBound) applyF32(x []float32) []float32 {
	if !c.Present {
		return x
	}
	return clampF32(x, c.Min, c.Max)
}

// matF32MixedNT is metal's Linear.Forward(f32_input) = Matmul(in, Transpose(weight_bf16)) → f32: the
// bf16 weight is promoted to f32 (an exact widen) and the nt steel GEMM (with split-K dispatch) runs
// in f32. in is [M=L, K=inDim] fp32; weight is [outDim, inDim] bf16; returns [L, outDim] fp32.
func matF32MixedNT(in []float32, weight []byte, L, outDim, inDim int) ([]float32, error) {
	if len(in) != L*inDim {
		return nil, core.NewError("native.matF32MixedNT: len(in) must equal L*inDim")
	}
	if len(weight) != outDim*inDim*bf16Size {
		return nil, core.NewError("native.matF32MixedNT: len(weight) must equal outDim*inDim*2 bytes")
	}
	return MatMulF32NT(in, bf16ToF32Slice(weight), L, inDim, outDim)
}

// clippedMatF32 is ClippableLinear.Forward in fp32: clip input → mixed-dtype matmul → clip output.
func clippedMatF32(in []float32, weight []byte, L, outDim, inDim int, clip ClipPair) ([]float32, error) {
	out, err := matF32MixedNT(clip.In.applyF32(in), weight, L, outDim, inDim)
	if err != nil {
		return nil, err
	}
	return clip.Out.applyF32(out), nil
}

// audioActivateF32 applies the Conformer activation on fp32 (gemma4AudioActivate): SiLU = x·σ(x).
func audioActivateF32(x []float32, act string) ([]float32, error) {
	switch act {
	case "relu":
		return reluF32(x), nil
	case "gelu", "gelu_pytorch_tanh":
		return Gelu(x)
	default: // silu / swish / ""
		s, err := Sigmoid(x)
		if err != nil {
			return nil, err
		}
		return Mul(x, s)
	}
}

// reluF32 is metal's ReLU (Maximum(x, 0)) in fp32 — the subsampler's Maximum has an f32 zero
// (FromValue), so it promotes its bf16 input to fp32 and the tower is fp32 from the first ReLU on.
func reluF32(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

// AudioSubsampleF32 runs the gemma4 audio subsampler returning FP32 [ceil(frames/4), hidden] — byte-
// identical to metal's Gemma4AudioSubSampleConvProjection.Forward, which promotes to fp32 at the first
// ReLU. Layer0's conv + LayerNorm stay bf16 (the input is bf16 log-mel); the ReLU promotes; Layer1 and
// the InputProj run fp32. The fp32 entry the encoder feeds into the Conformer layers.
func AudioSubsampleF32(features []byte, w *AudioSubsampleWeights, cfg AudioSubsampleConfig) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(features) != cfg.Frames*cfg.MelBins*bf16Size {
		return nil, core.NewError("native.AudioSubsampleF32: len(features) must equal Frames*MelBins*2 bytes")
	}
	t0, f0 := convOut(cfg.Frames), convOut(cfg.MelBins)
	// Layer0: bf16 conv + scale-only LayerNorm, then the fp32-promoting ReLU.
	c0, err := Conv2dBF16(features, w.Conv0, 1, cfg.Frames, cfg.MelBins, 1, cfg.OutC0, 3, 3, 2, 2, 1, 1)
	if err != nil {
		return nil, err
	}
	n0, err := LayerNormBF16(c0, w.Norm0W, w.Norm0B, t0*f0, cfg.OutC0, cfg.Eps)
	if err != nil {
		return nil, err
	}
	r0 := reluF32(bf16ToF32Slice(n0))

	// Layer1: fp32 (conv weight + norm widened from bf16).
	t1, f1 := convOut(t0), convOut(f0)
	c1, err := Conv2dF32(r0, bf16ToF32Slice(w.Conv1), 1, t0, f0, cfg.OutC0, cfg.OutC1, 3, 3, 2, 2, 1, 1)
	if err != nil {
		return nil, err
	}
	n1, err := LayerNormF32(c1, bf16ToF32Slice(w.Norm1W), bf16ToF32Slice(w.Norm1B), t1*f1, cfg.OutC1, cfg.Eps)
	if err != nil {
		return nil, err
	}
	r1 := reluF32(n1)

	// flatten [t1, f1·outC1] → InputProj (fp32 mixed-dtype matmul).
	return clippedMatF32(r1, w.InputProj, t1, cfg.Hidden, f1*cfg.OutC1, w.InputProjClip)
}

// sliceColsF32 extracts columns [c0:c1) from each row of [rows,cols] fp32.
func sliceColsF32(x []float32, rows, cols, c0, c1 int) []float32 {
	w := c1 - c0
	out := make([]float32, rows*w)
	for r := range rows {
		copy(out[r*w:r*w+w], x[r*cols+c0:r*cols+c1])
	}
	return out
}

// depthwiseConv1dF32 is the causal depthwise conv1d (NLC, left-pad K-1) in fp32 — the fp32 sibling of
// depthwiseConv1dBF16, matching metal.Conv1d(f32). out[t,c] = Σ_k in[t-(K-1)+k, c]·dw[c,k].
func depthwiseConv1dF32(in, dw []float32, L, ch, K int) []float32 {
	out := make([]float32, L*ch)
	for t := range L {
		for c := range ch {
			var acc float32
			for k := range K {
				if src := t - (K - 1) + k; src >= 0 {
					acc += in[src*ch+c] * dw[c*K+k]
				}
			}
			out[t*ch+c] = acc
		}
	}
	return out
}

// AudioLightConvF32 runs one Conformer LightConv on [L, hidden] FP32 — byte-identical to metal's
// Gemma4AudioLightConv.Forward: RMSNorm → LinearStart → GLU(gate·σ(gateIn)) → causal depthwise conv →
// clamp → RMSNorm → SiLU → LinearEnd → +x. No clamp before the first RMSNorm (unlike the FF).
func AudioLightConvF32(x []float32, w *AudioLightConvWeights, cfg AudioConfig) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if cfg.Hidden == 0 || cfg.Channels == 0 || cfg.KernelSize == 0 {
		return nil, core.NewError("native.AudioLightConvF32: cfg.Hidden, Channels, KernelSize must be set")
	}
	L, ch := len(x)/cfg.Hidden, cfg.Channels

	pre, err := RMSNorm(x, bf16ToF32Slice(w.PreNorm), L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, err
	}
	start, err := clippedMatF32(pre, w.LinearStart, L, 2*ch, cfg.Hidden, w.StartClip) // [L, 2·ch]
	if err != nil {
		return nil, err
	}
	gate := sliceColsF32(start, L, 2*ch, 0, ch)
	sig, err := Sigmoid(sliceColsF32(start, L, 2*ch, ch, 2*ch))
	if err != nil {
		return nil, err
	}
	glu, err := Mul(gate, sig) // GLU [L, ch]
	if err != nil {
		return nil, err
	}
	conv := depthwiseConv1dF32(glu, bf16ToF32Slice(w.DepthwiseWeight), L, ch, cfg.KernelSize)
	normed, err := RMSNorm(clampF32(conv, cfg.ClipMin, cfg.ClipMax), bf16ToF32Slice(w.ConvNorm), L, ch, cfg.Eps)
	if err != nil {
		return nil, err
	}
	act, err := audioActivateF32(normed, cfg.Act)
	if err != nil {
		return nil, err
	}
	end, err := clippedMatF32(act, w.LinearEnd, L, cfg.Hidden, ch, w.EndClip)
	if err != nil {
		return nil, err
	}
	return Add(end, x) // residual on the fp32 input
}

// AudioFeedForwardF32 runs one Conformer FeedForward on [L, hidden] FP32 — byte-identical to metal's
// Gemma4AudioFeedForward.Forward for ANY data (the fp32 path metal actually takes): clamp → RMSNorm →
// FFW1 → SiLU → FFW2 → clamp → RMSNorm → ·residual → +x. Weights stay bf16 on disk, widened per op.
func AudioFeedForwardF32(x []float32, w *AudioFeedForwardWeights, cfg AudioConfig) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if cfg.Hidden == 0 || cfg.FFInter == 0 {
		return nil, core.NewError("native.AudioFeedForwardF32: cfg.Hidden and cfg.FFInter must be set")
	}
	L := len(x) / cfg.Hidden

	pre, err := RMSNorm(clampF32(x, cfg.ClipMin, cfg.ClipMax), bf16ToF32Slice(w.PreNorm), L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, err
	}
	up, err := clippedMatF32(pre, w.FFW1, L, cfg.FFInter, cfg.Hidden, w.FFW1Clip)
	if err != nil {
		return nil, err
	}
	act, err := audioActivateF32(up, cfg.Act)
	if err != nil {
		return nil, err
	}
	down, err := clippedMatF32(act, w.FFW2, L, cfg.Hidden, cfg.FFInter, w.FFW2Clip)
	if err != nil {
		return nil, err
	}
	post, err := RMSNorm(clampF32(down, cfg.ClipMin, cfg.ClipMax), bf16ToF32Slice(w.PostNorm), L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, err
	}
	return Add(mulScalarF32(post, cfg.FFResidual), x) // residual on the fp32 input
}
