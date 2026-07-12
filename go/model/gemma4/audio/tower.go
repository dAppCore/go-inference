// SPDX-Licence-Identifier: EUPL-1.2

package audio

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// tower.go is the host float32 forward of the Gemma 4 Conformer audio tower, consuming the neutral
// model.LoadedAudio payload the gemma4 assembler produces (weights + geometry). It ports the
// engine/metal native tower module-for-module — subsample (bf16 layer0, then fp32) → 12 macaron
// Conformer layers → output projection WITH BIAS — as plain host arithmetic. The tower runs in f32
// from the subsampler's first ReLU; bf16 weights are widened per matmul (an exact cast). HF's
// Gemma4AudioModel goldens gate the composition at cosine >= 0.999.

// Subsample runs the Conformer audio subsampler on log-mel features [frames,melBins] bf16, returning
// f32 [ceil(frames/4), Hidden]. Layer0's conv + LayerNorm run bf16 (rounded), then the ReLU promotes
// to f32; layer1 + input_proj run f32. Mirrors engine/metal's AudioSubsampleF32.
func Subsample(features []byte, frames, melBins int, la *model.LoadedAudio) ([]float32, error) {
	if la == nil {
		return nil, core.NewError("audio.Subsample: nil LoadedAudio")
	}
	cfg := la.Cfg
	sub := la.Subsample
	if len(features) != frames*melBins*bf16Size {
		return nil, core.NewError("audio.Subsample: len(features) must equal frames*melBins*2 bytes")
	}
	outC0 := len(sub.Norm0W) / bf16Size
	outC1 := len(sub.Norm1W) / bf16Size
	if outC0 == 0 || outC1 == 0 {
		return nil, core.NewError("audio.Subsample: subsample norm widths must be non-zero")
	}
	t0, f0 := convOut(frames), convOut(melBins)

	// Layer0: bf16 conv + scale-only LayerNorm (both rounded to bf16), then the fp32-promoting ReLU.
	c0 := conv2dF32(bf16ToF32Slice(features), bf16ToF32Slice(sub.Conv0), 1, frames, melBins, 1, outC0, 3, 3, 2, 2, 1, 1)
	n0 := layerNorm(roundBf16(c0), sub.Norm0W, sub.Norm0B, t0*f0, outC0, cfg.Eps)
	r0 := relu(roundBf16(n0))

	// Layer1: f32 (conv + norm widened from bf16).
	t1, f1 := convOut(t0), convOut(f0)
	c1 := conv2dF32(r0, bf16ToF32Slice(sub.Conv1), 1, t0, f0, outC0, outC1, 3, 3, 2, 2, 1, 1)
	n1 := layerNorm(c1, sub.Norm1W, sub.Norm1B, t1*f1, outC1, cfg.Eps)
	r1 := relu(n1)

	// flatten [t1, f1·outC1] → input_proj (f32 mixed-dtype matmul).
	return linear(r1, sub.InputProj, t1, f1*outC1, cfg.Hidden), nil
}

// sliceCols extracts columns [c0:c1) from each row of [rows,cols] f32.
func sliceCols(x []float32, rows, cols, c0, c1 int) []float32 {
	w := c1 - c0
	out := make([]float32, rows*w)
	for r := range rows {
		copy(out[r*w:r*w+w], x[r*cols+c0:r*cols+c1])
	}
	return out
}

// depthwiseConv1d is the causal depthwise conv1d (NLC, left-pad K-1): out[t,c] = Σ_k in[t-(K-1)+k,c]·dw[c,k].
func depthwiseConv1d(in, dw []float32, l, ch, k int) []float32 {
	out := make([]float32, l*ch)
	for t := range l {
		for c := range ch {
			var acc float32
			for kk := range k {
				if src := t - (k - 1) + kk; src >= 0 {
					acc += in[src*ch+c] * dw[c*k+kk]
				}
			}
			out[t*ch+c] = acc
		}
	}
	return out
}

// feedForward runs one Conformer macaron FeedForward on f32 [L,Hidden]: clamp → RMSNorm → FFW1 → act →
// FFW2 → clamp → RMSNorm → ·residual → +x. Mirrors AudioFeedForwardF32.
func feedForward(x []float32, ff model.LoadedAudioFeedForward, cfg model.LoadedAudioConfig) []float32 {
	l := len(x) / cfg.Hidden
	pre := rmsNorm(clampF32(x, cfg.ClipMin, cfg.ClipMax), ff.PreNorm, l, cfg.Hidden, cfg.Eps)
	up := linear(pre, ff.FFW1, l, cfg.Hidden, cfg.FFInter)
	act := activate(up, cfg.Act)
	down := linear(act, ff.FFW2, l, cfg.FFInter, cfg.Hidden)
	post := rmsNorm(clampF32(down, cfg.ClipMin, cfg.ClipMax), ff.PostNorm, l, cfg.Hidden, cfg.Eps)
	return add(mulScalar(post, cfg.FFResidual), x)
}

// lightConv runs one Conformer LightConv on f32 [L,Hidden]: RMSNorm → LinearStart → GLU → causal
// depthwise conv → clamp → RMSNorm → act → LinearEnd → +x. No clamp before the first RMSNorm (unlike
// the FF). Mirrors AudioLightConvF32.
func lightConv(x []float32, lc model.LoadedAudioLightConv, cfg model.LoadedAudioConfig) []float32 {
	l, ch := len(x)/cfg.Hidden, cfg.Channels
	pre := rmsNorm(x, lc.PreNorm, l, cfg.Hidden, cfg.Eps)
	start := linear(pre, lc.LinearStart, l, cfg.Hidden, 2*ch) // [L, 2·ch]
	gate := sliceCols(start, l, 2*ch, 0, ch)
	sig := sigmoid(sliceCols(start, l, 2*ch, ch, 2*ch))
	glu := mul(gate, sig) // GLU [L, ch]
	conv := depthwiseConv1d(glu, bf16ToF32Slice(lc.DepthwiseWeight), l, ch, cfg.KernelSize)
	normed := rmsNorm(clampF32(conv, cfg.ClipMin, cfg.ClipMax), lc.ConvNorm, l, ch, cfg.Eps)
	act := activate(normed, cfg.Act)
	end := linear(act, lc.LinearEnd, l, ch, cfg.Hidden)
	return add(end, x) // residual on the f32 input
}

// Layer runs one Conformer block on f32 [L,Hidden] — the macaron sandwich: ff1 → clamp→RMSNorm(pre)→
// attn → clamp→RMSNorm(post)→+ff1 → lconv → ff2 → clamp→RMSNorm(out). Mirrors engine/metal's AudioLayer.
func Layer(x []float32, layer *model.LoadedAudioLayer, cfg model.LoadedAudioConfig) ([]float32, error) {
	if cfg.Hidden == 0 {
		return nil, core.NewError("audio.Layer: cfg.Hidden must be set")
	}
	l := len(x) / cfg.Hidden
	rmsClamped := func(b []float32, norm []byte) []float32 {
		return rmsNorm(clampF32(b, cfg.ClipMin, cfg.ClipMax), norm, l, cfg.Hidden, cfg.Eps)
	}
	h := feedForward(x, layer.FF1, cfg)
	pre := rmsClamped(h, layer.NormPreAttn)
	attn := attention(pre, layer.Attn, cfg)
	post := rmsClamped(attn, layer.NormPostAttn)
	res := add(post, h)
	conv := lightConv(res, layer.LConv, cfg)
	ff2 := feedForward(conv, layer.FF2, cfg)
	return rmsClamped(ff2, layer.NormOut), nil
}

// Encode runs the full audio tower on log-mel features [frames,melBins] bf16, returning
// [ceil(frames/4), OutputDim] f32: subsample → Conformer layers → output projection (+bias). Mirrors
// engine/metal's AudioEncode.
func Encode(features []byte, frames, melBins int, la *model.LoadedAudio) ([]float32, error) {
	if la == nil || la.OutputProj == nil {
		return nil, core.NewError("audio.Encode: LoadedAudio has no output projection")
	}
	cfg := la.Cfg
	h, err := Subsample(features, frames, melBins, la)
	if err != nil {
		return nil, err
	}
	for i := range la.Layers {
		if h, err = Layer(h, &la.Layers[i], cfg); err != nil {
			return nil, core.E("audio.Encode", core.Sprintf("layer %d", i), err)
		}
	}
	t := len(h) / cfg.Hidden
	out := matMulMixedNT(h, la.OutputProj, t, cfg.Hidden, cfg.OutputDim)
	addOutputProjBias(out, la.OutputProjBias, t, cfg.OutputDim)
	return out, nil
}

// addOutputProjBias adds audio_tower.output_proj.bias [outDim] (BF16) to every output row in place —
// HF's output_proj is a bias=True Linear (the e2b bias max|abs| is 14.875, so dropping it corrupts
// every clip). A nil/short bias is a no-op (packs that omit it).
func addOutputProjBias(out []float32, bias []byte, rows, outDim int) {
	if outDim <= 0 || len(bias) < outDim*bf16Size {
		return
	}
	b := bf16ToF32Slice(bias[:outDim*bf16Size])
	for r := range rows {
		row := out[r*outDim : (r+1)*outDim]
		for c := range row {
			row[c] += b[c]
		}
	}
}
