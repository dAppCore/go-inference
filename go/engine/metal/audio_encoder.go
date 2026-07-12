// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
)

// audio_encoder.go assembles the gemma4 Conformer audio tower from the byte-identical blocks
// (AudioFeedForward / AudioAttention / AudioLightConv / AudioSubsample) — the no-cgo port of metal's
// Gemma4AudioLayer + Gemma4AudioEncoder. The assembly itself is composition of proven byte-parity ops
// (clampBF16, RMSNormBF16, AddBF16, MatRowsBF16) plus the host sinusoid position table, so the whole
// tower stays byte-identical to pkg/metal.

// AudioLayerWeights bundles one Conformer block's sub-block weights + its three RMSNorms. The FF/attn/
// conv geometry is read from the shared AudioConfig.
type AudioLayerWeights struct {
	FF1, FF2                           *AudioFeedForwardWeights
	Attn                               *AudioAttentionWeights
	LConv                              *AudioLightConvWeights
	NormPreAttn, NormPostAttn, NormOut []byte
}

// AudioLayer runs one Conformer block on [L, hidden] FP32 — byte-identical to metal's
// Gemma4AudioLayer.Forward: ff1 → clamp→RMSNorm(pre)→attn→clamp→RMSNorm(post)→+ff1 → lconv → ff2 →
// clamp→RMSNorm(out). The tower runs in fp32 (the f32 GC clamp promotes the activation — see
// audio_f32.go); the clamp is the shared ±gradient-clipping (cfg.ClipMin/ClipMax).
func AudioLayer(x []float32, w *AudioLayerWeights, cfg AudioConfig) ([]float32, error) {
	L := len(x) / cfg.Hidden
	rmsClamped := func(b []float32, norm []byte) ([]float32, error) {
		return RMSNorm(clampF32(b, cfg.ClipMin, cfg.ClipMax), bf16ToF32Slice(norm), L, cfg.Hidden, cfg.Eps)
	}

	h, err := AudioFeedForwardF32(x, w.FF1, cfg)
	if err != nil {
		return nil, err
	}
	pre, err := rmsClamped(h, w.NormPreAttn)
	if err != nil {
		return nil, err
	}
	attn, err := AudioAttentionF32(pre, w.Attn, cfg)
	if err != nil {
		return nil, err
	}
	post, err := rmsClamped(attn, w.NormPostAttn)
	if err != nil {
		return nil, err
	}
	res, err := Add(post, h)
	if err != nil {
		return nil, err
	}
	conv, err := AudioLightConvF32(res, w.LConv, cfg)
	if err != nil {
		return nil, err
	}
	ff2, err := AudioFeedForwardF32(conv, w.FF2, cfg)
	if err != nil {
		return nil, err
	}
	return rmsClamped(ff2, w.NormOut)
}

// AudioEncoderWeights is the whole tower: subsampler, the Conformer layers, and the output
// projection into the multimodal embedding width. PosEmbed is the shared sinusoid table; if nil it is
// built from cfg (AudioPositionTable). OutputDim is the projection's output width.
type AudioEncoderWeights struct {
	Subsample  *AudioSubsampleWeights
	SubsampleC AudioSubsampleConfig
	Layers     []*AudioLayerWeights
	OutputProj []byte // [OutputDim, hidden]
	// OutputProjBias is audio_tower.output_proj.bias [OutputDim] BF16, added to every projected row —
	// HF's output_proj is a bias=True Linear. nil is a no-op (packs that omit it).
	OutputProjBias []byte
	OutputDim      int
}

// AudioEncode runs the full audio tower on log-mel features [frames, melBins] bf16, returning
// [ceil(frames/4), OutputDim] FP32 — byte-identical to metal's Gemma4AudioEncoder.Forward: subsample
// (bf16) → widen → Conformer layers (fp32) → OutputProj (fp32 mixed-dtype matmul). The per-layer
// attentions share PosEmbed (cfg-derived, set on each layer's Attn weights by the caller / loader).
func AudioEncode(features []byte, w *AudioEncoderWeights, cfg AudioConfig) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	h, err := AudioSubsampleF32(features, w.Subsample, w.SubsampleC) // subsampler promotes to fp32 at its first ReLU
	if err != nil {
		return nil, err
	}
	for i, layer := range w.Layers {
		if h, err = AudioLayer(h, layer, cfg); err != nil {
			return nil, core.E("native.AudioEncode", core.Sprintf("layer %d", i), err)
		}
	}
	T := len(h) / cfg.Hidden
	out, err := matF32MixedNT(h, w.OutputProj, T, w.OutputDim, cfg.Hidden) // OutputProj.Forward(f32)
	if err != nil {
		return nil, err
	}
	addOutputProjBias(out, w.OutputProjBias, T, w.OutputDim)
	return out, nil
}

// addOutputProjBias adds the audio_tower.output_proj.bias [outDim] (BF16) to every one of the rows
// output rows in place — HF's Gemma4AudioModel.output_proj is a bias=True Linear. A nil/short bias is
// a no-op (packs that omit it), so the tower stays byte-identical when there is no bias to apply.
func addOutputProjBias(out []float32, bias []byte, rows, outDim int) {
	if outDim <= 0 || len(bias) < outDim*2 {
		return
	}
	b := bf16ToF32Slice(bias[:outDim*2])
	for r := range rows {
		row := out[r*outDim : (r+1)*outDim]
		for c := range row {
			row[c] += b[c]
		}
	}
}

// AudioPositionTable builds the [count, hidden] sinusoid relative-position table the Conformer
// attention reads — byte-identical to metal's gemma4AudioPositionTable: positions [count-1 .. 0],
// [sin… cos…] over hidden/2 log-spaced timescales, host f32 (then fed to the attention as PosEmbed).
func AudioPositionTable(count, hidden int) []float32 {
	half := hidden / 2
	logIncrement := math.Log(10000.0) / float64(maxInt(half-1, 1))
	vals := make([]float32, count*hidden)
	for p := range count {
		position := float64(count - 1 - p)
		row := p * hidden
		for i := range half {
			scaled := position * math.Exp(float64(i)*-logIncrement)
			vals[row+i] = float32(math.Sin(scaled))
			vals[row+half+i] = float32(math.Cos(scaled))
		}
	}
	return vals
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
