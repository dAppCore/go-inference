// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// weights.go reads a WhisperForConditionalGeneration checkpoint's safetensors into the flat f32 slices
// the host forward (encoder.go/decoder.go) consumes. The tensor names below are read verbatim off the
// REAL openai/whisper-tiny checkpoint (never guessed): every entry, prefix, and bias-presence rule was
// confirmed against `model.safetensors`'s actual 167-tensor name list before this file was written —
// notably k_proj carries NO bias (q_proj/v_proj/out_proj do), and there is no separate lm_head/proj_out
// tensor: WhisperForConditionalGeneration ties proj_out.weight to model.decoder.embed_tokens.weight
// (config carries no explicit tie_word_embeddings=false override, and no checkpoint tensor exists for
// it), so DecodeLMHead reuses EmbedTokens. mamba2.LoadMambaModel (arch/mamba2/loader.go) is the direct
// precedent for this "own loader, own session, never enters model.Assemble" shape.

// LinearWeights is one nn.Linear projection: Weight is [out,in] row-major (the PyTorch/safetensors
// convention), Bias is nil when the checkpoint carries none (Whisper's k_proj).
type LinearWeights struct {
	Weight  []float32
	Bias    []float32
	In, Out int
}

// LayerNormWeights is one nn.LayerNorm's affine parameters, both [dim].
type LayerNormWeights struct {
	Weight []float32
	Bias   []float32
}

// AttnWeights is one WhisperAttention block: Q/V/Out carry bias, K never does.
type AttnWeights struct {
	Q, K, V, Out LinearWeights
}

// EncoderLayerWeights is one WhisperEncoderLayer: pre-LN self-attention, pre-LN FFN.
type EncoderLayerWeights struct {
	SelfAttnNorm LayerNormWeights
	SelfAttn     AttnWeights
	FinalNorm    LayerNormWeights
	FC1, FC2     LinearWeights
}

// DecoderLayerWeights is one WhisperDecoderLayer: pre-LN causal self-attention, pre-LN cross-attention
// over the encoder output, pre-LN FFN.
type DecoderLayerWeights struct {
	SelfAttnNorm  LayerNormWeights
	SelfAttn      AttnWeights
	CrossAttnNorm LayerNormWeights
	CrossAttn     AttnWeights
	FinalNorm     LayerNormWeights
	FC1, FC2      LinearWeights
}

// Weights is a whole loaded Whisper checkpoint's tensors, widened to f32 and organised by
// encoder/decoder stage — the shape encoder.go/decoder.go's host forward walks directly.
type Weights struct {
	// Encoder: conv×2 subsample, a full (non-causal) transformer stack, final top-level norm.
	Conv1Weight, Conv1Bias []float32 // conv1: [DModel, NumMelBins, 3] / [DModel]
	Conv2Weight, Conv2Bias []float32 // conv2: [DModel, DModel, 3] / [DModel], stride 2
	EncoderPos             []float32 // [MaxSourcePositions, DModel] — always read in full (§encoder.go)
	EncoderLayers          []EncoderLayerWeights
	EncoderFinalNorm       LayerNormWeights

	// Decoder: learned token + position embeddings, a causal+cross transformer stack, final
	// top-level norm. The LM head is TIED to EmbedTokens (see doc comment above) — no separate tensor.
	EmbedTokens      []float32 // [VocabSize, DModel]
	DecoderPos       []float32 // [MaxTargetPositions, DModel]
	DecoderLayers    []DecoderLayerWeights
	DecoderFinalNorm LayerNormWeights

	DModel, VocabSize, MaxSourcePositions, MaxTargetPositions int
	EncoderHeads, DecoderHeads                                int
}

// tensorF32 widens a bf16/f16/f32 safetensors tensor to a flat f32 slice — Whisper checkpoints ship F32
// (torch_dtype "float32" in every published config), but bf16/f16 conversions circulate on the Hub, so
// both widen the same way mamba2.tensorF32 does (arch/mamba2/loader.go).
func tensorF32(t safetensors.Tensor) ([]float32, error) {
	switch t.Dtype {
	case "BF16", "bfloat16":
		if len(t.Data)%2 != 0 {
			return nil, core.NewError("whisper.tensorF32: bf16 byte length odd")
		}
		out := make([]float32, len(t.Data)/2)
		for i := range out {
			b := uint16(t.Data[2*i]) | uint16(t.Data[2*i+1])<<8
			out[i] = math.Float32frombits(uint32(b) << 16)
		}
		return out, nil
	case "F16", "float16":
		if len(t.Data)%2 != 0 {
			return nil, core.NewError("whisper.tensorF32: f16 byte length odd")
		}
		out := make([]float32, len(t.Data)/2)
		for i := range out {
			b := uint16(t.Data[2*i]) | uint16(t.Data[2*i+1])<<8
			out[i] = float16ToFloat32(b)
		}
		return out, nil
	case "F32", "float32":
		if len(t.Data)%4 != 0 {
			return nil, core.NewError("whisper.tensorF32: f32 byte length not /4")
		}
		out := make([]float32, len(t.Data)/4)
		for i := range out {
			out[i] = math.Float32frombits(uint32(t.Data[4*i]) | uint32(t.Data[4*i+1])<<8 | uint32(t.Data[4*i+2])<<16 | uint32(t.Data[4*i+3])<<24)
		}
		return out, nil
	}
	return nil, core.NewError("whisper.tensorF32: unsupported dtype " + t.Dtype)
}

// float16ToFloat32 widens one IEEE-754 binary16 value (as its raw bits) to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	var bits uint32
	switch exp {
	case 0:
		if frac == 0 {
			bits = sign << 31
		} else {
			// subnormal: normalise by shifting the fraction into a normal float32
			for frac&0x400 == 0 {
				frac <<= 1
				exp--
			}
			exp++
			frac &= 0x3ff
			bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13)
		}
	case 0x1f:
		bits = (sign << 31) | 0x7f800000 | (frac << 13)
	default:
		bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13)
	}
	return math.Float32frombits(bits)
}

// weightLoader closes over a checkpoint's tensor map, resolving names with the whisper prefix
// convention (model.encoder.*/model.decoder.*) and reporting exactly which name was missing.
type weightLoader struct {
	tensors map[string]safetensors.Tensor
}

func (l weightLoader) get(name string) (safetensors.Tensor, bool) {
	t, ok := l.tensors[name]
	return t, ok
}

func (l weightLoader) f32req(name string) ([]float32, error) {
	t, ok := l.get(name)
	if !ok {
		return nil, core.NewError("whisper.LoadWeights: missing tensor " + name)
	}
	return tensorF32(t)
}

func (l weightLoader) linear(prefix string, in, out int, hasBias bool) (LinearWeights, error) {
	w, err := l.f32req(prefix + ".weight")
	if err != nil {
		return LinearWeights{}, err
	}
	if len(w) != in*out {
		return LinearWeights{}, core.NewError(core.Sprintf("whisper.LoadWeights: %s.weight has %d elements, want %d (%d×%d)", prefix, len(w), in*out, out, in))
	}
	lw := LinearWeights{Weight: w, In: in, Out: out}
	if hasBias {
		b, err := l.f32req(prefix + ".bias")
		if err != nil {
			return LinearWeights{}, err
		}
		if len(b) != out {
			return LinearWeights{}, core.NewError(core.Sprintf("whisper.LoadWeights: %s.bias has %d elements, want %d", prefix, len(b), out))
		}
		lw.Bias = b
	}
	return lw, nil
}

func (l weightLoader) layerNorm(prefix string, dim int) (LayerNormWeights, error) {
	w, err := l.f32req(prefix + ".weight")
	if err != nil {
		return LayerNormWeights{}, err
	}
	b, err := l.f32req(prefix + ".bias")
	if err != nil {
		return LayerNormWeights{}, err
	}
	if len(w) != dim || len(b) != dim {
		return LayerNormWeights{}, core.NewError(core.Sprintf("whisper.LoadWeights: %s has %d/%d elements, want %d", prefix, len(w), len(b), dim))
	}
	return LayerNormWeights{Weight: w, Bias: b}, nil
}

func (l weightLoader) attn(prefix string, d int) (AttnWeights, error) {
	q, err := l.linear(prefix+".q_proj", d, d, true)
	if err != nil {
		return AttnWeights{}, err
	}
	k, err := l.linear(prefix+".k_proj", d, d, false) // Whisper's k_proj is bias=False — see doc comment
	if err != nil {
		return AttnWeights{}, err
	}
	v, err := l.linear(prefix+".v_proj", d, d, true)
	if err != nil {
		return AttnWeights{}, err
	}
	o, err := l.linear(prefix+".out_proj", d, d, true)
	if err != nil {
		return AttnWeights{}, err
	}
	return AttnWeights{Q: q, K: k, V: v, Out: o}, nil
}

// LoadWeights reads every tensor a WhisperForConditionalGeneration checkpoint carries, widened to f32,
// against the geometry cfg.Arch's refusal message reports (d_model/heads/layers/mel bins/vocab). Never
// consults model.Assemble/model.LookupArch — mamba2's "own loader" shape (see doc comment above).
func LoadWeights(tensors map[string]safetensors.Tensor, cfg *Config) (*Weights, error) {
	if cfg == nil {
		return nil, core.NewError("whisper.LoadWeights: nil config")
	}
	d := cfg.DModel
	if d <= 0 || cfg.EncoderLayers <= 0 || cfg.DecoderLayers <= 0 || cfg.NumMelBins <= 0 || cfg.VocabSize <= 0 ||
		cfg.MaxSourcePositions <= 0 || cfg.MaxTargetPositions <= 0 || cfg.EncoderAttentionHeads <= 0 || cfg.DecoderAttentionHeads <= 0 {
		return nil, core.NewError("whisper.LoadWeights: config geometry is incomplete (d_model/layers/heads/mel_bins/vocab/positions must all be positive)")
	}
	l := weightLoader{tensors: tensors}

	conv1W, err := l.f32req("model.encoder.conv1.weight")
	if err != nil {
		return nil, err
	}
	if len(conv1W) != d*cfg.NumMelBins*3 {
		return nil, core.NewError(core.Sprintf("whisper.LoadWeights: model.encoder.conv1.weight has %d elements, want %d (%d×%d×3)", len(conv1W), d*cfg.NumMelBins*3, d, cfg.NumMelBins))
	}
	conv1B, err := l.f32req("model.encoder.conv1.bias")
	if err != nil {
		return nil, err
	}
	if len(conv1B) != d {
		return nil, core.NewError(core.Sprintf("whisper.LoadWeights: model.encoder.conv1.bias has %d elements, want %d", len(conv1B), d))
	}
	conv2W, err := l.f32req("model.encoder.conv2.weight")
	if err != nil {
		return nil, err
	}
	if len(conv2W) != d*d*3 {
		return nil, core.NewError(core.Sprintf("whisper.LoadWeights: model.encoder.conv2.weight has %d elements, want %d (%d×%d×3)", len(conv2W), d*d*3, d, d))
	}
	conv2B, err := l.f32req("model.encoder.conv2.bias")
	if err != nil {
		return nil, err
	}
	if len(conv2B) != d {
		return nil, core.NewError(core.Sprintf("whisper.LoadWeights: model.encoder.conv2.bias has %d elements, want %d", len(conv2B), d))
	}
	encPos, err := l.f32req("model.encoder.embed_positions.weight")
	if err != nil {
		return nil, err
	}
	if len(encPos) != cfg.MaxSourcePositions*d {
		return nil, core.NewError(core.Sprintf("whisper.LoadWeights: model.encoder.embed_positions.weight has %d elements, want %d (%d×%d)", len(encPos), cfg.MaxSourcePositions*d, cfg.MaxSourcePositions, d))
	}

	encLayers := make([]EncoderLayerWeights, cfg.EncoderLayers)
	for i := range encLayers {
		p := core.Sprintf("model.encoder.layers.%d", i)
		selfNorm, err := l.layerNorm(p+".self_attn_layer_norm", d)
		if err != nil {
			return nil, err
		}
		selfAttn, err := l.attn(p+".self_attn", d)
		if err != nil {
			return nil, err
		}
		finalNorm, err := l.layerNorm(p+".final_layer_norm", d)
		if err != nil {
			return nil, err
		}
		fc1, err := l.linear(p+".fc1", d, cfg.EncoderFFNDim, true)
		if err != nil {
			return nil, err
		}
		fc2, err := l.linear(p+".fc2", cfg.EncoderFFNDim, d, true)
		if err != nil {
			return nil, err
		}
		encLayers[i] = EncoderLayerWeights{SelfAttnNorm: selfNorm, SelfAttn: selfAttn, FinalNorm: finalNorm, FC1: fc1, FC2: fc2}
	}
	encFinalNorm, err := l.layerNorm("model.encoder.layer_norm", d)
	if err != nil {
		return nil, err
	}

	embedTokens, err := l.f32req("model.decoder.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	if len(embedTokens) != cfg.VocabSize*d {
		return nil, core.NewError(core.Sprintf("whisper.LoadWeights: model.decoder.embed_tokens.weight has %d elements, want %d (%d×%d)", len(embedTokens), cfg.VocabSize*d, cfg.VocabSize, d))
	}
	decPos, err := l.f32req("model.decoder.embed_positions.weight")
	if err != nil {
		return nil, err
	}
	if len(decPos) != cfg.MaxTargetPositions*d {
		return nil, core.NewError(core.Sprintf("whisper.LoadWeights: model.decoder.embed_positions.weight has %d elements, want %d (%d×%d)", len(decPos), cfg.MaxTargetPositions*d, cfg.MaxTargetPositions, d))
	}

	decLayers := make([]DecoderLayerWeights, cfg.DecoderLayers)
	for i := range decLayers {
		p := core.Sprintf("model.decoder.layers.%d", i)
		selfNorm, err := l.layerNorm(p+".self_attn_layer_norm", d)
		if err != nil {
			return nil, err
		}
		selfAttn, err := l.attn(p+".self_attn", d)
		if err != nil {
			return nil, err
		}
		crossNorm, err := l.layerNorm(p+".encoder_attn_layer_norm", d)
		if err != nil {
			return nil, err
		}
		crossAttn, err := l.attn(p+".encoder_attn", d)
		if err != nil {
			return nil, err
		}
		finalNorm, err := l.layerNorm(p+".final_layer_norm", d)
		if err != nil {
			return nil, err
		}
		fc1, err := l.linear(p+".fc1", d, cfg.DecoderFFNDim, true)
		if err != nil {
			return nil, err
		}
		fc2, err := l.linear(p+".fc2", cfg.DecoderFFNDim, d, true)
		if err != nil {
			return nil, err
		}
		decLayers[i] = DecoderLayerWeights{
			SelfAttnNorm: selfNorm, SelfAttn: selfAttn,
			CrossAttnNorm: crossNorm, CrossAttn: crossAttn,
			FinalNorm: finalNorm, FC1: fc1, FC2: fc2,
		}
	}
	decFinalNorm, err := l.layerNorm("model.decoder.layer_norm", d)
	if err != nil {
		return nil, err
	}

	return &Weights{
		Conv1Weight: conv1W, Conv1Bias: conv1B,
		Conv2Weight: conv2W, Conv2Bias: conv2B,
		EncoderPos: encPos, EncoderLayers: encLayers, EncoderFinalNorm: encFinalNorm,
		EmbedTokens: embedTokens, DecoderPos: decPos, DecoderLayers: decLayers, DecoderFinalNorm: decFinalNorm,
		DModel: d, VocabSize: cfg.VocabSize, MaxSourcePositions: cfg.MaxSourcePositions, MaxTargetPositions: cfg.MaxTargetPositions,
		EncoderHeads: cfg.EncoderAttentionHeads, DecoderHeads: cfg.DecoderAttentionHeads,
	}, nil
}
