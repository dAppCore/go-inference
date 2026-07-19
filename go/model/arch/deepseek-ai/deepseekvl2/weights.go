// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// projectorInputDim is the concat width MlpProjector's Linear(2048,1280) expects: CLIP's 1024-wide
// per-patch output concatenated with SAM's own 1024-wide (samNeckOut3) raw feature — see
// vision.go's VisionForward for exactly how the two towers' outputs are concatenated.
const projectorInputDim = clipHidden + samNeckOut3

// weights.go reads a DeepSeek-OCR checkpoint's safetensors into the flat f32 slices the host
// forward (vision_sam.go/vision_clip.go/vision.go/decoder.go) consumes. Every tensor name/shape
// below was read verbatim off the REAL deepseek-ai/DeepSeek-OCR checkpoint's
// model.safetensors.index.json (2710 tensors, never guessed) before this file was written — see
// each sub-loader's doc comment for its slice of that name list. The tower's own geometry
// (SAM 768/12/12, CLIP 1024/24/16, …) is NOT read from config.json (see VisionConfig's doc
// comment: deepencoder.py's build_sam_vit_b()/build_clip_l() hardcode it) — this loader hardcodes
// the SAME constants, and every read is shape-checked against them, so a checkpoint whose real
// tensors don't match fails loudly here rather than silently misreading.

// weightLoader closes over a checkpoint's tensor map, resolving names verbatim and reporting
// exactly which name was missing or shaped wrong — mirrors whisper's weightLoader
// (arch/openai/whisper/weights.go).
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
		return nil, core.NewError("deepseekvl2.LoadWeights: missing tensor " + name)
	}
	return tensorF32(t.Dtype, t.Data)
}

func (l weightLoader) f32shaped(name string, wantLen int) ([]float32, error) {
	v, err := l.f32req(name)
	if err != nil {
		return nil, err
	}
	if len(v) != wantLen {
		return nil, core.NewError(core.Sprintf("deepseekvl2.LoadWeights: %s has %d elements, want %d", name, len(v), wantLen))
	}
	return v, nil
}

// linearW reads one [out,in] weight (+ optional [out] bias, when hasBias) — the PyTorch/
// safetensors nn.Linear convention every projection in this checkpoint uses.
func (l weightLoader) linearW(prefix string, in, out int, hasBias bool) (w, b []float32, err error) {
	w, err = l.f32shaped(prefix+".weight", out*in)
	if err != nil {
		return nil, nil, err
	}
	if hasBias {
		b, err = l.f32shaped(prefix+".bias", out)
		if err != nil {
			return nil, nil, err
		}
	}
	return w, b, nil
}

// lnBiasW reads one [dim] LayerNorm weight+bias pair (SAM/CLIP's full-LayerNorm shape).
func (l weightLoader) lnBiasW(prefix string, dim int) (w, b []float32, err error) {
	w, err = l.f32shaped(prefix+".weight", dim)
	if err != nil {
		return nil, nil, err
	}
	b, err = l.f32shaped(prefix+".bias", dim)
	if err != nil {
		return nil, nil, err
	}
	return w, b, nil
}

// Weights is a whole loaded DeepSeek-OCR checkpoint: the dual-tower vision encoder (SAM+CLIP),
// the linear projector, the two learned assembly vectors (ImageNewline/ViewSeparator), and the
// MoE decoder — every tensor widened to f32, organised by the shape vision_sam.go/vision_clip.go/
// vision.go/decoder.go's host forward walks directly.
type Weights struct {
	SAM           SAMWeights
	CLIP          CLIPWeights
	ProjW         []float32 // [1280, 2048]
	ProjB         []float32 // [1280]
	ImageNewline  []float32 // [1280] — appended once per grid row (SAM/CLIP's soft-token separator)
	ViewSeparator []float32 // [1280] — appended once at the end of the assembled vision run
	Decoder       DecoderWeights
}

// LoadWeights reads every tensor a DeepSeek-OCR checkpoint carries, widened to f32, against the
// geometry cfg's top-level fields report (decoder side) and the hardcoded tower constants
// (samEmbedDim etc., vision_sam.go/vision_clip.go — see VisionConfig's doc comment for why the
// tower geometry is never config-derived). Never consults model.Assemble/model.LookupArch —
// mamba2/whisper's "own loader" shape (see ocr.go's doc comment).
func LoadWeights(tensors map[string]safetensors.Tensor, cfg *Config) (*Weights, error) {
	if cfg == nil {
		return nil, core.NewError("deepseekvl2.LoadWeights: nil config")
	}
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 || cfg.NumAttentionHeads <= 0 {
		return nil, core.NewError("deepseekvl2.LoadWeights: decoder config geometry is incomplete (hidden_size/num_hidden_layers/vocab_size/num_attention_heads must all be positive)")
	}
	l := weightLoader{tensors: tensors}

	sam, err := loadSAMWeights(l)
	if err != nil {
		return nil, core.E("deepseekvl2.LoadWeights", "SAM tower", err)
	}
	clip, err := loadCLIPWeights(l)
	if err != nil {
		return nil, core.E("deepseekvl2.LoadWeights", "CLIP tower", err)
	}
	projW, projB, err := l.linearW("model.projector.layers", projectorInputDim, cfg.HiddenSize, true)
	if err != nil {
		return nil, core.E("deepseekvl2.LoadWeights", "projector", err)
	}
	imageNewline, err := l.f32shaped("model.image_newline", cfg.HiddenSize)
	if err != nil {
		return nil, core.E("deepseekvl2.LoadWeights", "image_newline", err)
	}
	viewSeparator, err := l.f32shaped("model.view_seperator", cfg.HiddenSize) // sic: the checkpoint's own tensor name misspells "separator"
	if err != nil {
		return nil, core.E("deepseekvl2.LoadWeights", "view_seperator", err)
	}
	dec, err := loadDecoderWeights(l, cfg)
	if err != nil {
		return nil, core.E("deepseekvl2.LoadWeights", "decoder", err)
	}

	return &Weights{
		SAM: sam, CLIP: clip,
		ProjW: projW, ProjB: projB,
		ImageNewline: imageNewline, ViewSeparator: viewSeparator,
		Decoder: dec,
	}, nil
}
