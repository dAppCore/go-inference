// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "math"

// model.go holds the native decode model as it stands TODAY: a bf16/quant FORK (BF16Model vs
// QuantModel). This is the re-roll being collapsed into one shape-neutral model (weights self-describe
// their quant, decode dispatches per-weight through the backend-quant registry). Retired by that collapse.

// BF16Model is a gemma4 model's bf16 weights mapped onto the native structs: the per-layer
// DecodeLayerWeights plus the model-level tensors (embedding, final norm, LM head, and the per-
// layer-input tower for E2B/E4B). LMHead aliases Embed when the checkpoint ties them (Tied).
type BF16Model struct {
	Layers    []DecodeLayerWeights
	Embed     []byte // [vocab × dModel] bf16
	FinalNorm []byte // [dModel] bf16 (model.norm.weight)
	LMHead    []byte // [vocab × dModel] bf16 (lm_head.weight, or Embed when tied)
	Tied      bool   // LMHead is the tied embedding (no separate lm_head.weight)
	// gemma4 per-layer-input tower (E2B/E4B), bf16: the per-layer embedding table, the model-side
	// projection, and its norm. Empty when the model has no PLE tower (12B/26B/31B).
	EmbedPerLayer      []byte // [pliVocab × (nLayers·pliDim)] bf16
	PerLayerModelProjW []byte // [(nLayers·pliDim) × dModel] bf16
	PerLayerProjNormW  []byte // [pliDim] bf16
}

// HasPLE reports whether this model carries the per-layer-input tower (E2B/E4B).
func (g *BF16Model) HasPLE() bool { return g != nil && len(g.EmbedPerLayer) > 0 }

// foldRootSize multiplies a bf16 norm weight by RootSize = dModel^-0.5 (host), matching metal's
// cached Router.ScaleScaled = Scale·RootSize — the gemma4 MoE router norm MoERouterQuant expects
// pre-folded. nil passes through (an absent weight propagates the assembler's error).
func foldRootSize(w []byte, dModel int) []byte {
	if len(w) == 0 {
		return w
	}
	rootSize := float32(math.Pow(float64(dModel), -0.5))
	out := make([]byte, len(w))
	for i := 0; i+1 < len(w); i += 2 {
		h := f32ToBF16(bf16ToF32(w[i], w[i+1]) * rootSize)
		out[i], out[i+1] = byte(h), byte(h>>8)
	}
	return out
}

// QuantModel is a quantised gemma4 model mapped onto the native structs: the quantised decode
// layers plus the model-level tensors. In a 4-bit checkpoint the embedding is itself quantised
// (mlx quantises nn.Embedding) and gemma ties the LM head to it, so Embed/EmbedScales/
// EmbedBiases are the affine triple and LMHead* alias them when tied (the usual gemma4 case).
type QuantModel struct {
	Layers                             []QuantizedLayerWeights
	Embed, EmbedScales, EmbedBiases    []byte // quantised [vocab × dModel] input embedding
	FinalNorm                          []byte // bf16 [dModel] (model.norm.weight)
	LMHead, LMHeadScales, LMHeadBiases []byte // tied embedding, or a separate quant head
	Tied                               bool
	GroupSize, Bits                    int
	// per-layer-input tower (gemma4 E2B/E4B; nil for models without it). The per-layer
	// embedding is 4-bit, the model projection + norm bf16 — fed to PerLayerInputs each token.
	EmbedPerLayer, EmbedPerLayerScales, EmbedPerLayerBiases []byte // quant [vocabPLI × numLayers·pliDim]
	PerLayerModelProjW                                      []byte // [numLayers·pliDim × dModel]: bf16 (regular packs, e2b) OR packed 4-bit (qat packs, e4b)
	PerLayerModelProjScales, PerLayerModelProjBiases        []byte // affine scales/biases when the projection is quantised (e4b); nil ⇒ PerLayerModelProjW is bf16
	PerLayerModelProjGS, PerLayerModelProjBits              int    // affine geometry for the quantised projection
	PerLayerProjNormW                                       []byte // bf16 [pliDim]
}

// HasPLE reports whether this model carries the gemma4 per-layer-input tower.
func (g *QuantModel) HasPLE() bool { return len(g.EmbedPerLayer) > 0 }
