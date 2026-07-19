// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// weights.go reads a REAL zai-org/GLM-OCR checkpoint's safetensors into the flat f32 slices
// vision.go/textdecoder.go's host forward consumes. Every tensor name, prefix, and bias-
// presence rule below was confirmed against the real model.safetensors's 526-entry name list
// (never guessed) — see docs referenced from register.go. Two things this loader deliberately
// never reads: model.language_model.layers.16.* (the num_nextn_predict_layers=1 MTP layer —
// transformers' own GlmOcrPreTrainedModel._keys_to_ignore_on_load_unexpected names exactly this
// pattern; this package's bounded `for i := range TextConfig.NumHiddenLayers` loop never reaches
// index 16) and lm_head is loaded as its OWN tensor (tie_word_embeddings is false in every
// published config.json — confirmed a separate "lm_head.weight" entry exists, distinct from
// "model.language_model.embed_tokens.weight").
//
// Splitting: the text MLP's gate_up_proj and the vision attention's fused qkv are each decoded
// to f32 ONCE, then row-sliced (never re-decoded) into their Gate/Up or Q/K/V parts — PyTorch's
// `.chunk(2, dim=-1)` on gate_up_proj's OUTPUT features is exactly "first half of output ROWS is
// gate, second half is up" for a [out,in] row-major Linear weight, and the vision qkv
// projection's `.reshape(seq,3,heads,-1)` is exactly "first `hidden` output rows are every
// head's Q, next `hidden` rows K, next `hidden` rows V" for the same reason.

// LinearWeights is one nn.Linear projection: Weight is [Out,In] row-major (the PyTorch/
// safetensors convention), Bias is nil when the checkpoint carries none.
type LinearWeights struct {
	Weight  []float32
	Bias    []float32
	In, Out int
}

// RMSNormWeights is one GlmOcrRMSNorm's scale — no bias (T5LayerNorm-style, per the reference).
type RMSNormWeights struct {
	Weight []float32
}

// LayerNormWeights is one nn.LayerNorm's affine parameters (both [dim]) — used only by the
// vision patch merger's post_projection_norm.
type LayerNormWeights struct {
	Weight, Bias []float32
}

// VisionAttnWeights is one GlmOcrVisionAttention block: Q/K/V (sliced from the checkpoint's
// fused qkv.weight/qkv.bias, all three WITH bias — attention_bias is true for every published
// GLM-OCR vision_config), Proj (with bias), and the per-head RMSNorm applied to Q/K before rope
// (QNorm/KNorm, each length headDim).
type VisionAttnWeights struct {
	Q, K, V, Proj LinearWeights
	QNorm, KNorm  RMSNormWeights
}

// VisionMLPWeights is one GlmOcrVisionMlp: separate gate_proj/up_proj/down_proj (NOT fused,
// unlike the text decoder's MLP), all three WITH bias (attention_bias mirrors into this MLP's
// bias flag per the reference's GlmOcrVisionBlock construction).
type VisionMLPWeights struct {
	Gate, Up, Down LinearWeights
}

// VisionBlockWeights is one GlmOcrVisionBlock: pre-RMSNorm self-attention, pre-RMSNorm MLP.
type VisionBlockWeights struct {
	Norm1, Norm2 RMSNormWeights
	Attn         VisionAttnWeights
	MLP          VisionMLPWeights
}

// VisionMergerWeights is GlmOcrVisionPatchMerger: proj (no bias) → post_projection_norm
// (nn.LayerNorm, default eps — WITH bias) → GELU → a SwiGLU-shaped gate/up/down (no bias),
// projecting the downsampled vision hidden state into the text decoder's embedding space.
type VisionMergerWeights struct {
	Proj               LinearWeights
	PostProjectionNorm LayerNormWeights
	Gate, Up, Down     LinearWeights
}

// VisionWeights is a whole loaded GLM-OCR vision tower.
type VisionWeights struct {
	PatchEmbed    LinearWeights // Conv3d flattened to Linear[hidden, in_channels*temporal*patch*patch]
	Blocks        []VisionBlockWeights
	PostLayernorm RMSNormWeights
	Downsample    LinearWeights // Conv2d flattened to Linear[out_hidden, hidden*merge*merge]
	Merger        VisionMergerWeights
}

// TextAttnWeights is one GlmOcrTextAttention: Q/K/V/O, all WITHOUT bias (hardcoded bias=False
// in the reference's modular override, matching attention_bias:false in every published
// text_config).
type TextAttnWeights struct {
	Q, K, V, O LinearWeights
}

// TextMLPWeights is one GlmOcrTextMLP: gate/up (sliced from the checkpoint's fused
// gate_up_proj.weight — see the file doc comment) and down, all WITHOUT bias.
type TextMLPWeights struct {
	Gate, Up, Down LinearWeights
}

// TextLayerWeights is one GlmOcrTextDecoderLayer: the GLM-4 sandwich-norm shape — input_norm
// pre-attention, post_self_attn_norm on the attention branch BEFORE its residual add,
// post_attn_norm pre-MLP (despite the name — it runs after attention's residual, before the
// MLP), post_mlp_norm on the MLP branch before ITS residual add.
type TextLayerWeights struct {
	InputNorm, PostAttnNorm, PostSelfAttnNorm, PostMLPNorm RMSNormWeights
	Attn                                                   TextAttnWeights
	MLP                                                    TextMLPWeights
}

// TextWeights is a whole loaded GLM-OCR text decoder. LMHead is a SEPARATE tensor from
// EmbedTokens (tie_word_embeddings is false — see the file doc comment), so both are always
// widened and kept independently even though they share [VocabSize,HiddenSize].
type TextWeights struct {
	EmbedTokens []float32
	Layers      []TextLayerWeights
	FinalNorm   RMSNormWeights
	LMHead      LinearWeights
}

// Weights is a whole loaded GLM-OCR checkpoint (vision tower + text decoder), ready for
// vision.VisionForward / textdecoder.TextForward.
type Weights struct {
	Vision VisionWeights
	Text   TextWeights
}

// weightLoader closes over a checkpoint's tensor map, resolving names and reporting exactly
// which name (or which shape) was wrong — mirrors whisper.weightLoader (arch/openai/whisper/
// weights.go), but widens through the shared safetensors.DecodeFloat32 (F32/F16/BF16/F64) rather
// than a private per-package dtype-conversion copy, since GLM-OCR's checkpoint ships BF16 and
// that decode path is already shared, tested library code.
type weightLoader struct {
	tensors map[string]safetensors.Tensor
}

func numElements(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// f32req decodes tensor name to a flat f32 slice, requiring it to carry exactly `expected`
// elements — a shape mismatch is reported by name, not a downstream index panic.
func (l weightLoader) f32req(name string, expected int) ([]float32, error) {
	t, ok := l.tensors[name]
	if !ok {
		return nil, core.NewError("glmocr.LoadWeights: missing tensor " + name)
	}
	n := numElements(t.Shape)
	if n != expected {
		return nil, core.NewError(core.Sprintf("glmocr.LoadWeights: %s has %d elements, want %d", name, n, expected))
	}
	return safetensors.DecodeFloat32(t.Dtype, t.Data, n)
}

func (l weightLoader) linear(prefix string, in, out int, hasBias bool) (LinearWeights, error) {
	w, err := l.f32req(prefix+".weight", out*in)
	if err != nil {
		return LinearWeights{}, err
	}
	lw := LinearWeights{Weight: w, In: in, Out: out}
	if hasBias {
		b, err := l.f32req(prefix+".bias", out)
		if err != nil {
			return LinearWeights{}, err
		}
		lw.Bias = b
	}
	return lw, nil
}

func (l weightLoader) rmsNorm(prefix string, dim int) (RMSNormWeights, error) {
	w, err := l.f32req(prefix+".weight", dim)
	if err != nil {
		return RMSNormWeights{}, err
	}
	return RMSNormWeights{Weight: w}, nil
}

func (l weightLoader) layerNorm(prefix string, dim int) (LayerNormWeights, error) {
	w, err := l.f32req(prefix+".weight", dim)
	if err != nil {
		return LayerNormWeights{}, err
	}
	b, err := l.f32req(prefix+".bias", dim)
	if err != nil {
		return LayerNormWeights{}, err
	}
	return LayerNormWeights{Weight: w, Bias: b}, nil
}

// visionAttn splits the checkpoint's fused qkv.weight/qkv.bias [3*hidden,hidden]/[3*hidden] into
// Q/K/V — GlmOcrVisionAttention.forward's `.reshape(seq,3,heads,-1)` makes output rows
// [0,hidden) every head's Q, [hidden,2*hidden) K, [2*hidden,3*hidden) V (see the file doc
// comment); row-slicing the once-decoded f32 tensor reproduces that split exactly.
func (l weightLoader) visionAttn(prefix string, hidden, headDim int) (VisionAttnWeights, error) {
	qkvW, err := l.f32req(prefix+".qkv.weight", 3*hidden*hidden)
	if err != nil {
		return VisionAttnWeights{}, err
	}
	qkvB, err := l.f32req(prefix+".qkv.bias", 3*hidden)
	if err != nil {
		return VisionAttnWeights{}, err
	}
	q := LinearWeights{Weight: qkvW[0 : hidden*hidden], Bias: qkvB[0:hidden], In: hidden, Out: hidden}
	k := LinearWeights{Weight: qkvW[hidden*hidden : 2*hidden*hidden], Bias: qkvB[hidden : 2*hidden], In: hidden, Out: hidden}
	v := LinearWeights{Weight: qkvW[2*hidden*hidden : 3*hidden*hidden], Bias: qkvB[2*hidden : 3*hidden], In: hidden, Out: hidden}
	proj, err := l.linear(prefix+".proj", hidden, hidden, true)
	if err != nil {
		return VisionAttnWeights{}, err
	}
	qNorm, err := l.rmsNorm(prefix+".q_norm", headDim)
	if err != nil {
		return VisionAttnWeights{}, err
	}
	kNorm, err := l.rmsNorm(prefix+".k_norm", headDim)
	if err != nil {
		return VisionAttnWeights{}, err
	}
	return VisionAttnWeights{Q: q, K: k, V: v, Proj: proj, QNorm: qNorm, KNorm: kNorm}, nil
}

func (l weightLoader) visionMLP(prefix string, hidden, ff int) (VisionMLPWeights, error) {
	gate, err := l.linear(prefix+".gate_proj", hidden, ff, true)
	if err != nil {
		return VisionMLPWeights{}, err
	}
	up, err := l.linear(prefix+".up_proj", hidden, ff, true)
	if err != nil {
		return VisionMLPWeights{}, err
	}
	down, err := l.linear(prefix+".down_proj", ff, hidden, true)
	if err != nil {
		return VisionMLPWeights{}, err
	}
	return VisionMLPWeights{Gate: gate, Up: up, Down: down}, nil
}

func (l weightLoader) visionBlock(prefix string, hidden, ff, headDim int) (VisionBlockWeights, error) {
	norm1, err := l.rmsNorm(prefix+".norm1", hidden)
	if err != nil {
		return VisionBlockWeights{}, err
	}
	norm2, err := l.rmsNorm(prefix+".norm2", hidden)
	if err != nil {
		return VisionBlockWeights{}, err
	}
	attnW, err := l.visionAttn(prefix+".attn", hidden, headDim)
	if err != nil {
		return VisionBlockWeights{}, err
	}
	mlpW, err := l.visionMLP(prefix+".mlp", hidden, ff)
	if err != nil {
		return VisionBlockWeights{}, err
	}
	return VisionBlockWeights{Norm1: norm1, Norm2: norm2, Attn: attnW, MLP: mlpW}, nil
}

func (l weightLoader) visionMerger(prefix string, dim, contextDim int) (VisionMergerWeights, error) {
	proj, err := l.linear(prefix+".proj", dim, dim, false)
	if err != nil {
		return VisionMergerWeights{}, err
	}
	ln, err := l.layerNorm(prefix+".post_projection_norm", dim)
	if err != nil {
		return VisionMergerWeights{}, err
	}
	gate, err := l.linear(prefix+".gate_proj", dim, contextDim, false)
	if err != nil {
		return VisionMergerWeights{}, err
	}
	up, err := l.linear(prefix+".up_proj", dim, contextDim, false)
	if err != nil {
		return VisionMergerWeights{}, err
	}
	down, err := l.linear(prefix+".down_proj", contextDim, dim, false)
	if err != nil {
		return VisionMergerWeights{}, err
	}
	return VisionMergerWeights{Proj: proj, PostProjectionNorm: ln, Gate: gate, Up: up, Down: down}, nil
}

func (l weightLoader) textAttn(prefix string, hidden, heads, kvHeads, headDim int) (TextAttnWeights, error) {
	q, err := l.linear(prefix+".q_proj", hidden, heads*headDim, false)
	if err != nil {
		return TextAttnWeights{}, err
	}
	k, err := l.linear(prefix+".k_proj", hidden, kvHeads*headDim, false)
	if err != nil {
		return TextAttnWeights{}, err
	}
	v, err := l.linear(prefix+".v_proj", hidden, kvHeads*headDim, false)
	if err != nil {
		return TextAttnWeights{}, err
	}
	o, err := l.linear(prefix+".o_proj", heads*headDim, hidden, false)
	if err != nil {
		return TextAttnWeights{}, err
	}
	return TextAttnWeights{Q: q, K: k, V: v, O: o}, nil
}

// textMLP splits the checkpoint's fused gate_up_proj.weight [2*ff,hidden] into Gate/Up — see
// the file doc comment for why row-slicing the once-decoded f32 tensor is exact.
func (l weightLoader) textMLP(prefix string, hidden, ff int) (TextMLPWeights, error) {
	fused, err := l.f32req(prefix+".gate_up_proj.weight", 2*ff*hidden)
	if err != nil {
		return TextMLPWeights{}, err
	}
	gate := LinearWeights{Weight: fused[0 : ff*hidden], In: hidden, Out: ff}
	up := LinearWeights{Weight: fused[ff*hidden : 2*ff*hidden], In: hidden, Out: ff}
	down, err := l.linear(prefix+".down_proj", ff, hidden, false)
	if err != nil {
		return TextMLPWeights{}, err
	}
	return TextMLPWeights{Gate: gate, Up: up, Down: down}, nil
}

func (l weightLoader) textLayer(prefix string, hidden, ff, heads, kvHeads, headDim int) (TextLayerWeights, error) {
	inputNorm, err := l.rmsNorm(prefix+".input_layernorm", hidden)
	if err != nil {
		return TextLayerWeights{}, err
	}
	postAttnNorm, err := l.rmsNorm(prefix+".post_attention_layernorm", hidden)
	if err != nil {
		return TextLayerWeights{}, err
	}
	postSelfAttnNorm, err := l.rmsNorm(prefix+".post_self_attn_layernorm", hidden)
	if err != nil {
		return TextLayerWeights{}, err
	}
	postMLPNorm, err := l.rmsNorm(prefix+".post_mlp_layernorm", hidden)
	if err != nil {
		return TextLayerWeights{}, err
	}
	attnW, err := l.textAttn(prefix+".self_attn", hidden, heads, kvHeads, headDim)
	if err != nil {
		return TextLayerWeights{}, err
	}
	mlpW, err := l.textMLP(prefix+".mlp", hidden, ff)
	if err != nil {
		return TextLayerWeights{}, err
	}
	return TextLayerWeights{
		InputNorm: inputNorm, PostAttnNorm: postAttnNorm, PostSelfAttnNorm: postSelfAttnNorm, PostMLPNorm: postMLPNorm,
		Attn: attnW, MLP: mlpW,
	}, nil
}

// LoadWeights reads every tensor a GlmOcrForConditionalGeneration checkpoint carries (except
// the MTP layer — see the file doc comment), widened to f32, against the geometry cfg declares.
func LoadWeights(tensors map[string]safetensors.Tensor, cfg *Config) (*Weights, error) {
	if cfg == nil || cfg.TextConfig == nil || cfg.VisionConfig == nil {
		return nil, core.NewError("glmocr.LoadWeights: nil config/text_config/vision_config")
	}
	tc, vc := cfg.TextConfig, cfg.VisionConfig
	if vc.HiddenSize <= 0 || vc.Depth <= 0 || vc.NumHeads <= 0 || vc.PatchSize <= 0 || vc.TemporalPatchSize <= 0 ||
		vc.SpatialMergeSize <= 0 || vc.OutHiddenSize <= 0 || vc.IntermediateSize <= 0 || vc.InChannels <= 0 {
		return nil, core.NewError("glmocr.LoadWeights: vision_config geometry is incomplete")
	}
	if tc.HiddenSize <= 0 || tc.IntermediateSize <= 0 || tc.NumHiddenLayers <= 0 || tc.NumAttentionHeads <= 0 ||
		tc.NumKeyValueHeads <= 0 || tc.HeadDim <= 0 || tc.VocabSize <= 0 {
		return nil, core.NewError("glmocr.LoadWeights: text_config geometry is incomplete")
	}
	if vc.HiddenSize%vc.NumHeads != 0 {
		return nil, core.NewError("glmocr.LoadWeights: vision hidden_size not divisible by num_heads")
	}
	l := weightLoader{tensors: tensors}
	visHeadDim := vc.HiddenSize / vc.NumHeads

	patchDim := vc.InChannels * vc.TemporalPatchSize * vc.PatchSize * vc.PatchSize
	patchEmbed, err := l.linear("model.visual.patch_embed.proj", patchDim, vc.HiddenSize, true)
	if err != nil {
		return nil, err
	}
	blocks := make([]VisionBlockWeights, vc.Depth)
	for i := range blocks {
		prefix := core.Sprintf("model.visual.blocks.%d", i)
		blocks[i], err = l.visionBlock(prefix, vc.HiddenSize, vc.IntermediateSize, visHeadDim)
		if err != nil {
			return nil, err
		}
	}
	postLayernorm, err := l.rmsNorm("model.visual.post_layernorm", vc.HiddenSize)
	if err != nil {
		return nil, err
	}
	downDim := vc.HiddenSize * vc.SpatialMergeSize * vc.SpatialMergeSize
	downsample, err := l.linear("model.visual.downsample", downDim, vc.OutHiddenSize, true)
	if err != nil {
		return nil, err
	}
	merger, err := l.visionMerger("model.visual.merger", vc.OutHiddenSize, vc.OutHiddenSize*vc.InChannels)
	if err != nil {
		return nil, err
	}

	embedTokens, err := l.f32req("model.language_model.embed_tokens.weight", tc.VocabSize*tc.HiddenSize)
	if err != nil {
		return nil, err
	}
	layers := make([]TextLayerWeights, tc.NumHiddenLayers)
	for i := range layers {
		prefix := core.Sprintf("model.language_model.layers.%d", i)
		layers[i], err = l.textLayer(prefix, tc.HiddenSize, tc.IntermediateSize, tc.NumAttentionHeads, tc.NumKeyValueHeads, tc.HeadDim)
		if err != nil {
			return nil, err
		}
	}
	finalNorm, err := l.rmsNorm("model.language_model.norm", tc.HiddenSize)
	if err != nil {
		return nil, err
	}
	lmHead, err := l.linear("lm_head", tc.HiddenSize, tc.VocabSize, false)
	if err != nil {
		return nil, err
	}

	return &Weights{
		Vision: VisionWeights{
			PatchEmbed: patchEmbed, Blocks: blocks, PostLayernorm: postLayernorm,
			Downsample: downsample, Merger: merger,
		},
		Text: TextWeights{
			EmbedTokens: embedTokens, Layers: layers, FinalNorm: finalNorm, LMHead: lmHead,
		},
	}, nil
}
