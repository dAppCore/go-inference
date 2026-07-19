// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// weights.go reads a DOTS-OCR checkpoint's safetensors into the flat f32 slices the host forward
// (vision.go/decoder.go) consumes. Every tensor name/shape/bias-presence rule below was read
// verbatim off the REAL rednote-hilab/dots.ocr checkpoint (never guessed) — confirmed against the
// actual 643-tensor name list and per-tensor dtype/shape header before this file was written: the
// text decoder is an unmodified Qwen2ForCausalLM (model.embed_tokens / model.layers.N.* /
// model.norm / a STANDALONE lm_head.weight — tie_word_embeddings is false, unlike whisper's tied
// head), and the vision tower is vision_tower.{patch_embed,blocks.N,post_trunk_norm,merger}.*.
// Widening uses safetensors.DecodeFloatData (BF16/F16/F32 → f32) rather than a hand-rolled
// widener — the checkpoint ships BF16, DecodeFloatData already carries the NEON-accelerated,
// tested conversion this package would otherwise duplicate (see whisper.tensorF32's older,
// package-local widener for the shape this used to take before DecodeFloatData existed).

// LinearWeights is one nn.Linear projection: Weight is [out,in] row-major (the PyTorch/
// safetensors convention), Bias is nil when the checkpoint carries none for that projection.
type LinearWeights struct {
	Weight  []float32
	Bias    []float32
	In, Out int
}

// RMSNormWeights is one RMSNorm's scale-only affine parameter, [dim] (no bias — every norm in
// this architecture except the vision merger's ln_q is RMSNorm).
type RMSNormWeights struct {
	Weight []float32
}

// LayerNormWeights is a standard mean/variance LayerNorm's affine parameters, both [dim]. The
// ONLY LayerNorm in this architecture is the vision PatchMerger's ln_q (modeling_dots_vision.py:
// `LayerNorm(context_dim, eps=1e-6)`, hard-coded eps distinct from the vision tower's RMSNorm eps
// — see VisionWeights.MergerLNEps).
type LayerNormWeights struct {
	Weight []float32
	Bias   []float32
}

// VisionAttnWeights is one DotsVisionBlock's self-attention: QKV is the FUSED [3*dim,dim]
// projection split by output-row range (rows [0,dim)=Q, [dim,2dim)=K, [2dim,3dim)=V — the exact
// slicing VisionAttention.forward's `reshape(seq,3,heads,-1)` implies for a row-major [out,in]
// weight, confirmed against the real qkv.weight shape [4608,1536]=[3*1536,1536]). Bias is nil
// when the checkpoint's use_bias is false (every published DOTS-OCR checkpoint).
type VisionAttnWeights struct {
	Q, K, V, Proj LinearWeights
}

// VisionBlockWeights is one DotsVisionBlock: pre-norm self-attention, pre-norm SwiGLU FFN — both
// RMSNorm. Gate/Up/Down name DotsSwiGLUFFN's fc1/fc3/fc2 respectively (`silu(fc1(x)) * fc3(x)`
// then `fc2(...)`  — fc1 is the silu-gated branch, fc3 is the plain branch, fc2 is the down
// projection; renamed here to match this package's decoder.go MLP naming, not the checkpoint's
// own fc1/fc2/fc3 tensor names, which weights.go still reads verbatim).
type VisionBlockWeights struct {
	Norm1          RMSNormWeights
	Attn           VisionAttnWeights
	Norm2          RMSNormWeights
	Gate, Up, Down LinearWeights
}

// VisionWeights is the whole NaViT-style ViT tower: a linear patch embed (the real Conv2d(patch,
// patch, stride=patch) folds exactly into a dense [embed_dim, channels*temporal*patch*patch]
// projection — verified 0.0 max-abs-diff against the real nn.Conv2d module on real weights, see
// this lane's golden-capture notes) + RMSNorm, N transformer blocks, an optional post-trunk
// RMSNorm, and the PatchMerger (LayerNorm → Linear → GELU → Linear) that projects
// spatial_merge_size²-grouped patches down to the text decoder's hidden width.
type VisionWeights struct {
	PatchEmbed     LinearWeights // [EmbedDim, NumChannels*TemporalPatchSize*PatchSize*PatchSize]
	PatchEmbedNorm RMSNormWeights
	Blocks         []VisionBlockWeights
	PostTrunkNorm  RMSNormWeights // present iff Config.VisionConfig.PostNorm
	MergerLNQ      LayerNormWeights
	MergerFC1      LinearWeights // [HiddenSize*merge², HiddenSize*merge²]
	MergerFC2      LinearWeights // [HiddenSize(text), HiddenSize*merge²]
}

// DecoderLayerWeights is one Qwen2DecoderLayer: pre-norm GQA causal self-attention (q/k/v carry
// bias, o_proj never does — Qwen2Attention hard-codes this in the shipped transformers source
// regardless of config.attention_bias, see Config.AttentionBias's doc comment), pre-norm SwiGLU
// MLP (no bias anywhere in Qwen2MLP).
type DecoderLayerWeights struct {
	InputNorm      RMSNormWeights
	Q, K, V, O     LinearWeights
	PostAttnNorm   RMSNormWeights
	Gate, Up, Down LinearWeights
}

// Weights is a whole loaded DOTS-OCR checkpoint's tensors, widened to f32 and organised by
// vision/decoder stage — the shape vision.go/decoder.go's host forward walks directly.
type Weights struct {
	Vision VisionWeights

	EmbedTokens []float32 // [VocabSize, HiddenSize]
	Layers      []DecoderLayerWeights
	FinalNorm   RMSNormWeights
	LMHead      LinearWeights // [VocabSize, HiddenSize] — a STANDALONE tensor (tie_word_embeddings is false)

	HiddenSize, VocabSize int
	NumAttentionHeads     int
	NumKeyValueHeads      int
}

// weightLoader closes over a checkpoint's tensor map, resolving names verbatim and reporting
// exactly which name was missing or malformed.
type weightLoader struct {
	tensors map[string]safetensors.Tensor
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func (l weightLoader) f32req(name string) ([]float32, error) {
	t, ok := l.tensors[name]
	if !ok {
		return nil, core.NewError("dotsocr.LoadWeights: missing tensor " + name)
	}
	v, err := safetensors.DecodeFloatData(t.Dtype, t.Data, numel(t.Shape))
	if err != nil {
		return nil, core.E("dotsocr.LoadWeights", "decode "+name, err)
	}
	return v, nil
}

// linear reads prefix+".weight" (required, [out,in]) and prefix+".bias" (only when hasBias).
func (l weightLoader) linear(prefix string, in, out int, hasBias bool) (LinearWeights, error) {
	w, err := l.f32req(prefix + ".weight")
	if err != nil {
		return LinearWeights{}, err
	}
	if len(w) != in*out {
		return LinearWeights{}, core.NewError(core.Sprintf("dotsocr.LoadWeights: %s.weight has %d elements, want %d (%d×%d)", prefix, len(w), in*out, out, in))
	}
	lw := LinearWeights{Weight: w, In: in, Out: out}
	if hasBias {
		b, err := l.f32req(prefix + ".bias")
		if err != nil {
			return LinearWeights{}, err
		}
		if len(b) != out {
			return LinearWeights{}, core.NewError(core.Sprintf("dotsocr.LoadWeights: %s.bias has %d elements, want %d", prefix, len(b), out))
		}
		lw.Bias = b
	}
	return lw, nil
}

// splitLinear carves LinearWeights for the fused QKV row-range [outOffset, outOffset+outRows)
// out of a single [3*dim,dim] projection already read whole — see VisionAttnWeights' doc comment
// for why this is the correct split (no bias: DOTS-OCR's vision qkv is use_bias=false).
func splitLinear(fused []float32, in, outOffset, outRows int) LinearWeights {
	start := outOffset * in
	return LinearWeights{Weight: fused[start : start+outRows*in], In: in, Out: outRows}
}

func (l weightLoader) rmsNorm(name string, dim int) (RMSNormWeights, error) {
	w, err := l.f32req(name)
	if err != nil {
		return RMSNormWeights{}, err
	}
	if len(w) != dim {
		return RMSNormWeights{}, core.NewError(core.Sprintf("dotsocr.LoadWeights: %s has %d elements, want %d", name, len(w), dim))
	}
	return RMSNormWeights{Weight: w}, nil
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
		return LayerNormWeights{}, core.NewError(core.Sprintf("dotsocr.LoadWeights: %s has %d/%d elements, want %d", prefix, len(w), len(b), dim))
	}
	return LayerNormWeights{Weight: w, Bias: b}, nil
}

// loadVisionBlock reads one DotsVisionBlock: fused qkv split into Q/K/V by row range, proj, two
// pre-norms (RMSNorm), and the SwiGLU FFN (fc1/fc3 gate+up, fc2 down).
func (l weightLoader) loadVisionBlock(i int, vc *VisionConfig) (VisionBlockWeights, error) {
	p := core.Sprintf("vision_tower.blocks.%d", i)
	d := vc.EmbedDim
	hasBias := vc.UseBias

	qkvW, err := l.f32req(p + ".attn.qkv.weight")
	if err != nil {
		return VisionBlockWeights{}, err
	}
	if len(qkvW) != 3*d*d {
		return VisionBlockWeights{}, core.NewError(core.Sprintf("dotsocr.LoadWeights: %s.attn.qkv.weight has %d elements, want %d (3×%d×%d)", p, len(qkvW), 3*d*d, d, d))
	}
	attn := VisionAttnWeights{
		Q: splitLinear(qkvW, d, 0, d),
		K: splitLinear(qkvW, d, d, d),
		V: splitLinear(qkvW, d, 2*d, d),
	}
	if hasBias {
		qkvB, err := l.f32req(p + ".attn.qkv.bias")
		if err != nil {
			return VisionBlockWeights{}, err
		}
		if len(qkvB) != 3*d {
			return VisionBlockWeights{}, core.NewError(core.Sprintf("dotsocr.LoadWeights: %s.attn.qkv.bias has %d elements, want %d", p, len(qkvB), 3*d))
		}
		attn.Q.Bias, attn.K.Bias, attn.V.Bias = qkvB[0:d], qkvB[d:2*d], qkvB[2*d:3*d]
	}
	proj, err := l.linear(p+".attn.proj", d, d, hasBias)
	if err != nil {
		return VisionBlockWeights{}, err
	}
	attn.Proj = proj

	norm1, err := l.rmsNorm(p+".norm1.weight", d)
	if err != nil {
		return VisionBlockWeights{}, err
	}
	norm2, err := l.rmsNorm(p+".norm2.weight", d)
	if err != nil {
		return VisionBlockWeights{}, err
	}
	gate, err := l.linear(p+".mlp.fc1", d, vc.IntermediateSize, hasBias)
	if err != nil {
		return VisionBlockWeights{}, err
	}
	up, err := l.linear(p+".mlp.fc3", d, vc.IntermediateSize, hasBias)
	if err != nil {
		return VisionBlockWeights{}, err
	}
	down, err := l.linear(p+".mlp.fc2", vc.IntermediateSize, d, hasBias)
	if err != nil {
		return VisionBlockWeights{}, err
	}
	return VisionBlockWeights{Norm1: norm1, Attn: attn, Norm2: norm2, Gate: gate, Up: up, Down: down}, nil
}

// loadVision reads the whole vision_tower.* subtree.
func (l weightLoader) loadVision(vc *VisionConfig) (VisionWeights, error) {
	if vc == nil {
		return VisionWeights{}, core.NewError("dotsocr.LoadWeights: nil vision_config")
	}
	patchDim := vc.NumChannels * vc.TemporalPatchSize * vc.PatchSize * vc.PatchSize
	// The real Conv2d(patch_size,stride=patch_size) IS a dense [EmbedDim,patchDim] projection —
	// its [outC,inC,kh,kw] weight already flattens row-major to exactly that shape (verified
	// against the real module, see this package's doc comment).
	patchEmbed, err := l.linear("vision_tower.patch_embed.patchifier.proj", patchDim, vc.EmbedDim, true)
	if err != nil {
		return VisionWeights{}, err
	}
	patchEmbedNorm, err := l.rmsNorm("vision_tower.patch_embed.patchifier.norm.weight", vc.EmbedDim)
	if err != nil {
		return VisionWeights{}, err
	}
	blocks := make([]VisionBlockWeights, vc.NumHiddenLayers)
	for i := range blocks {
		b, err := l.loadVisionBlock(i, vc)
		if err != nil {
			return VisionWeights{}, err
		}
		blocks[i] = b
	}
	var postNorm RMSNormWeights
	if vc.PostNorm {
		postNorm, err = l.rmsNorm("vision_tower.post_trunk_norm.weight", vc.EmbedDim)
		if err != nil {
			return VisionWeights{}, err
		}
	}
	mergerDim := vc.EmbedDim * vc.SpatialMergeSize * vc.SpatialMergeSize
	lnq, err := l.layerNorm("vision_tower.merger.ln_q", vc.EmbedDim)
	if err != nil {
		return VisionWeights{}, err
	}
	fc1, err := l.linear("vision_tower.merger.mlp.0", mergerDim, mergerDim, true)
	if err != nil {
		return VisionWeights{}, err
	}
	fc2, err := l.linear("vision_tower.merger.mlp.2", mergerDim, vc.HiddenSize, true)
	if err != nil {
		return VisionWeights{}, err
	}
	return VisionWeights{
		PatchEmbed: patchEmbed, PatchEmbedNorm: patchEmbedNorm,
		Blocks: blocks, PostTrunkNorm: postNorm,
		MergerLNQ: lnq, MergerFC1: fc1, MergerFC2: fc2,
	}, nil
}

// loadDecoderLayer reads one Qwen2DecoderLayer: q/k/v carry bias (hard-coded true in the real
// Qwen2Attention regardless of config.attention_bias), o_proj and the whole MLP never do.
func (l weightLoader) loadDecoderLayer(i int, cfg *Config) (DecoderLayerWeights, error) {
	p := core.Sprintf("model.layers.%d", i)
	d := cfg.HiddenSize
	headDim := d / cfg.NumAttentionHeads
	kvDim := cfg.NumKeyValueHeads * headDim

	inputNorm, err := l.rmsNorm(p+".input_layernorm.weight", d)
	if err != nil {
		return DecoderLayerWeights{}, err
	}
	q, err := l.linear(p+".self_attn.q_proj", d, d, true)
	if err != nil {
		return DecoderLayerWeights{}, err
	}
	k, err := l.linear(p+".self_attn.k_proj", d, kvDim, true)
	if err != nil {
		return DecoderLayerWeights{}, err
	}
	v, err := l.linear(p+".self_attn.v_proj", d, kvDim, true)
	if err != nil {
		return DecoderLayerWeights{}, err
	}
	o, err := l.linear(p+".self_attn.o_proj", d, d, false)
	if err != nil {
		return DecoderLayerWeights{}, err
	}
	postAttnNorm, err := l.rmsNorm(p+".post_attention_layernorm.weight", d)
	if err != nil {
		return DecoderLayerWeights{}, err
	}
	gate, err := l.linear(p+".mlp.gate_proj", d, cfg.IntermediateSize, false)
	if err != nil {
		return DecoderLayerWeights{}, err
	}
	up, err := l.linear(p+".mlp.up_proj", d, cfg.IntermediateSize, false)
	if err != nil {
		return DecoderLayerWeights{}, err
	}
	down, err := l.linear(p+".mlp.down_proj", cfg.IntermediateSize, d, false)
	if err != nil {
		return DecoderLayerWeights{}, err
	}
	return DecoderLayerWeights{
		InputNorm: inputNorm, Q: q, K: k, V: v, O: o,
		PostAttnNorm: postAttnNorm, Gate: gate, Up: up, Down: down,
	}, nil
}

// LoadWeights reads every tensor a DOTS-OCR checkpoint carries, widened to f32, against the
// geometry cfg/cfg.VisionConfig report. Never consults model.Assemble/model.LookupArch — this
// package's Composed/Arch hooks (register.go/config.go) keep refusing unconditionally; OCR runs
// through this package's OWN loader (Load, ocr.go), mirroring whisper.LoadWeights/
// mamba2.LoadMambaModel's "own loader, own session" shape.
func LoadWeights(tensors map[string]safetensors.Tensor, cfg *Config) (*Weights, error) {
	if cfg == nil {
		return nil, core.NewError("dotsocr.LoadWeights: nil config")
	}
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 || cfg.VocabSize <= 0 {
		return nil, core.NewError("dotsocr.LoadWeights: config geometry is incomplete (hidden_size/num_hidden_layers/num_attention_heads/num_key_value_heads/vocab_size must all be positive)")
	}
	if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return nil, core.NewError("dotsocr.LoadWeights: hidden_size must divide by num_attention_heads")
	}
	vc := cfg.VisionConfig
	if vc == nil || vc.EmbedDim <= 0 || vc.NumHiddenLayers <= 0 || vc.NumAttentionHeads <= 0 || vc.PatchSize <= 0 || vc.SpatialMergeSize <= 0 {
		return nil, core.NewError("dotsocr.LoadWeights: vision_config geometry is incomplete (embed_dim/num_hidden_layers/num_attention_heads/patch_size/spatial_merge_size must all be positive)")
	}
	l := weightLoader{tensors: tensors}

	vision, err := l.loadVision(vc)
	if err != nil {
		return nil, err
	}

	embedTokens, err := l.f32req("model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	if len(embedTokens) != cfg.VocabSize*cfg.HiddenSize {
		return nil, core.NewError(core.Sprintf("dotsocr.LoadWeights: model.embed_tokens.weight has %d elements, want %d (%d×%d)", len(embedTokens), cfg.VocabSize*cfg.HiddenSize, cfg.VocabSize, cfg.HiddenSize))
	}

	layers := make([]DecoderLayerWeights, cfg.NumHiddenLayers)
	for i := range layers {
		dl, err := l.loadDecoderLayer(i, cfg)
		if err != nil {
			return nil, err
		}
		layers[i] = dl
	}
	finalNorm, err := l.rmsNorm("model.norm.weight", cfg.HiddenSize)
	if err != nil {
		return nil, err
	}

	var lmHead LinearWeights
	if cfg.TieWordEmbeddings {
		lmHead = LinearWeights{Weight: embedTokens, In: cfg.HiddenSize, Out: cfg.VocabSize}
	} else {
		lmHead, err = l.linear("lm_head", cfg.HiddenSize, cfg.VocabSize, false)
		if err != nil {
			return nil, err
		}
	}

	return &Weights{
		Vision:      vision,
		EmbedTokens: embedTokens,
		Layers:      layers,
		FinalNorm:   finalNorm,
		LMHead:      lmHead,
		HiddenSize:  cfg.HiddenSize, VocabSize: cfg.VocabSize,
		NumAttentionHeads: cfg.NumAttentionHeads, NumKeyValueHeads: cfg.NumKeyValueHeads,
	}, nil
}
