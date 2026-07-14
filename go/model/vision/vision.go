// SPDX-Licence-Identifier: EUPL-1.2

// Package vision holds the engine-neutral, pure-data vision payload a model package's assembler
// produces and a backend uploads. It carries no model.Linear (vision weights use this package's own
// Linear byte-view type), so it is a LEAF: it imports nothing from the model root, and the model root
// referencing these types (LoadedModel.Vision, ArchSpec.Vision) is a one-way dependency, never a cycle.
// The arch-specific assemblers (gemma4.AssembleVision etc.) live in their model package and import
// these types from here.
package vision

// Linear is one vision linear's weight plus optional affine-quant
// metadata and additive bias.
type Linear struct {
	Weight         []byte
	Scales, Biases []byte
	Bias           []byte
	OutDim, InDim  int
	GroupSize      int
	Bits           int
	Kind           string
}

// Layer is one vision encoder layer's weights.
type Layer struct {
	InputNorm, PostAttnNorm, PreFFNorm, PostFFNorm []byte
	Q, K, V, O                                     Linear
	QNorm, KNorm                                   []byte
	Gate, Up, Down                                 Linear
}

// Projector is the vision-to-text projector.
type Projector struct {
	Projection, Linear1, Linear2 Linear
}

// Config is the engine-neutral vision tower geometry.
type Config struct {
	Hidden                int
	PatchDim              int
	NumLayers             int
	NumHeads              int
	NumKVHeads            int
	HeadDim               int
	PatchSize             int
	NumChannels           int
	GridH                 int
	GridW                 int
	PositionEmbeddingSize int
	RopeBase              float32
	RMSNormEps            float32
	PoolKernel            int
	Standardize           bool
	EmbeddingScale        float32
	ImageTokenID          int32
	ImageBeginToken       string
	ImageToken            string
	ImageEndToken         string
	VideoTokenID          int32
	VideoToken            string
}

// Loaded is the neutral vision payload a backend can upload/build.
type Loaded struct {
	PatchEmbedding     []byte
	PatchConvWeight    []byte
	PositionEmbeddings []byte
	PostLayernorm      []byte
	StdBias, StdScale  []byte
	Layers             []Layer
	Projector          Projector
	Cfg                Config
}

// UnifiedConfig is the encoder-free (gemma4_unified) vision
// geometry: raw model patches project straight into the backbone with no
// encoder tower.
type UnifiedConfig struct {
	MMEmbedDim      int     // multimodal embed dim (the patch stage's width)
	TextHidden      int     // backbone hidden the projection lands in
	PosembSize      int     // pos_embedding positions per spatial axis
	PatchSize       int     // teacher patch pixels (16)
	ModelPatchSize  int     // pooled model patch pixels (48 = PoolKernel·PatchSize)
	PoolKernel      int     // teacher patches per model-patch side (3)
	MaxSoftTokens   int     // soft-token budget per image (280)
	LayerNormEps    float32 // patch_ln1/patch_ln2/pos_norm epsilon
	RMSNormEps      float32 // the scale-free pre-projection RMSNorm epsilon
	ImageTokenID    int32
	ImageBeginToken string
	ImageToken      string
	ImageEndToken   string
	// Video rides the SAME embedder: each sampled frame patchifies like an
	// image and splices at the video placeholder, one timestamped block per
	// frame (the reference's "MM:SS {boi}{video_token×N}{eoi}" join).
	VideoTokenID int32
	VideoToken   string
	// BidirectionalImages: the text config declares
	// use_bidirectional_attention == "vision" — image soft-token spans attend
	// bidirectionally in prefill (causal evaluation misreads large images).
	BidirectionalImages bool

	// Audio (encoder-free): each audio token is AudioSamplesPerToken raw
	// 16 kHz waveform samples projected straight into the backbone — no mel
	// front-end, no Conformer. Zero AudioSamplesPerToken ⇒ the pack carries
	// no unified audio head.
	AudioSamplesPerToken int
	AudioTokenID         int32
	AudioBeginToken      string
	AudioToken           string
	AudioEndToken        string
}

// Unified is the encoder-free vision payload (gemma4_unified):
// LayerNorm → patch dense (+bias) → LayerNorm → factorised per-axis position
// add → LayerNorm → scale-free RMSNorm → projection. The LayerNorms carry
// weight+bias; the pre-projection RMSNorm has NO parameters (with_scale=False
// upstream), so only its epsilon lives in the config.
type Unified struct {
	PatchLN1W, PatchLN1B []byte
	PatchDense           Linear // [MMEmbedDim × ModelPatchSize²·3] + Bias
	PatchLN2W, PatchLN2B []byte
	PosEmbedding         []byte // [PosembSize, 2, MMEmbedDim] bf16 (axis 0 = row, 1 = col)
	PosNormW, PosNormB   []byte
	Projection           Linear // embed_vision [TextHidden × MMEmbedDim], no bias
	// AudioProjection is the unified audio head — embed_audio
	// [TextHidden × AudioSamplesPerToken], no bias; nil Weight when absent.
	AudioProjection Linear
	Cfg             UnifiedConfig
}
