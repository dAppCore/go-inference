// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// loaded.go is the neutral loaded-weights set: the single hand-off between a model package's weight
// parsing and a backend's device upload (pkg/native, future go-rocm).
// It lives at the pkg/model ROOT, not a model subpackage — a LoadedModel is what EVERY arch produces,
// so a model-named home would force every backend + every other model to import that one model for a
// neutral type. The arch-specific fields (QK-norm, layer-scalar, the PLE tower, MoE) are optional:
// archs without them leave them nil (a minimal arch is the full set minus the extras).

// LoadedLayer is one decode layer's weights: projections as quant-agnostic Linear, norms as raw bf16
// bytes. KV-shared layers carry nil K/V (they read the owner's cache); dense layers carry Gate/Up/Down,
// MoE layers carry MoE instead.
type LoadedLayer struct {
	AttnNorm, PostAttnNorm []byte // input_layernorm, post_attention_layernorm
	QNorm, KNorm           []byte // self_attn.q_norm / k_norm (nil without QK-norm)
	LayerScalar            []byte // per-layer output scalar [1] (nil when absent)
	Q, K, V, O             *Linear

	MLPNorm, PostFFNorm []byte // pre/post feedforward norms (dense MLP)
	Gate, Up, Down      *Linear
	MoE                 *LoadedMoE // non-nil ⇒ MoE layer (Gate/Up/Down then unused)

	PerLayerGate, PerLayerProjection *Linear // per-layer-input gate (E2B/E4B PLE); nil without the tower
	PostPerLayerInputNorm            []byte
}

// LoadedMoE is a MoE layer's dual-branch FFN: a dense local MLP + the sparse experts, each with its
// own norms.
type LoadedMoE struct {
	PreFFNorm, PreFFNorm2, PostFFNorm1, PostFFNorm2, PostFFNorm []byte
	RouterScale, PerExpertScale                                 []byte
	LocalGate, LocalUp, LocalDown                               *Linear
	Router                                                      *Linear
	ExpGate, ExpUp, ExpGateUp, ExpDown                          *Linear // experts.switch_glu.*
}

// LoadedVisionLinear is one vision linear's weight plus optional affine-quant
// metadata and additive bias.
type LoadedVisionLinear struct {
	Weight         []byte
	Scales, Biases []byte
	Bias           []byte
	OutDim, InDim  int
	GroupSize      int
	Bits           int
	Kind           string
}

// LoadedVisionLayer is one vision encoder layer's weights.
type LoadedVisionLayer struct {
	InputNorm, PostAttnNorm, PreFFNorm, PostFFNorm []byte
	Q, K, V, O                                     LoadedVisionLinear
	QNorm, KNorm                                   []byte
	Gate, Up, Down                                 LoadedVisionLinear
}

// LoadedVisionProjector is the vision-to-text projector.
type LoadedVisionProjector struct {
	Projection, Linear1, Linear2 LoadedVisionLinear
}

// LoadedVisionConfig is the engine-neutral vision tower geometry.
type LoadedVisionConfig struct {
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

// LoadedVision is the neutral vision payload a backend can upload/build.
type LoadedVision struct {
	PatchEmbedding     []byte
	PatchConvWeight    []byte
	PositionEmbeddings []byte
	PostLayernorm      []byte
	StdBias, StdScale  []byte
	Layers             []LoadedVisionLayer
	Projector          LoadedVisionProjector
	Cfg                LoadedVisionConfig
}

// LoadedUnifiedVisionConfig is the encoder-free (gemma4_unified) vision
// geometry: raw model patches project straight into the backbone with no
// encoder tower.
type LoadedUnifiedVisionConfig struct {
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

// LoadedUnifiedVision is the encoder-free vision payload (gemma4_unified):
// LayerNorm → patch dense (+bias) → LayerNorm → factorised per-axis position
// add → LayerNorm → scale-free RMSNorm → projection. The LayerNorms carry
// weight+bias; the pre-projection RMSNorm has NO parameters (with_scale=False
// upstream), so only its epsilon lives in the config.
type LoadedUnifiedVision struct {
	PatchLN1W, PatchLN1B []byte
	PatchDense           LoadedVisionLinear // [MMEmbedDim × ModelPatchSize²·3] + Bias
	PatchLN2W, PatchLN2B []byte
	PosEmbedding         []byte // [PosembSize, 2, MMEmbedDim] bf16 (axis 0 = row, 1 = col)
	PosNormW, PosNormB   []byte
	Projection           LoadedVisionLinear // embed_vision [TextHidden × MMEmbedDim], no bias
	// AudioProjection is the unified audio head — embed_audio
	// [TextHidden × AudioSamplesPerToken], no bias; nil Weight when absent.
	AudioProjection LoadedVisionLinear
	Cfg             LoadedUnifiedVisionConfig
}

// LoadedAudioClipBound is one optional per-linear activation clamp.
type LoadedAudioClipBound struct {
	Min, Max float32
	Present  bool
}

// LoadedAudioClipPair holds the optional input/output clamps for a clippable audio linear.
type LoadedAudioClipPair struct {
	In, Out LoadedAudioClipBound
}

// LoadedAudioLinear is one audio linear's weight plus optional activation clamps.
type LoadedAudioLinear struct {
	Weight         []byte
	Scales, Biases []byte
	Clip           LoadedAudioClipPair
	OutDim, InDim  int
	GroupSize      int
	Bits           int
	Kind           string
}

// LoadedAudioSubsample is the Conformer audio subsampler payload.
type LoadedAudioSubsample struct {
	Conv0, Norm0W, Norm0B []byte
	Conv1, Norm1W, Norm1B []byte
	InputProj             LoadedAudioLinear
}

// LoadedAudioFeedForward is one macaron feed-forward block in a Conformer layer.
type LoadedAudioFeedForward struct {
	PreNorm, PostNorm []byte
	FFW1, FFW2        LoadedAudioLinear
}

// LoadedAudioAttention is one chunked relative-position attention block.
type LoadedAudioAttention struct {
	Q, K, V, Post LoadedAudioLinear
	RelativeKProj []byte
	QScalePerDim  []float32
	PosEmbed      []float32
	PosCount      int
}

// LoadedAudioLightConv is one Conformer light-convolution block.
type LoadedAudioLightConv struct {
	PreNorm, ConvNorm []byte
	LinearStart       LoadedAudioLinear
	LinearEnd         LoadedAudioLinear
	DepthwiseWeight   []byte
}

// LoadedAudioLayer is one Conformer encoder layer.
type LoadedAudioLayer struct {
	FF1, FF2              LoadedAudioFeedForward
	Attn                  LoadedAudioAttention
	LConv                 LoadedAudioLightConv
	NormPreAttn           []byte
	NormPostAttn, NormOut []byte
}

// LoadedAudioConfig is the engine-neutral audio tower geometry.
type LoadedAudioConfig struct {
	Hidden, FFInter, Channels, KernelSize      int
	Eps                                        float32
	Act                                        string
	FFResidual, ClipMin, ClipMax               float32
	NumHeads, HeadDim                          int
	ChunkSize, PastHorizon, FutureHorizon      int
	KScale, LogitCap, InvalidLogit             float32
	OutputDim, AudioTokenID                    int
	AudioBeginToken, AudioToken, AudioEndToken string
}

// LoadedAudio is the neutral audio payload a backend can upload/build.
type LoadedAudio struct {
	Subsample  LoadedAudioSubsample
	Layers     []LoadedAudioLayer
	OutputProj []byte
	// OutputProjBias is audio_tower.output_proj.bias [OutputDim] (BF16 bytes), added per row after
	// the encoder's output projection. HF's Gemma4AudioModel.output_proj is a bias=True Linear and
	// the checkpoint ships a non-negligible bias (e2b max|abs| 14.875), so dropping it corrupts every
	// clip; nil only for packs that omit it.
	OutputProjBias []byte
	Projector      LoadedAudioLinear
	Cfg            LoadedAudioConfig
}

// LoadedDiffusion is the neutral block-diffusion payload a backend can upload/build.
type LoadedDiffusion struct {
	SelfCondPreNorm                        []byte
	SelfCondGate, SelfCondUp, SelfCondDown *Linear
	EncoderLayerScalars                    [][]byte
	CanvasLength                           int32
	EOSTokens                              []int32
}

// LoadedModel is the whole backend-agnostic weight set: the Arch + every weight as a Linear or raw
// norm bytes, viewing the source mmap. The single assembler output every backend consumes.
type LoadedModel struct {
	Arch               Arch
	Embed              *Linear // token embedding (also the tied LM head when LMHead is nil)
	EmbedNorm          []byte  // optional layer norm applied immediately after token embedding (BLOOM)
	PositionEmbed      *Linear // learned absolute position table; nil for rotary architectures
	EmbedProjectionIn  *Linear // optional embedding-width to hidden-width projection
	EmbedProjectionOut *Linear // optional hidden-width to tied-head-width projection
	LMHead             *Linear // separate output projection, or nil ⇒ tied to Embed
	FinalNorm          []byte
	Layers             []LoadedLayer

	EmbedPerLayer     *Linear // PLE tower (E2B/E4B); nil when absent
	PerLayerModelProj *Linear
	PerLayerProjNorm  []byte
	Vision            *LoadedVision
	UnifiedVision     *LoadedUnifiedVision // encoder-free vision (gemma4_unified); nil when absent
	Audio             *LoadedAudio
	Diffusion         *LoadedDiffusion
}

// Tied reports whether the LM head reuses the token embedding (no separate lm_head weight).
func (m *LoadedModel) Tied() bool { return m.LMHead == nil }

// ValidateRequired checks the always-present weights are there — a missing one is a malformed
// checkpoint, surfaced as a clean load error rather than a nil-deref deep in the decode. OPTIONAL
// weights are deliberately not required: k/v on KV-shared layers, v on K==V layers, lm_head when tied,
// the PLE tower, and QK-norm — so a well-formed checkpoint of any family/quant passes and only a
// genuinely-incomplete one is rejected. Every arch's assembler calls this on its LoadedModel.
func (m *LoadedModel) ValidateRequired(arch Arch) error {
	if m.Embed == nil {
		return core.NewError("model.LoadedModel: missing model.embed_tokens")
	}
	if m.FinalNorm == nil && !arch.NoFinalNorm && !arch.NonParametricLayerNorm {
		return core.NewError("model.LoadedModel: missing model.norm.weight")
	}
	for i := range m.Layers {
		L := &m.Layers[i]
		if L.Q == nil || L.O == nil {
			return core.NewError(core.Sprintf("model.LoadedModel: layer %d missing input_layernorm/q_proj/o_proj", i))
		}
		if arch.NormPlacement == NormPlacementPost {
			if len(L.PostAttnNorm) == 0 {
				return core.NewError(core.Sprintf("model.LoadedModel: layer %d missing post_attention_layernorm", i))
			}
		} else if len(L.AttnNorm) == 0 && !arch.NonParametricLayerNorm {
			return core.NewError(core.Sprintf("model.LoadedModel: layer %d missing input_layernorm", i))
		}
		if arch.Layer[i].OwnsCache() && L.K == nil {
			return core.NewError(core.Sprintf("model.LoadedModel: layer %d missing k_proj (cache owner)", i))
		}
		if L.MoE == nil {
			if L.Gate == nil || L.Up == nil || L.Down == nil {
				return core.NewError(core.Sprintf("model.LoadedModel: layer %d missing a required dense-MLP weight", i))
			}
			if arch.NormPlacement == NormPlacementPost {
				if len(L.PostFFNorm) == 0 {
					return core.NewError(core.Sprintf("model.LoadedModel: layer %d missing post_feedforward_layernorm", i))
				}
			} else if len(L.MLPNorm) == 0 && !arch.NonParametricLayerNorm {
				return core.NewError(core.Sprintf("model.LoadedModel: layer %d missing MLP norm", i))
			}
		}
	}
	return nil
}
