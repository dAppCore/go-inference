// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/decode/tokenizer"
)

// NativeTokenModel binds the no-cgo decode backend + the embed/head bookend
// weights behind model.TokenModel, so model.Generate drives the whole token loop
// (embed → decode → head → sample) over the native path with no per-backend loop
// code. The decode runs whole-sequence through NativeBackend (model.Backend);
// the embed/head closures wrap the proven bookends — bf16 (EmbedTokensBF16 /
// LMHeadBF16) or 4-bit (EmbedTokensQuant / LMHeadQuant), set by the constructor,
// exactly as ArchSession/NewArchQuantSession carry their embed/head funcs.
// This is the native side of "the surface pkg/rocm drops into yields real
// tokens". E2B/E4B per-layer-input models work via the incremental session path
// (OpenSession + StepWithID); the whole-sequence DecodeForward does not do PLE.
type NativeTokenModel struct {
	*NativeBackend
	embed     func(id int32) ([]byte, error)
	embedInto func(dst []byte, id int32) ([]byte, error)
	head      func(hidden []byte) ([]byte, error)
	vocab     int
	// Optional loaded-weight quant metadata surfaced to the no-cgo serve adapter's Info path.
	// bf16 models leave these at zero, matching inference.ModelInfo's unquantised convention.
	quantBits  int
	quantGroup int
	// openSession builds a fresh persistent-cache decode session (ArchSession /
	// ArchQuantSession) — the incremental O(1)/token path model.Generate prefers
	// over the whole-sequence NativeBackend.DecodeForward. It takes the model's shardBuffers so the
	// session binds its weights as no-copy shard views (the directory-loaded model) rather than
	// uploading copies; a nil sb (in-memory model) uses the upload path.
	openSession func(*shardBuffers, *headEncoder) (model.DecodeStepper, error)
	// shards holds the memory-mapped checkpoint + per-shard no-copy Metal buffers when the model
	// was loaded zero-copy from a directory (LoadGemma4TokenModelDir). The embed/head closures and
	// the decode buffers reference VIEWS into these mmaps, so shards lives for the model's life
	// (and outlives any OpenSession session, which re-references the same weights). nil for a model
	// built from in-memory weight bytes. Close unmaps.
	shards *shardBuffers
	// headEnc is the zero-copy LM head (the per-token serve path: model.Generate's generateStepwise
	// calls m.Head every token). It binds the [vocab×dModel] head weight no-copy from the shard mmap,
	// resolved once — killing the per-token re-upload balloon. nil for an in-memory model (Head then
	// uses the upload closure). Concurrency-safe (no shared mutable state), so the shared model can
	// serve many request goroutines. Set by LoadGemma4TokenModelDir.
	headEnc   *headEncoder
	vision    *model.LoadedVision
	audio     *model.LoadedAudio
	diffusion *model.LoadedDiffusion
	bf16      *BF16Model
	quant     *QuantModel
	// tok is the optional text tokenizer, mirroring pkg/metal Model's held
	// tokenizer. It is nil unless attached (AttachTokenizer): the decode model
	// works in token-id space, so text↔ids is a serve-boundary concern the
	// caller wires. When set, the model satisfies the string-prompt
	// inference.KVSnapshotter / KVChunkSnapshotter contracts directly.
	tok *tokenizer.Tokenizer
}

type archSessionConfig struct {
	pagedKVPageSize int
	pagedKVPrealloc bool
}

// Close releases a directory-loaded model's memory-mapped checkpoint (no-op when the weights are
// in-memory bytes). The resident decode/serve weights live for the process in the serve shape, so
// this is for explicit teardown (tests, a model hot-swap that drains first); do not Close while a
// Generate is in flight.
func (m *NativeTokenModel) Close() error {
	if m == nil {
		return nil
	}
	return m.shards.Close()
}

var _ model.SessionModel = (*NativeTokenModel)(nil)

const largeVariantAttentionHeads = 16

// OpenSession opens a fresh incremental decode session (empty KV cache). This
// makes model.Generate run the native path O(1)/token (stepToken over a
// persistent cache) instead of re-decoding the whole sequence each token.
func (m *NativeTokenModel) OpenSession() (model.DecodeStepper, error) {
	return m.openSession(m.shards, m.headEnc)
}

// AttachTokenizer binds the text tokenizer the string-prompt KV-capture
// contracts need (inference.KVSnapshotter / KVChunkSnapshotter). It returns the
// model so callers can chain it onto a constructor. Passing nil detaches.
//
//	tm := native.NewBF16TokenModel(g, arch, maxLen).AttachTokenizer(tok)
func (m *NativeTokenModel) AttachTokenizer(tok *tokenizer.Tokenizer) *NativeTokenModel {
	if m == nil {
		return nil
	}
	m.tok = tok
	return m
}

// Tokenizer returns the attached text tokenizer, or nil when the model works in
// token-id space only (the default — the serve boundary owns text↔ids).
func (m *NativeTokenModel) Tokenizer() *tokenizer.Tokenizer {
	if m == nil {
		return nil
	}
	return m.tok
}

func (m *NativeTokenModel) AcceptsImageInput() bool {
	return m != nil && m.vision != nil
}

func (m *NativeTokenModel) ImagePlaceholderTokenID() int32 {
	if m == nil || m.vision == nil {
		return 0
	}
	return m.vision.Cfg.ImageTokenID
}

func (m *NativeTokenModel) ImagePlaceholderBlock(softTokens int) string {
	if m == nil || m.vision == nil || softTokens <= 0 {
		return ""
	}
	cfg := m.vision.Cfg
	return nativeVisionPlaceholderBlock(cfg.ImageBeginToken, cfg.ImageToken, cfg.ImageEndToken, softTokens)
}

func (m *NativeTokenModel) VideoPlaceholderTokenID() int32 {
	if m == nil || m.vision == nil {
		return 0
	}
	return m.vision.Cfg.VideoTokenID
}

func (m *NativeTokenModel) VideoPlaceholderBlock(softTokens int) string {
	if m == nil || m.vision == nil || softTokens <= 0 {
		return ""
	}
	cfg := m.vision.Cfg
	return nativeVisionPlaceholderBlock(cfg.ImageBeginToken, cfg.VideoToken, cfg.ImageEndToken, softTokens)
}

func nativeVisionPlaceholderBlock(begin, token, end string, softTokens int) string {
	if token == "" || softTokens <= 0 {
		return ""
	}
	var b core.Builder
	b.Grow(len(begin) + len(end) + softTokens*len(token))
	b.WriteString(begin)
	for i := 0; i < softTokens; i++ {
		b.WriteString(token)
	}
	b.WriteString(end)
	return b.String()
}

func (m *NativeTokenModel) ProjectImageFeatures(patches []byte) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel.ProjectImageFeatures: nil model")
	}
	weights, cfg, ok := nativeVisionFromLoaded(m.vision)
	if !ok {
		return nil, core.NewError("native.NativeTokenModel.ProjectImageFeatures: model has no vision payload")
	}
	return VisionTower(patches, weights, cfg)
}

func (m *NativeTokenModel) ProjectImagePixels(pixels []float32, height, width int) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel.ProjectImagePixels: nil model")
	}
	weights, cfg, ok := nativeVisionFromLoaded(m.vision)
	if !ok {
		return nil, core.NewError("native.NativeTokenModel.ProjectImagePixels: model has no vision payload")
	}
	return VisionTowerNHWC(pixels, height, width, weights, cfg)
}

func (m *NativeTokenModel) InjectImageFeatures(embeddings []byte, tokenIDs []int32, features []byte) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel.InjectImageFeatures: nil model")
	}
	if !m.AcceptsImageInput() {
		return nil, core.NewError("native.NativeTokenModel.InjectImageFeatures: model has no vision payload")
	}
	return VisionInjectFeatures(embeddings, tokenIDs, features, m.ImagePlaceholderTokenID(), m.HiddenSize())
}

func (m *NativeTokenModel) InjectVideoFeatures(embeddings []byte, tokenIDs []int32, features []byte) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel.InjectVideoFeatures: nil model")
	}
	if !m.AcceptsImageInput() {
		return nil, core.NewError("native.NativeTokenModel.InjectVideoFeatures: model has no vision payload")
	}
	return VisionInjectFeatures(embeddings, tokenIDs, features, m.VideoPlaceholderTokenID(), m.HiddenSize())
}

func (m *NativeTokenModel) AcceptsAudioInput() bool {
	return m != nil && m.audio != nil
}

func (m *NativeTokenModel) BlockDiffusionCapable() bool {
	return m != nil && m.diffusion != nil
}

func (m *NativeTokenModel) AudioPlaceholderTokenID() int32 {
	if m == nil || m.audio == nil {
		return 0
	}
	return int32(m.audio.Cfg.AudioTokenID)
}

func (m *NativeTokenModel) AudioPlaceholderBlock(softTokens int) string {
	if m == nil || m.audio == nil || softTokens <= 0 {
		return ""
	}
	cfg := m.audio.Cfg
	if cfg.AudioToken == "" {
		return ""
	}
	var b core.Builder
	b.Grow(len(cfg.AudioBeginToken) + len(cfg.AudioEndToken) + softTokens*len(cfg.AudioToken))
	b.WriteString(cfg.AudioBeginToken)
	for i := 0; i < softTokens; i++ {
		b.WriteString(cfg.AudioToken)
	}
	b.WriteString(cfg.AudioEndToken)
	return b.String()
}

func (m *NativeTokenModel) AudioSoftTokens(frames int) int {
	if m == nil || m.audio == nil || frames <= 0 {
		return 0
	}
	half := func(n int) int { return (n + 1) / 2 }
	return half(half(frames))
}

func (m *NativeTokenModel) ProjectAudioFeatures(features []byte, frames, melBins int) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel.ProjectAudioFeatures: nil model")
	}
	weights, cfg, projector, ok := nativeAudioFromLoaded(m.audio, frames, melBins)
	if !ok {
		return nil, core.NewError("native.NativeTokenModel.ProjectAudioFeatures: model has no audio payload")
	}
	encoded, err := AudioEncode(features, weights, cfg)
	if err != nil {
		return nil, err
	}
	return nativeAudioProjector(encoded, projector, weights.OutputDim, m.audio.Cfg.Eps)
}

func (m *NativeTokenModel) InjectAudioFeatures(embeddings []byte, tokenIDs []int32, features []byte) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel.InjectAudioFeatures: nil model")
	}
	if !m.AcceptsAudioInput() {
		return nil, core.NewError("native.NativeTokenModel.InjectAudioFeatures: model has no audio payload")
	}
	return AudioInjectFeatures(embeddings, tokenIDs, features, m.AudioPlaceholderTokenID(), m.HiddenSize())
}

// TokenEmbeddingsWithFeatures gathers scaled token embeddings and splices any
// pre-projected multimodal soft-token rows into their placeholder positions.
// The returned rows share one backing store and are ready for
// ArchSession.PrefillTokenEmbeddings.
func (m *NativeTokenModel) TokenEmbeddingsWithFeatures(tokenIDs []int32, imageFeatures, audioFeatures, videoFeatures []byte) ([][]byte, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel.TokenEmbeddingsWithFeatures: nil model")
	}
	if len(tokenIDs) == 0 {
		return nil, core.NewError("native.NativeTokenModel.TokenEmbeddingsWithFeatures: empty token ids")
	}
	row := m.EmbeddingBytes()
	if row <= 0 {
		return nil, core.NewError("native.NativeTokenModel.TokenEmbeddingsWithFeatures: invalid embedding width")
	}
	if m.embedInto == nil && m.embed == nil {
		return nil, core.NewError("native.NativeTokenModel.TokenEmbeddingsWithFeatures: model has no embedding bookend")
	}

	stream := make([]byte, len(tokenIDs)*row)
	for i, id := range tokenIDs {
		start := i * row
		if _, err := m.EmbedInto(stream[start:start+row], id); err != nil {
			return nil, err
		}
	}

	if len(imageFeatures) > 0 {
		if err := m.spliceTokenFeaturesInto(stream, tokenIDs, imageFeatures, m.ImagePlaceholderTokenID(), "image"); err != nil {
			return nil, err
		}
	}
	if len(audioFeatures) > 0 {
		if err := m.spliceTokenFeaturesInto(stream, tokenIDs, audioFeatures, m.AudioPlaceholderTokenID(), "audio"); err != nil {
			return nil, err
		}
	}
	if len(videoFeatures) > 0 {
		if err := m.spliceTokenFeaturesInto(stream, tokenIDs, videoFeatures, m.VideoPlaceholderTokenID(), "video"); err != nil {
			return nil, err
		}
	}

	rows := make([][]byte, len(tokenIDs))
	for i := range tokenIDs {
		start := i * row
		rows[i] = stream[start : start+row]
	}
	return rows, nil
}

func (m *NativeTokenModel) spliceTokenFeaturesInto(stream []byte, tokenIDs []int32, features []byte, tokenID int32, label string) error {
	row := m.EmbeddingBytes()
	if row <= 0 {
		return core.NewError("native.NativeTokenModel.TokenEmbeddingsWithFeatures: invalid embedding width")
	}
	if tokenID == 0 {
		return core.NewError("native.NativeTokenModel.TokenEmbeddingsWithFeatures: " + label + " token id is not configured")
	}
	if len(stream) != len(tokenIDs)*row {
		return core.NewError("native.NativeTokenModel.TokenEmbeddingsWithFeatures: token ids must match embedding rows")
	}
	if len(features)%row != 0 {
		return core.NewError("native.NativeTokenModel.TokenEmbeddingsWithFeatures: " + label + " feature rows must align to embedding width")
	}
	featureRows := len(features) / row
	slots := 0
	for _, id := range tokenIDs {
		if id == tokenID {
			slots++
		}
	}
	if slots != featureRows {
		return core.NewError("native.NativeTokenModel.TokenEmbeddingsWithFeatures: " + label + " feature count must equal token slots")
	}
	featureIdx := 0
	for pos, id := range tokenIDs {
		if id != tokenID {
			continue
		}
		copy(stream[pos*row:(pos+1)*row], features[featureIdx*row:(featureIdx+1)*row])
		featureIdx++
	}
	return nil
}

func nativeVisionFromLoaded(loaded *model.LoadedVision) (*VisionWeights, VisionConfig, bool) {
	if loaded == nil {
		return nil, VisionConfig{}, false
	}
	cfg := VisionConfig{
		Hidden:                loaded.Cfg.Hidden,
		PatchDim:              loaded.Cfg.PatchDim,
		NumLayers:             loaded.Cfg.NumLayers,
		NumHeads:              loaded.Cfg.NumHeads,
		NumKVHeads:            loaded.Cfg.NumKVHeads,
		HeadDim:               loaded.Cfg.HeadDim,
		PatchSize:             loaded.Cfg.PatchSize,
		NumChannels:           loaded.Cfg.NumChannels,
		GridH:                 loaded.Cfg.GridH,
		GridW:                 loaded.Cfg.GridW,
		PositionEmbeddingSize: loaded.Cfg.PositionEmbeddingSize,
		RopeBase:              loaded.Cfg.RopeBase,
		RMSNormEps:            loaded.Cfg.RMSNormEps,
		PoolKernel:            loaded.Cfg.PoolKernel,
		Standardize:           loaded.Cfg.Standardize,
		EmbeddingScale:        loaded.Cfg.EmbeddingScale,
		ImageTokenID:          loaded.Cfg.ImageTokenID,
		ImageBeginToken:       loaded.Cfg.ImageBeginToken,
		ImageToken:            loaded.Cfg.ImageToken,
		ImageEndToken:         loaded.Cfg.ImageEndToken,
		VideoTokenID:          loaded.Cfg.VideoTokenID,
		VideoToken:            loaded.Cfg.VideoToken,
	}
	weights := &VisionWeights{
		PatchEmbedding:     loaded.PatchEmbedding,
		PatchConvWeight:    loaded.PatchConvWeight,
		PositionEmbeddings: loaded.PositionEmbeddings,
		PostLayernorm:      loaded.PostLayernorm,
		StdBias:            loaded.StdBias,
		StdScale:           loaded.StdScale,
		Layers:             make([]VisionLayerWeights, len(loaded.Layers)),
		Projector: VisionProjectorWeights{
			Projection: nativeVisionProjectorLinear(loaded.Projector.Projection),
			Linear1:    nativeVisionProjectorLinear(loaded.Projector.Linear1),
			Linear2:    nativeVisionProjectorLinear(loaded.Projector.Linear2),
			Eps:        loaded.Cfg.RMSNormEps,
		},
	}
	for i := range loaded.Layers {
		src := &loaded.Layers[i]
		weights.Layers[i] = VisionLayerWeights{
			InputNorm:    src.InputNorm,
			PostAttnNorm: src.PostAttnNorm,
			PreFFNorm:    src.PreFFNorm,
			PostFFNorm:   src.PostFFNorm,
			WQ:           src.Q.Weight,
			WK:           src.K.Weight,
			WV:           src.V.Weight,
			WO:           src.O.Weight,
			BQ:           src.Q.Bias,
			BK:           src.K.Bias,
			BV:           src.V.Bias,
			BO:           src.O.Bias,
			QNorm:        src.QNorm,
			KNorm:        src.KNorm,
			WGate:        src.Gate.Weight,
			WUp:          src.Up.Weight,
			WDown:        src.Down.Weight,
			BGate:        src.Gate.Bias,
			BUp:          src.Up.Bias,
			BDown:        src.Down.Bias,
		}
	}
	return weights, cfg, true
}

func nativeVisionProjectorLinear(l model.LoadedVisionLinear) VisionProjectorLinear {
	return VisionProjectorLinear{
		Weight:    l.Weight,
		Scales:    l.Scales,
		Biases:    l.Biases,
		Bias:      l.Bias,
		OutDim:    l.OutDim,
		InDim:     l.InDim,
		GroupSize: l.GroupSize,
		Bits:      l.Bits,
	}
}

func nativeAudioClipBound(c model.LoadedAudioClipBound) ClipBound {
	return ClipBound{Min: c.Min, Max: c.Max, Present: c.Present}
}

func nativeAudioClipPair(c model.LoadedAudioClipPair) ClipPair {
	return ClipPair{In: nativeAudioClipBound(c.In), Out: nativeAudioClipBound(c.Out)}
}

func nativeAudioFromLoaded(loaded *model.LoadedAudio, frames, melBins int) (*AudioEncoderWeights, AudioConfig, model.LoadedAudioLinear, bool) {
	if loaded == nil || loaded.OutputProj == nil {
		return nil, AudioConfig{}, model.LoadedAudioLinear{}, false
	}
	cfg := AudioConfig{
		Hidden:        loaded.Cfg.Hidden,
		FFInter:       loaded.Cfg.FFInter,
		Channels:      loaded.Cfg.Channels,
		KernelSize:    loaded.Cfg.KernelSize,
		Eps:           loaded.Cfg.Eps,
		Act:           loaded.Cfg.Act,
		FFResidual:    loaded.Cfg.FFResidual,
		ClipMin:       loaded.Cfg.ClipMin,
		ClipMax:       loaded.Cfg.ClipMax,
		NumHeads:      loaded.Cfg.NumHeads,
		HeadDim:       loaded.Cfg.HeadDim,
		ChunkSize:     loaded.Cfg.ChunkSize,
		PastHorizon:   loaded.Cfg.PastHorizon,
		FutureHorizon: loaded.Cfg.FutureHorizon,
		KScale:        loaded.Cfg.KScale,
		LogitCap:      loaded.Cfg.LogitCap,
		InvalidLogit:  loaded.Cfg.InvalidLogit,
	}
	outC0 := len(loaded.Subsample.Norm0W) / bf16Size
	outC1 := len(loaded.Subsample.Norm1W) / bf16Size
	weights := &AudioEncoderWeights{
		Subsample: &AudioSubsampleWeights{
			Conv0:         loaded.Subsample.Conv0,
			Norm0W:        loaded.Subsample.Norm0W,
			Norm0B:        loaded.Subsample.Norm0B,
			Conv1:         loaded.Subsample.Conv1,
			Norm1W:        loaded.Subsample.Norm1W,
			Norm1B:        loaded.Subsample.Norm1B,
			InputProj:     loaded.Subsample.InputProj.Weight,
			InputProjClip: nativeAudioClipPair(loaded.Subsample.InputProj.Clip),
		},
		SubsampleC: AudioSubsampleConfig{
			Frames: frames, MelBins: melBins, OutC0: outC0, OutC1: outC1,
			Hidden: loaded.Cfg.Hidden, Eps: loaded.Cfg.Eps,
		},
		Layers:     make([]*AudioLayerWeights, len(loaded.Layers)),
		OutputProj: loaded.OutputProj,
		OutputDim:  loaded.Cfg.OutputDim,
	}
	for i := range loaded.Layers {
		src := &loaded.Layers[i]
		weights.Layers[i] = &AudioLayerWeights{
			FF1: &AudioFeedForwardWeights{
				PreNorm: src.FF1.PreNorm, PostNorm: src.FF1.PostNorm,
				FFW1: src.FF1.FFW1.Weight, FFW2: src.FF1.FFW2.Weight,
				FFW1Clip: nativeAudioClipPair(src.FF1.FFW1.Clip), FFW2Clip: nativeAudioClipPair(src.FF1.FFW2.Clip),
			},
			FF2: &AudioFeedForwardWeights{
				PreNorm: src.FF2.PreNorm, PostNorm: src.FF2.PostNorm,
				FFW1: src.FF2.FFW1.Weight, FFW2: src.FF2.FFW2.Weight,
				FFW1Clip: nativeAudioClipPair(src.FF2.FFW1.Clip), FFW2Clip: nativeAudioClipPair(src.FF2.FFW2.Clip),
			},
			Attn: &AudioAttentionWeights{
				QProj: src.Attn.Q.Weight, KProj: src.Attn.K.Weight, VProj: src.Attn.V.Weight, Post: src.Attn.Post.Weight,
				QClip: nativeAudioClipPair(src.Attn.Q.Clip), KClip: nativeAudioClipPair(src.Attn.K.Clip),
				VClip: nativeAudioClipPair(src.Attn.V.Clip), PostClip: nativeAudioClipPair(src.Attn.Post.Clip),
				RelativeKProj: src.Attn.RelativeKProj, QScalePerDim: src.Attn.QScalePerDim,
				PosEmbed: src.Attn.PosEmbed, PosCount: src.Attn.PosCount,
			},
			LConv: &AudioLightConvWeights{
				PreNorm: src.LConv.PreNorm, ConvNorm: src.LConv.ConvNorm,
				LinearStart: src.LConv.LinearStart.Weight, LinearEnd: src.LConv.LinearEnd.Weight,
				DepthwiseWeight: src.LConv.DepthwiseWeight,
				StartClip:       nativeAudioClipPair(src.LConv.LinearStart.Clip), EndClip: nativeAudioClipPair(src.LConv.LinearEnd.Clip),
			},
			NormPreAttn:  src.NormPreAttn,
			NormPostAttn: src.NormPostAttn,
			NormOut:      src.NormOut,
		}
	}
	return weights, cfg, loaded.Projector, true
}

func nativeAudioProjector(rows []float32, projector model.LoadedAudioLinear, inputDim int, eps float32) ([]byte, error) {
	if inputDim <= 0 || len(rows)%inputDim != 0 {
		return nil, core.NewError("native.NativeTokenModel.ProjectAudioFeatures: invalid audio projector geometry")
	}
	L := len(rows) / inputDim
	normed := append([]float32(nil), rows...)
	for i := 0; i < L; i++ {
		rmsNormVec(normed[i*inputDim:i*inputDim+inputDim], nil, eps)
	}
	if projector.Weight == nil {
		return f32ToBf16Slice(normed), nil
	}
	if len(projector.Scales) > 0 {
		if projector.InDim != inputDim || projector.OutDim <= 0 || projector.GroupSize <= 0 || projector.Bits <= 0 {
			return nil, core.NewError("native.NativeTokenModel.ProjectAudioFeatures: invalid quant audio projector geometry")
		}
		if len(projector.Biases) == 0 {
			return nil, core.NewError("native.NativeTokenModel.ProjectAudioFeatures: quant audio projector missing biases")
		}
		in := f32ToBf16Slice(normed)
		out := make([]byte, L*projector.OutDim*bf16Size)
		for r := 0; r < L; r++ {
			rowOut := out[r*projector.OutDim*bf16Size : (r+1)*projector.OutDim*bf16Size]
			rowIn := in[r*inputDim*bf16Size : (r+1)*inputDim*bf16Size]
			if _, err := QMVBF16Into(rowOut, rowIn, projector.Weight, projector.Scales, projector.Biases, projector.OutDim, inputDim, projector.GroupSize, projector.Bits); err != nil {
				return nil, err
			}
		}
		return out, nil
	}
	outDim := len(projector.Weight) / (inputDim * bf16Size)
	out, err := clippedMatF32(normed, projector.Weight, L, outDim, inputDim, nativeAudioClipPair(projector.Clip))
	if err != nil {
		return nil, err
	}
	return f32ToBf16Slice(out), nil
}

// NumLayers reports the transformer layer count from the backend-agnostic arch.
func (m *NativeTokenModel) NumLayers() int {
	if m == nil || m.NativeBackend == nil {
		return 0
	}
	return len(m.arch.Layer)
}

// NumQueryHeads reports the attention query-head count used by the native arch.
func (m *NativeTokenModel) NumQueryHeads() int {
	if m == nil || m.NativeBackend == nil {
		return 0
	}
	return m.arch.Heads
}

// HiddenSize reports the model hidden dimension from the backend-agnostic arch.
func (m *NativeTokenModel) HiddenSize() int {
	if m == nil || m.NativeBackend == nil {
		return 0
	}
	return m.arch.Hidden
}

// QuantBits reports the loaded token model's quant bit width, or 0 for bf16.
func (m *NativeTokenModel) QuantBits() int {
	if m == nil {
		return 0
	}
	return m.quantBits
}

// QuantGroup reports the loaded token model's quant group size, or 0 for bf16.
func (m *NativeTokenModel) QuantGroup() int {
	if m == nil {
		return 0
	}
	return m.quantGroup
}

// UsesFixedSlidingCache reports whether this arch declares sliding-window local
// attention and therefore uses bounded local-layer KV state.
func (m *NativeTokenModel) UsesFixedSlidingCache() bool {
	return m != nil && m.NativeBackend != nil && m.arch.SlidingWindow > 0
}

// NeedsThoughtChannelSuppressor mirrors the Gemma-4 large-variant prompt rule
// from the model topology: large variants declare at least 16 query heads.
func (m *NativeTokenModel) NeedsThoughtChannelSuppressor() bool {
	return m != nil && m.NativeBackend != nil && m.arch.Heads >= largeVariantAttentionHeads
}

// AttentionCacheLayout maps each transformer layer to the cache slot it reads,
// following the arch-derived owner/share topology. Layers whose owner has no
// valid cache in the requested range stay -1.
func (m *NativeTokenModel) AttentionCacheLayout(numLayers, numCaches int) []int {
	if numLayers < 0 {
		numLayers = 0
	}
	layout := make([]int, numLayers)
	for i := range layout {
		layout[i] = -1
	}
	if m == nil || m.NativeBackend == nil {
		return layout
	}
	for layerIdx := 0; layerIdx < numLayers && layerIdx < len(m.arch.Layer); layerIdx++ {
		ownerIdx := m.arch.Layer[layerIdx].KVShareFrom
		if ownerIdx < 0 || ownerIdx >= len(m.arch.Layer) {
			continue
		}
		cacheIdx := m.arch.Layer[ownerIdx].CacheIndex
		if cacheIdx < 0 || cacheIdx >= numCaches {
			continue
		}
		layout[layerIdx] = cacheIdx
	}
	return layout
}

// NewBF16TokenModel binds an assembled bf16 gemma4 (weights + arch) as a
// model.TokenModel — the contract-native generation path. Decode runs
// whole-sequence through NativeBackend (opts forwarded, e.g. WithICB); the LM
// head reads the arch's eps + soft-cap, the embed scale is √hidden. The arch
// must be PLE-free (12B/31B dense, 26B-A4B MoE, Ministral).
func NewBF16TokenModel(g *BF16Model, arch model.Arch, maxLen int, opts ...BackendOption) (*NativeTokenModel, error) {
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewBF16TokenModel: weights/arch layer count mismatch")
	}
	b, err := NewBF16Backend(arch, g.Layers, maxLen, opts...)
	if err != nil {
		return nil, err
	}
	sessionCfg := archSessionConfig{pagedKVPageSize: b.pagedKVPageSize, pagedKVPrealloc: b.pagedKVPrealloc}
	scale := embedScaleOf(arch)
	vocab, dModel, eps, softCap := arch.Vocab, arch.Hidden, arch.Eps, arch.SoftCap
	tm := &NativeTokenModel{
		NativeBackend: b,
		vocab:         vocab,
		bf16:          g,
		embed:         func(id int32) ([]byte, error) { return embedTokenBF16(g.Embed, id, vocab, dModel, scale) },
		embedInto: func(dst []byte, id int32) ([]byte, error) {
			return embedTokenBF16Into(dst, g.Embed, id, vocab, dModel, scale)
		},
		head: func(hidden []byte) ([]byte, error) {
			return LMHeadBF16(hidden, g.FinalNorm, g.LMHead, dModel, vocab, eps, softCap)
		},
		openSession: func(sb *shardBuffers, head *headEncoder) (model.DecodeStepper, error) {
			return newArchSessionShardsWithHeadConfig(g, arch, maxLen, sb, head, sessionCfg)
		},
	}
	he, herr := buildHeadEncoder(nil, g.FinalNorm, g.LMHead, nil, nil, dModel, vocab, 0, 0, eps, softCap, false)
	if herr != nil {
		return nil, herr
	}
	tm.headEnc = he
	return tm, nil
}

// NewQuantTokenModel binds an assembled 4-bit gemma4 (weights + arch) as a
// model.TokenModel — the quant sibling of NewBF16TokenModel. The embed/head wrap
// the 4-bit bookends (EmbedTokensQuant / LMHeadQuant) over the packed embedding +
// tied or separate head. E2B/E4B per-layer-input models are supported via the
// INCREMENTAL session path (OpenSession's ArchQuantSession threads the per-layer
// inputs through StepWithID); the whole-sequence DecodeForward fallback does not do
// PLE, so model.Generate (which prefers the session) is the path for those.
func NewQuantTokenModel(g *QuantModel, arch model.Arch, maxLen int, opts ...BackendOption) (*NativeTokenModel, error) {
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewQuantTokenModel: weights/arch layer count mismatch")
	}
	b, err := NewQuantBackend(arch, g.Layers, maxLen, opts...)
	if err != nil {
		return nil, err
	}
	sessionCfg := archSessionConfig{pagedKVPageSize: b.pagedKVPageSize, pagedKVPrealloc: b.pagedKVPrealloc}
	scale := embedScaleOf(arch)
	vocab, dModel, eps, softCap := arch.Vocab, arch.Hidden, arch.Eps, arch.SoftCap
	gs, bits := g.GroupSize, g.Bits
	tm := &NativeTokenModel{
		NativeBackend: b,
		vocab:         vocab,
		quantBits:     bits,
		quantGroup:    gs,
		quant:         g,
		embed: func(id int32) ([]byte, error) {
			return embedTokenQuant(g.Embed, g.EmbedScales, g.EmbedBiases, id, vocab, dModel, gs, bits, scale)
		},
		embedInto: func(dst []byte, id int32) ([]byte, error) {
			return embedTokenQuantInto(dst, g.Embed, g.EmbedScales, g.EmbedBiases, id, vocab, dModel, gs, bits, scale)
		},
		head: func(hidden []byte) ([]byte, error) {
			return LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, dModel, vocab, gs, bits, eps, softCap)
		},
		openSession: func(sb *shardBuffers, head *headEncoder) (model.DecodeStepper, error) {
			return newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, sb, head, sessionCfg)
		},
	}
	he, herr := buildHeadEncoder(nil, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, dModel, vocab, gs, bits, eps, softCap, true)
	if herr != nil {
		return nil, herr
	}
	tm.headEnc = he
	return tm, nil
}

// Vocab is the logit width Greedy/Sample read — the LM head's output dimension.
func (m *NativeTokenModel) Vocab() int { return m.vocab }

// Embed gathers a token id's scaled input embedding (dModel bf16 bytes).
func (m *NativeTokenModel) Embed(id int32) ([]byte, error) { return m.embed(id) }

// EmbeddingBytes reports the byte width of one token embedding row.
func (m *NativeTokenModel) EmbeddingBytes() int {
	if hidden := m.HiddenSize(); hidden > 0 {
		return hidden * bf16Size
	}
	return 0
}

// EmbedInto gathers a token id's scaled input embedding into caller-owned
// storage, avoiding the allocation made by Embed on hot multimodal prefill paths.
func (m *NativeTokenModel) EmbedInto(dst []byte, id int32) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel.EmbedInto: nil model")
	}
	if m.embedInto != nil {
		return m.embedInto(dst, id)
	}
	emb, err := m.Embed(id)
	if err != nil {
		return nil, err
	}
	if len(dst) != len(emb) {
		return nil, core.NewError("native.NativeTokenModel.EmbedInto: dst size mismatch")
	}
	copy(dst, emb)
	return dst, nil
}

// Head maps a final hidden state to vocab logits (final norm + projection +
// optional soft-cap), bf16 bytes throughout. It prefers the zero-copy head (the head weight bound
// no-copy from the shard mmap, resolved once) when the model was loaded from a directory — the
// per-token serve path runs through here, so this is where the LM-head re-upload balloon is killed.
// Falls back to the upload closure for an in-memory model.
func (m *NativeTokenModel) Head(hidden []byte) ([]byte, error) {
	if m.headEnc != nil {
		return m.headEnc.encode(hidden, false) // Head returns logits to the caller (may sample) → apply the softcap
	}
	return m.head(hidden)
}
