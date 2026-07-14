// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

// inference_model.go is engine/hip's composition root for the shared engine
// package — the engine/hip analogue of engine/metal's inference_model.go. It
// wraps a loaded Gemma4-Q4 hip model as the shared engine.TokenModel (open a
// retained hipEngineSession, release the weights) and assembles it, plus the
// model's ModelInfo and tokenizer, into a shared engine.TextModel that hands out
// KV-capturable sessions through the go-inference contracts.
//
// # Relationship to the existing "rocm" backend
//
// hip already registers the "rocm" inference.Backend (register_rocm.go), whose
// LoadModel returns the rich rocmModel (Generate/Chat/Classify/BatchGenerate/
// adapters/benchmark/evaluate). engine.TextModel is a THINNER serving surface —
// it is the shared, KV-portable session vehicle, not a replacement for rocmModel.
// This file therefore ADDS the engine-based composition (used by the HIP-gated
// conformance and available for a future serving swap) without changing hip's
// registered backend. Routing "rocm" through the shared engine is a serving
// decision with a richness trade-off — see the reconcile landing report.
package hip

import (
	"context"
	"iter"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
)

var (
	_ engine.TokenModel                = (*hipTokenModel)(nil)
	_ engine.VisionTokenModel          = (*hipTokenModel)(nil)
	_ engine.VideoTokenModel           = (*hipTokenModel)(nil)
	_ engine.AudioInputTokenModel      = (*hipTokenModel)(nil)
	_ engine.CacheModeReporter         = (*hipTokenModel)(nil)
	_ engine.PromptReuseCapableModel   = (*hipTokenModel)(nil)
	_ engine.StopTokenDeclarer         = (*hipTokenModel)(nil)
	_ engine.SamplingDefaultsDeclarer  = (*hipTokenModel)(nil)
	_ engine.ThoughtSuppressorDeclarer = (*hipTokenModel)(nil)
	_ engine.ChatTemplateDeclarer      = (*hipTokenModel)(nil)
	_ engine.LaneSetOpener             = (*hipTokenModel)(nil)
	_ engine.TrainerModel              = (*hipTokenModel)(nil)
)

// hipTokenModel wraps a loaded Gemma4-Q4 hip model as the shared
// engine.TokenModel: OpenEngineSession opens a retained hipEngineSession (the
// engine.Session the shared adapters drive), and Close releases the resident
// weights. declaredSampling carries the checkpoint's generation_config sampling
// intent, parsed once at load, which engine.TextModel folds into each request
// (engine.SamplingDefaultsDeclarer — see generation_config.go).
type hipTokenModel struct {
	loaded           *hipLoadedModel
	tokenizer        engine.TextTokenizer
	modelType        string
	declaredStops    []int32
	declaredSampling engine.SamplingDefaults
}

// newHipTokenModel binds a loaded model + tokenizer as an engine.TokenModel,
// parsing the checkpoint's generation_config sampling defaults from the loaded
// model's directory so DeclaredSamplingDefaults reports the model's declared
// intent (the zero value when the file is absent or declares none).
func newHipTokenModel(loaded *hipLoadedModel, tok engine.TextTokenizer, modelType string) *hipTokenModel {
	m := &hipTokenModel{loaded: loaded, tokenizer: tok, modelType: modelType}
	if loaded != nil {
		m.declaredStops = loadGenerationConfigStops(loaded.modelPath)
		for _, id := range hipGemma4Q4DefaultStopTokenIDs(loaded) {
			if !hipTokenIsStop(id, m.declaredStops) {
				m.declaredStops = append(m.declaredStops, id)
			}
		}
		m.declaredSampling = loadGenerationConfigSamplingDefaults(loaded.modelPath)
	}
	return m
}

func (m *hipTokenModel) NeedsThoughtChannelSuppressor() bool {
	return m != nil && m.loaded != nil && rocmGemma4NeedsThoughtChannelSuppressor(m.loaded.modelIdentity())
}

func (m *hipTokenModel) DeclaredChatTemplate() (engine.ChatTemplate, bool) {
	if m == nil || m.tokenizer == nil {
		return engine.ChatTemplate{}, false
	}
	architecture := m.modelType
	if m.loaded != nil && m.loaded.modelInfo.Architecture != "" {
		architecture = m.loaded.modelInfo.Architecture
	}
	if template, ok := hipArchitectureChatTemplate(architecture); ok {
		return template, true
	}
	return engine.GemmaChatTemplate(engine.DetectTurnTokens(m.tokenizer), m.NeedsThoughtChannelSuppressor()), true
}

func hipArchitectureChatTemplate(architecture string) (engine.ChatTemplate, bool) {
	templateID, ok := ROCmChatTemplateID(architecture)
	if !ok || templateID != "qwen" {
		return engine.ChatTemplate{}, false
	}
	return hipQwenChatTemplate(), true
}

func hipQwenChatTemplate() engine.ChatTemplate {
	return engine.ChatTemplate{
		Open:          "<|im_start|>",
		Close:         "<|im_end|>",
		UserRole:      "user",
		AssistantRole: "assistant",
		SystemRole:    "system",
		Thinking:      &engine.ChatThinking{OffSuffix: "<think>\n\n</think>\n\n"},
		Stops:         []string{"<|im_end|>"},
	}
}

func formatHIPArchitectureChatTemplate(messages []inference.Message, architecture string, enableThinking *bool) (string, bool) {
	template, ok := hipArchitectureChatTemplate(architecture)
	if !ok {
		return "", false
	}
	prompt := engine.RenderChatTurns(template, messages)
	if template.Thinking != nil && !template.ResolveThinking(enableThinking) {
		prompt += template.Thinking.OffSuffix
	}
	return prompt, true
}

// OpenEngineSession opens a fresh retained Gemma4-Q4 decode session as the
// engine.Session the shared adapters drive.
func (m *hipTokenModel) OpenEngineSession() (engine.Session, error) {
	if m == nil || m.loaded == nil {
		return nil, core.NewError("hip.TokenModel: model is not initialised")
	}
	return newHipEngineSession(m.loaded)
}

// OpenTrainer opens a retained Gemma4 output-head LoRA lifecycle through the
// shared engine.TrainerModel contract.
func (m *hipTokenModel) OpenTrainer(cfg inference.TrainingConfig) (engine.Trainer, error) {
	if m == nil {
		return nil, core.NewError("hip.TokenModel: model is not initialised")
	}
	return newHIPLoRATrainer(m.loaded, cfg)
}

// Close releases the loaded model's resident weights.
func (m *hipTokenModel) Close() error {
	if m == nil || m.loaded == nil {
		return nil
	}
	return m.loaded.Close()
}

func (m *hipTokenModel) SessionsReusePrompts() bool { return true }

func (m *hipTokenModel) SupportedCacheModes() []string {
	return []string{rocmKVCacheModeFP16, rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4}
}

func (m *hipTokenModel) AcceptsImageInput() bool {
	return m != nil && m.loaded != nil && m.loaded.AcceptsImageInput()
}

func (m *hipTokenModel) ImagePlaceholderTokenID() int32 {
	if !m.AcceptsImageInput() {
		return 0
	}
	if m.loaded.unifiedVision != nil {
		return m.loaded.unifiedVision.loaded.Cfg.ImageTokenID
	}
	return m.loaded.vision.loaded.Cfg.ImageTokenID
}

func (m *hipTokenModel) ImagePlaceholderBlock(softTokens int) string {
	if !m.AcceptsImageInput() {
		return ""
	}
	if m.loaded.unifiedVision != nil {
		cfg := m.loaded.unifiedVision.loaded.Cfg
		return hipMultimodalPlaceholderBlock(cfg.ImageBeginToken, cfg.ImageToken, cfg.ImageEndToken, softTokens)
	}
	cfg := m.loaded.vision.loaded.Cfg
	return hipMultimodalPlaceholderBlock(cfg.ImageBeginToken, cfg.ImageToken, cfg.ImageEndToken, softTokens)
}

func (m *hipTokenModel) ProjectImage(payload []byte) ([]byte, int, error) {
	if !m.AcceptsImageInput() {
		return nil, 0, core.NewError("hip.TokenModel.ProjectImage: loaded model has no vision tower")
	}
	var embeddings []float32
	var softTokens int
	var err error
	if m.loaded.unifiedVision != nil {
		embeddings, softTokens, err = m.loaded.unifiedVision.ProjectImage(payload)
	} else {
		embeddings, softTokens, err = m.loaded.vision.ProjectImage(payload)
	}
	if err != nil {
		return nil, 0, core.E("hip.TokenModel.ProjectImage", "project image", err)
	}
	features, err := hipFloat32Payload(embeddings)
	if err != nil {
		return nil, 0, core.E("hip.TokenModel.ProjectImage", "encode projected embeddings", err)
	}
	return features, softTokens, nil
}

func (m *hipTokenModel) VideoPlaceholderTokenID() int32 {
	if !m.AcceptsImageInput() {
		return 0
	}
	if m.loaded.unifiedVision != nil {
		return m.loaded.unifiedVision.loaded.Cfg.VideoTokenID
	}
	return m.loaded.vision.loaded.Cfg.VideoTokenID
}

func (m *hipTokenModel) VideoPlaceholderBlock(softTokens int) string {
	if !m.AcceptsImageInput() {
		return ""
	}
	if m.loaded.unifiedVision != nil {
		cfg := m.loaded.unifiedVision.loaded.Cfg
		return hipMultimodalPlaceholderBlock(cfg.ImageBeginToken, cfg.VideoToken, cfg.ImageEndToken, softTokens)
	}
	cfg := m.loaded.vision.loaded.Cfg
	return hipMultimodalPlaceholderBlock(cfg.ImageBeginToken, cfg.VideoToken, cfg.ImageEndToken, softTokens)
}

func (m *hipTokenModel) AcceptsAudioInput() bool {
	return m != nil && m.loaded != nil && m.loaded.AcceptsAudioInput()
}

func (m *hipTokenModel) AudioPlaceholderTokenID() int32 {
	if !m.AcceptsAudioInput() {
		return 0
	}
	if m.loaded.unifiedVision != nil && m.loaded.unifiedVision.AcceptsAudio() {
		return m.loaded.unifiedVision.loaded.Cfg.AudioTokenID
	}
	return int32(m.loaded.audio.loaded.Cfg.AudioTokenID)
}

func (m *hipTokenModel) AudioPlaceholderBlock(softTokens int) string {
	if !m.AcceptsAudioInput() || softTokens <= 0 {
		return ""
	}
	if m.loaded.unifiedVision != nil && m.loaded.unifiedVision.AcceptsAudio() {
		cfg := m.loaded.unifiedVision.loaded.Cfg
		return hipMultimodalPlaceholderBlock(cfg.AudioBeginToken, cfg.AudioToken, cfg.AudioEndToken, softTokens)
	}
	cfg := m.loaded.audio.loaded.Cfg
	return hipMultimodalPlaceholderBlock(cfg.AudioBeginToken, cfg.AudioToken, cfg.AudioEndToken, softTokens)
}

func (m *hipTokenModel) ProjectAudio(payload []byte) ([]byte, int, error) {
	if !m.AcceptsAudioInput() {
		return nil, 0, core.NewError("hip.TokenModel.ProjectAudio: loaded model has no audio tower")
	}
	waveform, err := hipDecodeWAVMono16k(payload)
	if err != nil {
		return nil, 0, core.E("hip.TokenModel.ProjectAudio", "decode WAV", err)
	}
	if m.loaded.unifiedVision != nil && m.loaded.unifiedVision.AcceptsAudio() {
		embeddings, softTokens, err := m.loaded.unifiedVision.ProjectAudioSamples(waveform)
		if err != nil {
			return nil, 0, core.E("hip.TokenModel.ProjectAudio", "project unified audio", err)
		}
		features, err := hipFloat32Payload(embeddings)
		if err != nil {
			return nil, 0, core.E("hip.TokenModel.ProjectAudio", "encode projected embeddings", err)
		}
		return features, softTokens, nil
	}
	embeddings, softTokens, err := m.loaded.audio.ProjectEmbeddings(waveform)
	if err != nil {
		return nil, 0, core.E("hip.TokenModel.ProjectAudio", "project audio", err)
	}
	hidden := m.loaded.modelInfo.HiddenSize
	if hidden <= 0 || softTokens <= 0 || len(embeddings) != softTokens*hidden {
		return nil, 0, core.NewError("hip.TokenModel.ProjectAudio: projected embedding geometry does not match text hidden size")
	}
	features, err := hipFloat32Payload(embeddings)
	if err != nil {
		return nil, 0, core.E("hip.TokenModel.ProjectAudio", "encode projected embeddings", err)
	}
	return features, softTokens, nil
}

func (m *hipTokenModel) TokenEmbeddingsWithFeatures(ids []int32, imageFeatures, audioFeatures, videoFeatures []byte) ([][]byte, error) {
	if m == nil || m.loaded == nil {
		return nil, core.NewError("hip.TokenModel.TokenEmbeddingsWithFeatures: model is not initialised")
	}
	if len(ids) == 0 {
		return nil, core.NewError("hip.TokenModel.TokenEmbeddingsWithFeatures: empty token ids")
	}
	if m.loaded.modelInfo.NumLayers <= 0 {
		return nil, core.NewError("hip.TokenModel.TokenEmbeddingsWithFeatures: loaded model layer count is required")
	}
	cfg, err := m.loaded.cachedGemma4Q4ForwardConfig(m.loaded.modelInfo.NumLayers)
	if err != nil {
		return nil, err
	}
	hidden := cfg.Layers[0].HiddenSize
	if hidden <= 0 {
		return nil, core.NewError("hip.TokenModel.TokenEmbeddingsWithFeatures: invalid embedding width")
	}
	device, err := hipRunGemma4Q4PrefillEmbeddingBatch(context.Background(), m.loaded.driver, cfg.Layers[0], ids)
	if err != nil {
		return nil, core.E("hip.TokenModel.TokenEmbeddingsWithFeatures", "gather token embeddings", err)
	}
	defer func() { _ = device.Close() }()
	rowBytes := hidden * 4
	stream := make([]byte, len(ids)*rowBytes)
	if err := m.loaded.driver.CopyDeviceToHost(device.Pointer(), stream); err != nil {
		return nil, core.E("hip.TokenModel.TokenEmbeddingsWithFeatures", "copy token embeddings", err)
	}
	if len(imageFeatures) > 0 {
		if err := m.spliceTokenFeaturesInto(stream, ids, imageFeatures, rowBytes, m.ImagePlaceholderTokenID(), "image"); err != nil {
			return nil, err
		}
	}
	if len(audioFeatures) > 0 {
		if err := m.spliceTokenFeaturesInto(stream, ids, audioFeatures, rowBytes, m.AudioPlaceholderTokenID(), "audio"); err != nil {
			return nil, err
		}
	}
	if len(videoFeatures) > 0 {
		if err := m.spliceTokenFeaturesInto(stream, ids, videoFeatures, rowBytes, m.VideoPlaceholderTokenID(), "video"); err != nil {
			return nil, err
		}
	}
	rows := make([][]byte, len(ids))
	for index := range ids {
		rows[index] = stream[index*rowBytes : (index+1)*rowBytes]
	}
	return rows, nil
}

func (m *hipTokenModel) spliceTokenFeaturesInto(stream []byte, ids []int32, features []byte, rowBytes int, tokenID int32, label string) error {
	if tokenID == 0 {
		return core.NewError("hip.TokenModel.TokenEmbeddingsWithFeatures: " + label + " token id is not configured")
	}
	if rowBytes <= 0 || len(stream) != len(ids)*rowBytes || len(features)%rowBytes != 0 {
		return core.NewError("hip.TokenModel.TokenEmbeddingsWithFeatures: invalid " + label + " feature geometry")
	}
	slots := 0
	for _, id := range ids {
		if id == tokenID {
			slots++
		}
	}
	if slots != len(features)/rowBytes {
		return core.NewError("hip.TokenModel.TokenEmbeddingsWithFeatures: " + label + " feature count must equal token slots")
	}
	featureIndex := 0
	for position, id := range ids {
		if id != tokenID {
			continue
		}
		copy(stream[position*rowBytes:(position+1)*rowBytes], features[featureIndex*rowBytes:(featureIndex+1)*rowBytes])
		featureIndex++
	}
	return nil
}

func hipMultimodalPlaceholderBlock(begin, token, end string, softTokens int) string {
	if token == "" || softTokens <= 0 {
		return ""
	}
	var block core.Builder
	block.Grow(len(begin) + len(end) + softTokens*len(token))
	block.WriteString(begin)
	for range softTokens {
		block.WriteString(token)
	}
	block.WriteString(end)
	return block.String()
}

// newHipEngineTextModel assembles a loaded Gemma4-Q4 hip model as the shared
// engine.TextModel (inference.TextModel + inference.SessionFactory). The
// ModelInfo is taken from the loaded model's own metadata (architecture, vocab,
// layer/hidden sizes, quant — the hip-specific input the engine-neutral wrapper
// cannot derive); maxLen is the loaded context window; tok is the tokenizer the
// text-prompt serve boundary needs (loaded separately, as engine/metal does).
func newHipEngineTextModel(loaded *hipLoadedModel, tok engine.TextTokenizer, modelType string) (*engine.TextModel, error) {
	if loaded == nil {
		return nil, core.NewError("hip.EngineTextModel: loaded model is nil")
	}
	info := loaded.modelInfo
	if info.Architecture == "" {
		info.Architecture = modelType
	}
	maxLen := loaded.contextSize
	if maxLen <= 0 {
		maxLen = defaultContextLengthCap
	}
	return engine.NewTextModel(newHipTokenModel(loaded, tok, modelType), tok, modelType, info, maxLen), nil
}

// NewSession exposes the shared retained-session contract on the production
// ROCm model when its loaded Gemma4 runtime was composed through engine.TextModel.
func (m *rocmModel) NewSession() inference.SessionHandle {
	if m == nil || m.engineModel == nil {
		return nil
	}
	return m.engineModel.NewSession()
}

func (m *rocmModel) FormatChatPromptWithThinking(messages []inference.Message, enableThinking *bool) string {
	if m == nil || m.engineModel == nil {
		return ""
	}
	return m.engineModel.FormatChatPromptWithThinking(messages, enableThinking)
}

func (m *rocmModel) FormatChatContinuationWithThinking(messages []inference.Message, enableThinking *bool) string {
	if m == nil || m.engineModel == nil {
		return ""
	}
	return m.engineModel.FormatChatContinuationWithThinking(messages, enableThinking)
}

func (m *rocmModel) SetChatInterceptor(fn func(context.Context, []inference.Message, ...inference.GenerateOption) (iter.Seq[inference.Token], bool)) {
	if m == nil {
		return
	}
	m.stateMutex.Lock()
	m.chatIntercept = fn
	m.stateMutex.Unlock()
}

func (m *rocmModel) interceptChat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) (iter.Seq[inference.Token], bool) {
	if m == nil {
		return nil, false
	}
	m.stateMutex.Lock()
	intercept := m.chatIntercept
	m.stateMutex.Unlock()
	if intercept == nil {
		return nil, false
	}
	return intercept(ctx, messages, opts...)
}

func (m *rocmModel) RecordChatMetrics(promptTokens, generated int, start, decodeStart time.Time) {
	if m == nil {
		return
	}
	m.recordMetrics(promptTokens, generated, start, decodeStart)
}

func (m *rocmModel) MaxLen() int {
	if m == nil {
		return 0
	}
	return m.contextLength
}

func (m *rocmModel) Tokenize(text string) ([]int32, error) {
	if m == nil || m.engineModel == nil {
		return nil, core.NewError("rocm.Tokenize: shared engine tokenizer is unavailable")
	}
	return m.engineModel.Tokenize(text)
}

func (m *rocmModel) DecodeToken(id int32) string {
	if m == nil || m.engineModel == nil {
		return ""
	}
	return m.engineModel.DecodeToken(id)
}

func (m *rocmModel) ResolvedStopTokens(requestStops []int32) []int32 {
	if m == nil || m.engineModel == nil {
		return append([]int32(nil), requestStops...)
	}
	return m.engineModel.ResolvedStopTokens(requestStops)
}

func (m *rocmModel) BatchStepAvailable() bool {
	if m != nil {
		if loaded, ok := m.native.(*hipLoadedModel); ok && loaded.gemma4LoRA != nil {
			return false
		}
	}
	return m != nil && m.engineModel != nil && m.engineModel.BatchStepAvailable()
}

func (m *rocmModel) OpenLaneSet(cfg inference.LaneSetConfig) (inference.LaneSet, error) {
	if m == nil || m.engineModel == nil {
		return nil, core.NewError("rocm.OpenLaneSet: shared engine model is unavailable")
	}
	return m.engineModel.OpenLaneSet(cfg)
}
