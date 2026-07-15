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
	_ engine.PromptReuseCapableModel   = (*hipTokenModel)(nil)
	_ engine.StopTokenDeclarer         = (*hipTokenModel)(nil)
	_ engine.SamplingDefaultsDeclarer  = (*hipTokenModel)(nil)
	_ engine.ThoughtSuppressorDeclarer = (*hipTokenModel)(nil)
	_ engine.ChatTemplateDeclarer      = (*hipTokenModel)(nil)
	_ engine.LaneSetOpener             = (*hipTokenModel)(nil)
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

// Training boundary: hipTokenModel deliberately does not implement
// engine.TrainerModel yet. The Gemma4-Q4 package forward remains experimental,
// and its exposed logits/final-hidden buffers do not form a retained Batch
// forward with response loss masking. RunNativeLoRABackwardPass consumes
// caller-provided activations and upstream gradients rather than gradients from
// that model forward, while the current adapter loader supports only tiny,
// small dense Qwen/Gemma LM-head, and BERT classifier adapters, not Gemma4.
// Adding OpenTrainer before those paths compose through Batch -> loss ->
// backward -> AdamW -> Save -> reload would advertise synthetic-array updates
// as model training.

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
	return engine.GemmaChatTemplate(engine.DetectTurnTokens(m.tokenizer), m.NeedsThoughtChannelSuppressor()), true
}

// OpenEngineSession opens a fresh retained Gemma4-Q4 decode session as the
// engine.Session the shared adapters drive.
func (m *hipTokenModel) OpenEngineSession() (engine.Session, error) {
	if m == nil || m.loaded == nil {
		return nil, core.NewError("hip.TokenModel: model is not initialised")
	}
	return newHipEngineSession(m.loaded)
}

// Close releases the loaded model's resident weights.
func (m *hipTokenModel) Close() error {
	if m == nil || m.loaded == nil {
		return nil
	}
	return m.loaded.Close()
}

func (m *hipTokenModel) SessionsReusePrompts() bool { return true }

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
	return m != nil && m.engineModel != nil && m.engineModel.BatchStepAvailable()
}

func (m *rocmModel) OpenLaneSet(cfg inference.LaneSetConfig) (inference.LaneSet, error) {
	if m == nil || m.engineModel == nil {
		return nil, core.NewError("rocm.OpenLaneSet: shared engine model is unavailable")
	}
	return m.engineModel.OpenLaneSet(cfg)
}
