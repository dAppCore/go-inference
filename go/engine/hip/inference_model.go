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
	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
)

var (
	_ engine.TokenModel               = (*hipTokenModel)(nil)
	_ engine.SamplingDefaultsDeclarer = (*hipTokenModel)(nil)
)

// hipTokenModel wraps a loaded Gemma4-Q4 hip model as the shared
// engine.TokenModel: OpenEngineSession opens a retained hipEngineSession (the
// engine.Session the shared adapters drive), and Close releases the resident
// weights. declaredSampling carries the checkpoint's generation_config sampling
// intent, parsed once at load, which engine.TextModel folds into each request
// (engine.SamplingDefaultsDeclarer — see generation_config.go).
type hipTokenModel struct {
	loaded           *hipLoadedModel
	tokenizer        *tokenizer.Tokenizer
	modelType        string
	declaredSampling engine.SamplingDefaults
}

// newHipTokenModel binds a loaded model + tokenizer as an engine.TokenModel,
// parsing the checkpoint's generation_config sampling defaults from the loaded
// model's directory so DeclaredSamplingDefaults reports the model's declared
// intent (the zero value when the file is absent or declares none).
func newHipTokenModel(loaded *hipLoadedModel, tok *tokenizer.Tokenizer, modelType string) *hipTokenModel {
	m := &hipTokenModel{loaded: loaded, tokenizer: tok, modelType: modelType}
	if loaded != nil {
		m.declaredSampling = loadGenerationConfigSamplingDefaults(loaded.modelPath)
	}
	return m
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

// newHipEngineTextModel assembles a loaded Gemma4-Q4 hip model as the shared
// engine.TextModel (inference.TextModel + inference.SessionFactory). The
// ModelInfo is taken from the loaded model's own metadata (architecture, vocab,
// layer/hidden sizes, quant — the hip-specific input the engine-neutral wrapper
// cannot derive); maxLen is the loaded context window; tok is the tokenizer the
// text-prompt serve boundary needs (loaded separately, as engine/metal does).
func newHipEngineTextModel(loaded *hipLoadedModel, tok *tokenizer.Tokenizer, modelType string) (*engine.TextModel, error) {
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
