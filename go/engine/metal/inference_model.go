// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// inference_model.go is engine/metal's composition root for the go-inference
// serving contracts. The engine-agnostic wrapper logic (prefill / generate /
// capture / restore / fork, and the inference.TextModel / SessionHandle surface)
// lives in the shared package engine and is reused by engine/hip; only the two
// metal-specific pieces live here: assembling the loaded model's ModelInfo, and
// opening a fresh *ArchSession as an engine.Session. Wrapping the no-cgo
// NativeTokenModel this way registers the "metal" backend from go-inference
// alone, no go-mlx composition root.
package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
)

var (
	_ engine.TokenModel = (*NativeTokenModel)(nil)
	_ engine.Session    = (*ArchSession)(nil)
)

// newNativeTextModel wraps a loaded no-cgo token model as the shared
// engine.TextModel (inference.TextModel + inference.SessionFactory). The
// tokenizer is the one attached to tm (AttachTokenizer) — text↔ids is the serve
// boundary the model carries once loaded. The ModelInfo is assembled from the
// model's own loaded metadata (vocab, layer/hidden sizes, quant), the one
// metal-specific input the engine-neutral engine.TextModel cannot derive.
func newNativeTextModel(tm *NativeTokenModel, modelType string) *engine.TextModel {
	info := inference.ModelInfo{
		Architecture: modelType,
		VocabSize:    tm.Vocab(),
		NumLayers:    len(tm.arch.Layer),
		HiddenSize:   tm.arch.Hidden,
		QuantBits:    tm.quantBits,
		QuantGroup:   tm.quantGroup,
	}
	return engine.NewTextModel(tm, tm.Tokenizer(), modelType, info, tm.maxLen)
}

// OpenEngineSession opens a fresh incremental decode session (empty KV cache) as
// the engine.Session the shared adapters drive — *NativeTokenModel's half of the
// engine.TokenModel contract. The token model is a model.SessionModel; OpenSession
// returns the engine's ArchSession stepper, which speaks kv.Snapshot directly, so
// no metal.* / kvconv conversion is needed.
func (m *NativeTokenModel) OpenEngineSession() (engine.Session, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel: model is not initialised")
	}
	stepper, err := m.OpenSession()
	if err != nil {
		return nil, err
	}
	sess, ok := stepper.(*ArchSession)
	if !ok {
		if closer, closeOK := stepper.(interface{ Close() error }); closeOK {
			_ = closer.Close()
		}
		return nil, core.NewError("native.NativeTokenModel: token model does not open an ArchSession")
	}
	return sess, nil
}
