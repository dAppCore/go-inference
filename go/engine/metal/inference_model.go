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
	_ engine.TokenModel               = (*NativeTokenModel)(nil)
	_ engine.PromptReuseCapableModel  = (*NativeTokenModel)(nil)
	_ engine.Session                  = (*ArchSession)(nil)
	_ engine.PromptReuseSession       = (*ArchSession)(nil)
	_ engine.CanonicalLandingSession  = (*ArchSession)(nil)
	_ engine.TrainerModel             = (*NativeTokenModel)(nil)
	_ engine.Trainer                  = (*LoRATrainer)(nil)
	_ engine.CacheModeReporter        = (*NativeTokenModel)(nil)
	_ engine.StopTokenDeclarer        = (*NativeTokenModel)(nil)
	_ engine.SamplingDefaultsDeclarer = (*NativeTokenModel)(nil)
	_ engine.ChatTemplateDeclarer     = (*NativeTokenModel)(nil)
	_ engine.DecodePhaseTracer        = (*ArchSession)(nil)
)

// DeclaredChatTemplate declares this checkpoint's chat dialect
// (engine.ChatTemplateDeclarer): the gemma turn template in the marker dialect
// the loaded tokenizer carries (gemma4 <|turn> vs the gemma3-era
// <start_of_turn>), pre-closing the empty thought channel when the large-variant
// geometry needs it (NeedsThoughtChannelSuppressor). Declaring it — rather than
// leaving engine.TextModel to detect — is the seam a second architecture will
// self-report its own dialect through; today it hands back exactly what the
// tokenizer-detected fallback would build, so gemma rendering is unchanged.
func (m *NativeTokenModel) DeclaredChatTemplate() (engine.ChatTemplate, bool) {
	if m == nil {
		return engine.ChatTemplate{}, false
	}
	tok := m.Tokenizer()
	if tok == nil {
		return engine.ChatTemplate{}, false
	}
	// ChatML for a <|im_start|> vocab (Qwen/Yi/Mistral), else the gemma dialect in the tokenizer's
	// marker flavour — a gemma checkpoint declares exactly what it did before, while a Qwen2 pack
	// stops being mis-framed as gemma (whose <start_of_turn> markers it echoes instead of answering).
	tmpl := engine.DetectChatTemplate(tok, engine.DetectTurnTokens(tok), m.NeedsThoughtChannelSuppressor())
	// The checkpoint's own default system prompt (Qwen2.5 injects one; gemma and
	// Qwen3.5/3.6 leave this ""), so a no-system chat frames the vendor default
	// exactly rather than omitting it. Empty is a no-op in the render loop.
	tmpl.DefaultSystem = m.defaultSystem
	return tmpl, true
}

// SupportedCacheModes reports the one KV cache mode this no-cgo engine runs: its
// own built-in cache, selected automatically (engine.CacheModeReporter). It
// exposes no runtime selector for the go-mlx-era alternatives (fp16 / q8 /
// kq8vq4 / turboquant) — those were pkg/metal load options that did not port —
// so a -kv-cache override naming any other mode is not honoured. Reporting the
// single real mode lets callers print an accurate note, and future engines that
// do honour a selector list theirs here through the same seam.
func (m *NativeTokenModel) SupportedCacheModes() []string {
	return []string{"native"}
}

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

// SessionsReusePrompts declares that this engine's ArchSession implements
// engine.PromptReuseSession (PrefillTokensCached, session_prompt_reuse.go) —
// the model-level probe behind the stateless lane's resident prompt cache.
func (m *NativeTokenModel) SessionsReusePrompts() bool {
	return true
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

// OpenTrainer opens a retained head-LoRA SFT trainer over this loaded model — the metal half of the
// engine.TrainerModel seam. The returned engine.Trainer (a *LoRATrainer) owns a fresh frozen base
// session and a zero-initialised head adapter; cfg supplies the LoRA rank/alpha and learning rate.
func (m *NativeTokenModel) OpenTrainer(cfg inference.TrainingConfig) (engine.Trainer, error) {
	return NewLoRATrainer(m, cfg)
}
