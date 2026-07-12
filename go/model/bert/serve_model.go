// SPDX-Licence-Identifier: EUPL-1.2

package bert

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ServeModel adapts a *Model to inference.TextModel so a host BERT encoder can
// be mounted on the OpenAI-compatible mux (compat.NewMux) and served over
// /v1/embeddings and /v1/rerank. It embeds *Model, so Embed and Rerank forward
// to the real encoder; the generative surface (Generate/Chat/Classify/
// BatchGenerate) is stubbed because an encoder has no autoregressive path —
// those routes report the model is embedding-only rather than pretending.
//
//	m, _ := bert.Load(dir)
//	handler := compat.NewMux(openai.NewStaticResolver(
//	    map[string]inference.TextModel{"bge-small": bert.NewServeModel(m)}))
type ServeModel struct {
	*Model
}

// NewServeModel wraps a loaded encoder as a servable TextModel.
func NewServeModel(model *Model) *ServeModel { return &ServeModel{Model: model} }

// Compile-time proof the adapter satisfies the serving contracts: a TextModel
// the mux can resolve, plus the embedding and rerank capabilities the
// /v1/embeddings and /v1/rerank gates assert.
var (
	_ inference.TextModel      = (*ServeModel)(nil)
	_ inference.EmbeddingModel = (*ServeModel)(nil)
	_ inference.RerankModel    = (*ServeModel)(nil)
)

// Generate yields no tokens — an encoder does not generate text. The route
// layer surfaces the empty stream; Err reports the embedding-only nature.
func (s *ServeModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return emptyTokenSeq
}

// Chat yields no tokens — see Generate.
func (s *ServeModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return emptyTokenSeq
}

// Classify is unsupported on an embedding-only model.
func (s *ServeModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.E("bert.ServeModel.Classify", "encoder is embedding-only; use Embed", nil))
}

// BatchGenerate is unsupported on an embedding-only model.
func (s *ServeModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.E("bert.ServeModel.BatchGenerate", "encoder is embedding-only; use Embed", nil))
}

// ModelType returns the architecture string from config.json (e.g. "bert").
func (s *ServeModel) ModelType() string { return s.cfg.ModelType }

// Info returns the encoder's architecture metadata.
func (s *ServeModel) Info() inference.ModelInfo {
	return inference.ModelInfo{
		Architecture: s.cfg.ModelType,
		VocabSize:    s.cfg.VocabSize,
		NumLayers:    s.cfg.NumHiddenLayers,
		HiddenSize:   s.cfg.HiddenSize,
	}
}

// Metrics returns zeroed counters — the encoder path reports token usage through
// the EmbeddingResult, not the generative metrics.
func (s *ServeModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }

// Err reports the embedding-only nature after a Generate/Chat call yielded no
// tokens, so a caller that mistakenly drove the generative path sees why.
func (s *ServeModel) Err() core.Result {
	return core.Fail(core.E("bert.ServeModel", "encoder is embedding-only; use /v1/embeddings or /v1/rerank", nil))
}

// Close releases nothing — the host encoder holds only heap weights.
func (s *ServeModel) Close() core.Result { return core.Ok(nil) }

// emptyTokenSeq is the shared no-token stream Generate and Chat return.
func emptyTokenSeq(func(inference.Token) bool) {}
