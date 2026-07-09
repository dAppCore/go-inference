// SPDX-Licence-Identifier: EUPL-1.2

package enginetest

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// fakeTextModel is the kit's minimal conformant TextModel — the self-test
// implementer and the worked example. State is one counter.
type fakeTextModel struct {
	generated int
}

func (f *fakeTextModel) tokens(cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		n := cfg.MaxTokens
		if n <= 0 || n > 4 {
			n = 4
		}
		f.generated = 0
		for i := 0; i < n; i++ {
			if !yield(inference.Token{ID: int32(i), Text: "x"}) {
				return
			}
			f.generated++
		}
	}
}

func (f *fakeTextModel) Generate(_ context.Context, _ string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return f.tokens(inference.ApplyGenerateOpts(opts))
}

func (f *fakeTextModel) Chat(_ context.Context, _ []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return f.tokens(inference.ApplyGenerateOpts(opts))
}

func (f *fakeTextModel) Classify(_ context.Context, prompts []string, _ ...inference.GenerateOption) core.Result {
	results := make([]inference.ClassifyResult, len(prompts))
	return core.Ok(results)
}

func (f *fakeTextModel) BatchGenerate(_ context.Context, prompts []string, _ ...inference.GenerateOption) core.Result {
	results := make([]inference.BatchResult, len(prompts))
	return core.Ok(results)
}

func (f *fakeTextModel) ModelType() string { return "fake" }

func (f *fakeTextModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "fake", VocabSize: 16, NumLayers: 1, HiddenSize: 8}
}

func (f *fakeTextModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{GeneratedTokens: f.generated}
}

func (f *fakeTextModel) Err() core.Result   { return core.Ok(nil) }
func (f *fakeTextModel) Close() core.Result { return core.Ok(nil) }

var _ inference.TextModel = (*fakeTextModel)(nil)

// TestTextModel_SuiteSelfTest_Good proves the TextModel conformance suite
// runs end-to-end against a minimal conformant implementer.
func TestTextModel_SuiteSelfTest_Good(t *testing.T) {
	TextModel(t, func(*testing.T) inference.TextModel {
		return &fakeTextModel{}
	})
}
