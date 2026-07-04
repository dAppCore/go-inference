// SPDX-Licence-Identifier: EUPL-1.2

package enginetest

import (
	"context"
	"testing"

	"dappco.re/go/inference"
)

// ModelFactory builds a fresh loaded model for one subtest. The suite owns
// the returned model's lifecycle (it will Close it).
type ModelFactory func(t *testing.T) inference.TextModel

// TextModel runs the conformance suite for one engine's
// [inference.TextModel] implementation: the lifecycle, shape, and error
// invariants any conformant model must satisfy, independent of output
// content. Batch surfaces (Classify, BatchGenerate) may return a clean
// failure Result on engines whose fixture cannot serve them — the suite
// then skips the shape checks with a note; panics and malformed Results
// always fail.
func TextModel(t *testing.T, factory ModelFactory) {
	t.Helper()
	ctx := context.Background()

	t.Run("GenerateIsBoundedAndCleans", func(t *testing.T) {
		m := factory(t)
		defer func() { _ = m.Close() }()
		count := 0
		for range m.Generate(ctx, "conformance", inference.WithMaxTokens(8)) {
			count++
		}
		if count == 0 {
			t.Fatal("Generate produced no tokens")
		}
		if count > 8 {
			t.Fatalf("Generate produced %d tokens, budget was 8", count)
		}
		if r := m.Err(); !r.OK {
			t.Fatalf("Err after clean generation = %+v, want OK", r)
		}
	})

	t.Run("ChatProducesTokens", func(t *testing.T) {
		m := factory(t)
		defer func() { _ = m.Close() }()
		count := 0
		for range m.Chat(ctx, []inference.Message{{Role: "user", Content: "hi"}}, inference.WithMaxTokens(4)) {
			count++
		}
		if count == 0 {
			t.Fatal("Chat produced no tokens")
		}
	})

	t.Run("MetricsReflectCompletedGeneration", func(t *testing.T) {
		m := factory(t)
		defer func() { _ = m.Close() }()
		for range m.Generate(ctx, "count me", inference.WithMaxTokens(4)) {
		}
		if got := m.Metrics().GeneratedTokens; got <= 0 {
			t.Fatalf("Metrics().GeneratedTokens = %d after a completed generation, want > 0", got)
		}
	})

	t.Run("InfoAndModelTypeAreSane", func(t *testing.T) {
		m := factory(t)
		defer func() { _ = m.Close() }()
		if m.ModelType() == "" {
			t.Fatal("ModelType() is empty — a conformant model identifies its architecture")
		}
		info := m.Info()
		if info.VocabSize < 0 || info.NumLayers < 0 || info.HiddenSize < 0 {
			t.Fatalf("Info() carries negative geometry: %+v", info)
		}
	})

	t.Run("ClassifyResultShape", func(t *testing.T) {
		m := factory(t)
		defer func() { _ = m.Close() }()
		prompts := []string{"a", "b"}
		r := m.Classify(ctx, prompts)
		if !r.OK {
			t.Skip("Classify returned a clean failure — fixture does not serve classification")
		}
		results, ok := r.Value.([]inference.ClassifyResult)
		if !ok {
			t.Fatalf("Classify OK Result carries %T, want []inference.ClassifyResult", r.Value)
		}
		if len(results) != len(prompts) {
			t.Fatalf("Classify returned %d results for %d prompts", len(results), len(prompts))
		}
	})

	t.Run("BatchGenerateResultShape", func(t *testing.T) {
		m := factory(t)
		defer func() { _ = m.Close() }()
		prompts := []string{"a", "b", "c"}
		r := m.BatchGenerate(ctx, prompts, inference.WithMaxTokens(4))
		if !r.OK {
			t.Skip("BatchGenerate returned a clean failure — fixture does not serve batching")
		}
		results, ok := r.Value.([]inference.BatchResult)
		if !ok {
			t.Fatalf("BatchGenerate OK Result carries %T, want []inference.BatchResult", r.Value)
		}
		if len(results) != len(prompts) {
			t.Fatalf("BatchGenerate returned %d results for %d prompts", len(results), len(prompts))
		}
	})

	t.Run("CloseIsClean", func(t *testing.T) {
		m := factory(t)
		if r := m.Close(); !r.OK {
			t.Fatalf("Close on fresh model = %+v, want OK", r)
		}
	})
}
