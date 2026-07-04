// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"

	"dappco.re/go/inference/model"
)

func TestArchSessionPrefillAppendGenerateFromCache(t *testing.T) {
	requireNativeRuntime(t)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4, 5}
	full := append(append([]int32{}, prefix...), suffix...)

	retained := newSessionStateFixture(t)
	if err := retained.PrefillTokens(prefix); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if retained.Pos() != len(prefix) {
		t.Fatalf("Pos after PrefillTokens = %d, want %d", retained.Pos(), len(prefix))
	}
	if !idsEqual(retained.cachedIDs, prefix) {
		t.Fatalf("cached ids after PrefillTokens = %v, want %v", retained.cachedIDs, prefix)
	}
	if err := retained.AppendTokens(suffix); err != nil {
		t.Fatalf("AppendTokens: %v", err)
	}
	if retained.Pos() != len(full) {
		t.Fatalf("Pos after AppendTokens = %d, want %d", retained.Pos(), len(full))
	}
	if !idsEqual(retained.cachedIDs, full) {
		t.Fatalf("cached ids after AppendTokens = %v, want %v", retained.cachedIDs, full)
	}

	got, err := retained.GenerateFromCache(4, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(full, 4, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("GenerateFromCache = %v, want cold retained-state continuation %v", got, want)
	}
	if retained.Pos() != len(full)+len(got) {
		t.Fatalf("Pos after GenerateFromCache = %d, want %d", retained.Pos(), len(full)+len(got))
	}
	if !idsEqual(retained.cachedIDs, append(append([]int32{}, full...), got...)) {
		t.Fatalf("cached ids after GenerateFromCache = %v, want full prompt plus generated %v", retained.cachedIDs, got)
	}
}

func TestArchSessionPrefillTokensResetsRetainedState(t *testing.T) {
	requireNativeRuntime(t)

	retained := newSessionStateFixture(t)
	if _, err := retained.Generate([]int32{9, 8, 7}, 2, -1); err != nil {
		t.Fatalf("seed Generate: %v", err)
	}
	prompt := []int32{1, 2, 3, 4}
	if err := retained.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens reset: %v", err)
	}
	got, err := retained.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after reset: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("GenerateFromCache after reset = %v, want cold prompt continuation %v", got, want)
	}
}

func TestArchSessionPrefillTokenEmbeddingsRetainsBoundary_Good(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4}

	retained := newSessionStateFixture(t)
	embeddings := make([][]byte, len(prompt))
	for i, id := range prompt {
		emb, err := retained.embedID(id)
		if err != nil {
			t.Fatalf("embedID(%d): %v", id, err)
		}
		embeddings[i] = append([]byte(nil), emb...)
	}
	if err := retained.PrefillTokenEmbeddings(prompt, embeddings); err != nil {
		t.Fatalf("PrefillTokenEmbeddings: %v", err)
	}
	if retained.Pos() != len(prompt) {
		t.Fatalf("Pos after PrefillTokenEmbeddings = %d, want %d", retained.Pos(), len(prompt))
	}
	if !idsEqual(retained.cachedIDs, prompt) {
		t.Fatalf("cached ids after PrefillTokenEmbeddings = %v, want %v", retained.cachedIDs, prompt)
	}
	if _, err := retained.BoundaryLogits(); err != nil {
		t.Fatalf("BoundaryLogits after PrefillTokenEmbeddings: %v", err)
	}
	got, err := retained.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after PrefillTokenEmbeddings: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("GenerateFromCache after explicit embeddings = %v, want %v", got, want)
	}
}

func TestArchSessionGenerateFromCacheTransformedUsesRetainedLogitsWithoutHidden(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := sess.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	raw, err := model.Greedy(logits, sess.arch.Vocab)
	if err != nil {
		t.Fatalf("Greedy: %v", err)
	}
	want := (raw + 1) % int32(sess.arch.Vocab)
	sess.retainedHidden = nil

	got, err := sess.GenerateFromCacheEachTransformed(1, -1, func(id int32) int32 {
		if id == raw {
			return want
		}
		return id
	}, nil)
	if err != nil {
		t.Fatalf("GenerateFromCacheEachTransformed with retained logits only: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("GenerateFromCacheEachTransformed generated %d tokens, want 1", len(got))
	}
	if got[0] != want {
		t.Fatalf("GenerateFromCacheEachTransformed token = %d, want transformed retained-logits token %d", got[0], want)
	}
	if !idsEqual(sess.cachedIDs, append(append([]int32{}, prompt...), want)) {
		t.Fatalf("cached ids after transformed retained-logits replay = %v, want prompt plus %d", sess.cachedIDs, want)
	}
}

func TestArchSessionGenerateFromCacheSuppressionUsesRetainedLogitsWithoutHidden(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	suppressed := int32(sess.arch.Vocab - 2)
	want := int32(sess.arch.Vocab - 1)
	logits := make([]float32, sess.arch.Vocab)
	for i := range logits {
		logits[i] = -8
	}
	logits[suppressed] = 9
	logits[want] = 6
	sess.retainedLogits = toBF16Bytes(logits)
	sess.retainedHidden = nil

	got, err := sess.GenerateFromCacheEachWithSuppressionAndTransform(1, -1, []int32{suppressed}, nil, nil)
	if err != nil {
		t.Fatalf("GenerateFromCacheEachWithSuppressionAndTransform with retained logits only: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("GenerateFromCacheEachWithSuppressionAndTransform generated %d tokens, want 1", len(got))
	}
	if got[0] != want {
		t.Fatalf("GenerateFromCacheEachWithSuppressionAndTransform token = %d, want retained-logits unsuppressed token %d", got[0], want)
	}
	if !idsEqual(sess.cachedIDs, append(append([]int32{}, prompt...), want)) {
		t.Fatalf("cached ids after suppressed retained-logits replay = %v, want prompt plus %d", sess.cachedIDs, want)
	}
}

func TestArchSessionAppendTokensUsesRestoredKVWithoutRetainedHidden(t *testing.T) {
	requireNativeRuntime(t)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4, 5}
	full := append(append([]int32(nil), prefix...), suffix...)

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prefix); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := saved.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}

	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	source.RetainedHidden = nil
	source.RetainedLogits = logits

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	if len(restored.retainedHidden) != 0 {
		t.Fatal("RestoreStateBlocks unexpectedly retained hidden")
	}
	if err := restored.AppendTokens(suffix); err != nil {
		t.Fatalf("AppendTokens after retained-logits restore: %v", err)
	}
	if len(restored.retainedLogits) != 0 {
		t.Fatalf("AppendTokens retained logits length = %d, want reset", len(restored.retainedLogits))
	}
	if len(restored.retainedHidden) != restored.arch.Hidden*bf16Size {
		t.Fatalf("AppendTokens retained hidden length = %d, want %d", len(restored.retainedHidden), restored.arch.Hidden*bf16Size)
	}
	if restored.Pos() != len(full) {
		t.Fatalf("Pos after AppendTokens = %d, want %d", restored.Pos(), len(full))
	}
	if !idsEqual(restored.cachedIDs, full) {
		t.Fatalf("cached ids after AppendTokens = %v, want %v", restored.cachedIDs, full)
	}

	control := newSessionStateFixture(t)
	if err := control.PrefillTokens(prefix); err != nil {
		t.Fatalf("control PrefillTokens: %v", err)
	}
	if err := control.AppendTokens(suffix); err != nil {
		t.Fatalf("control AppendTokens: %v", err)
	}
	if !bytes.Equal(restored.retainedHidden, control.retainedHidden) {
		t.Fatal("restored AppendTokens retained hidden did not match non-restored append")
	}

	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after AppendTokens: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(full, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("GenerateFromCache after restored append = %v, want cold continuation %v", got, want)
	}
}

func TestArchSessionRestoreStatePreservesGenerateFromCacheBoundary(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	blob, err := saved.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}
	restored := newSessionStateFixture(t)
	if err := restored.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreState: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("restored GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
}

func TestArchSessionGenerateRecordsResidentIDs(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3}

	sess := newSessionStateFixture(t)
	got, err := sess.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	wantResident := append(append([]int32(nil), prompt...), got...)
	if sess.Pos() != len(wantResident) {
		t.Fatalf("Pos after generate = %d, want %d", sess.Pos(), len(wantResident))
	}
	if !idsEqual(sess.cachedIDs, wantResident) {
		t.Fatalf("cached ids after generate = %v, want prompt plus generated %v", sess.cachedIDs, wantResident)
	}
}

func TestArchSessionGenerateSampledEachRecordsResidentIDs(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9}

	sess := newSessionStateFixture(t)
	got, err := sess.GenerateSampledEach(prompt, 3, nil, model.NewSampler(17), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	wantResident := append(append([]int32(nil), prompt...), got...)
	if sess.Pos() != len(wantResident) {
		t.Fatalf("Pos after sampled generate = %d, want %d", sess.Pos(), len(wantResident))
	}
	if !idsEqual(sess.cachedIDs, wantResident) {
		t.Fatalf("cached ids after sampled generate = %v, want prompt plus generated %v", sess.cachedIDs, wantResident)
	}
}
