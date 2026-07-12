// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
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

// TestArchSessionCloseRetainedHiddenPinned covers both closeRetainedHiddenPinned arms directly
// (#30 r4): aliased with the prompt cache's pinned hidden (WarmPromptCache — must NOT Close() a
// buffer the prompt cache still owns, only clear the session's own reference) and an ordinary
// session-owned buffer (no alias — Close()s the Metal buffer and clears). A nil receiver is the
// third arm (the method's own no-op guard, exercised the same way every nil-safe method in this
// file is).
func TestArchSessionCloseRetainedHiddenPinned(t *testing.T) {
	requireNativeRuntime(t)

	t.Run("aliased with prompt cache", func(t *testing.T) {
		warm := newSessionStateFixture(t)
		if err := warm.WarmPromptCache([]int32{1, 2, 3}); err != nil {
			t.Fatalf("WarmPromptCache: %v", err)
		}
		if warm.cachedPromptHiddenPinned == nil || warm.retainedHiddenPinned != warm.cachedPromptHiddenPinned {
			t.Fatal("fixture must alias retainedHiddenPinned with cachedPromptHiddenPinned to exercise the aliased arm")
		}
		aliased := warm.cachedPromptHiddenPinned
		warm.closeRetainedHiddenPinned()
		if warm.retainedHiddenPinned != nil || warm.retainedHidden != nil {
			t.Fatal("closeRetainedHiddenPinned did not clear the session's own reference")
		}
		if warm.cachedPromptHiddenPinned != aliased || warm.cachedPromptHiddenPinned.buf == nil {
			t.Fatal("closeRetainedHiddenPinned Close()d a buffer the prompt cache still owns")
		}
	})

	t.Run("owned buffer", func(t *testing.T) {
		sess := newSessionStateFixture(t)
		pinned, ok := sess.ensureRetainedHiddenPinned(sess.arch.Hidden * bf16Size)
		if !ok || pinned == nil || pinned.buf == nil {
			t.Fatal("ensureRetainedHiddenPinned did not allocate a pinned buffer")
		}
		sess.retainedHidden = pinned.bytes
		sess.closeRetainedHiddenPinned()
		if sess.retainedHiddenPinned != nil || sess.retainedHidden != nil {
			t.Fatal("closeRetainedHiddenPinned did not clear an owned buffer")
		}
	})

	t.Run("nil receiver", func(t *testing.T) {
		var sess *ArchSession
		sess.closeRetainedHiddenPinned() // must not panic
	})
}

// TestArchSessionCloseRetainedLogitsPinned is closeRetainedHiddenPinned's twin for the retained
// logits pinned buffer — same three arms (aliased with the prompt cache, owned, nil receiver).
func TestArchSessionCloseRetainedLogitsPinned(t *testing.T) {
	requireNativeRuntime(t)

	t.Run("aliased with prompt cache", func(t *testing.T) {
		warm := newSessionStateFixture(t)
		if err := warm.WarmPromptCache([]int32{1, 2, 3}); err != nil {
			t.Fatalf("WarmPromptCache: %v", err)
		}
		if warm.cachedPromptLogitsPinned == nil || warm.retainedLogitsPinned != warm.cachedPromptLogitsPinned {
			t.Fatal("fixture must alias retainedLogitsPinned with cachedPromptLogitsPinned to exercise the aliased arm")
		}
		aliased := warm.cachedPromptLogitsPinned
		warm.closeRetainedLogitsPinned()
		if warm.retainedLogitsPinned != nil || warm.retainedLogits != nil {
			t.Fatal("closeRetainedLogitsPinned did not clear the session's own reference")
		}
		if warm.cachedPromptLogitsPinned != aliased || warm.cachedPromptLogitsPinned.buf == nil {
			t.Fatal("closeRetainedLogitsPinned Close()d a buffer the prompt cache still owns")
		}
	})

	t.Run("owned buffer", func(t *testing.T) {
		sess := newSessionStateFixture(t)
		pinned, ok := sess.ensureRetainedLogitsPinned(sess.arch.Vocab * bf16Size)
		if !ok || pinned == nil || pinned.buf == nil {
			t.Fatal("ensureRetainedLogitsPinned did not allocate a pinned buffer")
		}
		sess.retainedLogits = pinned.bytes
		sess.closeRetainedLogitsPinned()
		if sess.retainedLogitsPinned != nil || sess.retainedLogits != nil {
			t.Fatal("closeRetainedLogitsPinned did not clear an owned buffer")
		}
	})

	t.Run("nil receiver", func(t *testing.T) {
		var sess *ArchSession
		sess.closeRetainedLogitsPinned() // must not panic
	})
}

// newBidirSpanFixture builds a plain (non-PLE) bf16 gemma4-shaped arch — WITH the q_norm/k_norm
// towers AND numerically-varied weights. The bidirectional-span lane hard-errors without q_norm/
// k_norm: stepTokensBatchedDenseResultWithInputViewsPLE requires the fused qknorm-rope fold
// (foldAttn, gated on s.lb[li].qNorm.buf != nil) whenever s.rowAttnCaps is armed — "bidirectional
// row caps need the batched-rope attention fold". bf16Gemma4TensorsVaried (not gemma4Tensors,
// which fills every tensor with ONE repeated byte for field-mapping tests — every dimension of
// every embedding row then collapses to the same constant, so attending to a different row count
// changes nothing and any bidir/causal comparison passes vacuously) is load-bearing here: real
// per-element variation is what makes the two attention patterns numerically distinguishable.
func newBidirSpanFixture(t testing.TB) *ArchSession {
	t.Helper()
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, maxLen = 2, 32
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := bf16Gemma4TensorsVaried(t, arch)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g := loadedToBF16(lm)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	t.Cleanup(func() { sess.Close() })
	sess.state.icb = nil
	return sess
}

// TestArchSessionPrefillTokenEmbeddingsBidirSpanEngages exercises the bidirectional image/video
// span prefill lane (#30 r4): bidirSpanTokens marks a placeholder token id whose contiguous runs
// attend BIDIRECTIONALLY (a span row sees through to the span's end, not just its own causal
// prefix) — prefillRetainedEmbeddingsBidir/-Chunk + bidirTokenSpans's real caller-side wiring
// (token_model.go sets bidirSpanTokens from a real unified-vision config; this test sets it
// directly, the same in-package seam TestArchSessionPrefillTokenEmbeddingsKeepsFullStackIgnoresSkip
// uses for the causal skip-suffix guard). No sliding window: the whole prompt lands in ONE
// bidirectional chunk — prefillRetainedEmbeddingsBidir's per-span chunk-boundary adjustment loop
// (shrink/grow to a span edge) only matters when a span straddles a windowed chunk split, which
// this fixture never forces; that sub-branch is a remaining item (see the r4 report).
func TestArchSessionPrefillTokenEmbeddingsBidirSpanEngages(t *testing.T) {
	requireNativeRuntime(t)
	const spanTok = 9
	ids := []int32{1, 2, spanTok, spanTok, spanTok, 3, 4}

	buildEmbeddings := func(sess *ArchSession) [][]byte {
		embeddings := make([][]byte, len(ids))
		for i, id := range ids {
			emb, err := sess.embedID(id)
			if err != nil {
				t.Fatalf("embedID(%d): %v", id, err)
			}
			embeddings[i] = append([]byte(nil), emb...)
		}
		return embeddings
	}

	bidir := newBidirSpanFixture(t)
	bidir.bidirSpanTokens = [2]int32{spanTok, 0}
	bidirEmbeddings := buildEmbeddings(bidir)
	if err := bidir.PrefillTokenEmbeddings(ids, bidirEmbeddings); err != nil {
		t.Fatalf("PrefillTokenEmbeddings (bidir): %v", err)
	}
	if bidir.Pos() != len(ids) {
		t.Fatalf("Pos after bidir prefill = %d, want %d", bidir.Pos(), len(ids))
	}
	if len(bidir.retainedHidden) != bidir.arch.Hidden*bf16Size {
		t.Fatalf("bidir retained hidden len = %d, want %d", len(bidir.retainedHidden), bidir.arch.Hidden*bf16Size)
	}

	causal := newBidirSpanFixture(t) // bidirSpanTokens left zero — takes the ordinary causal lane
	causalEmbeddings := buildEmbeddings(causal)
	if err := causal.PrefillTokenEmbeddings(ids, causalEmbeddings); err != nil {
		t.Fatalf("PrefillTokenEmbeddings (causal): %v", err)
	}

	// A real invariant, not a pinned byte value: bidirectional attention lets the span's own
	// rows see FUTURE span tokens, which a strictly causal mask never does, so the two boundary
	// hiddens (both fed by the same span's cache rows through the later layers) must differ — an
	// accidental no-op bidir path (e.g. a caps slice that silently stays causal) would make them
	// byte-identical.
	if bytes.Equal(bidir.retainedHidden, causal.retainedHidden) {
		t.Fatal("bidirectional-span prefill produced the SAME boundary hidden as the causal lane — the span caps had no effect")
	}
}
