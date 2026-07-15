// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"errors"
	"testing"
	"unsafe"

	"dappco.re/go/inference/model"
)

func TestPromptCacheInputGuards(t *testing.T) {
	var nilSession *ArchSession
	nilSession.ClearPromptCache()

	sess := &ArchSession{}
	if _, err := sess.GenerateCached(nil, 1, -1); err == nil {
		t.Fatal("GenerateCached(nil prompt) error = nil")
	}
	if _, err := sess.GenerateCached([]int32{1}, 0, -1); err == nil {
		t.Fatal("GenerateCached(maxNew=0) error = nil")
	}
	if _, err := sess.GenerateCachedSampledEach([]int32{1}, 1, nil, nil, model.SampleParams{}, nil, nil); err == nil {
		t.Fatal("GenerateCachedSampledEach(nil sampler) error = nil")
	}
	if _, err := sess.GenerateCachedSampledEach(nil, 1, nil, model.NewSampler(1), model.SampleParams{}, nil, nil); err == nil {
		t.Fatal("GenerateCachedSampledEach(nil prompt) error = nil")
	}
	if _, err := sess.GenerateCachedSampledEach([]int32{1}, 0, nil, model.NewSampler(1), model.SampleParams{}, nil, nil); err == nil {
		t.Fatal("GenerateCachedSampledEach(maxNew=0) error = nil")
	}
	if err := sess.WarmPromptCache(nil); err == nil {
		t.Fatal("WarmPromptCache(nil prompt) error = nil")
	}
	if err := sess.CompactCache(-1); err == nil {
		t.Fatal("CompactCache(negative keep) error = nil")
	}
	sess.cachedIDs = []int32{1, 2, 3}
	sess.pos = len(sess.cachedIDs)
	if err := sess.CompactCache(len(sess.cachedIDs)); err != nil {
		t.Fatalf("CompactCache(no-op keep) error = %v", err)
	}
	if sess.Pos() != 3 {
		t.Fatalf("CompactCache(no-op) pos = %d, want 3", sess.Pos())
	}
	if hit := sess.CachedPrefixLen(nil); hit != 0 {
		t.Fatalf("CachedPrefixLen(nil prompt) = %d, want 0", hit)
	}
	if err := sess.prefillCachedIDs(nil); err != nil {
		t.Fatalf("prefillCachedIDs(nil) error = %v", err)
	}
	sess.maxLen = 1
	sess.pos = 0
	sess.cachedIDs = []int32{9}
	if err := sess.WarmPromptCache([]int32{1, 2}); err == nil {
		t.Fatal("WarmPromptCache(over maxLen) error = nil")
	}
	if sess.Pos() != 0 {
		t.Fatalf("WarmPromptCache overflow Pos = %d, want 0", sess.Pos())
	}
	if len(sess.cachedIDs) != 0 {
		t.Fatalf("WarmPromptCache overflow cachedIDs = %v, want empty", sess.cachedIDs)
	}
	sess.maxLen = 1
	sess.pos = 1
	sess.cachedIDs = []int32{1}
	if _, err := sess.GenerateCached([]int32{1, 2}, 1, -1); err == nil {
		t.Fatal("GenerateCached(over maxLen) error = nil")
	}
	if sess.cachedIDs != nil {
		t.Fatalf("GenerateCached failed run cachedIDs = %v, want nil", sess.cachedIDs)
	}
	sess.maxLen = 1
	sess.pos = 1
	sess.cachedIDs = []int32{1}
	if _, err := sess.GenerateCachedSampledEach([]int32{1, 2}, 1, nil, model.NewSampler(1), model.SampleParams{}, nil, nil); err == nil {
		t.Fatal("GenerateCachedSampledEach(over maxLen) error = nil")
	}
	sess.cachedIDs = []int32{1, 2, 3}
	if hit := sess.CachedPrefixLen([]int32{1, 2, 3}); hit != 2 {
		t.Fatalf("CachedPrefixLen(exact prompt) = %d, want 2", hit)
	}
}

// TestGenerateCachedPrefixReuse proves native prompt caching: after a first turn warms the cache, a
// second prompt that shares a prefix is served by reusing that prefix's KV (re-prefilling only the
// suffix) and produces a TOKEN-IDENTICAL continuation to a cold Generate of the same full prompt — while
// CachedPrefixLen confirms the shared prefix was actually skipped.
func TestGenerateCachedPrefixReuse(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen = 64, 3, 96
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	mk := func() *ArchSession {
		s, err := NewArchSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		return s
	}

	shared := []int32{1, 2, 3, 4, 5}
	full := []int32{1, 2, 3, 4, 5, 6, 7} // extends `shared`

	// cold reference: a fresh session decodes the full prompt.
	cold, err := mk().Generate(full, 8, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}

	// warm session: a first turn on `shared`, then GenerateCached on `full` reuses the shared prefix.
	warm := mk()
	if _, err := warm.GenerateCached(shared, 6, -1); err != nil {
		t.Fatalf("warm turn 1: %v", err)
	}
	hit := warm.CachedPrefixLen(full)
	if hit != len(shared) {
		t.Fatalf("prompt-cache prefix hit = %d, want %d (the shared prefix)", hit, len(shared))
	}
	got, err := warm.GenerateCached(full, 8, -1)
	if err != nil {
		t.Fatalf("warm turn 2: %v", err)
	}
	if len(got) != len(cold) {
		t.Fatalf("length mismatch: warm=%d cold=%d", len(got), len(cold))
	}
	for i := range cold {
		if got[i] != cold[i] {
			t.Fatalf("token %d diverged: warm(cached)=%d cold=%d", i, got[i], cold[i])
		}
	}
	t.Logf("native prompt cache: reused %d-token prefix, continuation token-identical to a cold run over %d tokens", hit, len(got))
}

func TestGenerateCachedExactPromptUsesCachedHidden(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	warm := newSessionStateFixture(t)
	if _, err := warm.GenerateCached(prompt, 4, -1); err != nil {
		t.Fatalf("warm GenerateCached: %v", err)
	}
	if hit := warm.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}

	got, err := warm.GenerateCached(prompt, 4, -1)
	if err != nil {
		t.Fatalf("exact cached GenerateCached: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 4, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("generated length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token %d after exact prompt cache = %d, want %d", i, got[i], want[i])
		}
	}
	if warm.Pos() != len(prompt)+len(got) {
		t.Fatalf("Pos after exact prompt cache = %d, want %d", warm.Pos(), len(prompt)+len(got))
	}
}

func TestGenerateCachedEachExactPromptStopsAfterFirstYield(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	warm := newSessionStateFixture(t)
	if _, err := warm.GenerateCached(prompt, 4, -1); err != nil {
		t.Fatalf("warm GenerateCached: %v", err)
	}
	if hit := warm.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}

	var yielded []int32
	got, err := warm.GenerateCachedEach(prompt, 4, -1, func(id int32) bool {
		yielded = append(yielded, id)
		return false
	})
	if err != nil {
		t.Fatalf("GenerateCachedEach: %v", err)
	}
	if len(got) != 1 || !idsEqual(got, yielded) {
		t.Fatalf("GenerateCachedEach got/yielded = %v/%v, want one matching streamed token", got, yielded)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 1, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("GenerateCachedEach first token = %v, want cold first token %v", got, want)
	}
	if warm.Pos() != len(prompt)+1 {
		t.Fatalf("Pos after stopped cached stream = %d, want prompt plus one generated token (%d)", warm.Pos(), len(prompt)+1)
	}
	if !idsEqual(warm.cachedIDs, append(append([]int32{}, prompt...), got...)) {
		t.Fatalf("cachedIDs after stopped stream = %v, want prompt plus %v", warm.cachedIDs, got)
	}
}

func TestGenerateCachedExactPromptUsesCachedLogits(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	warm := newSessionStateFixture(t)
	// This test counts HOST head-closure calls to pin the cached-logits flow of the
	// host decode lane. bf16 sessions record the arch ICB + GPU next-inputs seam now
	// and decode through the device headEnc (the host closure never fires) — force
	// the host lane; the ICB lane's cached-generate behaviour is covered by the ICB
	// parity suite.
	warm.state.icb = nil
	warm.encNextInputsGPU = nil
	if _, err := warm.GenerateCached(prompt, 3, -1); err != nil {
		t.Fatalf("warm GenerateCached: %v", err)
	}
	head := warm.head
	headCalls := 0
	warm.greedy = nil
	warm.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
		headCalls++
		return head(hidden, skipSoftcap)
	}

	got, err := warm.GenerateCached(prompt, 3, -1)
	if err != nil {
		t.Fatalf("exact cached GenerateCached: %v", err)
	}
	if headCalls != len(got)-1 {
		t.Fatalf("exact prompt-cache head calls = %d, want %d", headCalls, len(got)-1)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("generated length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token %d after exact prompt cached logits = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestGenerateCachedExactPromptUsesCachedLogitsWithoutHidden(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	fallback := newSessionStateFixture(t)
	if err := fallback.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache fallback: %v", err)
	}
	fallback.cachedPromptLogits = nil
	fallbackToken, err := fallback.GenerateCached(prompt, 1, -1)
	if err != nil {
		t.Fatalf("fallback GenerateCached: %v", err)
	}
	if len(fallbackToken) != 1 {
		t.Fatalf("fallback generated %d tokens, want 1", len(fallbackToken))
	}

	warm := newSessionStateFixture(t)
	if err := warm.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	target := (fallbackToken[0] + 1) % int32(warm.arch.Vocab)
	logits := make([]float32, warm.arch.Vocab)
	for i := range logits {
		logits[i] = -4
	}
	logits[target] = 4
	warm.cachedPromptLogits = toBF16Bytes(logits)
	warm.cachedPromptHidden = nil

	got, err := warm.GenerateCached(prompt, 1, -1)
	if err != nil {
		t.Fatalf("GenerateCached cached logits without hidden: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("cached-logits generated %d tokens, want 1", len(got))
	}
	if got[0] != target {
		t.Fatalf("cached-logits first token = %d, want synthetic cached-logits token %d", got[0], target)
	}
}

func TestGenerateCachedExactPromptUsesCachedLogitsWithSuppressionWithoutHidden(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	warm := newSessionStateFixture(t)
	if err := warm.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if warm.arch.Vocab < 2 {
		t.Fatalf("fixture vocab = %d, want at least 2", warm.arch.Vocab)
	}
	suppressed := int32(warm.arch.Vocab - 2)
	want := int32(warm.arch.Vocab - 1)
	logits := make([]float32, warm.arch.Vocab)
	for i := range logits {
		logits[i] = -8
	}
	logits[suppressed] = 9
	logits[want] = 6
	warm.cachedPromptLogits = toBF16Bytes(logits)
	warm.cachedPromptHidden = nil

	got, err := warm.GenerateCachedEachWithSuppression(prompt, 1, -1, []int32{suppressed}, nil)
	if err != nil {
		t.Fatalf("GenerateCachedEachWithSuppression cached logits without hidden: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("cached-logits generated %d tokens, want 1", len(got))
	}
	if got[0] != want {
		t.Fatalf("cached-logits suppressed first token = %d, want synthetic unsuppressed token %d", got[0], want)
	}
}

func TestGenerateCachedExactPromptUsesCachedLogitsWithTransformWithoutHidden(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	fallback := newSessionStateFixture(t)
	if err := fallback.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache fallback: %v", err)
	}
	fallback.cachedPromptLogits = nil
	fallbackToken, err := fallback.GenerateCached(prompt, 1, -1)
	if err != nil {
		t.Fatalf("fallback GenerateCached: %v", err)
	}
	if len(fallbackToken) != 1 {
		t.Fatalf("fallback generated %d tokens, want 1", len(fallbackToken))
	}

	warm := newSessionStateFixture(t)
	if err := warm.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if warm.arch.Vocab < 3 {
		t.Fatalf("fixture vocab = %d, want at least 3", warm.arch.Vocab)
	}
	target := (fallbackToken[0] + 1) % int32(warm.arch.Vocab)
	want := (fallbackToken[0] + 2) % int32(warm.arch.Vocab)
	logits := make([]float32, warm.arch.Vocab)
	for i := range logits {
		logits[i] = -6
	}
	logits[target] = 6
	warm.cachedPromptLogits = toBF16Bytes(logits)
	warm.cachedPromptHidden = nil

	transform := func(id int32) int32 {
		if id == target {
			return want
		}
		return id
	}
	got, err := warm.GenerateCachedEachTransformed(prompt, 1, -1, transform, nil)
	if err != nil {
		t.Fatalf("GenerateCachedEachTransformed cached logits without hidden: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("cached-logits generated %d tokens, want 1", len(got))
	}
	if got[0] != want {
		t.Fatalf("cached-logits transformed first token = %d, want synthetic transformed token %d", got[0], want)
	}
}

func TestGenerateCachedSampledEachExactPromptSkipsPromptReencode(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 5, TopP: 0.75}

	warm := newSessionStateFixture(t)
	if err := warm.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if hit := warm.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}

	embed := warm.embed
	embedCalls := 0
	warm.embed = func(id int32) ([]byte, error) {
		embedCalls++
		return embed(id)
	}

	got, err := warm.GenerateCachedSampledEach(prompt, 3, nil, model.NewSampler(123), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateCachedSampledEach: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.GenerateSampledEach(prompt, 3, nil, model.NewSampler(123), params, nil, nil)
	if err != nil {
		t.Fatalf("cold GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("sampled cached tokens = %v, want cold tokens %v", got, want)
	}
	if embedCalls > len(got) {
		t.Fatalf("cached sampled exact prompt embed calls = %d, want <= generated tokens %d", embedCalls, len(got))
	}
	wantResident := append(append([]int32(nil), prompt...), got...)
	if !idsEqual(warm.cachedIDs, wantResident) {
		t.Fatalf("cachedIDs after sampled exact prompt = %v, want %v", warm.cachedIDs, wantResident)
	}
}

func TestGenerateCachedSampledExactPromptUsesCachedLogits(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0}

	fallback := newSessionStateFixture(t)
	if err := fallback.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache fallback: %v", err)
	}
	fallback.cachedPromptLogits = nil
	fallbackToken, err := fallback.GenerateCachedSampledEach(prompt, 1, nil, model.NewSampler(1), params, nil, nil)
	if err != nil {
		t.Fatalf("fallback GenerateCachedSampledEach: %v", err)
	}
	if len(fallbackToken) != 1 {
		t.Fatalf("fallback generated %d tokens, want 1", len(fallbackToken))
	}

	warm := newSessionStateFixture(t)
	if err := warm.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	target := (fallbackToken[0] + 1) % int32(warm.arch.Vocab)
	logits := make([]float32, warm.arch.Vocab)
	for i := range logits {
		logits[i] = -4
	}
	logits[target] = 4
	warm.cachedPromptLogits = toBF16Bytes(logits)
	warm.cachedPromptHidden = nil

	got, err := warm.GenerateCachedSampledEach(prompt, 1, nil, model.NewSampler(1), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateCachedSampledEach cached logits: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("cached-logits generated %d tokens, want 1", len(got))
	}
	if got[0] != target {
		t.Fatalf("cached-logits first token = %d, want synthetic cached-logits token %d", got[0], target)
	}
}

func TestGenerateCachedSampledSuffixUsesRetainedPromptHiddenNoCopy(t *testing.T) {
	requireNativeRuntime(t)
	shared := []int32{1, 2, 3}
	full := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 5, TopP: 0.75}

	warm := newSessionStateFixture(t)
	if err := warm.WarmPromptCache(shared); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if warm.headEnc == nil {
		t.Fatal("session fixture did not build resident head encoder")
	}
	if hit := warm.CachedPrefixLen(full); hit != len(shared) {
		t.Fatalf("prompt-cache prefix hit = %d, want %d", hit, len(shared))
	}

	if _, err := warm.generateCachedSampled(full, 1, nil, model.NewSampler(123), params, nil, nil, false); err != nil {
		t.Fatalf("generateCachedSampled suffix: %v", err)
	}
	if warm.retainedHiddenBuffer() == nil {
		t.Fatal("sampled suffix replay did not retain the final prompt hidden in a no-copy buffer")
	}
	if len(warm.cachedPromptHidden) == 0 || len(warm.retainedHidden) == 0 {
		t.Fatal("sampled suffix replay did not record prompt-boundary hidden")
	}
	if !bytes.Equal(warm.cachedPromptHidden, warm.retainedHidden) {
		t.Fatal("sampled suffix cached hidden did not match the retained prompt-boundary hidden")
	}
	if unsafe.Pointer(&warm.cachedPromptHidden[0]) == unsafe.Pointer(&warm.retainedHidden[0]) &&
		warm.cachedPromptHiddenPinned != warm.retainedHiddenPinned {
		t.Fatal("sampled suffix cached hidden aliases retained hidden without shared no-copy ownership")
	}
}

// TestCompactCacheContinuation proves cache compaction is correct: after decoding a sequence and
// compacting to the most recent `keep` tokens, the session continues TOKEN-IDENTICALLY to a fresh
// session prefilled with exactly those kept tokens (the eviction + re-prefill re-rotates RoPE correctly).
func TestCompactCacheContinuation(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen, keep = 64, 3, 96, 4
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	mk := func() *ArchSession {
		s, err := NewArchSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		return s
	}

	// session A: decode a long-ish sequence, then compact to the most recent `keep` tokens.
	a := mk()
	if _, err := a.GenerateCached([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 6, -1); err != nil {
		t.Fatalf("A turn: %v", err)
	}
	resident := append([]int32(nil), a.cachedIDs...)
	kept := resident[len(resident)-keep:]
	if err := a.CompactCache(keep); err != nil {
		t.Fatalf("CompactCache: %v", err)
	}
	if a.Pos() != keep {
		t.Fatalf("post-compaction pos = %d, want %d", a.Pos(), keep)
	}
	cont := []int32{30, 31}
	genA, err := a.Generate(cont, 8, -1)
	if err != nil {
		t.Fatalf("A continue: %v", err)
	}

	// reference: a fresh session prefilled with exactly the kept tokens, same continuation.
	b := mk()
	full := append(append([]int32(nil), kept...), cont...)
	genB, err := b.Generate(full, 8, -1)
	if err != nil {
		t.Fatalf("B: %v", err)
	}
	if len(genA) != len(genB) {
		t.Fatalf("length mismatch: A=%d B=%d", len(genA), len(genB))
	}
	for i := range genA {
		if genA[i] != genB[i] {
			t.Fatalf("token %d diverged after compaction: A=%d B=%d", i, genA[i], genB[i])
		}
	}
	t.Logf("native cache compaction: kept %d recent tokens, continuation token-identical to a fresh session with that context", keep)
}

// TestClearPromptCacheDropsNativePrefixState pins the native engine equivalent
// of metal.Model.ClearPromptCache: clear the retained token-prefix metadata and
// rewind the decode cursor so the next cached generate starts cold.
func TestClearPromptCacheDropsNativePrefixState(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen = 64, 3, 96
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}

	warmPrompt := []int32{1, 2, 3, 4, 5}
	if _, err := sess.GenerateCached(warmPrompt, 6, -1); err != nil {
		t.Fatalf("warm GenerateCached: %v", err)
	}
	if hit := sess.CachedPrefixLen(warmPrompt); hit == 0 {
		t.Fatal("warm cache did not record a prefix")
	}

	sess.ClearPromptCache()
	if sess.Pos() != 0 {
		t.Fatalf("Pos after ClearPromptCache = %d, want 0", sess.Pos())
	}
	if hit := sess.CachedPrefixLen(warmPrompt); hit != 0 {
		t.Fatalf("CachedPrefixLen after ClearPromptCache = %d, want 0", hit)
	}
	if _, err := sess.GenerateCached(warmPrompt, 2, -1); err != nil {
		t.Fatalf("cold GenerateCached after ClearPromptCache: %v", err)
	}
	if sess.Pos() != len(warmPrompt)+2 {
		t.Fatalf("Pos after cold GenerateCached = %d, want %d", sess.Pos(), len(warmPrompt)+2)
	}
}

func TestWarmPromptCachePrefillsResidentPrefix(t *testing.T) {
	requireNativeRuntime(t)
	warm := newSessionStateFixture(t)
	prefix := []int32{1, 2, 3, 4, 5}
	if err := warm.WarmPromptCache(prefix); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if warm.Pos() != len(prefix) {
		t.Fatalf("Pos after WarmPromptCache = %d, want %d", warm.Pos(), len(prefix))
	}
	if len(warm.cachedIDs) != len(prefix) {
		t.Fatalf("resident ids after WarmPromptCache = %d, want %d", len(warm.cachedIDs), len(prefix))
	}
	extended := append(append([]int32(nil), prefix...), 6)
	if hit := warm.CachedPrefixLen(extended); hit != len(prefix) {
		t.Fatalf("CachedPrefixLen after WarmPromptCache = %d, want %d", hit, len(prefix))
	}

	got, err := warm.GenerateCached(extended, 4, -1)
	if err != nil {
		t.Fatalf("GenerateCached after WarmPromptCache: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(extended, 4, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("generated length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token %d after warm cache = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestWarmPromptCacheExactPromptStoresHiddenLogits(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	warm := newSessionStateFixture(t)
	// Counts HOST head-closure calls — the host decode lane's cached-logits flow (see
	// TestGenerateCachedExactPromptUsesCachedLogits); bf16 sessions fuse the head
	// device-side now, so force the host lane.
	warm.state.icb = nil
	warm.encNextInputsGPU = nil
	if err := warm.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if hit := warm.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("warmed exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}
	head := warm.head
	headCalls := 0
	warm.greedy = nil
	warm.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
		headCalls++
		return head(hidden, skipSoftcap)
	}

	got, err := warm.GenerateCached(prompt, 3, -1)
	if err != nil {
		t.Fatalf("GenerateCached after WarmPromptCache: %v", err)
	}
	if headCalls != len(got)-1 {
		t.Fatalf("warmed exact prompt-cache head calls = %d, want %d", headCalls, len(got)-1)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("generated length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token %d after warmed exact prompt cache = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestWarmPromptCacheUsesRetainedHiddenNoCopyLogits(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	warm := newSessionStateFixture(t)
	head := warm.head
	headCalls := 0
	warm.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
		headCalls++
		return head(hidden, skipSoftcap)
	}
	if err := warm.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if headCalls != 0 {
		t.Fatalf("WarmPromptCache generic head calls = %d, want retained no-copy head path", headCalls)
	}
	if warm.retainedHiddenBuffer() == nil {
		t.Fatal("WarmPromptCache did not retain prompt hidden in a no-copy buffer")
	}
	if len(warm.cachedPromptHidden) == 0 || len(warm.retainedHidden) == 0 {
		t.Fatal("WarmPromptCache did not record prompt-boundary hidden")
	}
	if !bytes.Equal(warm.cachedPromptHidden, warm.retainedHidden) {
		t.Fatal("WarmPromptCache cached hidden does not match retained prompt-boundary hidden")
	}
	if unsafe.Pointer(&warm.cachedPromptHidden[0]) == unsafe.Pointer(&warm.retainedHidden[0]) &&
		warm.cachedPromptHiddenPinned != warm.retainedHiddenPinned {
		t.Fatal("WarmPromptCache cached hidden aliases retained hidden without shared no-copy ownership")
	}
	if warm.retainedLogitsBuffer() == nil {
		t.Fatal("WarmPromptCache did not retain prompt logits in a no-copy buffer")
	}
	if len(warm.cachedPromptLogits) == 0 || len(warm.retainedLogits) == 0 {
		t.Fatal("WarmPromptCache did not record prompt-boundary logits")
	}
	if !bytes.Equal(warm.cachedPromptLogits, warm.retainedLogits) {
		t.Fatal("WarmPromptCache cached logits do not match retained prompt-boundary logits")
	}
	if unsafe.Pointer(&warm.cachedPromptLogits[0]) == unsafe.Pointer(&warm.retainedLogits[0]) &&
		warm.cachedPromptLogitsPinned != warm.retainedLogitsPinned {
		t.Fatal("WarmPromptCache cached logits alias retained logits without shared no-copy ownership")
	}
}

func TestWarmPromptCacheReusesResidentIDBacking(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	prefix := []int32{1, 2, 3, 4, 5}
	if err := sess.WarmPromptCache(prefix); err != nil {
		t.Fatalf("WarmPromptCache first: %v", err)
	}
	if len(sess.cachedIDs) == 0 {
		t.Fatal("WarmPromptCache left no resident ids")
	}
	before := unsafe.Pointer(&sess.cachedIDs[0])
	beforeCap := cap(sess.cachedIDs)

	if err := sess.WarmPromptCache(prefix); err != nil {
		t.Fatalf("WarmPromptCache second: %v", err)
	}
	if cap(sess.cachedIDs) != beforeCap {
		t.Fatalf("resident id capacity = %d, want %d", cap(sess.cachedIDs), beforeCap)
	}
	if after := unsafe.Pointer(&sess.cachedIDs[0]); after != before {
		t.Fatalf("resident id backing changed from %p to %p", before, after)
	}
}

func TestPrefillCachedIDsUsesEmbedInto(t *testing.T) {
	requireNativeRuntime(t)
	control := newSessionStateFixture(t)
	candidate := newSessionStateFixture(t)
	control.state.icb = nil
	candidate.state.icb = nil
	prefix := []int32{1, 2, 3, 4, 5}

	if err := control.prefillCachedIDs(prefix); err != nil {
		t.Fatalf("control prefillCachedIDs: %v", err)
	}
	candidate.embed = func(int32) ([]byte, error) {
		return nil, errors.New("allocating embed path called")
	}
	candidate.embedFuncPtr = 0
	if err := candidate.prefillCachedIDs(prefix); err != nil {
		t.Fatalf("candidate prefillCachedIDs: %v", err)
	}

	var want, got []byte
	var err error
	withAutoreleasePool(func() {
		want, err = control.stepIDInPool(6)
		if err != nil {
			return
		}
		got, err = candidate.stepIDInPool(6)
	})
	if err != nil {
		t.Fatalf("post-prefill stepIDInPool: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("embedInto prefillCachedIDs cache differs from allocating reference")
	}
}

func TestWarmPromptCacheAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	prefix := []int32{1, 2, 3, 4, 5}
	if err := sess.WarmPromptCache(prefix); err != nil {
		t.Fatalf("WarmPromptCache warmup: %v", err)
	}

	var warmErr error
	allocs := testing.AllocsPerRun(3, func() {
		sess.pos = 0
		sess.cachedIDs = sess.cachedIDs[:0]
		warmErr = sess.WarmPromptCache(prefix)
	})
	if warmErr != nil {
		t.Fatalf("WarmPromptCache: %v", warmErr)
	}
	if allocs > 29200 {
		t.Fatalf("WarmPromptCache allocations = %.0f, want <= 29200", allocs)
	}
}

func TestGenerateCachedReusesResidentIDBacking(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	prompt := []int32{1, 2, 3, 4, 5}
	if _, err := sess.GenerateCached(prompt, 6, -1); err != nil {
		t.Fatalf("warm GenerateCached: %v", err)
	}
	if len(sess.cachedIDs) == 0 {
		t.Fatal("warm GenerateCached left no resident ids")
	}
	before := unsafe.Pointer(&sess.cachedIDs[0])
	beforeCap := cap(sess.cachedIDs)

	if _, err := sess.GenerateCached(prompt, 2, -1); err != nil {
		t.Fatalf("cached GenerateCached: %v", err)
	}
	if cap(sess.cachedIDs) != beforeCap {
		t.Fatalf("resident id capacity = %d, want %d", cap(sess.cachedIDs), beforeCap)
	}
	if after := unsafe.Pointer(&sess.cachedIDs[0]); after != before {
		t.Fatalf("resident id backing changed from %p to %p", before, after)
	}
}

func TestCompactCacheReusesRetainedIDBacking(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	if _, err := sess.GenerateCached([]int32{1, 2, 3, 4, 5, 6, 7, 8}, 6, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	const keep = 4
	if len(sess.cachedIDs) < keep {
		t.Fatalf("resident ids = %d, want at least %d", len(sess.cachedIDs), keep)
	}
	before := unsafe.Pointer(&sess.cachedIDs[len(sess.cachedIDs)-keep])
	want := append([]int32(nil), sess.cachedIDs[len(sess.cachedIDs)-keep:]...)

	if err := sess.CompactCache(keep); err != nil {
		t.Fatalf("CompactCache: %v", err)
	}
	if len(sess.cachedIDs) != keep {
		t.Fatalf("resident ids after compaction = %d, want %d", len(sess.cachedIDs), keep)
	}
	if after := unsafe.Pointer(&sess.cachedIDs[0]); after != before {
		t.Fatalf("retained id backing changed from %p to %p", before, after)
	}
	for i, id := range want {
		if sess.cachedIDs[i] != id {
			t.Fatalf("retained id %d = %d, want %d", i, sess.cachedIDs[i], id)
		}
	}
}

func TestCompactCacheAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	if _, err := sess.GenerateCached([]int32{1, 2, 3, 4, 5, 6, 7, 8}, 6, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	resident := append([]int32(nil), sess.cachedIDs...)
	const keep = 4
	var compactErr error
	allocs := testing.AllocsPerRun(3, func() {
		sess.cachedIDs = resident
		sess.pos = len(resident)
		compactErr = sess.CompactCache(keep)
	})
	if compactErr != nil {
		t.Fatalf("CompactCache: %v", compactErr)
	}
	if allocs > 22973 {
		t.Fatalf("CompactCache allocations = %.0f, want <= 22973", allocs)
	}
}

func TestSameByteBacking_Good(t *testing.T) {
	backing := make([]byte, 8)
	if !sameByteBacking(backing[:2], backing[:5]) {
		t.Fatal("sameByteBacking slices with the same first byte = false, want true")
	}
}

func TestSameByteBacking_Bad(t *testing.T) {
	if sameByteBacking(make([]byte, 2), make([]byte, 2)) {
		t.Fatal("sameByteBacking independent allocations = true, want false")
	}
}

func TestSameByteBacking_Ugly(t *testing.T) {
	// A zero-length slice with capacity has backing storage, but no data byte
	// to compare against. The helper must decline it without indexing it.
	if sameByteBacking(make([]byte, 0, 2), make([]byte, 0, 2)) {
		t.Fatal("sameByteBacking zero-length slices = true, want false")
	}
}

func TestByteBackingPointer_Good(t *testing.T) {
	b := make([]byte, 0, 3)
	if got := byteBackingPointer(b); got == nil {
		t.Fatal("byteBackingPointer zero-length slice with capacity = nil, want backing pointer")
	}
}

func TestByteBackingPointer_Bad(t *testing.T) {
	if got := byteBackingPointer(nil); got != nil {
		t.Fatalf("byteBackingPointer(nil) = %p, want nil", got)
	}
}

func TestByteBackingPointer_Ugly(t *testing.T) {
	if got := byteBackingPointer(make([]byte, 0)); got != nil {
		t.Fatalf("byteBackingPointer zero-capacity slice = %p, want nil", got)
	}
}

func TestPrefillCachedIDs_Bad(t *testing.T) {
	sess := &ArchSession{pos: 1, maxLen: 1}
	if err := sess.prefillCachedIDs([]int32{1}); err == nil {
		t.Fatal("prefillCachedIDs beyond maxLen error = nil")
	}
}

func TestPromptCacheLogitsFromRetainedHidden_Good(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	var hidden, logits []byte
	var err error
	withAutoreleasePool(func() {
		hidden, err = sess.prefillPromptRetainedInPool([]int32{1, 2, 3})
		if err != nil {
			return
		}
		logits, err = sess.promptCacheLogitsFromRetainedHidden(hidden)
	})
	if err != nil {
		t.Fatalf("promptCacheLogitsFromRetainedHidden: %v", err)
	}
	if len(logits) != sess.arch.Vocab*bf16Size {
		t.Fatalf("prompt-cache logits bytes = %d, want %d", len(logits), sess.arch.Vocab*bf16Size)
	}
	if !bytes.Equal(logits, sess.retainedLogits) {
		t.Fatal("promptCacheLogitsFromRetainedHidden did not retain the returned logits")
	}
}

func TestPromptCacheLogitsFromRetainedHidden_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	if _, err := sess.promptCacheLogitsFromRetainedHidden([]byte{0}); err == nil {
		t.Fatal("promptCacheLogitsFromRetainedHidden short hidden error = nil")
	}
}

func TestArchSessionCachedPrefixLen_Good(t *testing.T) {
	sess := &ArchSession{arch: model.Arch{Hidden: 2}, cachedIDs: []int32{4, 5, 6}, cachedPromptIDs: []int32{4, 5, 6}, cachedPromptHidden: []byte{1, 2, 3, 4}}
	if got := sess.CachedPrefixLen([]int32{4, 5, 6}); got != 3 {
		t.Fatalf("CachedPrefixLen(exact retained prompt) = %d, want 3", got)
	}
}

func TestArchSessionCachedPrefixLen_Bad(t *testing.T) {
	sess := &ArchSession{arch: model.Arch{Hidden: 2}, cachedIDs: []int32{4, 5, 6}, cachedPromptIDs: []int32{4, 5, 6}}
	if got := sess.CachedPrefixLen([]int32{4, 5, 6}); got != 2 {
		t.Fatalf("CachedPrefixLen(exact prompt without retained hidden) = %d, want 2", got)
	}
}

func TestArchSessionCachedPromptLogitsForSampledReplay_Good(t *testing.T) {
	logits := []byte{1, 2, 3, 4}
	sess := &ArchSession{arch: model.Arch{Vocab: 2}, cachedPromptIDs: []int32{4, 5}, cachedPromptLogits: logits}
	// With no soft-cap, the cached boundary logits preserve the sampled replay
	// path even for non-greedy parameters; the head has no extra transform.
	if got := sess.cachedPromptLogitsForSampledReplay([]int32{4, 5}, model.SampleParams{Temperature: 1}); !bytes.Equal(got, logits) {
		t.Fatalf("cachedPromptLogitsForSampledReplay(no soft-cap) = %v, want %v", got, logits)
	}
}

func TestArchSessionCachedPromptLogitsForSampledReplay_Bad(t *testing.T) {
	sess := &ArchSession{arch: model.Arch{Vocab: 2, SoftCap: 1}, cachedPromptIDs: []int32{4, 5}, cachedPromptLogits: []byte{1, 2, 3, 4}}
	if got := sess.cachedPromptLogitsForSampledReplay([]int32{4, 5}, model.SampleParams{Temperature: 1}); got != nil {
		t.Fatalf("cachedPromptLogitsForSampledReplay(transformed sampled replay) = %v, want nil", got)
	}
}
