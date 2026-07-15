// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
)

const (
	mtpFixtureDModel = 512
	mtpFixtureNHeads = 8
	mtpFixtureNKV    = 4
	mtpFixtureHead   = 64
	mtpFixtureDFF    = 1024
	mtpFixtureVocab  = 64
	mtpFixtureLayers = 3
	mtpFixtureMaxLen = 96

	mtpWordedPromptText  = "speculative decoding works with a few words"
	mtpWordedPromptWords = 7
)

var mtpWordedPromptTokens = [...]int32{2, 18, 7, 41, 13, 5, 29}

func mtpWordedPromptIDs() []int32 {
	return mtpWordedPromptTokens[:]
}

func TestMTPWordedPromptFixtureUsesAFewWords(t *testing.T) {
	prompt := mtpWordedPromptIDs()
	if mtpWordedPromptWords < 5 {
		t.Fatalf("MTP worded prompt %q has %d words, want a few words", mtpWordedPromptText, mtpWordedPromptWords)
	}
	if len(prompt) != mtpWordedPromptWords {
		t.Fatalf("MTP worded prompt token count = %d, want one stable token id per word", len(prompt))
	}
	for i, id := range prompt {
		if id <= 0 || int(id) >= mtpFixtureVocab {
			t.Fatalf("MTP worded prompt token %d = %d outside fixture vocab %d", i, id, mtpFixtureVocab)
		}
	}
}

func newMTPDecodeFixture(t testing.TB) func() *ArchSession {
	t.Helper()
	return newMTPDecodeFixtureWithArch(t, nil)
}

func newMTPDecodeFixtureWithArch(t testing.TB, configure func(*model.Arch)) func() *ArchSession {
	t.Helper()
	return newMTPDecodeFixtureWithArchAndLayers(t, configure, nil)
}

// newMTPDecodeFixtureWithArchAndLayers is newMTPDecodeFixtureWithArch with a hook over the layer
// weights too — the attention-fold tests use it to mint the gemma4 q/k norms the fold gates on.
func newMTPDecodeFixtureWithArchAndLayers(t testing.TB, configure func(*model.Arch), configureLayers func([]DecodeLayerWeights)) func() *ArchSession {
	t.Helper()
	return newMTPDecodeFixtureWithArchLayersMaxLen(t, configure, configureLayers, mtpFixtureMaxLen)
}

// newMTPDecodeFixtureWithArchLayersMaxLen additionally sizes the session cache — the deferred-ring
// tests need headroom for a full ring plus a steelGEMMMinRows-wide batch.
func newMTPDecodeFixtureWithArchLayersMaxLen(t testing.TB, configure func(*model.Arch), configureLayers func([]DecodeLayerWeights), maxLen int) func() *ArchSession {
	t.Helper()
	layers := make([]DecodeLayerWeights, mtpFixtureLayers)
	types := make([]string, mtpFixtureLayers)
	for li := range layers {
		layers[li] = forwardLayer(mtpFixtureDModel, mtpFixtureNHeads, mtpFixtureNKV, mtpFixtureHead, mtpFixtureDFF, (li+1)*100)
		types[li] = "full_attention"
	}
	if configureLayers != nil {
		configureLayers(layers)
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(mtpFixtureVocab*mtpFixtureDModel, 21))
	g := &BF16Model{
		Layers:    layers,
		Embed:     embed,
		FinalNorm: toBF16Bytes(syntheticFloat32(mtpFixtureDModel, 22)),
		LMHead:    embed,
		Tied:      true,
	}
	arch := model.Arch{
		Hidden: mtpFixtureDModel, Heads: mtpFixtureNHeads, KVHeads: mtpFixtureNKV, HeadDim: mtpFixtureHead, FF: mtpFixtureDFF, Vocab: mtpFixtureVocab,
		GlobalHeadDim: mtpFixtureHead, GlobalKVHeads: mtpFixtureNKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: mtpFixtureHead, RotaryDimLocal: mtpFixtureHead,
		Layer: specs,
	}
	if configure != nil {
		configure(&arch)
	}
	return func() *ArchSession {
		s, err := NewArchSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		head := &headEncoder{
			finalNorm: copyView(g.FinalNorm),
			weight:    copyView(g.LMHead),
			dModel:    arch.Hidden,
			vocab:     arch.Vocab,
			eps:       arch.Eps,
			softCap:   arch.SoftCap,
		}
		s.headEnc = head
		s.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
			return head.encode(hidden, skipSoftcap)
		}
		s.greedy = func(hidden []byte, suppress []int32) (int32, bool, error) {
			return head.greedyInPool(hidden, suppress)
		}
		s.markDefaultHeadFunc()
		s.markDefaultGreedyFunc()
		return s
	}
}

func TestMTPDecodeInputGuards(t *testing.T) {
	session := func(maxLen int) *ArchSession { return &ArchSession{maxLen: maxLen} }
	prompt := mtpWordedPromptIDs()
	tests := []struct {
		name   string
		target *ArchSession
		draft  *ArchSession
		prompt []int32
		maxNew int
		k      int
	}{
		{name: "nil target", target: nil, draft: session(8), prompt: prompt, maxNew: 1, k: 1},
		{name: "nil draft", target: session(8), draft: nil, prompt: prompt, maxNew: 1, k: 1},
		{name: "empty prompt", target: session(8), draft: session(8), prompt: nil, maxNew: 1, k: 1},
		{name: "zero maxNew", target: session(8), draft: session(8), prompt: prompt, maxNew: 0, k: 1},
		{name: "zero k", target: session(8), draft: session(8), prompt: prompt, maxNew: 1, k: 0},
		{name: "target headroom", target: session(1), draft: session(8), prompt: prompt, maxNew: 1, k: 1},
		{name: "draft headroom", target: session(8), draft: session(1), prompt: prompt, maxNew: 1, k: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := MTPDecode(tt.target, tt.draft, tt.prompt, tt.maxNew, -1, tt.k); err == nil {
				t.Fatal("MTPDecode error = nil")
			}
		})
	}
}

func TestMTPDecodeSampledInputGuards(t *testing.T) {
	session := func(maxLen int) *ArchSession { return &ArchSession{maxLen: maxLen} }
	prompt := mtpWordedPromptIDs()
	targetSampler := model.NewSampler(1)
	draftSampler := model.NewSampler(2)
	sharedSampler := model.NewSampler(3)
	tests := []struct {
		name          string
		target        *ArchSession
		draft         *ArchSession
		prompt        []int32
		maxNew        int
		k             int
		targetSampler *model.Sampler
		draftSampler  *model.Sampler
	}{
		{name: "nil target", target: nil, draft: session(8), prompt: prompt, maxNew: 1, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "nil draft", target: session(8), draft: nil, prompt: prompt, maxNew: 1, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "nil target sampler", target: session(8), draft: session(8), prompt: prompt, maxNew: 1, k: 1, draftSampler: draftSampler},
		{name: "nil draft sampler", target: session(8), draft: session(8), prompt: prompt, maxNew: 1, k: 1, targetSampler: targetSampler},
		{name: "shared sampler", target: session(8), draft: session(8), prompt: prompt, maxNew: 1, k: 1, targetSampler: sharedSampler, draftSampler: sharedSampler},
		{name: "empty prompt", target: session(8), draft: session(8), prompt: nil, maxNew: 1, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "zero maxNew", target: session(8), draft: session(8), prompt: prompt, maxNew: 0, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "zero k", target: session(8), draft: session(8), prompt: prompt, maxNew: 1, k: 0, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "target headroom", target: session(1), draft: session(8), prompt: prompt, maxNew: 1, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "draft headroom", target: session(8), draft: session(1), prompt: prompt, maxNew: 1, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := MTPDecodeSampled(tt.target, tt.draft, tt.prompt, tt.maxNew, nil, tt.targetSampler, tt.draftSampler, model.SampleParams{}, tt.k); err == nil {
				t.Fatal("MTPDecodeSampled error = nil")
			}
		})
	}
}

// TestMTPDecodeBatchedTokenIdentity is the headline MTP invariant: speculative decode emits EXACTLY
// the token stream plain greedy Generate would, while engaging the batched verify (one pass over the
// resident stack per draft block, not K stepGreedy rounds). It builds a synthetic dense bf16 session
// (no PLE, no ICB on the bf16 path) so verifyBatched takes the batched path, uses draft==target weights
// so every draft is accepted (exercising the full accept loop + the batched verify), and asserts the
// MTP token stream equals Generate's token-for-token.
func TestMTPDecodeBatchedTokenIdentity(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 16
	mk := newMTPDecodeFixture(t)

	prompt := mtpWordedPromptIDs()

	// reference: plain greedy Generate on a fresh session.
	ref, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	// MTP speculative decode with draft==target weights → every draft accepted, batched verify engaged.
	res, err := MTPDecode(mk(), mk(), prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}

	if len(res.Tokens) != len(ref) {
		t.Fatalf("MTP emitted %d tokens, Generate emitted %d", len(res.Tokens), len(ref))
	}
	for i := range ref {
		if res.Tokens[i] != ref[i] {
			t.Fatalf("token %d diverged: MTP=%d Generate=%d", i, res.Tokens[i], ref[i])
		}
	}
	if res.Accepted == 0 {
		t.Fatal("no drafts accepted — the speculative/batched path did not engage")
	}
	// draft == target weights, so every proposed token IS the target's greedy → all must be accepted.
	// A drop below full acceptance means the draft cache drifted out of alignment with the target.
	if res.Accepted != res.Drafted {
		t.Fatalf("draft==target should accept every draft, got %d/%d (draft cache misaligned)", res.Accepted, res.Drafted)
	}
	t.Log(core.Sprintf("MTP batched == Generate over %d tokens; accepted %d/%d drafted in %d rounds",
		len(ref), res.Accepted, res.Drafted, res.Rounds))
}

func TestMTPDecodeEachYieldsCommittedTokens(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 10
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)
	ref, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	var yielded []int32
	res, err := MTPDecodeEach(mk(), mk(), prompt, maxNew, -1, K, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("MTPDecodeEach: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("MTPDecodeEach tokens %v != Generate %v", res.Tokens, ref)
	}
	if !mtpIDsEqual(yielded, res.Tokens) {
		t.Fatalf("MTPDecodeEach yielded %v != result tokens %v", yielded, res.Tokens)
	}
}

func TestMTPDecodeUsesExactContextTailHeadroom(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 2
	prompt := mtpWordedPromptIDs()
	maxLen := len(prompt) + maxNew
	mk := newMTPDecodeFixture(t)
	limit := func(s *ArchSession) *ArchSession {
		s.maxLen = maxLen
		return s
	}

	ref, err := limit(mk()).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate exact tail reference: %v", err)
	}
	res, err := MTPDecode(limit(mk()), limit(mk()), prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode exact tail: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("MTP exact tail tokens %v != Generate %v", res.Tokens, ref)
	}
}

func TestMTPDensePromptPrefillWordedHiddenMatchesSequential(t *testing.T) {
	requireNativeRuntime(t)
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)
	ref := mk()
	sess := mk()

	if err := ref.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	got, ok, err := sess.prefillMTPPrompt(prompt, true)
	if err != nil {
		t.Fatalf("prefillMTPPrompt: %v", err)
	}
	if !ok {
		t.Fatal("prefillMTPPrompt ok = false")
	}
	if !bytes.Equal(got, ref.retainedHidden) {
		t.Fatal("prefillMTPPrompt hidden differs from sequential prompt prefill")
	}
	for _, id := range []int32{13, 37, 41} {
		wantHidden, err := ref.stepID(id)
		if err != nil {
			t.Fatalf("reference stepID(%d): %v", id, err)
		}
		gotHidden, err := sess.stepID(id)
		if err != nil {
			t.Fatalf("dense-prefill stepID(%d): %v", id, err)
		}
		if !bytes.Equal(gotHidden, wantHidden) {
			t.Fatalf("hidden after stepping %d differs after dense prompt prefill", id)
		}
	}
}

func TestMTPDecodeSampledMatchesGenerateSampled(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 12
	const seed uint64 = 53
	prompt := mtpWordedPromptIDs()
	params := model.SampleParams{
		Temperature:   0.8,
		TopK:          7,
		TopP:          0.75,
		MinP:          0.01,
		RepeatPenalty: 1.2,
		SuppressTokens: []int32{
			2,
			7,
		},
	}
	mk := newMTPDecodeFixture(t)

	ref, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	res, err := MTPDecodeSampled(mk(), mk(), prompt, maxNew, nil, model.NewSampler(seed), model.NewSampler(seed+1), params, K)
	if err != nil {
		t.Fatalf("MTPDecodeSampled: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("sampled MTP tokens %v != GenerateSampledEach %v (accepted=%d drafted=%d rounds=%d)", res.Tokens, ref, res.Accepted, res.Drafted, res.Rounds)
	}
	if res.Drafted == 0 {
		t.Fatal("sampled MTP proposed no draft tokens")
	}
}

func TestMTPDecodeSampledEachYieldsCommittedTokens(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 10
	const seed uint64 = 53
	prompt := mtpWordedPromptIDs()
	params := model.SampleParams{
		Temperature:   0.8,
		TopK:          7,
		TopP:          0.75,
		MinP:          0.01,
		RepeatPenalty: 1.2,
		SuppressTokens: []int32{
			2,
			7,
		},
	}
	mk := newMTPDecodeFixture(t)
	ref, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	var yielded []int32
	res, err := MTPDecodeSampledEach(mk(), mk(), prompt, maxNew, nil, model.NewSampler(seed), model.NewSampler(seed+1), params, K, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("MTPDecodeSampledEach: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("MTPDecodeSampledEach tokens %v != GenerateSampledEach %v (accepted=%d drafted=%d rounds=%d)", res.Tokens, ref, res.Accepted, res.Drafted, res.Rounds)
	}
	if !mtpIDsEqual(yielded, res.Tokens) {
		t.Fatalf("MTPDecodeSampledEach yielded %v != result tokens %v", yielded, res.Tokens)
	}
}

func TestMTPSampledPickerMatchesGenerateSampledOnWordedPrompt(t *testing.T) {
	requireNativeRuntime(t)
	const maxNew = 12
	const seed uint64 = 53
	prompt := mtpWordedPromptIDs()
	params := model.SampleParams{
		Temperature:   0.8,
		TopK:          7,
		TopP:          0.75,
		MinP:          0.01,
		RepeatPenalty: 1.2,
		SuppressTokens: []int32{
			2,
			7,
		},
	}
	mk := newMTPDecodeFixture(t)

	ref, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	sess := mk()
	hidden, ok, err := sess.prefillMTPPrompt(prompt, true)
	if err != nil {
		t.Fatalf("prefillMTPPrompt: %v", err)
	}
	if !ok {
		t.Fatal("prefillMTPPrompt ok = false")
	}
	history := sess.sampleHistoryScratchFor(params, maxNew)
	var got []int32
	sampler := model.NewSampler(seed)
	for len(got) < maxNew {
		pickParams := sess.mtpSamplePickParams(params, nil, len(got))
		next, err := sess.sampleMTPTokenFromHidden(hidden, sampler, pickParams, history)
		if err != nil {
			t.Fatalf("sampleMTPTokenFromHidden token %d: %v", len(got), err)
		}
		got = append(got, next)
		if params.RepeatPenalty > 1 {
			history = append(history, next)
		}
		hidden, err = sess.stepID(next)
		if err != nil {
			t.Fatalf("stepID(%d): %v", next, err)
		}
	}
	if !mtpIDsEqual(got, ref) {
		t.Fatalf("MTP sampled picker tokens %v != GenerateSampledEach %v", got, ref)
	}
}

func TestMTPDecodeSlidingRingWrapMatchesGenerate(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 3, 10
	mk := newMTPDecodeFixtureWithArch(t, func(arch *model.Arch) {
		arch.SlidingWindow = 4
		arch.Layer[0].Attention = model.SlidingAttention
	})
	prompt := mtpWordedPromptIDs()

	ref, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate sliding reference: %v", err)
	}
	res, err := MTPDecode(mk(), mk(), prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode sliding: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("sliding MTP tokens %v != Generate %v", res.Tokens, ref)
	}
	if res.Accepted == 0 {
		t.Fatal("sliding MTP accepted no drafts; batched verify did not engage")
	}
}

func TestMTPDecodeDraftEqualsTargetAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 16
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)
	target := mk()
	draft := mk()

	var decodeErr error
	allocs := testing.AllocsPerRun(2, func() {
		target.pos = 0
		draft.pos = 0
		var res *MTPResult
		res, decodeErr = MTPDecode(target, draft, prompt, maxNew, -1, K)
		if decodeErr == nil && res.Accepted != res.Drafted {
			decodeErr = core.NewError("MTP draft==target did not accept every draft")
		}
	})
	if decodeErr != nil {
		t.Fatalf("MTPDecode: %v", decodeErr)
	}
	if allocs > 253400 {
		t.Fatalf("MTPDecode allocations = %.0f, want <= 253400", allocs)
	}
}

func TestMTPGreedyOfUsesDirectGreedyWhenAvailable(t *testing.T) {
	s := &ArchSession{
		arch: model.Arch{Vocab: 16},
		head: func([]byte, bool) ([]byte, error) {
			return nil, core.NewError("full logits head should not be called")
		},
		greedy: func([]byte, []int32) (int32, bool, error) {
			return 7, true, nil
		},
	}

	got, err := s.greedyOf([]byte{1, 2})
	if err != nil {
		t.Fatalf("greedyOf: %v", err)
	}
	if got != 7 {
		t.Fatalf("greedyOf = %d, want direct greedy token 7", got)
	}
}

// mtpSequentialFallbackSession forces s onto the byte-identical sequential verify
// lane by flipping the test-only verifyBatchedDisabledForTest guard, so
// verifyBatchedHiddens / verifyBatchedInto decline (ok=false) and MTPDecode /
// the assistant pair step token-by-token. This is the honest hook: it does not
// rely on any arch property (every resident arch — dense and PLE — now batches),
// so the sequential lane is exercised on the same fixture the batched lane uses.
func mtpSequentialFallbackSession(s *ArchSession) *ArchSession {
	s.verifyBatchedDisabledForTest = true
	return s
}

func mtpIDsEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestMTPDecodeSequentialFallbackTokenIdentity(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 12
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)

	ref, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	res, err := MTPDecode(mtpSequentialFallbackSession(mk()), mtpSequentialFallbackSession(mk()), prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("sequential fallback MTP tokens %v != Generate %v", res.Tokens, ref)
	}
	if res.Accepted != res.Drafted {
		t.Fatalf("draft==target sequential fallback accepted %d/%d", res.Accepted, res.Drafted)
	}
}

func TestMTPVerifyBatchedWrapperAndFallback(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	dense := mk()
	for _, id := range []int32{1, 2, 3} {
		if _, err := dense.stepID(id); err != nil {
			t.Fatalf("prefill dense stepID(%d): %v", id, err)
		}
	}
	greedys, ok, err := dense.verifyBatched([]int32{4, 5})
	if err != nil {
		t.Fatalf("verifyBatched dense: %v", err)
	}
	if !ok {
		t.Fatal("verifyBatched dense ok = false")
	}
	if len(greedys) != 2 {
		t.Fatalf("verifyBatched dense returned %d greedys, want 2", len(greedys))
	}
	for i, id := range greedys {
		if id < 0 || int(id) >= mtpFixtureVocab {
			t.Fatalf("greedy %d = %d outside vocab", i, id)
		}
	}

	fallback := mtpSequentialFallbackSession(mk())
	if _, ok, err = fallback.verifyBatched([]int32{4}); err != nil {
		t.Fatalf("verifyBatched fallback: %v", err)
	} else if ok {
		t.Fatal("verifyBatched fallback ok = true")
	}
	if _, ok, err = dense.verifyBatched(nil); err == nil {
		t.Fatal("verifyBatched empty error = nil")
	} else if ok {
		t.Fatal("verifyBatched empty ok = true")
	}
}

func TestMTPVerifyBatchedSlidingRingWrapMatchesSequential(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixtureWithArch(t, func(arch *model.Arch) {
		arch.SlidingWindow = 4
		arch.Layer[0].Attention = model.SlidingAttention
	})
	ref := mk()
	sess := mk()
	prompt := mtpWordedPromptIDs()
	if err := ref.PrefillTokens(prompt); err != nil {
		t.Fatalf("reference PrefillTokens: %v", err)
	}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("candidate PrefillTokens: %v", err)
	}
	if sess.Pos() != len(prompt) {
		t.Fatalf("prefill pos = %d, want %d", sess.Pos(), len(prompt))
	}
	ids := []int32{4, 5}
	want := make([]int32, len(ids))
	for i, id := range ids {
		hidden, err := ref.stepID(id)
		if err != nil {
			t.Fatalf("reference stepID(%d): %v", id, err)
		}
		want[i], err = ref.greedyOf(hidden)
		if err != nil {
			t.Fatalf("reference greedyOf(%d): %v", id, err)
		}
	}
	greedys, ok, err := sess.verifyBatchedInto(ids, make([]int32, len(ids)))
	if err != nil {
		t.Fatalf("verifyBatchedInto sliding wrap: %v", err)
	}
	if !ok {
		t.Fatal("verifyBatchedInto sliding wrap ok = false")
	}
	if !mtpIDsEqual(greedys, want) {
		t.Fatalf("verifyBatchedInto sliding wrap greedys = %v, want sequential %v", greedys, want)
	}
	if sess.Pos() != len(prompt) {
		t.Fatalf("verifyBatchedInto sliding wrap changed pos = %d, want %d", sess.Pos(), len(prompt))
	}
}

// TestMTPVerifyBatchedSlidingRingWrapStagedFoldEngages pins the attention fold's STAGED lane
// (#252): a sliding layer whose ring would evict during the batch folds its K/V projections into
// staging rows (the fused norm+rope's full-row write lands each row into its slot in order),
// byte-identical to the per-row halves — and the fold must actually engage: the gate
// preconditions are asserted loudly and the folded pass must encode strictly fewer dispatches.
func TestMTPVerifyBatchedSlidingRingWrapStagedFoldEngages(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("fused qknorm-rope kernel unavailable")
	}
	mk := newMTPDecodeFixtureWithArchAndLayers(t, func(arch *model.Arch) {
		arch.SlidingWindow = 4
		arch.Layer[0].Attention = model.SlidingAttention
		arch.ValueNorm = true
	}, func(layers []DecodeLayerWeights) {
		for li := range layers { // the gemma4 per-head QK-norms the attention fold gates on
			layers[li].QNormW = toBF16Bytes(syntheticFloat32(mtpFixtureHead, 900+li))
			layers[li].KNormW = toBF16Bytes(syntheticFloat32(mtpFixtureHead, 950+li))
		}
	})
	run := func(disableFold bool) ([]int32, int64, *ArchSession) {
		t.Helper()
		sess := mk()
		// Pins the STAGED attention-fold lane. bf16 sessions record the arch ICB now, and
		// small-batch verify on an ICB session deliberately keeps the per-row interleave
		// (the byte contract with the chained lane — see stepTokensBatchedDense) with the
		// folds off — so exercise the fold on the non-ICB lane it serves.
		sess.state.icb = nil
		if err := sess.PrefillTokens(mtpWordedPromptIDs()); err != nil {
			t.Fatalf("PrefillTokens: %v", err)
		}
		if !disableFold {
			// gate preconditions — if the fixture stops satisfying them the staged lane is
			// silently untested, so fail loudly on the missing piece instead.
			st := sess.state
			if st.lb[0].qNorm.buf == nil || st.lb[0].kNorm.buf == nil {
				t.Fatal("fixture lacks q/k norms — attention fold gate unmet, staged lane untested")
			}
			if st.valueNormOnes == nil {
				t.Fatal("fixture lacks the value norm — staged K/V landing gate unmet, staged lane untested")
			}
		}
		prevFold, prevTiming := batchedMLPFoldDisabledForTest, pieceTimingOn
		batchedMLPFoldDisabledForTest = disableFold
		pieceTimingOn = true
		dispatchCountForTest = 0
		defer func() {
			batchedMLPFoldDisabledForTest = prevFold
			pieceTimingOn = prevTiming
		}()
		ids := []int32{4, 5}
		greedys, ok, err := sess.verifyBatchedInto(ids, make([]int32, len(ids)))
		if err != nil {
			t.Fatalf("verifyBatchedInto (disableFold=%v): %v", disableFold, err)
		}
		if !ok {
			t.Fatalf("verifyBatchedInto declined (disableFold=%v)", disableFold)
		}
		return append([]int32(nil), greedys...), dispatchCountForTest, sess
	}
	foldGreedys, foldDispatches, folded := run(false)
	rowGreedys, rowDispatches, perRow := run(true)
	if foldDispatches >= rowDispatches {
		t.Fatalf("attention fold did not engage on the wrap batch: folded=%d dispatches, per-row=%d", foldDispatches, rowDispatches)
	}
	if !mtpIDsEqual(foldGreedys, rowGreedys) {
		t.Fatalf("folded wrap greedys = %v, per-row = %v (fold contract is byte-identity)", foldGreedys, rowGreedys)
	}
	va, err := perRow.stateLayerViews()
	if err != nil {
		t.Fatalf("per-row views: %v", err)
	}
	vb, err := folded.stateLayerViews()
	if err != nil {
		t.Fatalf("folded views: %v", err)
	}
	for i := range va {
		for j := range va[i].keyBytes {
			if va[i].keyBytes[j] != vb[i].keyBytes[j] {
				t.Fatalf("layer %d K diverges at byte %d (staged ring landing broke cache identity)", i, j)
			}
		}
		for j := range va[i].valueBytes {
			if va[i].valueBytes[j] != vb[i].valueBytes[j] {
				t.Fatalf("layer %d V diverges at byte %d (staged ring landing broke cache identity)", i, j)
			}
		}
	}
}

// TestStepTokensBatchedDenseDeferredRingLandingMatchesPerRow pins the staged sliding tail's
// deferred-landing lane (#252): at steelGEMMMinRows over a FULL ring, K/V stay in per-layer
// staging (roped/normed there), one two-segment ring SDPA replaces the K per-row landing+SDPA
// interleave, and the ring lands in bulk afterwards. The LANDED cache must be byte-identical to
// the per-row path (the landing copies the same roped bytes the per-row kernel writes); the
// boundary hidden is tolerance-checked (fp accumulation order differs — the token-identity
// trade). Engagement is pinned via the ring dispatch counter.
func TestStepTokensBatchedDenseDeferredRingLandingMatchesPerRow(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasSDPAMultiQRing(mtpFixtureHead) || !gpuHasCopyKernel() {
		t.Fatal("deferred-ring kernels missing — rebuild dist/lib/lthn_kernels.metallib (task build:kernels)")
	}
	const slideW = steelGEMMMinRows // ring exactly one batch wide: the tail chunk fills it end to end
	mk := newMTPDecodeFixtureWithArchLayersMaxLen(t, func(arch *model.Arch) {
		arch.SlidingWindow = slideW
		for i := range arch.Layer {
			arch.Layer[i].Attention = model.SlidingAttention
		}
		arch.ValueNorm = true
	}, func(layers []DecodeLayerWeights) {
		for li := range layers {
			layers[li].QNormW = toBF16Bytes(syntheticFloat32(mtpFixtureHead, 900+li))
			layers[li].KNormW = toBF16Bytes(syntheticFloat32(mtpFixtureHead, 950+li))
		}
	}, 3*steelGEMMMinRows)

	prompt := make([]int32, slideW)
	tail := make([]int32, steelGEMMMinRows)
	for i := range prompt {
		prompt[i] = int32(3 + (i*7)%29)
	}
	for i := range tail {
		tail[i] = int32(5 + (i*11)%23)
	}

	run := func(disable bool) ([]byte, int64, *ArchSession) {
		t.Helper()
		sess := mk()
		if err := sess.PrefillTokens(prompt); err != nil {
			t.Fatalf("PrefillTokens: %v", err)
		}
		prev, prevTiming := stagedRingDisabledForTest, pieceTimingOn
		stagedRingDisabledForTest = disable
		pieceTimingOn = true
		stagedRingDispatchesForTest = 0
		defer func() {
			stagedRingDisabledForTest = prev
			pieceTimingOn = prevTiming
		}()
		hidden, ok, err := sess.prefillRetainedTokensBatchedDenseOne(tail, "test.deferredRing")
		if err != nil {
			t.Fatalf("staged tail (disableRing=%v): %v", disable, err)
		}
		if !ok {
			t.Fatalf("staged tail DECLINED (disableRing=%v)", disable)
		}
		return append([]byte(nil), hidden...), stagedRingDispatchesForTest, sess
	}

	ringHidden, ringDispatches, ringSess := run(false)
	rowHidden, rowDispatches, rowSess := run(true)
	if ringDispatches == 0 {
		t.Fatal("deferred-ring lane did not engage (ring dispatch counter stayed 0)")
	}
	if rowDispatches != 0 {
		t.Fatalf("kill switch leaked: per-row run counted %d ring dispatches", rowDispatches)
	}
	if ringSess.Pos() != rowSess.Pos() {
		t.Fatalf("pos after staged tail: ring=%d perRow=%d", ringSess.Pos(), rowSess.Pos())
	}

	// the landed ring must be byte-identical — the deferred landing copies exactly the roped/
	// normed bytes the per-row landing kernel writes into each slot.
	va, err := rowSess.stateLayerViews()
	if err != nil {
		t.Fatalf("per-row views: %v", err)
	}
	vb, err := ringSess.stateLayerViews()
	if err != nil {
		t.Fatalf("ring views: %v", err)
	}
	for i := range va {
		assertRingKVMatch(t, "full-ring", i, va[i].keyBytes, vb[i].keyBytes, va[i].valueBytes, vb[i].valueBytes)
	}

	// the boundary hidden is the token-identity surface: same math, different fp accumulation
	// order — a few bf16 ulps, never structural divergence.
	if len(ringHidden) != len(rowHidden) {
		t.Fatalf("hidden sizes differ: ring=%d perRow=%d", len(ringHidden), len(rowHidden))
	}
	ringF := make([]float32, len(ringHidden)/bf16Size)
	rowF := make([]float32, len(rowHidden)/bf16Size)
	bf16ToF32Into(ringF, ringHidden)
	bf16ToF32Into(rowF, rowHidden)
	for i := range ringF {
		diff := ringF[i] - rowF[i]
		if diff < 0 {
			diff = -diff
		}
		limit := 0.03 * absf32(rowF[i])
		if limit < 1e-2 {
			limit = 1e-2
		}
		if diff > limit {
			t.Fatalf("deferred-ring hidden diverges at element %d: ring=%g perRow=%g (|diff|=%g > %g)", i, ringF[i], rowF[i], diff, limit)
		}
	}
}

// TestStepTokensBatchedDenseDeferredRingCrossingMatchesPerRow pins the wrap-CROSSING deferred
// chunk (#252): a single batch wider than the sliding window (basePos 0, K = 1.5·slideW — the
// merged tail the chunker now produces instead of a skinny follow-up chunk) runs the generalised
// ring kernel: empty pre-batch ring, per-query staged window [max(0, s-slideW+1) .. s], and only
// the last slideW rows landing. Landed ring byte-identity + tolerance hiddens vs the per-row
// staged interleave (the sequential-semantics oracle).
func TestStepTokensBatchedDenseDeferredRingCrossingMatchesPerRow(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasSDPAMultiQRing(mtpFixtureHead) || !gpuHasCopyKernel() {
		t.Fatal("deferred-ring kernels missing — rebuild dist/lib/lthn_kernels.metallib (task build:kernels)")
	}
	const slideW = steelGEMMMinRows
	kRows := slideW + slideW/2 // crosses the wrap: rows [slideW..kRows) evict during the batch
	mk := newMTPDecodeFixtureWithArchLayersMaxLen(t, func(arch *model.Arch) {
		arch.SlidingWindow = slideW
		for i := range arch.Layer {
			arch.Layer[i].Attention = model.SlidingAttention
		}
		arch.ValueNorm = true
	}, func(layers []DecodeLayerWeights) {
		for li := range layers {
			layers[li].QNormW = toBF16Bytes(syntheticFloat32(mtpFixtureHead, 900+li))
			layers[li].KNormW = toBF16Bytes(syntheticFloat32(mtpFixtureHead, 950+li))
		}
	}, 3*steelGEMMMinRows)

	tail := make([]int32, kRows)
	for i := range tail {
		tail[i] = int32(5 + (i*11)%23)
	}

	run := func(disable bool) ([]byte, int64, *ArchSession) {
		t.Helper()
		sess := mk()
		prev, prevTiming := stagedRingDisabledForTest, pieceTimingOn
		stagedRingDisabledForTest = disable
		pieceTimingOn = true
		stagedRingDispatchesForTest = 0
		defer func() {
			stagedRingDisabledForTest = prev
			pieceTimingOn = prevTiming
		}()
		hidden, ok, err := sess.prefillRetainedTokensBatchedDenseOne(tail, "test.deferredRingCrossing")
		if err != nil {
			t.Fatalf("crossing chunk (disableRing=%v): %v", disable, err)
		}
		if !ok {
			t.Fatalf("crossing chunk DECLINED (disableRing=%v)", disable)
		}
		return append([]byte(nil), hidden...), stagedRingDispatchesForTest, sess
	}

	ringHidden, ringDispatches, ringSess := run(false)
	rowHidden, rowDispatches, rowSess := run(true)
	if ringDispatches == 0 {
		t.Fatal("crossing deferred-ring lane did not engage (ring dispatch counter stayed 0)")
	}
	if rowDispatches != 0 {
		t.Fatalf("kill switch leaked: per-row run counted %d ring dispatches", rowDispatches)
	}
	if ringSess.Pos() != rowSess.Pos() {
		t.Fatalf("pos after crossing chunk: ring=%d perRow=%d", ringSess.Pos(), rowSess.Pos())
	}

	va, err := rowSess.stateLayerViews()
	if err != nil {
		t.Fatalf("per-row views: %v", err)
	}
	vb, err := ringSess.stateLayerViews()
	if err != nil {
		t.Fatalf("ring views: %v", err)
	}
	for i := range va {
		assertRingKVMatch(t, "crossing", i, va[i].keyBytes, vb[i].keyBytes, va[i].valueBytes, vb[i].valueBytes)
	}

	if len(ringHidden) != len(rowHidden) {
		t.Fatalf("hidden sizes differ: ring=%d perRow=%d", len(ringHidden), len(rowHidden))
	}
	ringF := make([]float32, len(ringHidden)/bf16Size)
	rowF := make([]float32, len(rowHidden)/bf16Size)
	bf16ToF32Into(ringF, ringHidden)
	bf16ToF32Into(rowF, rowHidden)
	for i := range ringF {
		diff := ringF[i] - rowF[i]
		if diff < 0 {
			diff = -diff
		}
		limit := 0.03 * absf32(rowF[i])
		if limit < 1e-2 {
			limit = 1e-2
		}
		if diff > limit {
			t.Fatalf("crossing hidden diverges at element %d: ring=%g perRow=%g (|diff|=%g > %g)", i, ringF[i], rowF[i], diff, limit)
		}
	}
}

// assertRingKVMatch pins the deferred-landing KV contract per layer: LAYER 0's landed rows must
// be byte-identical to the per-row lane (same inputs, same projection+rope — only the landing
// mechanics differ), while later layers inherit the SDPA's token-identity hiddens through their
// projections, so their landed rows are tolerance-checked (a few bf16 ulps, never structural).
func assertRingKVMatch(t *testing.T, lane string, li int, kA, kB, vA, vB []byte) {
	t.Helper()
	if li == 0 {
		for j := range kA {
			if kA[j] != kB[j] {
				t.Fatalf("%s layer 0 K diverges at byte %d (the landing must be byte-exact at layer 0)", lane, j)
			}
		}
		for j := range vA {
			if vA[j] != vB[j] {
				t.Fatalf("%s layer 0 V diverges at byte %d (the landing must be byte-exact at layer 0)", lane, j)
			}
		}
		return
	}
	// later layers compound the reordering noise through the synthetic fixture's unnormalised
	// weights (observed ~2% at layer 1, ~15% at layer 2 on small elements) — the bound's job is
	// catching landing/layout breaks, which diverge by orders of magnitude, not percents.
	check := func(name string, a, b []byte) {
		af := make([]float32, len(a)/bf16Size)
		bf := make([]float32, len(b)/bf16Size)
		bf16ToF32Into(af, a)
		bf16ToF32Into(bf, b)
		for j := range af {
			diff := af[j] - bf[j]
			if diff < 0 {
				diff = -diff
			}
			limit := 0.2 * absf32(bf[j])
			if limit < 5e-2 {
				limit = 5e-2
			}
			if diff > limit {
				t.Fatalf("%s layer %d %s diverges at element %d: %g vs %g (|diff|=%g > %g)", lane, li, name, j, af[j], bf[j], diff, limit)
			}
		}
	}
	check("K", kA, kB)
	check("V", vA, vB)
}

func TestMTPVerifyBatchedHiddensMatchesSequential(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	ref := mk()
	sess := mk()
	prompt := mtpWordedPromptIDs()
	if err := ref.PrefillTokens(prompt); err != nil {
		t.Fatalf("reference PrefillTokens: %v", err)
	}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("candidate PrefillTokens: %v", err)
	}
	ids := []int32{4, 5, 6}
	want := make([][]byte, len(ids))
	for i, id := range ids {
		hidden, err := ref.stepID(id)
		if err != nil {
			t.Fatalf("reference stepID(%d): %v", id, err)
		}
		want[i] = append([]byte(nil), hidden...)
	}
	got, ok, err := sess.verifyBatchedHiddens(ids)
	if err != nil {
		t.Fatalf("verifyBatchedHiddens: %v", err)
	}
	if !ok {
		t.Fatal("verifyBatchedHiddens ok = false")
	}
	if len(got) != len(want) {
		t.Fatalf("verifyBatchedHiddens returned %d rows, want %d", len(got), len(want))
	}
	for i := range got {
		eqBytes(t, core.Sprintf("batched hidden row %d", i), got[i], want[i])
	}
	if sess.Pos() != len(prompt) {
		t.Fatalf("verifyBatchedHiddens changed pos = %d, want %d", sess.Pos(), len(prompt))
	}
}

func TestMTPSampledDenseBatchRowPickerMatchesHiddenOnWordedPrompt(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	sess := mk()
	prompt := mtpWordedPromptIDs()
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(%q): %v", mtpWordedPromptText, err)
	}
	ids := []int32{4, 5, 6}
	hiddens, ok, err := sess.verifyBatchedHiddens(ids)
	if err != nil {
		t.Fatalf("verifyBatchedHiddens: %v", err)
	}
	if !ok {
		t.Fatal("verifyBatchedHiddens ok = false")
	}
	params := model.SampleParams{
		Temperature:    0.8,
		TopK:           7,
		TopP:           0.75,
		MinP:           0.01,
		RepeatPenalty:  1.2,
		SuppressTokens: []int32{2, 7},
	}
	history := []int32{3, 5, 8}
	const row = 1
	want, err := sess.sampleMTPTokenFromHidden(hiddens[row], model.NewSampler(83), params, history)
	if err != nil {
		t.Fatalf("sampleMTPTokenFromHidden: %v", err)
	}
	got, direct, err := sess.sampleMTPTokenFromDenseBatchRow(row, model.NewSampler(83), params, history)
	if err != nil {
		t.Fatalf("sampleMTPTokenFromDenseBatchRow: %v", err)
	}
	if !direct {
		t.Fatal("sampleMTPTokenFromDenseBatchRow declined the worded prompt batch row")
	}
	if got != want {
		t.Fatalf("sampleMTPTokenFromDenseBatchRow = %d, want hidden sample %d", got, want)
	}
}

func TestMTPVerifyBatchedUsesEmbedInto(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	control := mk()
	candidate := mk()
	for _, sess := range []*ArchSession{control, candidate} {
		for _, id := range mtpWordedPromptIDs() {
			if _, err := sess.stepID(id); err != nil {
				t.Fatalf("prefill stepID(%d): %v", id, err)
			}
		}
	}
	ids := []int32{4, 5, 6, 7}
	want := make([]int32, len(ids))
	if _, ok, err := control.verifyBatchedInto(ids, want); err != nil {
		t.Fatalf("control verifyBatchedInto: %v", err)
	} else if !ok {
		t.Fatal("control verifyBatchedInto ok = false")
	}

	candidate.embed = func(int32) ([]byte, error) {
		return nil, core.NewError("allocating embed path called")
	}
	candidate.embedFuncPtr = 0
	got := make([]int32, len(ids))
	if _, ok, err := candidate.verifyBatchedInto(ids, got); err != nil {
		t.Fatalf("candidate verifyBatchedInto: %v", err)
	} else if !ok {
		t.Fatal("candidate verifyBatchedInto ok = false")
	}
	if !mtpIDsEqual(got, want) {
		t.Fatalf("verifyBatchedInto embedInto greedys %v != allocating reference %v", got, want)
	}
}

func TestMTPPrefillPromptUsesEmbedInto(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	control := mk()
	candidate := mk()
	ids := mtpWordedPromptIDs()
	want, ok, err := control.prefillMTPPrompt(ids, true)
	if err != nil {
		t.Fatalf("control prefillMTPPrompt: %v", err)
	}
	if !ok {
		t.Fatal("control prefillMTPPrompt ok = false")
	}

	candidate.embed = func(int32) ([]byte, error) {
		return nil, core.NewError("allocating embed path called")
	}
	candidate.embedFuncPtr = 0
	got, ok, err := candidate.prefillMTPPrompt(ids, true)
	if err != nil {
		t.Fatalf("candidate prefillMTPPrompt: %v", err)
	}
	if !ok {
		t.Fatal("candidate prefillMTPPrompt ok = false")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("prefillMTPPrompt embedInto hidden differs from allocating reference")
	}
}

func TestMTPPrefillPromptRetainsLastHiddenNoCopy(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	sess := mk()
	ids := mtpWordedPromptIDs()

	hidden, ok, err := sess.prefillMTPPrompt(ids, true)
	if err != nil {
		t.Fatalf("prefillMTPPrompt: %v", err)
	}
	if !ok {
		t.Fatal("prefillMTPPrompt ok = false")
	}
	if sess.retainedHiddenPinned == nil || sess.retainedHiddenPinned.buf == nil {
		t.Fatal("prefillMTPPrompt did not retain a pinned last hidden")
	}
	if len(hidden) != len(sess.retainedHiddenPinned.bytes) {
		t.Fatalf("prefillMTPPrompt hidden len = %d, want retained pinned len %d", len(hidden), len(sess.retainedHiddenPinned.bytes))
	}
	if unsafe.Pointer(&hidden[0]) != unsafe.Pointer(&sess.retainedHiddenPinned.bytes[0]) {
		t.Fatal("prefillMTPPrompt hidden does not alias retained pinned backing")
	}
	if sess.retainedHiddenBufferFor(hidden) == nil {
		t.Fatal("prefillMTPPrompt retained hidden is not exposed as a no-copy buffer")
	}
}

func TestMTPStepIDRetainsHiddenNoCopy(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	sess := mk()

	hidden, err := sess.stepID(3)
	if err != nil {
		t.Fatalf("stepID: %v", err)
	}
	if sess.retainedHiddenPinned == nil || sess.retainedHiddenPinned.buf == nil {
		t.Fatal("stepID did not retain a pinned hidden")
	}
	if len(hidden) != len(sess.retainedHiddenPinned.bytes) {
		t.Fatalf("stepID hidden len = %d, want retained pinned len %d", len(hidden), len(sess.retainedHiddenPinned.bytes))
	}
	if unsafe.Pointer(&hidden[0]) != unsafe.Pointer(&sess.retainedHiddenPinned.bytes[0]) {
		t.Fatal("stepID hidden does not alias retained pinned backing")
	}
	if sess.retainedHiddenBufferFor(hidden) == nil {
		t.Fatal("stepID retained hidden is not exposed as a no-copy buffer")
	}
}

func TestGreedyFallbackUsesHeadLogitsScratch(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	sess := mk()

	hidden, err := sess.stepID(3)
	if err != nil {
		t.Fatalf("stepID: %v", err)
	}
	if sess.retainedHiddenBufferFor(hidden) == nil {
		t.Fatal("test setup did not retain hidden as a no-copy buffer")
	}
	sess.greedy = nil
	sess.sampleHeadLogits = nil

	got, err := sess.greedyFromHiddenInPool(hidden, nil)
	if err != nil {
		t.Fatalf("greedyFromHiddenInPool: %v", err)
	}
	if got < 0 || int(got) >= sess.arch.Vocab {
		t.Fatalf("greedyFromHiddenInPool token = %d outside vocab %d", got, sess.arch.Vocab)
	}
	if len(sess.sampleHeadLogits) != sess.arch.Vocab*bf16Size {
		t.Fatalf("fallback logits scratch len = %d, want %d", len(sess.sampleHeadLogits), sess.arch.Vocab*bf16Size)
	}
}

func TestMTPVerifyBatchedDirectHeadAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	dense := mk()
	for _, id := range mtpWordedPromptIDs() {
		if _, err := dense.stepID(id); err != nil {
			t.Fatalf("prefill dense stepID(%d): %v", id, err)
		}
	}
	ids := []int32{4, 5, 6, 7}
	greedys := make([]int32, len(ids))
	if _, ok, err := dense.verifyBatchedInto(ids, greedys); err != nil {
		t.Fatalf("verifyBatched warmup: %v", err)
	} else if !ok {
		t.Fatal("verifyBatched warmup ok = false")
	}

	var verifyErr error
	var verifyOK bool
	allocs := testing.AllocsPerRun(5, func() {
		_, verifyOK, verifyErr = dense.verifyBatchedInto(ids, greedys)
	})
	if verifyErr != nil {
		t.Fatalf("verifyBatched: %v", verifyErr)
	}
	if !verifyOK {
		t.Fatal("verifyBatched ok = false")
	}
	if allocs > 680 {
		t.Fatalf("verifyBatched allocations = %.0f, want <= 680", allocs)
	}
}

func TestMTPVerifyBatchedFallbackReusesPinnedHiddenRows(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	dense := mk()
	for _, id := range mtpWordedPromptIDs() {
		if _, err := dense.stepID(id); err != nil {
			t.Fatalf("prefill dense stepID(%d): %v", id, err)
		}
	}
	dense.greedy = func(hidden []byte, suppress []int32) (int32, bool, error) {
		return dense.headEnc.greedyInPool(hidden, suppress)
	}
	if dense.canUseDirectHeadGreedy() {
		t.Fatal("test setup still has direct head greedy enabled")
	}

	ids := []int32{4, 5, 6, 7}
	if _, ok, err := dense.verifyBatchedInto(ids, make([]int32, len(ids))); err != nil {
		t.Fatalf("verifyBatched fallback: %v", err)
	} else if !ok {
		t.Fatal("verifyBatched fallback ok = false")
	}
	if dense.mtpVerifyHiddenPinned == nil || dense.mtpVerifyHiddenPinned.buf == nil {
		t.Fatal("verifyBatched fallback did not retain pinned packed hidden rows")
	}
	if len(dense.mtpVerifyHiddenRows) != len(ids) {
		t.Fatalf("verifyBatched fallback retained %d rows, want %d", len(dense.mtpVerifyHiddenRows), len(ids))
	}
	base := unsafe.Pointer(&dense.mtpVerifyHiddenPinned.bytes[0])
	rowBytes := dense.arch.Hidden * bf16Size
	for i, row := range dense.mtpVerifyHiddenRows {
		if len(row) != rowBytes {
			t.Fatalf("hidden row %d length = %d, want %d", i, len(row), rowBytes)
		}
		if unsafe.Pointer(&row[0]) != unsafe.Pointer(&dense.mtpVerifyHiddenPinned.bytes[i*rowBytes]) {
			t.Fatalf("hidden row %d does not alias the pinned packed backing at %p", i, base)
		}
	}
}

func TestMTPDecodeWordedPromptEOSMatchesGenerate(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 8
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)

	first, err := mk().Generate(prompt, 1, -1)
	if err != nil {
		t.Fatalf("Generate first token: %v", err)
	}
	res, err := MTPDecode(mk(), mk(), prompt, maxNew, int(first[0]), K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, first) {
		t.Fatalf("MTP EOS tokens %v != first greedy %v", res.Tokens, first)
	}
	if len(res.Tokens) != 1 {
		t.Fatalf("MTP EOS emitted %d tokens, want 1", len(res.Tokens))
	}
}

func TestMTPDecodeEOSRollsBackTargetPosition(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 8
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)

	first, err := mk().Generate(prompt, 1, -1)
	if err != nil {
		t.Fatalf("Generate first token: %v", err)
	}
	target := mk()
	draft := mk()
	res, err := MTPDecode(target, draft, prompt, maxNew, int(first[0]), K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if len(res.Tokens) != 1 {
		t.Fatalf("MTP EOS emitted %d tokens, want 1", len(res.Tokens))
	}
	wantPos := len(prompt) + len(res.Tokens)
	if target.Pos() != wantPos {
		t.Fatalf("target pos after EOS = %d, want prompt+emitted %d", target.Pos(), wantPos)
	}
}

func TestMTPDecodeEOSRetainsDraftBoundaryForContinuation(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 8
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)

	want, err := mk().Generate(prompt, 2, -1)
	if err != nil {
		t.Fatalf("Generate reference: %v", err)
	}
	target := mk()
	draft := mk()
	res, err := MTPDecode(target, draft, prompt, maxNew, int(want[0]), K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, want[:1]) {
		t.Fatalf("MTP EOS tokens = %v, want first greedy %v", res.Tokens, want[:1])
	}
	wantPos := len(prompt) + len(res.Tokens)
	if draft.Pos() != wantPos {
		t.Fatalf("draft pos after EOS = %d, want prompt+emitted %d", draft.Pos(), wantPos)
	}
	got, err := draft.GenerateFromCache(1, -1)
	if err != nil {
		t.Fatalf("draft GenerateFromCache after MTPDecode: %v", err)
	}
	if !mtpIDsEqual(got, want[1:]) {
		t.Fatalf("draft GenerateFromCache after MTPDecode = %v, want next token %v", got, want[1:])
	}
}

func TestMTPDecodePopulatesTargetKVSnapshotTokens(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 3, 4
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)
	target := mk()
	draft := mk()

	res, err := MTPDecode(target, draft, prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	snapshot, err := target.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions after MTPDecode: %v", err)
	}
	want := append(append([]int32(nil), prompt...), res.Tokens...)
	if !mtpIDsEqual(snapshot.Tokens, want) {
		t.Fatalf("snapshot tokens after MTPDecode = %v, want %v", snapshot.Tokens, want)
	}
	if snapshot.TokenOffset != len(want) {
		t.Fatalf("snapshot token offset = %d, want %d", snapshot.TokenOffset, len(want))
	}
}

func TestMTPDecodeMaxNewRetainsBoundaryForContinuation(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 2
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)

	want, err := mk().Generate(prompt, maxNew+1, -1)
	if err != nil {
		t.Fatalf("Generate reference: %v", err)
	}
	target := mk()
	draft := mk()
	res, err := MTPDecode(target, draft, prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, want[:maxNew]) {
		t.Fatalf("MTP tokens = %v, want prefix %v", res.Tokens, want[:maxNew])
	}
	got, err := target.GenerateFromCache(1, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after MTPDecode: %v", err)
	}
	if !mtpIDsEqual(got, want[maxNew:]) {
		t.Fatalf("GenerateFromCache after MTPDecode = %v, want next token %v", got, want[maxNew:])
	}
}

func TestMTPDecodeDensePromptPrefillAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 1
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)
	target := mk()
	draft := mk()

	var decodeErr error
	allocs := testing.AllocsPerRun(3, func() {
		target.pos = 0
		draft.pos = 0
		var res *MTPResult
		res, decodeErr = MTPDecode(target, draft, prompt, maxNew, -1, K)
		if decodeErr == nil && len(res.Tokens) != maxNew {
			decodeErr = core.NewError("MTPDecode prompt-prefill fixture emitted wrong token count")
		}
	})
	if decodeErr != nil {
		t.Fatalf("MTPDecode: %v", decodeErr)
	}
	if allocs > 64108 {
		t.Fatalf("MTPDecode dense prompt-prefill allocations = %.0f, want <= 64108", allocs)
	}
}
