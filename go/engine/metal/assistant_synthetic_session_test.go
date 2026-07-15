// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/mtp"
	"dappco.re/go/inference/model/safetensors"
)

// assistant_synthetic_session_test.go builds a REUSABLE synthetic speculative-decode
// pair — a bf16 target ArchSession plus an attached AssistantPair drafter — entirely
// from pure-Go synthetic tensors (no HF download, no metal_runtime-only asset). Every
// other AssistantPair-drives-a-live-session test in this package (assistant_live_test.go,
// assistant_quant_parity_test.go) needs a real gemma-4-E2B checkpoint pulled via
// enginegate.HFModelPath, so the whole *FromSession family — DraftBlockFromSession,
// VerifyDraftBlockFromSession, GenerateFromSession(Each), the sampled twins and the
// low-accept finish tails — sat dark under CI. This fixture opens them: the drafter's
// proposals never MATCH the (independent, synthetic) target, so acceptance is ~0, which
// is exactly what exercises the reject/replacement + plain-finish arms these tests pin
// behaviourally (there is no trusted reference to check TOKENS against — the pins are
// the loop invariants: tokens in vocab, block lengths, maxNew fill, sink parity).

const (
	synthPairDModel  = 8  // target hidden == assistant backbone_hidden_size
	synthPairNHeads  = 2  // target attention heads
	synthPairNKV     = 2  // target kv heads (drafter cross-attends this stream)
	synthPairHeadDim = 64 // shared head dim (target K/V rows the drafter reads)
	synthPairDFF     = 16 // target feed-forward width
	synthPairVocab   = 8  // shared vocab (drafter + target lm heads)
	synthPairMaxLen  = 64
)

// syntheticAttentionAssistant loads the package's one-sliding-layer attention drafter
// (backbone 8, hidden 128, headDim 64) with self-consistent synthetic bf16 weights, so
// DraftStepActivations runs a real attention+MLP forward rather than an all-zero shell.
func syntheticAttentionAssistant(t *testing.T) *AssistantModel {
	t.Helper()
	const hidden, backbone, nHeads, headDim, dFF, vocab = 128, synthPairDModel, synthPairNHeads, synthPairHeadDim, 256, synthPairVocab
	tensors := nativeAssistantAttentionTensors()
	p := "model.layers.0"
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{vocab, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(vocab, hidden))}
	tensors["model.norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 83))}
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, backbone * 2}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, backbone*2))}
	tensors["post_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{backbone, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(backbone, hidden))}
	tensors[p+".input_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 89))}
	tensors[p+".post_attention_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 97))}
	tensors[p+".pre_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 101))}
	tensors[p+".post_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 103))}
	tensors[p+".layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: toBF16Bytes([]float32{0.625})}
	tensors[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(nHeads*headDim, hidden))}
	tensors[p+".self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, nHeads*headDim))}
	tensors[p+".self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(syntheticFloat32(headDim, 107))}
	tensors[p+".mlp.gate_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.up_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.down_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, dFF}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, dFF))}
	assistant, err := LoadAssistantDir(writeNativeAssistantAttentionDir(t, tensors))
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	return assistant
}

// syntheticAssistantTargetSession builds a bf16 target ArchSession with a single
// sliding-attention layer whose dims (hidden 8 == backbone, headDim 64, kv heads 2)
// match syntheticAttentionAssistant, so validateTargetSessionArch passes and the drafter
// cross-attends the session's live K/V. Head + greedy are wired exactly as the token
// model would, so prepareAssistantPrompt / boundaryNormedHidden / verify all run.
func syntheticAssistantTargetSession(t *testing.T) (*ArchSession, model.Arch) {
	t.Helper()
	specs := model.DeriveLayers([]string{"sliding_attention"}, 0)
	layers := []DecodeLayerWeights{forwardLayer(synthPairDModel, synthPairNHeads, synthPairNKV, synthPairHeadDim, synthPairDFF, 100)}
	embed := toBF16Bytes(syntheticFloat32(synthPairVocab*synthPairDModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(synthPairDModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: synthPairDModel, Heads: synthPairNHeads, KVHeads: synthPairNKV, HeadDim: synthPairHeadDim,
		FF: synthPairDFF, Vocab: synthPairVocab, GlobalHeadDim: synthPairHeadDim, GlobalKVHeads: synthPairNKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: synthPairHeadDim, RotaryDimLocal: synthPairHeadDim, SlidingWindow: 16, Layer: specs,
	}
	sess, err := NewArchSession(g, arch, synthPairMaxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	head := &headEncoder{finalNorm: copyView(g.FinalNorm), weight: copyView(g.LMHead), dModel: arch.Hidden, vocab: arch.Vocab, eps: arch.Eps}
	sess.headEnc = head
	sess.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) { return head.encode(hidden, skipSoftcap) }
	sess.greedy = func(hidden []byte, suppress []int32) (int32, bool, error) { return head.greedyInPool(hidden, suppress) }
	sess.markDefaultHeadFunc()
	sess.markDefaultGreedyFunc()
	return sess, arch
}

// newSyntheticAssistantPair is the reusable fixture: a fresh unprefilled target session,
// its attached AssistantPair, and a short prompt (ids within vocab). Callers that hit the
// *FromSession draft/verify paths prefill via PrepareAssistantPrompt first; the
// Generate*FromSession entry points prefill internally.
func newSyntheticAssistantPair(t *testing.T) (*ArchSession, *AssistantPair, []int32) {
	t.Helper()
	requireNativeRuntime(t)
	sess, arch := syntheticAssistantTargetSession(t)
	assistant := syntheticAttentionAssistant(t)
	t.Cleanup(func() { assistant.Close() })
	pair := &AssistantPair{TargetArch: arch, Assistant: assistant}
	return sess, pair, []int32{1, 2, 3, 4, 5}
}

func synthTokensInVocab(t *testing.T, what string, ids []int32) {
	t.Helper()
	for i, id := range ids {
		if id < 0 || int(id) >= synthPairVocab {
			t.Fatalf("%s[%d] = %d outside vocab %d", what, i, id, synthPairVocab)
		}
	}
}

func TestSyntheticAssistantPairMethodAndPrepareAssistantPrompt(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	if m := pair.Method(); m != mtp.MTPDraftModel {
		t.Fatalf("Method = %q, want %q (unstamped config defaults to draft-model)", m, mtp.MTPDraftModel)
	}
	if err := sess.PrepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("PrepareAssistantPrompt: %v", err)
	}
	if sess.Pos() != len(prompt) {
		t.Fatalf("PrepareAssistantPrompt pos = %d, want %d", sess.Pos(), len(prompt))
	}
	// The boundary seed the drafter reads must be materialised by the prepare.
	if seed, err := sess.BoundaryNormedHidden(); err != nil {
		t.Fatalf("BoundaryNormedHidden: %v", err)
	} else if len(seed) != synthPairDModel*bf16Size {
		t.Fatalf("boundary seed len = %d, want %d", len(seed), synthPairDModel*bf16Size)
	}
}

func TestSyntheticAssistantPairDraftBlockAndVerifyFromSession(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	if err := sess.PrepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("PrepareAssistantPrompt: %v", err)
	}
	last := prompt[len(prompt)-1]

	const maxDraft = 4
	block, err := pair.DraftBlockFromSession(sess, last, maxDraft)
	if err != nil {
		t.Fatalf("DraftBlockFromSession: %v", err)
	}
	if len(block.Tokens) != maxDraft {
		t.Fatalf("draft block produced %d tokens, want %d", len(block.Tokens), maxDraft)
	}
	synthTokensInVocab(t, "draft", block.Tokens)

	// A single draft step must agree with the first token of the block (same seed,
	// same drafter, deterministic greedy) — the per-step path and the block path share
	// draftStepFromProjectedInto.
	step, err := pair.DraftStepFromSession(sess, last)
	if err != nil {
		t.Fatalf("DraftStepFromSession: %v", err)
	}
	if step.Token != block.Tokens[0] {
		t.Fatalf("DraftStepFromSession token = %d, want first block token %d", step.Token, block.Tokens[0])
	}

	posBefore := sess.Pos()
	vr, err := pair.VerifyDraftBlockFromSession(sess, block.Tokens)
	if err != nil {
		t.Fatalf("VerifyDraftBlockFromSession: %v", err)
	}
	if len(vr.TargetTokens) == 0 {
		t.Fatal("verify returned no target tokens")
	}
	synthTokensInVocab(t, "verify.TargetTokens", vr.TargetTokens)
	if vr.AcceptedCount < 0 || vr.AcceptedCount > len(block.Tokens) {
		t.Fatalf("accepted count = %d outside [0,%d]", vr.AcceptedCount, len(block.Tokens))
	}
	if vr.AcceptedCount+vr.RejectedCount != len(block.Tokens) {
		t.Fatalf("accepted %d + rejected %d != drafted %d", vr.AcceptedCount, vr.RejectedCount, len(block.Tokens))
	}
	// Verify must not commit tokens to the session cache — it rolls back to the boundary.
	if sess.Pos() != posBefore {
		t.Fatalf("verify changed session pos %d -> %d (must roll back)", posBefore, sess.Pos())
	}
}

func TestSyntheticAssistantPairGenerateFromSession(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	const maxNew, draftTokens = 12, 4
	res, err := pair.GenerateFromSession(sess, prompt, maxNew, -1, draftTokens, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession: %v", err)
	}
	if len(res.Tokens) != maxNew {
		t.Fatalf("generated %d tokens, want %d (eos disabled)", len(res.Tokens), maxNew)
	}
	synthTokensInVocab(t, "generate", res.Tokens)
	if res.DraftTokens == 0 {
		t.Fatal("speculative lane never engaged (no draft tokens)")
	}
	if res.PromptTokens != len(prompt) {
		t.Fatalf("PromptTokens = %d, want %d", res.PromptTokens, len(prompt))
	}
	// Accepted + rejected proposals cannot exceed what was drafted.
	if res.AcceptedTokens+res.RejectedTokens > res.DraftTokens {
		t.Fatalf("accepted %d + rejected %d > drafted %d", res.AcceptedTokens, res.RejectedTokens, res.DraftTokens)
	}
}

func TestSyntheticAssistantPairGenerateFromSessionEachYieldsCommittedTokens(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	const maxNew, draftTokens = 10, 4
	var yielded []int32
	res, err := pair.GenerateFromSessionEach(sess, prompt, maxNew, -1, draftTokens, nil, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateFromSessionEach: %v", err)
	}
	if !mtpIDsEqual(yielded, res.Tokens) {
		t.Fatalf("yielded %v != result tokens %v", yielded, res.Tokens)
	}
}

func TestSyntheticAssistantPairGenerateFromSessionEachStopsOnSinkFalse(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	const stopAfter = 3
	var yielded []int32
	res, err := pair.GenerateFromSessionEach(sess, prompt, 12, -1, 4, nil, func(id int32) bool {
		yielded = append(yielded, id)
		return len(yielded) < stopAfter
	})
	if err != nil {
		t.Fatalf("GenerateFromSessionEach: %v", err)
	}
	if len(res.Tokens) > stopAfter {
		t.Fatalf("sink stop ignored: emitted %d tokens, want <= %d", len(res.Tokens), stopAfter)
	}
}

// TestSyntheticAssistantPairLowAcceptFinishFromTargetCache pins the permanent-bail tail:
// with re-engagement disabled (LTHN_MTP_REENGAGE=0 semantics), a drafter that stays weak
// for the patience window retires and nativeAssistantFinishLowAcceptFromTargetCache
// finishes the request with plain target decode — filling exactly to maxNew.
func TestSyntheticAssistantPairLowAcceptFinishFromTargetCache(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	prev := mtpReengageDisabled
	mtpReengageDisabled = true
	t.Cleanup(func() { mtpReengageDisabled = prev })

	const maxNew, draftTokens = 20, 4
	res, err := pair.GenerateFromSession(sess, prompt, maxNew, -1, draftTokens, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession: %v", err)
	}
	if len(res.Tokens) != maxNew {
		t.Fatalf("low-accept finish produced %d tokens, want %d", len(res.Tokens), maxNew)
	}
	synthTokensInVocab(t, "finish", res.Tokens)
	// The plain-decode tail commits target tokens directly.
	if res.TargetTokens == 0 {
		t.Fatal("low-accept finish committed no target tokens")
	}
}

func TestSyntheticAssistantPairGenerateSampledFromSession(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9, MinP: 0.01, RepeatPenalty: 1.1}
	const maxNew, draftTokens = 12, 4
	res, err := pair.GenerateSampledFromSession(sess, prompt, maxNew, nil, model.NewSampler(53), params, draftTokens)
	if err != nil {
		t.Fatalf("GenerateSampledFromSession: %v", err)
	}
	if len(res.Tokens) != maxNew {
		t.Fatalf("sampled generated %d tokens, want %d", len(res.Tokens), maxNew)
	}
	synthTokensInVocab(t, "sampled", res.Tokens)
	if res.DraftTokens == 0 {
		t.Fatal("sampled speculative lane never engaged")
	}
}

func TestSyntheticAssistantPairGenerateSampledFromSessionEachYields(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	params := model.SampleParams{Temperature: 0.7, TopK: 5, RepeatPenalty: 1.2, SuppressTokens: []int32{0}}
	var yielded []int32
	res, err := pair.GenerateSampledFromSessionEach(sess, prompt, 10, nil, model.NewSampler(71), params, 4, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateSampledFromSessionEach: %v", err)
	}
	if !mtpIDsEqual(yielded, res.Tokens) {
		t.Fatalf("sampled yielded %v != result tokens %v", yielded, res.Tokens)
	}
	// SuppressTokens{0} must hold across the whole committed stream.
	for i, id := range res.Tokens {
		if id == 0 {
			t.Fatalf("sampled token[%d] = 0 but token 0 is suppressed", i)
		}
	}
}

func TestSyntheticAssistantPairLowAcceptFinishSampledFromTargetCache(t *testing.T) {
	sess, pair, prompt := newSyntheticAssistantPair(t)
	prev := mtpReengageDisabled
	mtpReengageDisabled = true
	t.Cleanup(func() { mtpReengageDisabled = prev })

	params := model.SampleParams{Temperature: 0.8, TopK: 4, RepeatPenalty: 1.1}
	const maxNew, draftTokens = 20, 4
	res, err := pair.GenerateSampledFromSession(sess, prompt, maxNew, nil, model.NewSampler(89), params, draftTokens)
	if err != nil {
		t.Fatalf("GenerateSampledFromSession: %v", err)
	}
	if len(res.Tokens) != maxNew {
		t.Fatalf("sampled low-accept finish produced %d tokens, want %d", len(res.Tokens), maxNew)
	}
	synthTokensInVocab(t, "sampledFinish", res.Tokens)
}
