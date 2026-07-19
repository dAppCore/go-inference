// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

func BenchmarkMTPDecodeDraftEqualsTarget(b *testing.B) {
	requireNativeRuntime(b)
	const K, maxNew = 4, 16
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(b)
	target := mk()
	draft := mk()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target.pos = 0
		draft.pos = 0
		res, err := MTPDecode(target, draft, prompt, maxNew, -1, K)
		if err != nil {
			b.Fatalf("MTPDecode: %v", err)
		}
		if res.Accepted != res.Drafted {
			b.Fatalf("accepted %d drafted %d", res.Accepted, res.Drafted)
		}
	}
}

func BenchmarkMTPDecodeDensePromptPrefill(b *testing.B) {
	requireNativeRuntime(b)
	const K, maxNew = 4, 1
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(b)
	target := mk()
	draft := mk()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target.pos = 0
		draft.pos = 0
		res, err := MTPDecode(target, draft, prompt, maxNew, -1, K)
		if err != nil {
			b.Fatalf("MTPDecode: %v", err)
		}
		if len(res.Tokens) != maxNew {
			b.Fatalf("tokens = %d, want %d", len(res.Tokens), maxNew)
		}
	}
}

func BenchmarkMTPDecodeSampledDirectRows(b *testing.B) {
	requireNativeRuntime(b)
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
	mk := newMTPDecodeFixture(b)
	target := mk()
	draft := mk()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target.pos = 0
		draft.pos = 0
		res, err := MTPDecodeSampled(target, draft, prompt, maxNew, nil, model.NewSampler(seed), model.NewSampler(seed+1), params, K)
		if err != nil {
			b.Fatalf("MTPDecodeSampled: %v", err)
		}
		if len(res.Tokens) != maxNew {
			b.Fatalf("tokens = %d, want %d", len(res.Tokens), maxNew)
		}
	}
}

func BenchmarkAssistantPairGenerateSampledLowAcceptFallback(b *testing.B) {
	requireNativeRuntime(b)
	pair, mk := newNativeAssistantGenerateFixture(b)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5}
	prompt, seed, _ := nativeAssistantSampledPromptWithRejectedFirstDraft(b, pair, mk, params)
	const maxNew = 6
	const draftTokens = 2
	target := mk()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := pair.GenerateSampledFromSession(target, prompt, maxNew, nil, model.NewSampler(seed), params, draftTokens)
		if err != nil {
			b.Fatalf("GenerateSampledFromSession: %v", err)
		}
		if len(got.Tokens) != maxNew {
			b.Fatalf("tokens = %d, want %d", len(got.Tokens), maxNew)
		}
		if got.DraftCalls != nativeAssistantLowAcceptPatience || got.TargetVerifyCalls != nativeAssistantLowAcceptPatience {
			b.Fatalf("draft/verify calls = %d/%d, want %d weak blocks before target-cache fallback", got.DraftCalls, got.TargetVerifyCalls, nativeAssistantLowAcceptPatience)
		}
	}
}

func BenchmarkMTPDecodeSequentialFallback(b *testing.B) {
	requireNativeRuntime(b)
	const K, maxNew = 4, 12
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(b)
	target := mtpSequentialFallbackSession(mk())
	draft := mtpSequentialFallbackSession(mk())

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target.pos = 0
		draft.pos = 0
		res, err := MTPDecode(target, draft, prompt, maxNew, -1, K)
		if err != nil {
			b.Fatalf("MTPDecode: %v", err)
		}
		if res.Accepted != res.Drafted {
			b.Fatalf("accepted %d drafted %d", res.Accepted, res.Drafted)
		}
	}
}

func BenchmarkMTPVerifyBatchedFallbackReusedHiddenRows(b *testing.B) {
	requireNativeRuntime(b)
	mk := newMTPDecodeFixture(b)
	dense := mk()
	for _, id := range mtpWordedPromptIDs() {
		if _, err := dense.stepID(id); err != nil {
			b.Fatalf("prefill dense stepID(%d): %v", id, err)
		}
	}
	dense.greedy = func(hidden []byte, suppress []int32) (int32, bool, error) {
		return dense.headEnc.greedyInPool(hidden, suppress)
	}
	ids := []int32{4, 5, 6, 7}
	greedys := make([]int32, len(ids))
	if _, ok, err := dense.verifyBatchedInto(ids, greedys); err != nil {
		b.Fatalf("verifyBatched warmup: %v", err)
	} else if !ok {
		b.Fatal("verifyBatched warmup ok = false")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok, err := dense.verifyBatchedInto(ids, greedys); err != nil {
			b.Fatalf("verifyBatched: %v", err)
		} else if !ok {
			b.Fatal("verifyBatched ok = false")
		}
	}
}
