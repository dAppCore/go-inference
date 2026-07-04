// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

func BenchmarkWarmPromptCacheRetainedIDs(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	prefix := []int32{1, 2, 3, 4, 5}
	if err := s.WarmPromptCache(prefix); err != nil {
		b.Fatalf("WarmPromptCache warmup: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := s.WarmPromptCache(prefix); err != nil {
			b.Fatalf("WarmPromptCache: %v", err)
		}
	}
}

func BenchmarkGenerateCachedExactPromptHiddenLogitsReplay(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	prompt := []int32{1, 2, 3, 4, 5}
	if _, err := s.GenerateCached(prompt, 2, -1); err != nil {
		b.Fatalf("GenerateCached warmup: %v", err)
	}
	if hit := s.CachedPrefixLen(prompt); hit != len(prompt) {
		b.Fatalf("exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.GenerateCached(prompt, 2, -1); err != nil {
			b.Fatalf("GenerateCached exact: %v", err)
		}
	}
}

func BenchmarkGenerateCachedAfterWarmPromptCacheExactReplay(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	prompt := []int32{1, 2, 3, 4, 5}
	if err := s.WarmPromptCache(prompt); err != nil {
		b.Fatalf("WarmPromptCache: %v", err)
	}
	if hit := s.CachedPrefixLen(prompt); hit != len(prompt) {
		b.Fatalf("exact prompt-cache hit after warm = %d, want %d", hit, len(prompt))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.GenerateCached(prompt, 2, -1); err != nil {
			b.Fatalf("GenerateCached exact after warm: %v", err)
		}
	}
}

func BenchmarkGenerateSampledPromptReplayNoCache(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 5, TopP: 0.75}
	sampler := model.NewSampler(1)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.pos = 0
		if _, err := s.GenerateSampledEach(prompt, 2, nil, sampler, params, nil, nil); err != nil {
			b.Fatalf("GenerateSampledEach prompt replay: %v", err)
		}
	}
}

func BenchmarkGenerateCachedSampledExactPromptReplay(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 5, TopP: 0.75}
	if err := s.WarmPromptCache(prompt); err != nil {
		b.Fatalf("WarmPromptCache: %v", err)
	}
	if hit := s.CachedPrefixLen(prompt); hit != len(prompt) {
		b.Fatalf("exact prompt-cache hit after warm = %d, want %d", hit, len(prompt))
	}
	sampler := model.NewSampler(1)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.GenerateCachedSampledEach(prompt, 2, nil, sampler, params, nil, nil); err != nil {
			b.Fatalf("GenerateCachedSampledEach exact after warm: %v", err)
		}
	}
}

func BenchmarkGenerateCachedExactPromptFourTokens(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	prompt := []int32{1, 2, 3, 4, 5}
	if _, err := s.GenerateCached(prompt, 4, -1); err != nil {
		b.Fatalf("GenerateCached warmup: %v", err)
	}
	if hit := s.CachedPrefixLen(prompt); hit != len(prompt) {
		b.Fatalf("exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.GenerateCached(prompt, 4, -1); err != nil {
			b.Fatalf("GenerateCached exact: %v", err)
		}
	}
}

func BenchmarkGenerateCachedEachExactPromptStopAfterOneOfFour(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	prompt := []int32{1, 2, 3, 4, 5}
	if _, err := s.GenerateCached(prompt, 4, -1); err != nil {
		b.Fatalf("GenerateCached warmup: %v", err)
	}
	if hit := s.CachedPrefixLen(prompt); hit != len(prompt) {
		b.Fatalf("exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}
	stopAfterOne := func(int32) bool { return false }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.GenerateCachedEach(prompt, 4, -1, stopAfterOne); err != nil {
			b.Fatalf("GenerateCachedEach exact stop: %v", err)
		}
	}
}

func BenchmarkGenerateCachedOneTokenSuffixReplay(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	prompts := [2][]int32{
		{1, 2, 3, 4, 5},
		{1, 2, 3, 4, 6},
	}
	if _, err := s.GenerateCached(prompts[0], 2, -1); err != nil {
		b.Fatalf("GenerateCached warmup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prompt := prompts[i&1]
		if _, err := s.GenerateCached(prompt, 2, -1); err != nil {
			b.Fatalf("GenerateCached suffix: %v", err)
		}
	}
}

func BenchmarkGenerateCachedSampledOneTokenSuffixReplay(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	prompts := [2][]int32{
		{1, 2, 3, 4, 5},
		{1, 2, 3, 4, 6},
	}
	params := model.SampleParams{Temperature: 0.8, TopK: 5, TopP: 0.75}
	if err := s.WarmPromptCache(prompts[0]); err != nil {
		b.Fatalf("WarmPromptCache: %v", err)
	}
	sampler := model.NewSampler(1)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prompt := prompts[i&1]
		if _, err := s.GenerateCachedSampledEach(prompt, 1, nil, sampler, params, nil, nil); err != nil {
			b.Fatalf("GenerateCachedSampledEach suffix: %v", err)
		}
	}
}

func BenchmarkCompactCacheRetainedIDs(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5, 6, 7, 8}, 6, -1); err != nil {
		b.Fatalf("GenerateCached warmup: %v", err)
	}
	resident := append([]int32(nil), s.cachedIDs...)
	const keep = 4
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.cachedIDs = resident
		s.pos = len(resident)
		if err := s.CompactCache(keep); err != nil {
			b.Fatalf("CompactCache: %v", err)
		}
	}
}
