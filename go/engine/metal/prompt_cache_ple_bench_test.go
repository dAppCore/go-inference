// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import "testing"

func benchWarmPromptCachePLESequential(b *testing.B, gpuInputs bool) {
	requireNativeRuntime(b)
	sess := newPromptCachePLEFixture(b)
	prefix := []int32{1, 5, 3, 7}
	oldChainDisabled := chainedGPUInputsDisabled
	chainedGPUInputsDisabled = !gpuInputs
	defer func() { chainedGPUInputsDisabled = oldChainDisabled }()
	if err := sess.WarmPromptCache(prefix); err != nil {
		b.Fatalf("WarmPromptCache warmup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess.pos = 0
		sess.cachedIDs = sess.cachedIDs[:0]
		if err := sess.WarmPromptCache(prefix); err != nil {
			b.Fatalf("WarmPromptCache: %v", err)
		}
	}
}

func BenchmarkWarmPromptCachePLESequential(b *testing.B) {
	benchWarmPromptCachePLESequential(b, true)
}

func BenchmarkWarmPromptCachePLESequentialHost(b *testing.B) {
	benchWarmPromptCachePLESequential(b, false)
}

func BenchmarkWarmPromptCachePLESequentialGPUInputs(b *testing.B) {
	benchWarmPromptCachePLESequential(b, true)
}

func benchPrefillTokensPLE(b *testing.B, gpuInputs bool) {
	requireNativeRuntime(b)
	sess := newPromptCachePLEFixture(b)
	prefix := []int32{1, 5, 3, 7}
	oldChainDisabled := chainedGPUInputsDisabled
	chainedGPUInputsDisabled = !gpuInputs
	defer func() { chainedGPUInputsDisabled = oldChainDisabled }()
	if err := sess.PrefillTokens(prefix); err != nil {
		b.Fatalf("PrefillTokens warmup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := sess.PrefillTokens(prefix); err != nil {
			b.Fatalf("PrefillTokens: %v", err)
		}
	}
}

func BenchmarkPrefillTokensPLESequentialHost(b *testing.B) {
	benchPrefillTokensPLE(b, false)
}

func BenchmarkPrefillTokensPLESequentialGPUInputs(b *testing.B) {
	benchPrefillTokensPLE(b, true)
}
