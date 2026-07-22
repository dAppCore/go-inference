// SPDX-Licence-Identifier: EUPL-1.2

package mistral

import "testing"

// The mistral declaration package was whole un-benched (no _bench_test.go). It is config-only —
// Ministral-3 runs through the native executor reusing model.Arch — so its benchable surface is
// the pure-Go arch build (Config.Arch → DeriveLayers + the YaRN frequency remap) and YaRNInvFreqs
// itself. Both are CPU-only (no GPU, no model), run once per model load, and run in core go qa.
// These are the AX-11 alloc baselines a later change to the config resolution is held to.

// benchMinistralConfig is a realistic Ministral-3-shaped config WITH YaRN long-context enabled, so
// Config.Arch exercises the full path including YaRNInvFreqs (factor > 1).
func benchMinistralConfig() Config {
	return Config{
		HiddenSize: 4096, NumHiddenLayers: 36, IntermediateSize: 12288,
		NumAttentionHeads: 32, NumKeyValueHeads: 8, HeadDim: 128,
		VocabSize: 131072, RMSNormEps: 1e-5,
		RopeParameters: &RopeParams{
			RopeTheta: 1_000_000, RopeType: "yarn",
			Factor: 8, BetaFast: 32, BetaSlow: 1, OriginalMaxPositionEmbeddings: 32768,
		},
	}
}

// BenchmarkConfigArch measures the Ministral arch build (Config.Arch → DeriveLayers over all-global
// layer types + the YaRN inverse-frequency resolution), allocated once per model load. CPU-only.
func BenchmarkConfigArch(b *testing.B) {
	cfg := benchMinistralConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := cfg.Arch(); err != nil {
			b.Fatalf("Arch: %v", err)
		}
	}
}

// BenchmarkYaRNInvFreqs measures the YaRN per-dim inverse-frequency remap in isolation — the
// NTK-by-parts blend (a Pow + a ramp per half-dim), computed once per model load for a long-context
// pack. Realistic Ministral-3 params (head_dim 128 → 64 freqs, 8x extension).
func BenchmarkYaRNInvFreqs(b *testing.B) {
	const dim = 128
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := YaRNInvFreqs(1_000_000, 8, 32, 1, 32768, dim)
		if len(f) != dim/2 {
			b.Fatalf("YaRNInvFreqs len = %d, want %d", len(f), dim/2)
		}
	}
}
