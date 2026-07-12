// SPDX-Licence-Identifier: EUPL-1.2

package granitemoe

import "testing"

var benchmarkArch any

func BenchmarkConfig_Arch(b *testing.B) {
	cfg := Config{ModelType: "granitemoe", HiddenSize: 1024, IntermediateSize: 512, NumHiddenLayers: 24, NumAttentionHeads: 16, NumKeyValueHeads: 8, NumLocalExperts: 32, NumExpertsPerTok: 8, VocabSize: 49152, RMSNormEps: 1e-6, RopeTheta: 1_500_000, LogitsScaling: 6, ResidualMultiplier: .22, EmbeddingMultiplier: 12, AttentionMultiplier: .015625}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		arch, err := cfg.Arch()
		if err != nil {
			b.Fatal(err)
		}
		benchmarkArch = arch
	}
}
