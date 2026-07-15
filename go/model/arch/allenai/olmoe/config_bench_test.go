// SPDX-Licence-Identifier: EUPL-1.2

package olmoe

import "testing"

var sinkArchHidden int

func BenchmarkConfig_Arch(b *testing.B) {
	cfg := Config{
		HiddenSize: 2048, IntermediateSize: 1024, NumHiddenLayers: 16,
		NumAttentionHeads: 16, NumKeyValueHeads: 16,
		NumExperts: 64, NumExpertsPerTok: 8, VocabSize: 50304,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		arch, err := cfg.Arch()
		if err != nil {
			b.Fatal(err)
		}
		sinkArchHidden = arch.Hidden
	}
}
