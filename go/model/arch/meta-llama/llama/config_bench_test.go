// SPDX-Licence-Identifier: EUPL-1.2

package llama

import "testing"

var (
	sinkArch  any
	sinkFreqs []float32
)

func BenchmarkConfig_Arch(b *testing.B) {
	cfg := Config{
		HiddenSize: 4096, IntermediateSize: 14336, NumHiddenLayers: 32,
		NumAttentionHeads: 32, NumKeyValueHeads: 8, VocabSize: 128256,
		RopeTheta: 500000,
		RopeScaling: &RopeScaling{
			RopeType: "llama3", Factor: 8, LowFreqFactor: 1, HighFreqFactor: 4,
			OriginalMaxPositionEmbeddings: 8192,
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		arch, _ := cfg.Arch()
		sinkArch = arch
	}
}

func BenchmarkLlama3InvFreqs(b *testing.B) {
	r := RopeScaling{
		RopeType: "llama3", Factor: 8, LowFreqFactor: 1, HighFreqFactor: 4,
		OriginalMaxPositionEmbeddings: 8192,
	}
	b.ReportAllocs()
	for b.Loop() {
		sinkFreqs = Llama3InvFreqs(500000, r, 128)
	}
}
