// SPDX-Licence-Identifier: EUPL-1.2

package cohere

import "testing"

var sinkArch any

func BenchmarkConfig_Arch(b *testing.B) {
	cfg := Config{ModelType: "cohere2", HiddenSize: 4096, IntermediateSize: 14336, NumHiddenLayers: 32, NumAttentionHeads: 32, NumKeyValueHeads: 8, VocabSize: 256000, SlidingWindow: 4096, SlidingWindowPattern: 4, LogitScale: 0.25}
	b.ReportAllocs()
	for b.Loop() {
		arch, _ := cfg.Arch()
		sinkArch = arch
	}
}
