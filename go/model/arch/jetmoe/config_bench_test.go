// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import "testing"

var sinkArch any

func BenchmarkConfig_Arch(b *testing.B) {
	cfg := Config{HiddenSize: 2048, FFNHiddenSize: 5632, NumHiddenLayers: 24, NumAttentionHeads: 32, NumKeyValueHeads: 16, KVChannels: 128, MoENumExperts: 8, MoETopK: 2, VocabSize: 32000}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		arch, _ := cfg.Arch()
		sinkArch = arch
	}
}
