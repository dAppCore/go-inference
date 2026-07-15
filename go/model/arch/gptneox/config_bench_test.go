// SPDX-Licence-Identifier: EUPL-1.2

package gptneox

import "testing"

var sinkArch any

func BenchmarkConfig_Arch(b *testing.B) {
	cfg := Config{ModelType: "gpt_neox", HiddenSize: 512, IntermediateSize: 2048, NumHiddenLayers: 6, NumAttentionHeads: 8, VocabSize: 50304, RotaryPct: .25}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkArch, _ = cfg.Arch()
	}
}
