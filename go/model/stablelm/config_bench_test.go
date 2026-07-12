// SPDX-Licence-Identifier: EUPL-1.2

package stablelm

import "testing"

var sinkArch any

func BenchmarkConfig_Arch(b *testing.B) {
	c := Config{ModelType: "stablelm", HiddenSize: 512, IntermediateSize: 2048, NumHiddenLayers: 6, NumAttentionHeads: 8, VocabSize: 50304, PartialRotaryFactor: .25}
	b.ReportAllocs()
	for range b.N {
		sinkArch, _ = c.Arch()
	}
}
