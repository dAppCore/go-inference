// SPDX-Licence-Identifier: EUPL-1.2

package smollm3

import "testing"

var sinkArch any

func BenchmarkConfig_Arch(b *testing.B) {
	c := Config{ModelType: "smollm3", HiddenSize: 512, IntermediateSize: 2048, NumHiddenLayers: 8, NumAttentionHeads: 8, NumKeyValueHeads: 2, VocabSize: 50304, NoRopeLayerInterval: 4}
	b.ReportAllocs()
	for range b.N {
		sinkArch, _ = c.Arch()
	}
}
