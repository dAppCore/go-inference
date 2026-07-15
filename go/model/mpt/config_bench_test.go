// SPDX-Licence-Identifier: EUPL-1.2

package mpt

import "testing"

var sinkArch any

func BenchmarkConfig_Arch(b *testing.B) {
	c := Config{ModelType: "mpt", DModel: 512, NHeads: 8, NLayers: 6, ExpansionRatio: 4, VocabSize: 50304, AttnConfig: AttentionConfig{ALiBi: true}}
	b.ReportAllocs()
	for range b.N {
		sinkArch, _ = c.Arch()
	}
}
