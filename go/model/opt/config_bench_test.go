// SPDX-Licence-Identifier: EUPL-1.2

package opt

import "testing"

var sinkArch any

func BenchmarkConfig_Arch(b *testing.B) {
	config := Config{Hidden: 768, EmbedDim: 768, Heads: 12, Layers: 12, FF: 3072, Positions: 2048, Vocab: 50272}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkArch, _ = config.Arch()
	}
}
