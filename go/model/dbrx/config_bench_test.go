// SPDX-Licence-Identifier: EUPL-1.2

package dbrx

import (
	"testing"

	"dappco.re/go/inference/model"
)

var sinkArch model.Arch

func BenchmarkConfig_Arch(b *testing.B) {
	cfg := Config{DModel: 6144, Heads: 48, Layers: 40, VocabSize: 100352, Attention: AttentionConfig{KVHeads: 8, RopeTheta: 500000}, FFN: FFNConfig{HiddenSize: 10752, Experts: 16, TopK: 4}}
	b.ReportAllocs()
	for b.Loop() {
		sinkArch, _ = cfg.Arch()
	}
}
