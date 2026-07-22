// SPDX-Licence-Identifier: EUPL-1.2

package qwenmoe

import (
	"testing"

	"dappco.re/go/inference/model"
)

var sinkArch model.Arch

func BenchmarkConfig_Arch(b *testing.B) {
	cfg := Config{HiddenSize: 2048, NumHiddenLayers: 48, NumAttentionHeads: 32, NumKeyValueHeads: 4, HeadDim: 128, VocabSize: 151936, NumExperts: 128, NumExpertsPerTok: 8, MoEIntermediateSize: 768, NormTopKProb: true}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		sinkArch, _ = cfg.Arch()
	}
}
