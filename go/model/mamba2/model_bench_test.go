// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import "testing"

// BenchmarkMambaDecodeStep measures one token of recurrent decode through the whole model (all layers,
// embed → blocks → no head), the O(1)/token streaming-generation cost.
func BenchmarkMambaDecodeStep(b *testing.B) {
	cfg := BlockConfig{NumHeads: 8, HeadDim: 64, StateDim: 64, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const D, vocab, nLayers = 512, 1024, 8
	m := mkModel(cfg, D, vocab, nLayers)
	s := NewSession(m)
	if _, err := s.Forward([]int32{1, 2, 3, 4}); err != nil { // warm the recurrent state
		b.Fatal(err)
	}
	tok := []int32{5}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Forward(tok); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkMambaGenerate measures end-to-end greedy generation (prefill + decode + head per token).
func BenchmarkMambaGenerate(b *testing.B) {
	cfg := BlockConfig{NumHeads: 8, HeadDim: 64, StateDim: 64, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 512, 1024, 4)
	prompt := []int32{1, 2, 3, 4}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := NewSession(m).Generate(prompt, 8, -1); err != nil {
			b.Fatal(err)
		}
	}
}
