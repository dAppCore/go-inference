// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import "testing"

// blockBenchCfg + blockBenchD are a moderate block geometry (D model dim, dInner = H·P).
var blockBenchCfg = BlockConfig{NumHeads: 8, HeadDim: 64, StateDim: 64, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}

const blockBenchD = 512

// BenchmarkBlockForwardF32Prefill measures the full Mamba-2 block over a prefill chunk (in-proj + conv +
// scan + gated-norm + out-proj). The host matmul projections dominate — this is the perf surface a device
// path optimises.
func BenchmarkBlockForwardF32Prefill(b *testing.B) {
	const L = 64
	w := mkBlockWeights(blockBenchCfg, blockBenchD)
	x := syn(L*blockBenchD, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := BlockForwardF32(x, w, blockBenchCfg, nil, nil, L, blockBenchD); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkBlockForwardF32Decode measures one decode step (L=1) with carried conv + SSM state.
func BenchmarkBlockForwardF32Decode(b *testing.B) {
	w := mkBlockWeights(blockBenchCfg, blockBenchD)
	x := syn(blockBenchD, 1)
	_, nc, ns, err := BlockForwardF32(x, w, blockBenchCfg, nil, nil, 1, blockBenchD)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, nc, ns, err = BlockForwardF32(x, w, blockBenchCfg, nc, ns, 1, blockBenchD); err != nil {
			b.Fatal(err)
		}
	}
}
