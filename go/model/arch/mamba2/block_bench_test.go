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

// BenchmarkBlockForwardF32Decode measures one decode step (L=1) with carried conv + SSM state on the
// steady-state scratch path: a caller-owned BlockScratch (sized by the warmup call) is reused every token
// so the in-proj and out-proj outputs write into resident buffers — the real session's per-token path.
func BenchmarkBlockForwardF32Decode(b *testing.B) {
	w := mkBlockWeights(blockBenchCfg, blockBenchD)
	x := syn(blockBenchD, 1)
	sc := &BlockScratch{}
	_, nc, ns, err := BlockForwardScratchF32(x, w, blockBenchCfg, nil, nil, 1, blockBenchD, sc)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, nc, ns, err = BlockForwardScratchF32(x, w, blockBenchCfg, nc, ns, 1, blockBenchD, sc); err != nil {
			b.Fatal(err)
		}
	}
}
