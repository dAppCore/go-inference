// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import "testing"

// The BlockForwardF32 bench baselines the whole RWKV-7 time-mix block per decode token
// (AX-11): the six input projections (R/W/K/A/B/V), the log-decay transform, the WKV7
// recurrence, and the output projection. The projections dominate (dense GEMM through the
// host matNT default); this pins the block's per-token allocation — each projMatMul result +
// the wDecay buffer + the recurrence output + the advanced state. Dims: D=1024, 16 heads,
// K=V=64, decode step L=1. Pure Go (no device GEMM wired).

func benchBlockWeights(D, H, K, V int) *BlockWeights {
	hk, hv := H*K, H*V
	return &BlockWeights{
		RProj: benchRwkvF32(hk * D), WProj: benchRwkvF32(hk * D), KProj: benchRwkvF32(hk * D),
		AProj: benchRwkvF32(hk * D), BProj: benchRwkvF32(hk * D),
		VProj: benchRwkvF32(hv * D), OutProj: benchRwkvF32(D * hv),
	}
}

// BenchmarkBlockForwardF32_Decode — one token through the full time-mix block on the steady-state
// scratch path: a caller-owned BlockScratch (sized by the warmup call) is reused every token so the
// seven projection GEMM outputs (r/w/k/v/a/b + out) write into resident buffers. This pins the residual
// per-token allocation once the projection outputs are eliminated (the WKV7 output + advanced state).
func BenchmarkBlockForwardF32_Decode(b *testing.B) {
	const D, H, K, V = 1024, benchRwkvHeads, benchRwkvK, benchRwkvV
	w := benchBlockWeights(D, H, K, V)
	cfg := BlockConfig{NumHeads: H, KeyDim: K, ValueDim: V}
	x := benchRwkvF32(1 * D)
	sc := &BlockScratch{}
	if _, _, err := BlockForwardScratchF32(x, w, cfg, nil, 1, D, sc); err != nil { // size the scratch
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := BlockForwardScratchF32(x, w, cfg, nil, 1, D, sc); err != nil {
			b.Fatal(err)
		}
	}
}
