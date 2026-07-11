// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import "testing"

// The GatedDeltaForwardF32 bench baselines the Qwen 3.6 gated-delta ("linear_attention")
// block per decode token (AX-11): in-proj QKV → causal conv → SiLU → GQA split → l2-norm →
// α/β gates → the delta-rule recurrence → gated RMSNorm → out-proj. It is one of the two
// mixers of the Qwen 3.6 hybrid (the fleet's peer to gemma4), so its per-token allocation —
// the many projection results + the q/k/v/α/β/gated intermediates — is a real decode cost.
// Dims: D=1024, KeyHeads=4 / ValueHeads=8 / HeadDim=128, conv kernel 4, decode L=1. Pure Go
// (projMatMul falls through to the host matNT default).

func benchQwenF32(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*131)%4096-2048) * 0.001
	}
	return s
}

func benchGatedDeltaWeights(D, KH, VH, HD, K int) *GatedDeltaWeights {
	qDim := KH * HD
	vDim := VH * HD
	convDim := 2*qDim + vDim
	return &GatedDeltaWeights{
		InProjQKV: benchQwenF32(convDim * D), ConvWeight: benchQwenF32(convDim * K), ConvBias: benchQwenF32(convDim),
		InProjA: benchQwenF32(VH * D), ALog: benchQwenF32(VH), DtBias: benchQwenF32(VH),
		InProjB: benchQwenF32(VH * D), InProjZ: benchQwenF32(vDim * D), Norm: benchQwenF32(HD), OutProj: benchQwenF32(D * vDim),
	}
}

// BenchmarkGatedDeltaForwardF32_Decode — one token through the gated-delta block from a fresh
// state: the full projection + conv + recurrence + gated-norm chain. The intermediate buffers
// are the per-token allocation the linear-attention layer pays.
func BenchmarkGatedDeltaForwardF32_Decode(b *testing.B) {
	const D, KH, VH, HD, K = 1024, 4, 8, 128, 4
	w := benchGatedDeltaWeights(D, KH, VH, HD, K)
	cfg := GatedDeltaConfig{KeyHeads: KH, ValueHeads: VH, HeadDim: HD, ConvKernel: K, Eps: 1e-6}
	x := benchQwenF32(1 * D)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := GatedDeltaForwardF32(x, w, cfg, nil, nil, 1, D); err != nil {
			b.Fatal(err)
		}
	}
}
