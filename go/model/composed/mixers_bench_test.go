// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model"
)

// The mixers bench baselines the gated-delta Mixer adapter (AX-11): gatedDeltaMixer.Forward
// wraps attn.GatedDeltaForwardF32 for the composed session, adding the per-token state
// boxing — the prior.(gatedDeltaState) type assertion in and the gatedDeltaState wrap out
// that the session threads opaquely. This measures the Qwen 3.6 linear-attention layer's
// per-token cost AS THE SESSION DRIVES IT (interface dispatch + state box included). Dims: a
// KeyHeads=4 / ValueHeads=8 / HeadDim=128 gated-delta layer over D=1024, decode step L=1.

func benchGatedDeltaMixer() Mixer {
	const (
		D  = 1024
		KH = 4
		VH = 8
		HD = 128
		K  = 4
	)
	qDim := KH * HD
	vDim := VH * HD
	convDim := 2*qDim + vDim
	w := &model.GatedDeltaWeights{
		InProjQKV:  benchF32(convDim * D),
		ConvWeight: benchF32(convDim * K),
		ConvBias:   benchF32(convDim),
		InProjA:    benchF32(VH * D),
		ALog:       benchF32(VH),
		DtBias:     benchF32(VH),
		InProjB:    benchF32(VH * D),
		InProjZ:    benchF32(vDim * D),
		Norm:       benchF32(HD),
		OutProj:    benchF32(D * vDim),
	}
	cfg := model.GatedDeltaConfig{KeyHeads: KH, ValueHeads: VH, HeadDim: HD, ConvKernel: K, Eps: 1e-6}
	return NewGatedDeltaMixer(w, cfg)
}

// BenchmarkGatedDeltaMixer_Forward — one decode token through the gated-delta layer via the
// Mixer interface (fresh state): the projections + conv + delta recurrence, plus the
// gatedDeltaState box the session threads. The per-token allocation of the linear-attention path.
func BenchmarkGatedDeltaMixer_Forward(b *testing.B) {
	m := benchGatedDeltaMixer()
	h := benchF32(1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := m.Forward(h, 1, 1024, nil); err != nil {
			b.Fatal(err)
		}
	}
}
