// SPDX-Licence-Identifier: EUPL-1.2

package composed

import "testing"

// The attention benches baseline the full_attention mixer's per-token CPU work (AX-11):
// Forward is a decode step (L=1) attending over a grown KV cache — the q/k/v projections,
// per-head QK-norm + partial rotary, the causal softmax over the cached keys, and the
// cache grow (the prior k/v copied into a new buffer each step). rmsNormHead and
// applyRotaryHalf are the per-head in-place primitives. Dims: D=1024, 8 heads of 128 (GQA
// 2 KV heads), 32 rotary dims, a 128-token prior cache — a realistic mid-context decode step.

const (
	benchAttnHeads   = 8
	benchAttnKVHeads = 2
	benchAttnHeadDim = 128
	benchAttnRotary  = 32
	benchAttnPrior   = 128 // cached tokens the decode step attends over
)

func benchAttnMixer() Mixer {
	const D = benchAttnHeads * benchAttnHeadDim // 1024
	w := &AttnWeights{
		QProj: benchF32(benchAttnHeads * benchAttnHeadDim * D),
		KProj: benchF32(benchAttnKVHeads * benchAttnHeadDim * D),
		VProj: benchF32(benchAttnKVHeads * benchAttnHeadDim * D),
		OProj: benchF32(D * benchAttnHeads * benchAttnHeadDim),
		QNorm: benchF32(benchAttnHeadDim),
		KNorm: benchF32(benchAttnHeadDim),
	}
	cfg := AttnConfig{Heads: benchAttnHeads, KVHeads: benchAttnKVHeads, HeadDim: benchAttnHeadDim, RotaryDim: benchAttnRotary, RopeTheta: 1e6, NormEps: 1e-6}
	return NewAttnMixer(w, cfg)
}

// benchAttnPriorState builds a prior KV cache of benchAttnPrior tokens (rotary already
// applied to the cached keys, as the mixer stores them).
func benchAttnPriorState() attnState {
	kv := benchAttnKVHeads * benchAttnHeadDim
	return attnState{k: benchF32(benchAttnPrior * kv), v: benchF32(benchAttnPrior * kv), n: benchAttnPrior}
}

// BenchmarkAttnMixer_Forward — a single-token decode step over a 128-token cache on the steady-state
// scratch path: a caller-owned attnScratch (seeded on the prior, sized by the warmup call) is reused
// every token so the q/k/v/o projections write into resident buffers. What remains pins the residual
// per-token cost: the O(N) cache-grow copy (fresh [N·KVH·HD] k and v each step), the scores buffer, and
// the boxed state return.
func BenchmarkAttnMixer_Forward(b *testing.B) {
	m := benchAttnMixer()
	h := benchF32(benchAttnHeads * benchAttnHeadDim)
	prior := benchAttnPriorState()
	prior.sc = &attnScratch{}
	if _, _, err := m.Forward(h, 1, benchAttnHeads*benchAttnHeadDim, prior); err != nil { // size the scratch
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := m.Forward(h, 1, benchAttnHeads*benchAttnHeadDim, prior); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkRMSNormHead — the per-head in-place RMSNorm (run per Q and K head every token):
// a scan + rescale over [HeadDim], zero allocation.
func BenchmarkRMSNormHead(b *testing.B) {
	x := benchF32(benchAttnHeadDim)
	w := benchF32(benchAttnHeadDim)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rmsNormHead(x, w, 1e-6)
	}
}

// BenchmarkApplyRotaryHalf — the per-head partial rotary at one position: cos/sin per
// rotated pair over rotaryDim/2 pairs, in place, zero allocation.
func BenchmarkApplyRotaryHalf(b *testing.B) {
	x := benchF32(benchAttnHeadDim)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRotaryHalf(x, 100, benchAttnRotary, 1e6)
	}
}
