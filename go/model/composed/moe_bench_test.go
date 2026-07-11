// SPDX-Licence-Identifier: EUPL-1.2

package composed

import "testing"

// The MoE benches baseline the Qwen 3.6 mixture-of-experts FFN (AX-11), all per decode
// token: MoEMLP.forward routes a token (softmax over the router logits, top-k select +
// renormalise) and sums the selected experts' SwiGLU plus the always-on shared expert;
// swigluExpert is one expert's SwiGLU; topKIndices is the partial selection. The per-token
// allocation story lives here — swigluExpert's h+out and topKIndices' index buffer are
// allocated inside the token loop, so a decode over many tokens multiplies them. Dims: a
// small MoE layer (D=1024, 8 experts of FF=1408, top-2 + shared).

const (
	benchMoED       = 1024
	benchMoEExperts = 8
	benchMoEFF      = 1408
	benchMoETopK    = 2
)

func benchMoEExpert() MoEExpert {
	return MoEExpert{Gate: benchF32(benchMoEFF * benchMoED), Up: benchF32(benchMoEFF * benchMoED), Down: benchF32(benchMoED * benchMoEFF)}
}

func benchMoEMLP() *MoEMLP {
	experts := make([]MoEExpert, benchMoEExperts)
	for e := range experts {
		experts[e] = benchMoEExpert()
	}
	shared := benchMoEExpert()
	return &MoEMLP{Router: benchF32(benchMoEExperts * benchMoED), Experts: experts, Shared: &shared, TopK: benchMoETopK}
}

// BenchmarkMoEMLP_Forward — one token routed through the full MoE: router scoring, the top-k
// select + renormalise, the selected + shared experts' SwiGLU. The per-token allocation
// (out, probs, and the per-expert swigluExpert scratch + the topKIndices buffer) is the cost.
func BenchmarkMoEMLP_Forward(b *testing.B) {
	m := benchMoEMLP()
	x := benchF32(benchMoED)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.forward(x, 1, benchMoED)
	}
}

// BenchmarkSwigluExpert — one expert's SwiGLU over a single token: the h[FF] f64 scratch +
// the [D] out. Called TopK+1 times per token, so its allocation multiplies with the routing.
func BenchmarkSwigluExpert(b *testing.B) {
	e := benchMoEExpert()
	xt := benchF32(benchMoED)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = swigluExpert(xt, e, benchMoED)
	}
}

// BenchmarkTopKIndices — the partial top-k selection over the expert scores: allocates an
// index buffer sized to the expert count, then a k-pass selection. Run once per token.
func BenchmarkTopKIndices(b *testing.B) {
	v := make([]float64, benchMoEExperts)
	for i := range v {
		v[i] = float64((i*7)%13) * 0.1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = topKIndices(v, benchMoETopK)
	}
}
