// SPDX-Licence-Identifier: EUPL-1.2

package composed

import "math"

// moe.go is the Qwen 3.6 (qwen3_6_moe) Mixture-of-Experts feed-forward — the MoE variant of the layer's
// FFN slot. A router scores the experts; the top-k are selected, their softmax weights renormalised over
// the selection (norm_topk_prob), and their SwiGLU outputs summed by weight; an always-on shared expert
// (a plain SwiGLU) is added directly. Mirrors metal's qwen3_moe combine. Host f32.

// MoEExpert is one SwiGLU expert (Gate/Up [FF,D], Down [D,FF]; FF = len(Gate)/D).
type MoEExpert struct{ Gate, Up, Down []float32 }

// MoEMLP routes a token to TopK of its experts plus the shared expert:
//
//	out = Σ_{k∈topk} (softmax(router·x)_k / Σ_topk) · SwiGLU_k(x)  +  SwiGLU_shared(x)
type MoEMLP struct {
	Router  []float32 // [NumExperts, D]
	Experts []MoEExpert
	Shared  *MoEExpert // nil ⇒ no shared expert
	TopK    int
}

// swigluExpert runs one SwiGLU expert over a single token xt [D] → [D].
func swigluExpert(xt []float32, e MoEExpert, D int) []float32 {
	FF := len(e.Gate) / D
	h := make([]float64, FF)
	for f := range FF {
		gr := e.Gate[f*D : f*D+D]
		ur := e.Up[f*D : f*D+D]
		var g, u float64
		for d := range D {
			g += float64(xt[d]) * float64(gr[d])
			u += float64(xt[d]) * float64(ur[d])
		}
		h[f] = silu(g) * u
	}
	out := make([]float32, D)
	for d := range D {
		dr := e.Down[d*FF : d*FF+FF]
		var acc float64
		for f := range FF {
			acc += h[f] * float64(dr[f])
		}
		out[d] = float32(acc)
	}
	return out
}

func (m *MoEMLP) forward(x []float32, L, D int) []float32 {
	nE := len(m.Experts)
	out := make([]float32, L*D)
	probs := make([]float64, nE)
	for t := range L {
		xt := x[t*D : (t+1)*D]
		// router logits → softmax numerators (the denominator cancels in the top-k renormalisation).
		maxL := math.Inf(-1)
		for e := range nE {
			rr := m.Router[e*D : e*D+D]
			var acc float64
			for d := range D {
				acc += float64(xt[d]) * float64(rr[d])
			}
			probs[e] = acc
			if acc > maxL {
				maxL = acc
			}
		}
		for e := range nE {
			probs[e] = math.Exp(probs[e] - maxL)
		}
		idx := topKIndices(probs, m.TopK)
		var sumW float64
		for _, e := range idx {
			sumW += probs[e]
		}
		ot := out[t*D : (t+1)*D]
		for _, e := range idx {
			w := probs[e] / sumW // renormalise the selected softmax probs to sum 1
			eo := swigluExpert(xt, m.Experts[e], D)
			for d := range D {
				ot[d] += float32(w * float64(eo[d]))
			}
		}
		if m.Shared != nil {
			so := swigluExpert(xt, *m.Shared, D)
			for d := range D {
				ot[d] += so[d]
			}
		}
	}
	return out
}

// topKIndices returns the indices of the k largest values in v (partial selection — k is small).
func topKIndices(v []float64, k int) []int {
	if k > len(v) {
		k = len(v)
	}
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < len(idx); j++ {
			if v[idx[j]] > v[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	return idx[:k]
}
