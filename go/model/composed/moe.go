// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"

	"dappco.re/go/inference/model"
)

// moe.go is the Qwen 3.6 (qwen3_5_moe) Mixture-of-Experts feed-forward — the MoE variant of the layer's
// FFN slot. A router scores the experts; the top-k are selected and combined by their softmax weights —
// renormalised over the selection when NormTopKProb (norm_topk_prob, the reference default), else the raw
// full-softmax weights. An always-on shared expert (a plain SwiGLU) is added, sigmoid-gated by
// shared_expert_gate·x when that weight is present (the reference gate) and ungated otherwise. Mirrors
// metal's qwen3_moe combine. Host f32.

// MoEExpert is one SwiGLU expert (Gate/Up [FF,D], Down [D,FF]; FF = len(Gate)/D).
type MoEExpert struct{ Gate, Up, Down []float32 }

// MoEMLP routes a token to TopK of its experts plus the shared expert:
//
//	w_k = softmax(router·x)_k / (NormTopKProb ? Σ_topk : Σ_all)
//	out = Σ_{k∈topk} w_k · SwiGLU_k(x)  +  σ(SharedGate·x) · SwiGLU_shared(x)
//
// SharedGate nil ⇒ the shared expert is added ungated (σ ≡ 1).
type MoEMLP struct {
	Router       []float32 // [NumExperts, D]
	Experts      []MoEExpert
	Shared       *MoEExpert // nil ⇒ no shared expert
	SharedGate   []float32  // [D] shared_expert_gate.weight; nil ⇒ shared added ungated
	TopK         int
	NormTopKProb bool // renormalise the top-k router weights over the selection (reference default true)
	Gating       model.MoEGating
}

// swigluExpertInto runs one SwiGLU expert over a single token xt [D], writing the [D] result
// into out (fully overwritten each call) with h [FF] as hidden scratch. Both buffers are
// caller-provided and reused across experts + tokens, so per-token MoE routing allocates no
// per-expert scratch. h must be at least FF (= len(e.Gate)/D) long; out at least D.
func swigluExpertInto(xt []float32, e MoEExpert, D int, h []float64, out []float32) {
	FF := len(e.Gate) / D
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
	for d := range D {
		dr := e.Down[d*FF : d*FF+FF]
		var acc float64
		for f := range FF {
			acc += h[f] * float64(dr[f])
		}
		out[d] = float32(acc)
	}
}

// swigluExpert runs one SwiGLU expert over a single token xt [D] → [D], allocating the h+out
// buffers. swigluExpertInto is the buffer-reusing core MoEMLP.forward drives per token.
func swigluExpert(xt []float32, e MoEExpert, D int) []float32 {
	out := make([]float32, D)
	swigluExpertInto(xt, e, D, make([]float64, len(e.Gate)/D), out)
	return out
}

func (m *MoEMLP) forward(x []float32, L, D int) []float32 {
	nE := len(m.Experts)
	out := make([]float32, L*D)
	// Per-token routing scratch, hoisted out of the token loop so a multi-token decode
	// allocates it once, not per token: the top-k index buffer, the expert hidden buffer
	// (sized to the widest expert), and the single expert-output buffer. Each is fully
	// overwritten per use, so reuse is byte-identical to a fresh allocation per call.
	idx := make([]int, nE)
	maxFF := 0
	for i := range m.Experts {
		if ff := len(m.Experts[i].Gate) / D; ff > maxFF {
			maxFF = ff
		}
	}
	if m.Shared != nil {
		if ff := len(m.Shared.Gate) / D; ff > maxFF {
			maxFF = ff
		}
	}
	// probs (router numerators, [nE]) and hbuf (expert hidden, [maxFF]) are both f64 scratch,
	// each fully overwritten before use — one backing slab carved into two capped windows saves
	// one alloc per forward call (per MoE layer per token) with no byte change.
	f64buf := make([]float64, nE+maxFF)
	probs := f64buf[0:nE:nE]
	hbuf := f64buf[nE : nE+maxFF : nE+maxFF]
	eo := make([]float32, D)
	for t := range L {
		xt := x[t*D : (t+1)*D]
		sel, denom := m.routeInto(xt, D, probs, idx)
		ot := out[t*D : (t+1)*D]
		for _, e := range sel {
			w := probs[e] / denom
			swigluExpertInto(xt, m.Experts[e], D, hbuf, eo)
			for d := range D {
				ot[d] += float32(w * float64(eo[d]))
			}
		}
		if m.Shared != nil {
			swigluExpertInto(xt, *m.Shared, D, hbuf, eo)
			g := float32(1) // ungated (SharedGate nil) ⇒ σ ≡ 1, byte-identical to a direct add
			if m.SharedGate != nil {
				var acc float64
				for d := range D {
					acc += float64(xt[d]) * float64(m.SharedGate[d])
				}
				g = float32(1.0 / (1.0 + math.Exp(-acc)))
			}
			for d := range D {
				ot[d] += g * eo[d]
			}
		}
	}
	return out
}

// routeInto is the allocation-free production router shared by forward and
// distribution receipts. It returns the selected experts and their softmax
// denominator, using the model-declared top-k normalisation policy.
func (m *MoEMLP) routeInto(xt []float32, D int, probs []float64, idx []int) ([]int, float64) {
	nE := len(m.Experts)
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
	if m.Gating == model.MoEGatingSigmoid {
		for e := range nE {
			probs[e] = 1 / (1 + math.Exp(-probs[e]))
		}
	} else {
		for e := range nE {
			probs[e] = math.Exp(probs[e] - maxL)
		}
	}
	sel := topKInto(probs, m.TopK, idx)
	denom := float64(1)
	if m.NormTopKProb {
		denom = 0
		for _, e := range sel {
			denom += probs[e]
		}
	} else if m.Gating != model.MoEGatingSigmoid {
		denom = 0
		for e := range nE {
			denom += probs[e]
		}
	}
	return sel, denom
}

// topKInto selects the indices of the k largest values in v into the caller-provided idx
// buffer (len(idx) must be >= len(v)), returning idx[:k]. The allocation-free core of
// topKIndices — MoEMLP.forward reuses one idx buffer across tokens.
func topKInto(v []float64, k int, idx []int) []int {
	if k > len(v) {
		k = len(v)
	}
	for i := range v {
		idx[i] = i
	}
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < len(v); j++ {
			if v[idx[j]] > v[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	return idx[:k]
}

// topKIndices returns the indices of the k largest values in v (partial selection — k is
// small), allocating the index buffer. topKInto is the buffer-reusing core.
func topKIndices(v []float64, k int) []int {
	return topKInto(v, k, make([]int, len(v)))
}
