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
// metal's qwen3_moe combine. Host f32; a quant checkpoint's routed + shared experts stay PACKED
// (model.QuantWeight — the same mlx-affine representation the dense MLP/attention projections carry) and
// dispatch through matNTQuant, composed.go's quant matvec seam, instead of being dequantised at load — a
// grouped-int4 MoE checkpoint's experts are the dominant tensor class (routinely an ~8x blow-up widened),
// so packed-native here is what makes a bigger-than-RAM sparse MoE servable at all.

// MoEExpert is one SwiGLU expert (Gate/Up [FF,D], Down [D,FF]; FF = len(Gate)/D). GateQ/UpQ/DownQ are its
// packed forms in a quant checkpoint (model.QuantWeight — mlx affine codes + bf16 group scales/biases);
// nil ⇒ dense f32 in Gate/Up/Down. Exactly one representation is populated per expert (mirrors MLP's
// GateQ/UpQ/DownQ) — moeExpertFF and the forward loop's dispatch both branch on GateQ to pick it.
//
// GateUpQ is the OPTIONAL fused form of a packed expert's gate+up: the [2·FF, D] concat
// (model.ConcatQuantRows) synthesised at load when the arch opts into the fusion (Arch.FuseExpertGateUp,
// via fuseExpertGateUp). When set, GateQ and UpQ are nil and swigluExpertQuantInto dispatches ONE quant
// matvec at 2·FF and splits the halves — one launch per routed expert instead of two. DownQ is unchanged.
type MoEExpert struct {
	Gate, Up, Down    []float32
	GateQ, UpQ, DownQ *model.QuantWeight
	GateUpQ           *model.QuantWeight
}

// packed reports whether the expert's projections are PACKED (quantised): GateQ set (separate gate/up)
// or GateUpQ set (fused). A dense expert has both nil and runs swigluExpertInto. The forward loop's
// per-expert dispatch branches on this.
func (e *MoEExpert) packed() bool { return e.GateQ != nil || e.GateUpQ != nil }

// fuseExpertGateUp replaces a packed expert's separate GateQ/UpQ with the single [gate‖up] GateUpQ
// (model.ConcatQuantRows), dropping the originals — their mmap views are no longer read, so the forward
// makes one quant matvec per expert instead of two (the composed MoE fusion; the single-expert twin of
// engine/metal's fuseExpertGateUpQuant). No-op for a dense expert, one already fused, or if either half
// is absent. Materialises the fused concat on the heap, trading the gate/up mmap zero-copy for the
// fused-path launch — which is why it is opt-in per model (Arch.FuseExpertGateUp), not automatic.
func fuseExpertGateUp(e *MoEExpert) {
	if e.GateQ == nil || e.UpQ == nil || e.GateUpQ != nil {
		return
	}
	if gu := model.ConcatQuantRows(e.GateQ, e.UpQ); gu != nil {
		e.GateUpQ, e.GateQ, e.UpQ = gu, nil, nil
	}
}

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

	// GateBatchedQ/UpBatchedQ/DownBatchedQ are the WHOLE switch_mlp.{gate,up,down}_proj tensors kept as
	// one packed [numExperts, …] quant weight each (every routed expert concatenated) — the batched form
	// the engine's single-dispatch routed-MoE kernel (MoEExpertsDevice) consumes, collapsing the topK×3
	// per-projection GPU submits per layer into ONE device call. MoEBits/MoEGroupSize are that pack's
	// affine geometry, shared by gate/up/down. Populated by the loader only for the mlx batched
	// (switch_mlp) layout; a per-expert checkpoint leaves them nil. Empty (GateBatchedQ nil) OR no bound
	// MoEExpertsDevice ⇒ the device batched path is unavailable and forward runs the per-expert host loop
	// over Experts (which always carries the same experts, sliced out — the fallback).
	GateBatchedQ, UpBatchedQ, DownBatchedQ *model.QuantWeight
	MoEBits, MoEGroupSize                  int
}

// ownBatchedQuant deep-copies the batched routed-expert quant tensors (GateBatchedQ/UpBatchedQ/DownBatchedQ)
// to owned heap buffers, so they outlive the input checkpoint tensors — the owned-copy contract of the
// non-zero-copy loader (LoadComposed unmaps the checkpoint right after the build, which would leave a
// zero-copy VIEW dangling). The zero-copy loader (LoadComposedDir) keeps the views and retains the mapping
// instead. No-op when the batched form is absent (a per-expert or dense checkpoint).
func (m *MoEMLP) ownBatchedQuant() {
	own := func(qw *model.QuantWeight) {
		if qw == nil {
			return
		}
		qw.Packed = append([]byte(nil), qw.Packed...)
		qw.Scales = append([]byte(nil), qw.Scales...)
		qw.Biases = append([]byte(nil), qw.Biases...)
	}
	own(m.GateBatchedQ)
	own(m.UpBatchedQ)
	own(m.DownBatchedQ)
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

// moeExpertFF returns expert e's hidden width FF: GateUpQ.OutDim/2 for a fused packed expert (gate and up
// share the concat's 2·FF rows), GateQ.OutDim for an unfused packed one (Gate is nil, so len(Gate)/D reads
// as 0), else len(Gate)/D for a dense one — the same derivation swigluExpertInto's FF local uses inline.
func moeExpertFF(e *MoEExpert, D int) int {
	if e.GateUpQ != nil {
		return e.GateUpQ.OutDim / 2
	}
	if e.GateQ != nil {
		return e.GateQ.OutDim
	}
	return len(e.Gate) / D
}

// swigluExpertQuantInto is swigluExpertInto's packed twin: e's gate/up/down stay PACKED
// (model.QuantWeight), dispatched through matNTQuant — the SAME quant matvec seam every other packed
// projection in this package uses (the device hook when the native backend is bound, else
// matNTQuantHost's per-ROW host dequant fallback), so a routed expert's packed weight is never widened to
// a whole f32 copy (matNTQuantHost dequantises one output row — [K] — at a time). Writes the [D] result
// into out (fully overwritten each call); the gate/up/hidden scratch is allocated per call, the same cost
// the packed dense MLP path (MLP.forward's GateQ branch) already pays.
//
// matNTQuant rounds to f32 at each of the three matvec boundaries, where swigluExpertInto's hand-inlined
// dense loop stays f64 from xt to out throughout — so a packed expert's result differs from the SAME
// weights dequantised and run through swigluExpertInto by a few ULPs (a rounding-TIER difference, not a
// bug: TestSwigluExpertQuantInto_MatchesDequantised pins the tolerance).
//
// A FUSED expert (GateUpQ set, GateQ/UpQ nil — fuseExpertGateUp) makes ONE matvec at 2·FF and slices the
// [gate‖up] halves out of the single result: because matNTQuant dequantises each output row independently,
// the fused halves are BYTE-IDENTICAL to the two separate gate/up matvecs (TestSwigluExpertQuantInto_-
// FusedMatchesUnfused pins the byte equality), so the fusion is a pure launch-count saving.
func swigluExpertQuantInto(xt []float32, e MoEExpert, D int, out []float32) {
	var g, u []float32
	if e.GateUpQ != nil {
		FF := e.GateUpQ.OutDim / 2
		gu := matNTQuant(nil, xt, e.GateUpQ, 1, D, 2*FF)
		g, u = gu[:FF], gu[FF:2*FF]
	} else {
		FF := e.GateQ.OutDim
		g = matNTQuant(nil, xt, e.GateQ, 1, D, FF)
		u = matNTQuant(nil, xt, e.UpQ, 1, D, FF)
	}
	FF := len(g)
	h := make([]float32, FF)
	for f := range FF {
		h[f] = float32(silu(float64(g[f])) * float64(u[f]))
	}
	matNTQuant(out, h, e.DownQ, 1, FF, D)
}

func (m *MoEMLP) forward(x []float32, L, D int) []float32 {
	nE := len(m.Experts)
	out := make([]float32, L*D)
	// Per-token routing scratch, hoisted out of the token loop so a multi-token decode
	// allocates it once, not per token: the top-k index buffer, the expert hidden buffer
	// (sized to the widest DENSE expert — a packed expert's own scratch is matNTQuant's, see
	// swigluExpertQuantInto), and the single expert-output buffer. Each is fully overwritten
	// per use, so reuse is byte-identical to a fresh allocation per call.
	idx := make([]int, nE)
	maxFF := 0
	for i := range m.Experts {
		if ff := moeExpertFF(&m.Experts[i], D); ff > maxFF {
			maxFF = ff
		}
	}
	if m.Shared != nil {
		if ff := moeExpertFF(m.Shared, D); ff > maxFF {
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
		// Batched device path: the whole routed MoE — select the topK experts, compute each expert's
		// SwiGLU, and combine by the router weights — in ONE device dispatch, collapsing the topK×3
		// per-projection submits this layer would otherwise fire. Engaged only when the loader populated
		// the batched packed-expert tensors AND a backend bound MoEExpertsDevice; the kernel applies the
		// combine weights itself, so its [D] result adds straight into ot. A device error falls back to
		// the per-expert host loop for THIS token (never crashes). The shared expert always stays on the
		// host path below — it is not routed through the kernel.
		routed := false
		if m.GateBatchedQ != nil && MoEExpertsDevice != nil {
			weights := make([]float64, len(sel))
			for k := range sel {
				weights[k] = probs[sel[k]] / denom
			}
			if combined, err := MoEExpertsDevice(xt, sel, weights, m.GateBatchedQ, m.UpBatchedQ, m.DownBatchedQ, len(m.Experts), len(sel), D, moeExpertFF(&m.Experts[0], D)); err == nil {
				for d := range D {
					ot[d] += combined[d]
				}
				routed = true
			}
		}
		if !routed {
			for _, e := range sel {
				w := probs[e] / denom
				if m.Experts[e].packed() {
					swigluExpertQuantInto(xt, m.Experts[e], D, eo)
				} else {
					swigluExpertInto(xt, m.Experts[e], D, hbuf, eo)
				}
				for d := range D {
					ot[d] += float32(w * float64(eo[d]))
				}
			}
		}
		if m.Shared != nil {
			if m.Shared.packed() {
				swigluExpertQuantInto(xt, *m.Shared, D, eo)
			} else {
				swigluExpertInto(xt, *m.Shared, D, hbuf, eo)
			}
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
