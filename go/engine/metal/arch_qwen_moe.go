// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// arch_qwen_moe.go decodes a qwen3_5_moe MoE FFN layer on the HOST — the SiLU switch-MLP routed experts +
// the always-on shared expert. The gemma DEVICE MoE (encMoEBlockQuantDevice) can't serve it: gemma's block
// assumes five sandwich norms, GELU experts and a router.scale, none of which qwen has. Correctness-first,
// ported from model/composed MoEMLP.forward: PreFFNorm(=post_attention_layernorm) → softmax top-k router →
// Σ_k w_k·SwiGLU_silu(expert_k) + σ(x·SharedSigmoid)·SwiGLU_silu(shared) → residual. A qwen MoE layer is
// marked by a bound shared expert, so gemma's device MoE is untouched. Device fusion is a later slice.

// sliceExpertRows returns a VIEW of expert e's rows in a batched [totalRows, K] quant weight. The per-row
// packed/scales/biases byte strides are derived dtype-agnostically as len(bytes)/totalRows, so it needs no
// knowledge of the scale/bias element type — no copy, the views alias the batched weight's bytes.
func sliceExpertRows(w QuantWeight, e, totalRows, rowsPerExpert int) QuantWeight {
	rp := len(w.Packed) / totalRows
	rs := len(w.Scales) / totalRows
	off := e * rowsPerExpert
	out := QuantWeight{
		GroupSize: w.GroupSize, Bits: w.Bits,
		Packed: w.Packed[off*rp : (off+rowsPerExpert)*rp],
		Scales: w.Scales[off*rs : (off+rowsPerExpert)*rs],
	}
	if len(w.Biases) > 0 {
		rb := len(w.Biases) / totalRows
		out.Biases = w.Biases[off*rb : (off+rowsPerExpert)*rb]
	}
	return out
}

// swigluSiLUHost computes down @ (silu(gate@x) · (up@x)) for one [d] row — a single expert's SiLU SwiGLU
// through the host quant matmul seam. gate/up are [ff,d], down is [d,ff].
func swigluSiLUHost(x []float32, gate, up, down QuantWeight, d, ff int) ([]float32, error) {
	g, err := projQuantAttn(gate, x, d, ff, nil)
	if err != nil {
		return nil, err
	}
	u, err := projQuantAttn(up, x, d, ff, nil)
	if err != nil {
		return nil, err
	}
	h := make([]float32, ff)
	for f := 0; f < ff; f++ {
		s := float64(g[f])
		h[f] = float32((s / (1.0 + math.Exp(-s))) * float64(u[f]))
	}
	return projQuantAttn(down, h, ff, d, nil)
}

// sharedGateSigmoid resolves the shared expert's σ scalar gate: σ(x · SharedSigmoid). Handles the gate as
// quant (projQuantAttn) or bf16 (a host dot over the widened [d] weight); an unbound gate ⇒ σ ≡ 1.
func sharedGateSigmoid(w QuantWeight, x []float32, d int) (float64, error) {
	if len(w.Packed) == 0 {
		return 1, nil
	}
	if w.Bits > 0 {
		gl, err := projQuantAttn(w, x, d, 1, nil)
		if err != nil {
			return 0, err
		}
		return 1.0 / (1.0 + math.Exp(-float64(gl[0]))), nil
	}
	gw := bf16VecToF32(w.Packed)
	var acc float64
	for i := 0; i < d && i < len(gw); i++ {
		acc += float64(x[i]) * float64(gw[i])
	}
	return 1.0 / (1.0 + math.Exp(-acc)), nil
}

// encQwenMoEHalf computes one qwen3_5_moe MoE FFN for a single decode token: reads the post-mixer residual
// from hBuf, writes the layer output (h + routed + shared) to out. Host path (correctness-first).
func (s *archDecodeState) encQwenMoEHalf(li int, moe *MoEQuantLayerWeights, out metal.MTLBuffer) error {
	if moe == nil {
		return core.NewError("native.encQwenMoEHalf: nil MoE weights")
	}
	D, nE, topK, ff := s.dModel, moe.NumExperts, moe.TopK, moe.ExpertDFF
	if nE <= 0 || topK <= 0 || ff <= 0 {
		return core.NewError("native.encQwenMoEHalf: bad MoE geometry")
	}
	h := bf16BufToF32(s.hBuf, 0, D)
	normed := rmsNormHostF32(h, bf16VecToF32(moe.PreFFNormW), s.eps)

	// Router: logits over every expert, softmax, top-k, renormalised over the selection (qwen default).
	logits, err := projQuantAttn(moe.Router, normed, D, nE, nil)
	if err != nil {
		return err
	}
	maxL := math.Inf(-1)
	for e := 0; e < nE; e++ {
		if float64(logits[e]) > maxL {
			maxL = float64(logits[e])
		}
	}
	probs := make([]float64, nE)
	for e := 0; e < nE; e++ {
		probs[e] = math.Exp(float64(logits[e]) - maxL)
	}
	sel := make([]int, 0, topK)
	used := make([]bool, nE)
	for k := 0; k < topK; k++ {
		best, bi := math.Inf(-1), -1
		for e := 0; e < nE; e++ {
			if !used[e] && probs[e] > best {
				best, bi = probs[e], e
			}
		}
		if bi < 0 {
			break
		}
		used[bi] = true
		sel = append(sel, bi)
	}
	denom := 0.0
	for _, e := range sel {
		denom += probs[e]
	}
	if denom == 0 {
		denom = 1
	}

	// Routed experts: Σ_k (probs[k]/denom) · SwiGLU_silu(expert_k). The batched switch_mlp is sliced per
	// selected expert — gate/up are [nE·ff, D], down is [nE·D, ff].
	acc := make([]float32, D)
	for _, e := range sel {
		w := probs[e] / denom
		ge := sliceExpertRows(moe.ExpGate, e, nE*ff, ff)
		ue := sliceExpertRows(moe.ExpUp, e, nE*ff, ff)
		de := sliceExpertRows(moe.ExpDown, e, nE*D, D)
		eo, err := swigluSiLUHost(normed, ge, ue, de, D, ff)
		if err != nil {
			return err
		}
		for d := 0; d < D; d++ {
			acc[d] += float32(w * float64(eo[d]))
		}
	}

	// Shared expert: σ(normed·SharedSigmoid) · SwiGLU_silu(shared). Shared FF == moe_intermediate (ff).
	if len(moe.SharedGate.Packed) > 0 {
		se, err := swigluSiLUHost(normed, moe.SharedGate, moe.SharedUp, moe.SharedDown, D, ff)
		if err != nil {
			return err
		}
		g, err := sharedGateSigmoid(moe.SharedSigmoid, normed, D)
		if err != nil {
			return err
		}
		gf := float32(g)
		for d := 0; d < D; d++ {
			acc[d] += gf * se[d]
		}
	}

	// Residual (h + routed + shared) → out, bf16.
	ob := unsafe.Slice((*byte)(out.Contents()), D*2)
	for d := 0; d < D; d++ {
		u := f32ToBF16(h[d] + acc[d])
		ob[2*d], ob[2*d+1] = byte(u), byte(u>>8)
	}
	return nil
}
