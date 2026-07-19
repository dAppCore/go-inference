// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// arch_gptoss_moe.go decodes a gpt_oss MoE FFN layer for the factory session — the clamped-SwiGLU
// routed experts with their additive biases. Neither existing MoE lane can serve it: the gemma
// device MoE (encMoEBlockQuantDevice) assumes a local dense MLP + five sandwich norms + a router
// norm, none of which gpt_oss has; the qwen half (encQwenMoEHalf) runs plain-SiLU experts and is
// marked by a shared expert gpt_oss lacks. Shape mirrors encQwenMoEHalf: PreFFNorm
// (=post_attention_layernorm) → router logits + BIAS → softmax → top-k → renormalise (host) → ONE
// MoEExpertsQuantClampedSiLU dispatch (clamped-sigmoid SwiGLU + per-expert biases) → residual.
//
// Reference (fetched 2026-07-19, cited in moe_clamped_swiglu.go / moe_block.go docs):
//
//	mlx-lm gpt_oss.py — "self.router = nn.Linear(config.hidden_size, config.num_local_experts,
//	bias=True)"; "g = self.router(x); experts, indices = mlx_topk(g, k=...); expert_weights =
//	mx.softmax(experts, ...)" — bias INSIDE the router linear, so it lands BEFORE top-k; softmax
//	over the selected top-k == softmax-over-all + renormalise-the-selection (the identity
//	gptoss's router_test.go proves; MoEGatingSoftmax + NormaliseMoETopK declare exactly that).
//	SwitchGLU(..., bias=True) — the per-expert projection biases MoEExpertsQuantClampedSiLU adds.

// gptOssRouterTopK computes the gpt_oss router decision from raw logits: logits += bias (bf16
// [nE], nil = none), softmax over ALL experts, pick the top-k probabilities, renormalise the
// selection to sum 1. Returns the selected expert indices and their bf16 combine weights — the
// exact (idx, weights) contract MoEExpertsQuantClampedSiLU consumes. Factored from the layer half
// so the router maths (bias-before-top-k included) is unit-testable without a session or GPU.
func gptOssRouterTopK(logits []float32, bias []byte, topK int) ([]int32, []byte, error) {
	nE := len(logits)
	if nE == 0 || topK <= 0 || topK > nE {
		return nil, nil, core.NewError("native.gptOssRouterTopK: need 0 < topK <= len(logits)")
	}
	if len(bias) != 0 && len(bias) != nE*bf16Size {
		return nil, nil, core.NewError("native.gptOssRouterTopK: bias must be len(logits) bf16 values or nil")
	}
	biased := make([]float64, nE)
	if len(bias) > 0 {
		bf := bf16ToF32Slice(bias)
		for e := range logits {
			biased[e] = float64(logits[e]) + float64(bf[e])
		}
	} else {
		for e := range logits {
			biased[e] = float64(logits[e])
		}
	}
	maxL := math.Inf(-1)
	for _, l := range biased {
		if l > maxL {
			maxL = l
		}
	}
	probs := make([]float64, nE)
	for e, l := range biased {
		probs[e] = math.Exp(l - maxL)
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
	idx := make([]int32, len(sel))
	wts := make([]byte, len(sel)*bf16Size)
	for i, e := range sel {
		idx[i] = int32(e)
		r := f32ToBF16(float32(probs[e] / denom))
		wts[2*i], wts[2*i+1] = byte(r), byte(r>>8)
	}
	return idx, wts, nil
}

// encGptOssMoEHalf computes one gpt_oss MoE FFN for a single decode token: reads the post-attention
// residual from hBuf, writes the layer output (h + routed) to out. Router host-orchestrated (one
// device round trip via projQuantAttn's host quant seam); the routed top-K is ONE
// MoEExpertsQuantClampedSiLU dispatch carrying the clamped activation + per-expert biases.
// Correctness-first host path, mirroring encQwenMoEHalf's shape (a device lane is the perf
// follow-up once the live gate proves the maths).
func (s *archDecodeState) encGptOssMoEHalf(li int, moe *MoEQuantLayerWeights, out metal.MTLBuffer) error {
	if moe == nil {
		return core.NewError("native.encGptOssMoEHalf: nil MoE weights")
	}
	D, nE, topK, ff := s.dModel, moe.NumExperts, moe.TopK, moe.ExpertDFF
	if nE <= 0 || topK <= 0 || ff <= 0 {
		return core.NewError("native.encGptOssMoEHalf: bad MoE geometry")
	}
	if moe.SwigluLimit <= 0 {
		return core.NewError("native.encGptOssMoEHalf: SwigluLimit must be > 0 (config swiglu_limit)")
	}
	h := bf16BufToF32(s.hBuf, 0, D)
	normed := rmsNormHostF32(h, bf16VecToF32(moe.PreFFNormW), s.eps)

	// Router: logits over every expert (+ the additive router bias), softmax, top-k, renormalise.
	logits, err := projQuantAttn(moe.Router, normed, D, nE, nil)
	if err != nil {
		return err
	}
	idx, wts, err := gptOssRouterTopK(logits, moe.RouterBias, topK)
	if err != nil {
		return err
	}

	// Routed experts: ONE device dispatch computes Σ_k w_k · ClampedSwiGLU(expert_k; biases) over
	// the batched expert tensors (see MoEExpertsQuantClampedSiLU — per-expert rows AND bias slices
	// addressed by byte offset in-kernel-loop, no host slicing).
	normedBF := f32ToBf16Slice(normed)
	routedBytes, err := MoEExpertsQuantClampedSiLU(normedBF, idx, wts,
		moe.ExpGate, moe.ExpUp, moe.ExpDown,
		moe.ExpGateBias, moe.ExpUpBias, moe.ExpDownBias,
		nE, len(idx), D, ff, moe.ExpGate.GroupSize, moe.ExpGate.Bits, moe.SwigluLimit)
	if err != nil {
		return err
	}
	acc := bf16ToF32Slice(routedBytes)

	// Residual (h + routed) → out, bf16. gpt_oss has no shared expert and no post-FF norm.
	ob := unsafe.Slice((*byte)(out.Contents()), D*2)
	for d := 0; d < D; d++ {
		u := f32ToBF16(h[d] + acc[d])
		ob[2*d], ob[2*d+1] = byte(u), byte(u>>8)
	}
	return nil
}
