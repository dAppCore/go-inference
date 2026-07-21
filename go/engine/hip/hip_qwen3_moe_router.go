// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import "math"

// hipQwen3MoERouterSelect selects the top-k experts from logits (one score per expert,
// raw pre-softmax) and returns their indices plus the combine weight each contributes to
// the routed sum — host-orchestrated (one device round trip already produced logits; this
// is pure Go), mirroring engine/metal's arch_qwen_moe.go encQwenMoEHalf router shape.
//
// normalise is the checkpoint's declared norm_topk_prob (model.Arch.NormaliseMoETopK,
// #65): true softmaxes over ALL experts then renormalises the gathered top-k to sum to
// one — mathematically identical to softmaxing over just the selected k (qwen's default,
// and the shape most Qwen3-MoE checkpoints declare). false softmaxes over ALL experts and
// gathers the top-k WITHOUT renormalising, so the returned weights do not sum to one
// (OLMoE's norm_topk_prob=false shape — engine/metal/router.go's softmaxAllThenGatherInto
// documents the same divergence). Both orders select the SAME top-k indices; they only
// differ in the weight each contributes.
//
// idx is returned highest-scoring first, ties broken by lower index. Panics never occur:
// an empty logits or non-positive topK returns (nil, nil).
func hipQwen3MoERouterSelect(logits []float32, topK int, normalise bool) (idx []int, weights []float32) {
	numExperts := len(logits)
	if numExperts == 0 || topK <= 0 {
		return nil, nil
	}
	if topK > numExperts {
		topK = numExperts
	}
	maxLogit := float32(math.Inf(-1))
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	probs := make([]float64, numExperts)
	var total float64
	for e, v := range logits {
		p := math.Exp(float64(v - maxLogit))
		probs[e] = p
		total += p
	}

	idx = make([]int, 0, topK)
	used := make([]bool, numExperts)
	for k := 0; k < topK; k++ {
		best, bestIdx := -1.0, -1
		for e := 0; e < numExperts; e++ {
			if used[e] {
				continue
			}
			if probs[e] > best {
				best, bestIdx = probs[e], e
			}
		}
		if bestIdx < 0 {
			break
		}
		used[bestIdx] = true
		idx = append(idx, bestIdx)
	}

	weights = make([]float32, len(idx))
	if normalise {
		var selectedSum float64
		for _, e := range idx {
			selectedSum += probs[e]
		}
		if selectedSum == 0 {
			selectedSum = 1
		}
		for i, e := range idx {
			weights[i] = float32(probs[e] / selectedSum)
		}
	} else {
		if total == 0 {
			total = 1
		}
		for i, e := range idx {
			weights[i] = float32(probs[e] / total)
		}
	}
	return idx, weights
}

// hipQwen3MoESwiGLUHostReference computes down @ (silu(gate@x) . (up@x)) for one expert
// on the host, in plain float64 arithmetic — the correctness oracle the device path
// (RMSNorm/projection/SwiGLU kernels chained in hip_qwen3_moe_layer.go) is tested against.
// gate/up are row-major [ff, d], down is row-major [d, ff].
func hipQwen3MoESwiGLUHostReference(x, gate, up, down []float32, d, ff int) []float32 {
	g := make([]float64, ff)
	u := make([]float64, ff)
	for row := 0; row < ff; row++ {
		var gs, us float64
		for col := 0; col < d; col++ {
			gs += float64(x[col]) * float64(gate[row*d+col])
			us += float64(x[col]) * float64(up[row*d+col])
		}
		g[row], u[row] = gs, us
	}
	h := make([]float64, ff)
	for i := range h {
		h[i] = (g[i] / (1 + math.Exp(-g[i]))) * u[i]
	}
	out := make([]float32, d)
	for row := 0; row < d; row++ {
		var sum float64
		for col := 0; col < ff; col++ {
			sum += h[col] * float64(down[row*ff+col])
		}
		out[row] = float32(sum)
	}
	return out
}
