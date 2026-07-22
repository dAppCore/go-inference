// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"
	"testing"

	core "dappco.re/go"
)

// TestHIPQwen3MoERouterSelect_Normalise_Good pins the norm_topk_prob=true combine order:
// softmax over the selected top-k sums to one. logits = [1,3,2,0] selects experts {1,2}
// (scores 3,2); softmax over just those two collapses to a 2-way sigmoid, a well-known
// closed form (sigmoid(1) = 1/(1+e^-1)), so the expected weights are computed
// independently of hipQwen3MoERouterSelect's own implementation.
func TestHIPQwen3MoERouterSelect_Normalise_Good(t *testing.T) {
	logits := []float32{1, 3, 2, 0}
	idx, weights := hipQwen3MoERouterSelect(logits, 2, true)

	core.AssertEqual(t, 2, len(idx))
	core.AssertEqual(t, 1, idx[0]) // highest score (3) first
	core.AssertEqual(t, 2, idx[1]) // second-highest (2)

	wantW1 := float32(1 / (1 + math.Exp(-1))) // sigmoid(3-2)
	wantW2 := 1 - wantW1
	assertFloat32Near(t, wantW1, weights[0])
	assertFloat32Near(t, wantW2, weights[1])

	sum := weights[0] + weights[1]
	assertFloat32Near(t, 1.0, sum)
}

// TestHIPQwen3MoERouterSelect_NoNormalise_Good pins the norm_topk_prob=false combine
// order (#65's OLMoE shape, honoured here for a qwen3_moe checkpoint that declares it
// explicitly): softmax over ALL experts, then gather the top-k WITHOUT renormalising —
// the SAME top-k indices as the normalise=true case, but weights that do not sum to one.
func TestHIPQwen3MoERouterSelect_NoNormalise_Good(t *testing.T) {
	logits := []float32{1, 3, 2, 0}
	idx, weights := hipQwen3MoERouterSelect(logits, 2, false)

	core.AssertEqual(t, 2, len(idx))
	core.AssertEqual(t, 1, idx[0])
	core.AssertEqual(t, 2, idx[1])

	total := math.Exp(1) + math.Exp(3) + math.Exp(2) + math.Exp(0)
	wantW1 := float32(math.Exp(3) / total)
	wantW2 := float32(math.Exp(2) / total)
	assertFloat32Near(t, wantW1, weights[0])
	assertFloat32Near(t, wantW2, weights[1])

	sum := weights[0] + weights[1]
	if sum >= 1.0 {
		t.Fatalf("norm_topk_prob=false weights must NOT sum to one (the divergence from normalise=true): got %v", sum)
	}
}

// TestHIPQwen3MoERouterSelect_BothOrders_SameSelection_Good proves both combine-order
// semantics select the IDENTICAL top-k expert set — they diverge only in the weight each
// contributes, never in which experts are chosen (matching engine/metal/router.go's
// documented invariant).
func TestHIPQwen3MoERouterSelect_BothOrders_SameSelection_Good(t *testing.T) {
	logits := []float32{-2, 5, 0.5, 3, -1, 1.2}
	idxTrue, _ := hipQwen3MoERouterSelect(logits, 3, true)
	idxFalse, _ := hipQwen3MoERouterSelect(logits, 3, false)

	core.AssertEqual(t, len(idxTrue), len(idxFalse))
	for i := range idxTrue {
		core.AssertEqual(t, idxTrue[i], idxFalse[i])
	}
}

// TestHIPQwen3MoERouterSelect_TopKClampedToExpertCount_Good rejects neither an empty
// input nor a topK exceeding the expert count; the latter clamps rather than erroring
// (mirrors the router's job of always returning a usable selection).
func TestHIPQwen3MoERouterSelect_TopKClampedToExpertCount_Good(t *testing.T) {
	logits := []float32{1, 2}
	idx, weights := hipQwen3MoERouterSelect(logits, 5, true)
	core.AssertEqual(t, 2, len(idx))
	core.AssertEqual(t, 2, len(weights))
}

// TestHIPQwen3MoERouterSelect_Empty_Bad returns no selection for an empty expert set.
func TestHIPQwen3MoERouterSelect_Empty_Bad(t *testing.T) {
	idx, weights := hipQwen3MoERouterSelect(nil, 2, true)
	if idx != nil || weights != nil {
		t.Fatalf("expected nil selection for empty logits, got idx=%v weights=%v", idx, weights)
	}
}

// TestHIPQwen3MoESwiGLUHostReference_Good pins the host oracle's arithmetic against a
// hand-computed 1-D expert (d=1, ff=1): gate=2, up=3, silu(2)=2/(1+e^-2), down scales by
// 1 — a value chosen so every step (matvec, SiLU, matvec) is independently checkable.
func TestHIPQwen3MoESwiGLUHostReference_Good(t *testing.T) {
	x := []float32{1}
	gate := []float32{2} // gate = x . [2] = 2
	up := []float32{3}   // up = x . [3] = 3
	down := []float32{1} // down = h . [1] = h
	got := hipQwen3MoESwiGLUHostReference(x, gate, up, down, 1, 1)

	silu2 := float32(2 / (1 + math.Exp(-2)))
	want := silu2 * 3
	core.AssertEqual(t, 1, len(got))
	assertFloat32Near(t, want, got[0])
}
