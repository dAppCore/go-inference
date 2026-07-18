// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model"
)

// moe_batched_seam_test.go gates the composed→engine batched-MoE PLUMBING host-side: MoEMLP.forward's device
// branch (moe.go) that routes the whole top-K MoE through composed.MoEExpertsDevice in one call, and the
// batched *model.QuantWeight form the loader builds. The metallib/GPU is unavailable here, so the seam is
// bound to a HOST stub implementing the same contract (a plain per-expert weighted sum over the SAME packed
// bytes) — proving forward feeds idx/weights/combine correctly. The device kernel's own numeric parity is
// gated in engine/metal (moe_quant_test.go); this test is the wiring proof one level up.

// TestMoEMLP_Forward_BatchedSeamMatchesPerExpert asserts forward with the batched device seam bound equals
// forward with it nil (the per-expert host loop) to a tight tolerance: the batched routed path must combine
// the SAME experts by the SAME weights as the loop it replaces, and must leave the shared expert (always on
// the host path) untouched. The MoEMLP carries BOTH representations — the per-expert packed experts AND the
// batched tensors (the per-expert packed bytes concatenated) — so both forwards run over identical weights.
func TestMoEMLP_Forward_BatchedSeamMatchesPerExpert(t *testing.T) {
	const D, FF, nE, topK, bits, gs = 64, 96, 6, 2, 4, 32

	packedExperts := make([]MoEExpert, nE)
	for e := range nE {
		packedExperts[e], _ = mkMoEExpertQuant(t, D, FF, bits, gs, e*10+1)
	}
	// Concatenate the per-expert packed tensors into the batched [numExperts, …] form the seam consumes —
	// the exact inverse of the per-expert slicing the loader's switch_mlp branch keeps as the host fallback.
	batched := func(pick func(MoEExpert) *model.QuantWeight, outDim, inDim int) *model.QuantWeight {
		var packed, scales, biases []byte
		for e := range nE {
			qw := pick(packedExperts[e])
			packed = append(packed, qw.Packed...)
			scales = append(scales, qw.Scales...)
			biases = append(biases, qw.Biases...)
		}
		return &model.QuantWeight{Packed: packed, Scales: scales, Biases: biases, Bits: bits, GroupSize: gs, OutDim: outDim, InDim: inDim}
	}
	gateBatched := batched(func(e MoEExpert) *model.QuantWeight { return e.GateQ }, FF, D)
	upBatched := batched(func(e MoEExpert) *model.QuantWeight { return e.UpQ }, FF, D)
	downBatched := batched(func(e MoEExpert) *model.QuantWeight { return e.DownQ }, D, FF)

	sharedPacked, _ := mkMoEExpertQuant(t, D, FF, bits, gs, 900)
	router := syn(nE*D, 500)
	m := &MoEMLP{
		Router: router, Experts: packedExperts, Shared: &sharedPacked, TopK: topK, NormTopKProb: true,
		GateBatchedQ: gateBatched, UpBatchedQ: upBatched, DownBatchedQ: downBatched, MoEBits: bits, MoEGroupSize: gs,
	}

	// Host stub honouring the MoEExpertsDevice contract: slice each selected expert out of the batched
	// tensors, run its packed SwiGLU (the SAME swigluExpertQuantInto the per-expert loop drives), and Σ
	// weights[k]·expert_k — so any mismatch is a wiring bug (wrong idx, wrong weight, wrong combine), not a
	// numeric-tier difference.
	MoEExpertsDevice = func(xt []float32, sel []int, weights []float64, gate, up, down *model.QuantWeight, numExperts, topK, dModel, dFF int) ([]float32, error) {
		b, g := gate.Bits, gate.GroupSize
		gatePacked, gateScale := dFF*dModel*b/8, dFF*(dModel/g)*2
		downPacked, downScale := dModel*dFF*b/8, dModel*(dFF/g)*2
		sliceQ := func(w *model.QuantWeight, e, packedPer, scalePer, outDim, inDim int) *model.QuantWeight {
			return &model.QuantWeight{
				Packed: w.Packed[e*packedPer : (e+1)*packedPer],
				Scales: w.Scales[e*scalePer : (e+1)*scalePer],
				Biases: w.Biases[e*scalePer : (e+1)*scalePer],
				Bits:   b, GroupSize: g, OutDim: outDim, InDim: inDim,
			}
		}
		acc := make([]float64, dModel)
		eo := make([]float32, dModel)
		for k, e := range sel {
			exp := MoEExpert{
				GateQ: sliceQ(gate, e, gatePacked, gateScale, dFF, dModel),
				UpQ:   sliceQ(up, e, gatePacked, gateScale, dFF, dModel),
				DownQ: sliceQ(down, e, downPacked, downScale, dModel, dFF),
			}
			swigluExpertQuantInto(xt, exp, dModel, eo)
			for d := range dModel {
				acc[d] += weights[k] * float64(eo[d])
			}
		}
		res := make([]float32, dModel)
		for d := range dModel {
			res[d] = float32(acc[d])
		}
		return res, nil
	}
	t.Cleanup(func() { MoEExpertsDevice = nil })

	const L = 5 // multi-token: top-2-of-6 routing sends different tokens to different expert subsets
	x := syn(L*D, 777)
	gotSeam := m.forward(x, L, D)

	MoEExpertsDevice = nil
	wantPerExpert := m.forward(x, L, D)

	maxRel := relError(t, "MoEMLP.forward(batched-seam)", gotSeam, wantPerExpert, 1e-4)
	t.Logf("batched-seam vs per-expert loop: %d tokens x %d dims, top-%d of %d experts + shared, max relative error %.3e", L, D, topK, nE, maxRel)
}
