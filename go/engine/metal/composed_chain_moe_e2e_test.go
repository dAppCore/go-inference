// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/quant/mlxaffine"
)

// TestComposedChainMoELayerVsPerSeam is the END-TO-END gate the isolated body test could not be:
// a full ComposedSession over a synthetic 2-layer all-MoE quant model (a gated-delta layer + an
// attention layer, each with a batched-quant MoE FFN + a sigmoid-gated shared expert) stepped ONE
// L=1 decode token, on the whole-token CHAIN vs the per-seam (non-chain) path. It asserts the chain
// final hidden and the head logits are (a) FINITE — the "model.Greedy: all tokens are suppressed"
// failure is non-finite logits — and (b) argmax-equal to the per-seam path within a bf16 band. This
// catches a MoE-tail wiring bug (the tail not writing the chained-forward hidden, or the head
// reading the wrong buffer) that a body-in-isolation test cannot see.
func TestComposedChainMoELayerVsPerSeam(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — composed chain MoE e2e")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable — composed chain MoE e2e: %v", err)
	}
	const D, vocab = 512, 64
	const nE, topK, expertDFF, sharedFF, gs, bits = 4, 2, 128, 128, 64, 4
	if !gatedDeltaStepUsable(64, 64, 8, 24) {
		t.Skip("gated-delta kernel unavailable in this metallib")
	}
	gateKey := lthnGatherQMVKey{groupSize: gs, bits: bits, expertRows: expertDFF, fast: expertDFF%8 == 0 && D%512 == 0, batchedX: false}
	if _, ok := lthnGatherQMVPipeline(gateKey); !ok {
		t.Skip("lean gather kernel unavailable (custom library not loaded)")
	}

	// batched routed experts (mlx-affine 4-bit) + the per-expert slices for the host per-seam loop.
	batched := func(outDim, inDim, saltBase int) (*model.QuantWeight, []*model.QuantWeight) {
		var p, s, b []byte
		var per []*model.QuantWeight
		for e := 0; e < nE; e++ {
			pe, se, be, err := mlxaffine.QuantizeTensor(cbSyn(outDim*inDim, saltBase+e*7), outDim, inDim, bits, gs)
			if err != nil {
				t.Fatalf("QuantizeTensor: %v", err)
			}
			per = append(per, &model.QuantWeight{Packed: pe, Scales: se, Biases: be, Bits: bits, GroupSize: gs, OutDim: outDim, InDim: inDim})
			p, s, b = append(p, pe...), append(s, se...), append(b, be...)
		}
		return &model.QuantWeight{Packed: p, Scales: s, Biases: b, Bits: bits, GroupSize: gs, OutDim: outDim, InDim: inDim}, per
	}
	buildMoE := func(seed int) *composed.MoEMLP {
		gb, gper := batched(expertDFF, D, seed+100)
		ub, uper := batched(expertDFF, D, seed+200)
		db, dper := batched(D, expertDFF, seed+300)
		experts := make([]composed.MoEExpert, nE)
		for e := range experts {
			experts[e] = composed.MoEExpert{GateQ: gper[e], UpQ: uper[e], DownQ: dper[e]}
		}
		return &composed.MoEMLP{
			Router: cbSyn(nE*D, seed+500), Experts: experts,
			Shared:     &composed.MoEExpert{GateQ: mustQuant(t, seed+400, sharedFF, D), UpQ: mustQuant(t, seed+401, sharedFF, D), DownQ: mustQuant(t, seed+402, D, sharedFF)},
			SharedGate: cbSyn(D, seed+501),
			TopK:       topK, NormTopKProb: true, Gating: model.MoEGatingSoftmax,
			GateBatchedQ: gb, UpBatchedQ: ub, DownBatchedQ: db, MoEBits: bits, MoEGroupSize: gs,
		}
	}

	gcfg := model.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 64, ConvKernel: 4, Eps: 1e-5}
	convDim, vDim := gcfg.ConvDim(), gcfg.VDim()
	gw := &model.GatedDeltaWeights{
		ConvWeight: cbSyn(convDim*gcfg.ConvKernel, 11), ConvBias: cbSyn(convDim, 12),
		ALog: cbSyn(gcfg.ValueHeads, 13), DtBias: cbSyn(gcfg.ValueHeads, 14), Norm: cbSyn(gcfg.HeadDim, 15),
		InProjQKVQ: mustQuant(t, 21, convDim, D), InProjZQ: mustQuant(t, 22, vDim, D),
		InProjAQ: mustQuant(t, 23, gcfg.ValueHeads, D), InProjBQ: mustQuant(t, 24, gcfg.ValueHeads, D),
		OutProjQ: mustQuant(t, 25, D, vDim),
	}
	const AH, AKVH, AHD = 4, 2, 128
	m := &composed.ComposedModel{
		EmbedQ: mustQuant(t, 1, vocab, D), NormF: cbSyn(D, 2), OutputQ: mustQuant(t, 3, vocab, D),
		D: D, Vocab: vocab, Eps: 1e-6, Quantised: true,
		Layers: []composed.Layer{
			{
				InputNorm: cbSyn(D, 31), PostAttnNorm: cbSyn(D, 32),
				MLP:   buildMoE(1000),
				Mixer: composed.NewGatedDeltaMixer(gw, gcfg),
			},
			{
				InputNorm: cbSyn(D, 51), PostAttnNorm: cbSyn(D, 52),
				MLP: buildMoE(2000),
				Mixer: composed.NewAttnMixer(&composed.AttnWeights{
					QProjQ: mustQuant(t, 71, AH*AHD, D), KProjQ: mustQuant(t, 72, AKVH*AHD, D), VProjQ: mustQuant(t, 73, AKVH*AHD, D),
					OProjQ: mustQuant(t, 74, D, AH*AHD), QNorm: cbSyn(AHD, 75), KNorm: cbSyn(AHD, 76),
				}, composed.AttnConfig{Heads: AH, KVHeads: AKVH, HeadDim: AHD, RotaryDim: AHD, RopeTheta: 1e6, NormEps: 1e-6}),
			},
		},
	}

	beginCalls := 0
	run := func(chain bool) (hidden, pending []float32) {
		savedBegin, savedEnd := composed.ComposedChainBeginDevice, composed.ComposedChainEndDevice
		defer func() { composed.ComposedChainBeginDevice, composed.ComposedChainEndDevice = savedBegin, savedEnd }()
		if chain {
			composed.ComposedChainBeginDevice = func(h []float32, L, D int) (any, error) {
				beginCalls++
				return savedBegin(h, L, D)
			}
		} else {
			composed.ComposedChainBeginDevice, composed.ComposedChainEndDevice = nil, nil // chainable()==false → per-seam
		}
		sess := composed.NewSession(m)
		if _, err := sess.Forward([]int32{1, 2, 3, 4}); err != nil { // prefill L>1
			t.Fatalf("prefill Forward(chain=%v): %v", chain, err)
		}
		h, err := sess.Forward([]int32{5}) // decode L=1
		if err != nil {
			t.Fatalf("decode Forward(chain=%v): %v", chain, err)
		}
		return append([]float32(nil), h...), sess.PendingHeadLogits()
	}

	hChain, foldLogits := run(true)
	if beginCalls == 0 {
		t.Fatal("chain path never engaged (forwardChain not taken) — the e2e test would be vacuous")
	}
	hSeam, _ := run(false)

	finite := func(name string, v []float32) {
		t.Helper()
		for i, x := range v {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("%s[%d] is non-finite (%v) — this is the 'all tokens suppressed' failure", name, i, x)
			}
		}
	}
	argmax := func(v []float32) int {
		best := 0
		for i := 1; i < len(v); i++ {
			if v[i] > v[best] {
				best = i
			}
		}
		return best
	}

	finite("chain hidden", hChain)
	finite("per-seam hidden", hSeam)
	logitsChain := composed.HeadLogitsHost(m, hChain)
	logitsSeam := composed.HeadLogitsHost(m, hSeam)
	finite("chain logits", logitsChain)
	finite("per-seam logits", logitsSeam)

	if a, b := argmax(logitsChain), argmax(logitsSeam); a != b {
		t.Fatalf("head argmax differs: chain %d vs per-seam %d (chain hidden is wrong)", a, b)
	}
	if rel := gdScaledDiff(t, "hidden", hChain, hSeam); rel > 3e-2 {
		t.Fatalf("chain vs per-seam hidden diverged: scaled max diff %.3e", rel)
	}
	// head fold parity (only when the fold engaged and set PendingHeadLogits).
	if foldLogits != nil {
		finite("chain fold logits", foldLogits)
		if a, b := argmax(foldLogits), argmax(logitsChain); a != b {
			t.Fatalf("chain head-fold argmax %d != host-over-chain-hidden argmax %d", a, b)
		}
	}
	t.Logf("composed chain MoE e2e: chain hidden finite, argmax %d agrees with per-seam, hidden scaled-diff %.3e, fold logits set=%v",
		argmax(logitsChain), gdScaledDiff(t, "hidden", hChain, hSeam), foldLogits != nil)
}
