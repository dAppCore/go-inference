// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TestStackedSFTReducesLoss is the multi-layer proof: a STACK of two MLP blocks + an lm_head, trained
// with cross-entropy + AdamW, where the backward CHAINS across both layers (lm_head → layer1 → layer0).
// The single-block backwards are gradient-checked elsewhere; this proves they compose down a stack and
// the whole thing learns — the N-layer chaining the full-stack SFT needs, in miniature. Loss must fall.
func TestStackedSFTReducesLoss(t *testing.T) {
	requireNativeRuntime(t)
	const M, dModel, dFF, vocab, steps = 8, 8, 16, 4, 200
	eps := float32(1e-5)
	x := syntheticFloat32(M*dModel, 1)
	targets := make([]int32, M)
	for i := range targets {
		targets[i] = int32((i * 5) % vocab)
	}

	// two MLP layers + an lm_head, each weight with its own optimiser.
	type layer struct{ normW, wGate, wUp, wDown []float32 }
	mkLayer := func(salt int) layer {
		return layer{
			normW: syntheticFloat32(dModel, salt),
			wGate: scaleSlice(syntheticFloat32(dFF*dModel, salt+1), 0.3),
			wUp:   scaleSlice(syntheticFloat32(dFF*dModel, salt+2), 0.3),
			wDown: scaleSlice(syntheticFloat32(dModel*dFF, salt+3), 0.3),
		}
	}
	layers := []layer{mkLayer(10), mkLayer(20)}
	wHead := scaleSlice(syntheticFloat32(vocab*dModel, 30), 0.3)

	opt := func(n int) *AdamW { return NewAdamW(n, 0.02, 0.0) }
	oN := []*AdamW{opt(dModel), opt(dModel)}
	oG := []*AdamW{opt(dFF * dModel), opt(dFF * dModel)}
	oU := []*AdamW{opt(dFF * dModel), opt(dFF * dModel)}
	oD := []*AdamW{opt(dModel * dFF), opt(dModel * dFF)}
	oHead := opt(vocab * dModel)

	var first, last float32
	for s := 0; s < steps; s++ {
		// forward, saving each layer's input (the residual stream).
		h0 := x
		h1, err := MLPBlockForwardF32(h0, layers[0].normW, layers[0].wGate, layers[0].wUp, layers[0].wDown, M, dModel, dFF, eps)
		if err != nil {
			t.Fatalf("fwd L0 step %d: %v", s, err)
		}
		h2, err := MLPBlockForwardF32(h1, layers[1].normW, layers[1].wGate, layers[1].wUp, layers[1].wDown, M, dModel, dFF, eps)
		if err != nil {
			t.Fatalf("fwd L1 step %d: %v", s, err)
		}
		logits, err := MatMulF32NT(h2, wHead, M, dModel, vocab)
		if err != nil {
			t.Fatalf("head step %d: %v", s, err)
		}
		loss, dLogits, err := CrossEntropyBackwardF32(logits, targets, M, vocab)
		if err != nil {
			t.Fatalf("loss step %d: %v", s, err)
		}
		if s == 0 {
			first = loss
		}
		last = loss

		// backward: chain lm_head → layer1 → layer0.
		dh2, dWHead, err := LinearBackwardF32(dLogits, h2, wHead, M, dModel, vocab)
		if err != nil {
			t.Fatalf("head bwd step %d: %v", s, err)
		}
		g1, err := MLPBlockBackwardF32(dh2, h1, layers[1].normW, layers[1].wGate, layers[1].wUp, layers[1].wDown, M, dModel, dFF, eps)
		if err != nil {
			t.Fatalf("bwd L1 step %d: %v", s, err)
		}
		g0, err := MLPBlockBackwardF32(g1.DH, h0, layers[0].normW, layers[0].wGate, layers[0].wUp, layers[0].wDown, M, dModel, dFF, eps)
		if err != nil {
			t.Fatalf("bwd L0 step %d: %v", s, err)
		}

		// optimiser step on every weight in the stack.
		gs := []*MLPBlockGrads{g0, g1}
		for li := range layers {
			_ = oN[li].Step(layers[li].normW, gs[li].DNormW)
			_ = oG[li].Step(layers[li].wGate, gs[li].DWGate)
			_ = oU[li].Step(layers[li].wUp, gs[li].DWUp)
			_ = oD[li].Step(layers[li].wDown, gs[li].DWDown)
		}
		_ = oHead.Step(wHead, dWHead)
	}
	if last >= first*0.5 {
		t.Fatalf("stacked SFT did not reduce loss enough: first=%.4f last=%.4f", first, last)
	}
	t.Logf("native stacked (2-layer + head) SFT works: backward chains across the stack, cross-entropy %.4f → %.4f over %d steps", first, last, steps)
}

func scaleSlice(s []float32, f float32) []float32 {
	for i := range s {
		s[i] *= f
	}
	return s
}
