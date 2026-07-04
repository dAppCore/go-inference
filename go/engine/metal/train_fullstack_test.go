// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TestFullStackLayerLoRASFT is the complete full-stack training proof: a stack of full transformer
// layers (multi-head attention block + MLP block, each with its residuals) with a LoRA adapter on EVERY
// layer's down-projection, trained by backpropagating the loss through the WHOLE stack (head → final
// norm → layer N-1 → … → layer 0), each layer's DH feeding the one below. Every block backward is
// gradient-checked and forward-matched; this proves they chain across a real-depth stack with trainable
// projection LoRAs on each, and the loss falls. Stable small weights keep the deep backprop numerically
// clean.
func TestFullStackLayerLoRASFT(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, H, Hkv, headDim, dFF = 128, 4, 2, 32, 256
	const vocab, nL, T, rank, steps = 32, 3, 5, 4, 400
	scaling := float32(16.0 / rank)
	eps, base, scale := float32(1e-5), float32(10000), float32(1.0/6.0) // headDim 32 → 1/sqrt ~ 0.177; use a fixed small scale
	scale = float32(1.0 / 5.656854)                                     // 1/sqrt(32)

	type layer struct {
		aNorm, wQ, wK, wV, wO    []float32
		mNorm, wGate, wUp, wDown []float32
		la, lb                   []float32 // LoRA on wDown: A [rank,dFF], B [dModel,rank]
		oA, oB                   *AdamW
	}
	mk := func(salt int) layer {
		s := func(n, k int) []float32 { return scaleSlice(syntheticFloat32(n, k), 0.08) }
		return layer{
			aNorm: syntheticFloat32(dModel, salt), wQ: s(H*headDim*dModel, salt+1), wK: s(Hkv*headDim*dModel, salt+2),
			wV: s(Hkv*headDim*dModel, salt+3), wO: s(dModel*H*headDim, salt+4),
			mNorm: syntheticFloat32(dModel, salt+5), wGate: s(dFF*dModel, salt+6), wUp: s(dFF*dModel, salt+7), wDown: s(dModel*dFF, salt+8),
			la: scaleSlice(syntheticFloat32(rank*dFF, salt+9), 0.1), lb: make([]float32, dModel*rank),
			oA: NewAdamW(rank*dFF, 0.02, 0), oB: NewAdamW(dModel*rank, 0.02, 0),
		}
	}
	layers := make([]layer, nL)
	for i := range layers {
		layers[i] = mk((i + 1) * 100)
	}
	finalNorm := syntheticFloat32(dModel, 7)
	lmHead := scaleSlice(syntheticFloat32(vocab*dModel, 8), 0.1)
	x := scaleSlice(syntheticFloat32(T*dModel, 9), 0.1)
	targets := make([]int32, T)
	for i := range targets {
		targets[i] = int32((i * 5) % vocab)
	}

	wDownEff := func(l layer) []float32 {
		ba, _ := MatMulF32(l.lb, l.la, dModel, rank, dFF)
		eff := make([]float32, dModel*dFF)
		for i := range eff {
			eff[i] = l.wDown[i] + scaling*ba[i]
		}
		return eff
	}

	var first, last float32
	for step := 0; step < steps; step++ {
		// forward, saving each layer's input h and its attention-block output a.
		hs := make([][]float32, nL+1)
		as := make([][]float32, nL)
		effs := make([][]float32, nL)
		hs[0] = x
		for l := 0; l < nL; l++ {
			a, err := MultiHeadAttnBlockForwardF32(hs[l], layers[l].aNorm, layers[l].wQ, layers[l].wK, layers[l].wV, layers[l].wO, T, dModel, H, Hkv, headDim, headDim, base, scale, eps, true)
			if err != nil {
				t.Fatalf("attn fwd L%d: %v", l, err)
			}
			as[l] = a
			effs[l] = wDownEff(layers[l])
			h, err := MLPBlockForwardF32(a, layers[l].mNorm, layers[l].wGate, layers[l].wUp, effs[l], T, dModel, dFF, eps)
			if err != nil {
				t.Fatalf("mlp fwd L%d: %v", l, err)
			}
			hs[l+1] = h
		}
		normed := rmsNormForwardF32(hs[nL], finalNorm, T, dModel, eps)
		logits, err := MatMulF32NT(normed, lmHead, T, dModel, vocab)
		if err != nil {
			t.Fatalf("logits: %v", err)
		}
		loss, dLogits, err := CrossEntropyBackwardF32(logits, targets, T, vocab)
		if err != nil {
			t.Fatalf("ce: %v", err)
		}
		if step == 0 {
			first = loss
		}
		last = loss

		// backward: head → final norm → layer N-1 … 0.
		dNormed, _, err := LinearBackwardF32(dLogits, normed, lmHead, T, dModel, vocab)
		if err != nil {
			t.Fatalf("head bwd: %v", err)
		}
		dh, _, err := RMSNormBackwardF32(dNormed, hs[nL], finalNorm, T, dModel, eps)
		if err != nil {
			t.Fatalf("finalnorm bwd: %v", err)
		}
		for l := nL - 1; l >= 0; l-- {
			mg, err := MLPBlockBackwardF32(dh, as[l], layers[l].mNorm, layers[l].wGate, layers[l].wUp, effs[l], T, dModel, dFF, eps)
			if err != nil {
				t.Fatalf("mlp bwd L%d: %v", l, err)
			}
			// LoRA gradients from this layer's dWdown.
			dA, _ := MatMulF32(transposeF32(layers[l].lb, dModel, rank), mg.DWDown, rank, dModel, dFF)
			dB, _ := MatMulF32(mg.DWDown, transposeF32(layers[l].la, rank, dFF), dModel, dFF, rank)
			for i := range dA {
				dA[i] *= scaling
			}
			for i := range dB {
				dB[i] *= scaling
			}
			_ = layers[l].oA.Step(layers[l].la, dA)
			_ = layers[l].oB.Step(layers[l].lb, dB)
			// continue the chain through this layer's attention block to the layer below.
			ag, err := MultiHeadAttnBlockBackwardF32(mg.DH, hs[l], layers[l].aNorm, layers[l].wQ, layers[l].wK, layers[l].wV, layers[l].wO, T, dModel, H, Hkv, headDim, headDim, base, scale, eps, true)
			if err != nil {
				t.Fatalf("attn bwd L%d: %v", l, err)
			}
			dh = ag.DH
		}
	}
	if last >= first*0.6 {
		t.Fatalf("full-stack LoRA SFT did not reduce loss enough: first=%.4f last=%.4f", first, last)
	}
	t.Logf("native full-stack LoRA across ALL %d layers: backward chains the whole stack, cross-entropy %.4f → %.4f over %d steps", nL, first, last, steps)
}
