// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// stableLayerWeights builds a dense layer with SMALL weights so a stacked forward keeps activations
// normalised (forwardLayer's ±1 weights explode them to ~1e5 where bf16/f32 diverge) — needed to train
// through the layer backward numerically cleanly.
func stableLayerWeights(dModel, nHeads, nKV, headDim, dFF, salt int) DecodeLayerWeights {
	qDim, kvDim := nHeads*headDim, nKV*headDim
	mk := func(n, s int) []byte {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*s+7)%101-50) * 0.002 // ±0.1
		}
		return toBF16Bytes(f)
	}
	return DecodeLayerWeights{
		AttnNormW: mk(dModel, salt+13), WQ: mk(qDim*dModel, salt+53),
		WK: mk(kvDim*dModel, salt+71), WV: mk(kvDim*dModel, salt+83), WO: mk(dModel*qDim, salt+17),
		MLPNormW: mk(dModel, salt+19), WGate: mk(dFF*dModel, salt+61),
		WUp: mk(dFF*dModel, salt+29), WDown: mk(dModel*dFF, salt+47),
	}
}

// TestRealSessionProjectionLoRASFT is the FULL-STACK projection-LoRA proof: a LoRA adapter on a layer's
// DOWN-PROJECTION (a resident weight, not the head) is trained by backpropagating the loss through that
// layer's real block backward (head → final norm → MLP block) over the engine's frozen activations from
// ForwardCaptureHiddens. The down-proj's effective weight is Wdown + (alpha/rank)·B·A; the LoRA gradients
// come from the block backward's dWdown (dA = scaling·Bᵀ·dWdown, dB = scaling·dWdown·Aᵀ). The loss must
// fall — proof native trains a LoRA on a real ArchSession's PROJECTION through the chained backward, the
// remaining full-stack training item. Stable small-weight model so the layer backprop is numerically clean.
func TestRealSessionProjectionLoRASFT(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 256, 4, 2, 64, 512
	const vocab, nL, maxLen, rank, steps = 48, 2, 64, 8, 400
	scaling := float32(16.0 / rank)
	eps := float32(1e-5)
	H, Hkv := nHeads, nKV

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = stableLayerWeights(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(scaleSlice(syntheticFloat32(vocab*dModel, 21), 0.1))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: eps, AttnScale: float32(1.0 / 8.0), RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	ids := []int32{1, 2, 3, 4, 5, 6}
	T := len(ids)
	scale := float32(1.0 / 8.0)

	_, perLayer, err := sess.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	// the LAST layer is the trainable one; its input is the frozen output of the layer below.
	lastIn := bf16ToF32Slice(perLayer[nL-2]) // [T,dModel]
	lw := g.Layers[nL-1]
	aNorm, wQ, wK, wV, wO := bf16ToF32Slice(lw.AttnNormW), bf16ToF32Slice(lw.WQ), bf16ToF32Slice(lw.WK), bf16ToF32Slice(lw.WV), bf16ToF32Slice(lw.WO)
	mNorm, wGate, wUp, wDown := bf16ToF32Slice(lw.MLPNormW), bf16ToF32Slice(lw.WGate), bf16ToF32Slice(lw.WUp), bf16ToF32Slice(lw.WDown)
	finalNorm, lmHead := bf16ToF32Slice(g.FinalNorm), bf16ToF32Slice(g.LMHead)

	// the attention half of the last layer is frozen — recompute it once (host forward matches the engine).
	attnOut, err := MultiHeadAttnBlockForwardF32(lastIn, aNorm, wQ, wK, wV, wO, T, dModel, H, Hkv, headDim, headDim, 10000, scale, eps, true)
	if err != nil {
		t.Fatalf("attn fwd: %v", err)
	}
	targets := make([]int32, T)
	for i := range targets {
		targets[i] = int32((i * 5) % vocab)
	}

	// trainable LoRA on Wdown [dModel,dFF]: A [rank,dFF], B [dModel,rank] (B zero → starts at base).
	aL := scaleSlice(syntheticFloat32(rank*dFF, 11), 0.1)
	bL := make([]float32, dModel*rank)
	optA, optB := NewAdamW(rank*dFF, 0.02, 0.0), NewAdamW(dModel*rank, 0.02, 0.0)

	var first, last float32
	for s := range steps {
		// effective down-proj = Wdown + scaling·(B·A).
		ba, err := MatMulF32(bL, aL, dModel, rank, dFF) // [dModel,dFF]
		if err != nil {
			t.Fatalf("BA %d: %v", s, err)
		}
		wDownEff := make([]float32, dModel*dFF)
		for i := range wDownEff {
			wDownEff[i] = wDown[i] + scaling*ba[i]
		}
		// forward: MLP block (with LoRA'd down-proj) → final norm → head.
		mlpOut, err := MLPBlockForwardF32(attnOut, mNorm, wGate, wUp, wDownEff, T, dModel, dFF, eps)
		if err != nil {
			t.Fatalf("mlp fwd %d: %v", s, err)
		}
		normedF := rmsNormForwardF32(mlpOut, finalNorm, T, dModel, eps)
		logits, err := MatMulF32NT(normedF, lmHead, T, dModel, vocab)
		if err != nil {
			t.Fatalf("logits %d: %v", s, err)
		}
		loss, dLogits, err := CrossEntropyBackwardF32(logits, targets, T, vocab)
		if err != nil {
			t.Fatalf("ce %d: %v", s, err)
		}
		if s == 0 {
			first = loss
		}
		last = loss
		// backward: head (frozen) → final norm → MLP block → dWdown.
		dNormedF, _, err := LinearBackwardF32(dLogits, normedF, lmHead, T, dModel, vocab)
		if err != nil {
			t.Fatalf("head bwd %d: %v", s, err)
		}
		dMlpOut, _, err := RMSNormBackwardF32(dNormedF, mlpOut, finalNorm, T, dModel, eps)
		if err != nil {
			t.Fatalf("finalnorm bwd %d: %v", s, err)
		}
		mg, err := MLPBlockBackwardF32(dMlpOut, attnOut, mNorm, wGate, wUp, wDownEff, T, dModel, dFF, eps)
		if err != nil {
			t.Fatalf("mlp bwd %d: %v", s, err)
		}
		// LoRA gradients from dWdown: dA = scaling·Bᵀ·dWdown, dB = scaling·dWdown·Aᵀ.
		dA, err := MatMulF32(transposeF32(bL, dModel, rank), mg.DWDown, rank, dModel, dFF)
		if err != nil {
			t.Fatalf("dA %d: %v", s, err)
		}
		dB, err := MatMulF32(mg.DWDown, transposeF32(aL, rank, dFF), dModel, dFF, rank)
		if err != nil {
			t.Fatalf("dB %d: %v", s, err)
		}
		for i := range dA {
			dA[i] *= scaling
		}
		for i := range dB {
			dB[i] *= scaling
		}
		_ = optA.Step(aL, dA)
		_ = optB.Step(bL, dB)
	}
	if last >= first*0.6 {
		t.Fatalf("projection-LoRA SFT did not reduce loss enough: first=%.4f last=%.4f", first, last)
	}
	t.Logf("native full-stack projection LoRA on a REAL ArchSession: LoRA(Wdown) via the chained block backward, cross-entropy %.4f → %.4f over %d steps", first, last, steps)
}
