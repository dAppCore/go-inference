// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// TestRealSessionHeadLoRASFT is real-model training on an actual gemma ArchSession (not a synthetic
// stack): the FROZEN base forward is the engine's own ForwardCaptureHiddens, and a LoRA adapter on the
// output head is trained with cross-entropy + AdamW to fit targets on the engine's real final hidden.
// The loss must fall — proof the native training stack drives a real ArchSession end to end. (The final
// RMSNorm normalises the hidden before the head, so this is numerically stable even on the synthetic
// exploding-activation weights.) Backpropagating further — LoRA on the layer projections via the chained
// block backwards over ForwardCaptureHiddens — is the full-stack extension on this same proven seam.
func TestRealSessionHeadLoRASFT(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen, rank, steps = 64, 3, 64, 8, 300
	scaling := float32(16.0 / rank)
	eps := float32(1e-5)

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: eps, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	T := len(ids)

	// FROZEN base: the engine's real forward, captured once. The last layer's hidden feeds the head.
	_, perLayer, err := sess.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	hLast := bf16ToF32Slice(perLayer[nL-1]) // [T,dModel]
	finalNorm := bf16ToF32Slice(g.FinalNorm)
	lmHead := bf16ToF32Slice(g.LMHead) // [vocab,dModel]

	// frozen head pre-activations: normed = RMSNorm(hLast), base logits = normed·lmHeadᵀ (computed once).
	normed := rmsNormForwardF32(hLast, finalNorm, T, dModel, eps)
	baseLogits, err := MatMulF32NT(normed, lmHead, T, dModel, vocab)
	if err != nil {
		t.Fatalf("base logits: %v", err)
	}
	targets := make([]int32, T)
	for i := range targets {
		targets[i] = int32((i * 7) % vocab)
	}

	// trainable LoRA head adapter (A random, B zero → starts at the base).
	a := syntheticFloat32(rank*dModel, 11)
	for i := range a {
		a[i] *= 0.2
	}
	b := make([]float32, vocab*rank)
	optA, optB := NewAdamW(rank*dModel, 0.05, 0.0), NewAdamW(vocab*rank, 0.05, 0.0)

	var first, last float32
	for s := range steps {
		xA, delta, err := LoRAForwardF32(normed, a, b, T, dModel, vocab, rank, scaling)
		if err != nil {
			t.Fatalf("lora fwd %d: %v", s, err)
		}
		logits := make([]float32, T*vocab)
		for i := range logits {
			logits[i] = baseLogits[i] + delta[i]
		}
		loss, dLogits, err := CrossEntropyBackwardF32(logits, targets, T, vocab)
		if err != nil {
			t.Fatalf("ce %d: %v", s, err)
		}
		if s == 0 {
			first = loss
		}
		last = loss
		dA, dB, _, err := LoRABackwardF32(dLogits, normed, a, b, xA, T, dModel, vocab, rank, scaling)
		if err != nil {
			t.Fatalf("lora bwd %d: %v", s, err)
		}
		_ = optA.Step(a, dA)
		_ = optB.Step(b, dB)
	}
	if last >= first*0.5 {
		t.Fatalf("real-session LoRA SFT did not reduce loss enough: first=%.4f last=%.4f", first, last)
	}
	t.Logf("native training drives a REAL ArchSession: engine forward (frozen) + head LoRA, cross-entropy %.4f → %.4f over %d AdamW steps", first, last, steps)
}
