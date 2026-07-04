// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestSessionKVCacheByteIdentical proves the native session's resident KV cache (the "session + kvconv
// off metal") is BYTE-IDENTICAL to the whole-sequence forward: stepping tokens one at a time through a
// persistent ArchSession (writing each token's K/V into the growing resident cache and attending it)
// must produce exactly the hiddens DecodeForwardArch produces stepping the same tokens over its own
// fresh cache. Identical to the byte means the resident-cache continuation is faithful — the property
// the serve path (serve --native) relies on for multi-turn conversations.
func TestSessionKVCacheByteIdentical(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const nL, T, maxLen = 3, 6, 32
	base, scale, eps := float32(10000), float32(0.125), float32(1e-5)

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)

	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}

	// reference: the whole-sequence forward over its own fresh growing cache.
	ref, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch: %v", err)
	}

	// the session: a persistent resident KV cache, stepped one token at a time (the kvconv path).
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(syntheticFloat32(8*dModel, 1)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 2)),
		LMHead:    toBF16Bytes(syntheticFloat32(8*dModel, 1)),
		Tied:      true,
	}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: 8,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: eps, AttnScale: scale, RopeBase: base, RopeScale: 1, RopeLocalBase: base,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	for tk := 0; tk < T; tk++ {
		h, err := sess.Step(inputs[tk]) // step the embedding over the resident cache, advancing pos
		if err != nil {
			t.Fatalf("session Step %d: %v", tk, err)
		}
		eqBytes(t, core.Sprintf("session resident-cache hidden tok%d vs whole-sequence forward", tk), h, ref[tk])
	}
	t.Logf("session + kvconv byte-identical: %d-token resident-cache decode == DecodeForwardArch token-for-token", T)
}
