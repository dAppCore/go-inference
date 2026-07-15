// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"os"
	"testing"

	g4 "dappco.re/go/inference/model/gemma4"
)

// TestGemma4PartialRotaryDecode gates that partial rotary is WIRED into the decode per
// attention type: on an all-sliding model the decode hidden depends on the LOCAL rotary dim
// (RotaryDimLocal), not the global (RotaryDim). So shrinking RotaryDimLocal changes the hidden
// (partial rotary is live), while shrinking the global RotaryDim does NOT (it never reaches a
// sliding layer). Compares hidden bytes (the op itself is gated byte-exact in TestRoPEDimsPartial).
func TestGemma4PartialRotaryDecode(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const maxLen = 16
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6, SlidingWindow: 4,
		LayerTypes: []string{"sliding_attention", "sliding_attention"},
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	embed := toBF16Bytes(mk(vocab*dModel, 11))
	prompt := []int32{1, 5, 3}
	attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
	embedScale := float32(math.Sqrt(float64(dModel)))

	lastHidden := func(rotDim, rotDimLocal int) []byte {
		var h []byte
		withAutoreleasePool(func() {
			lb, moe, _ := buildBF16ArchLayerBufs(layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen, arch.SlidingWindow, nil)
			st := newArchDecodeState(arch.Layer, lb, moe, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, rotDim, rotDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, false, 0)
			for p, id := range prompt {
				embs, err := EmbedTokensBF16(embed, []int32{id}, arch.Vocab, arch.Hidden, embedScale)
				if err != nil {
					t.Fatalf("EmbedTokensBF16: %v", err)
				}
				hh, err := st.stepToken(embs[0], p)
				if err != nil {
					t.Fatalf("stepToken(rotDim=%d,rotDimLocal=%d,pos=%d): %v", rotDim, rotDimLocal, p, err)
				}
				h = hh
			}
		})
		return h
	}
	hFullLocal := lastHidden(headDim, headDim)       // full rotary everywhere
	hPartialLocal := lastHidden(headDim, headDim/2)  // partial on the (used) sliding layers
	hGlobalPartial := lastHidden(headDim/2, headDim) // partial GLOBAL only — sliding layers ignore it

	if bytes.Equal(hFullLocal, hPartialLocal) {
		t.Fatal("shrinking the local rotary dim had no effect — partial rotary is not wired into the sliding decode")
	}
	if !bytes.Equal(hFullLocal, hGlobalPartial) {
		t.Fatal("the global rotary dim leaked into the sliding layers — per-type rotary is wrong")
	}
	t.Logf("partial rotary wired per-type: all-sliding hidden tracks RotaryDimLocal (full≠partial) and ignores the global RotaryDim (full≡global-partial)")
}
