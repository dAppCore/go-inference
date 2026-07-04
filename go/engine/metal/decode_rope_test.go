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

// TestDecodeRopePerType gates per-attention-type RoPE: on an all-sliding model the decode
// hidden state depends on the LOCAL theta (RopeLocalBase), never the global (RopeBase). So
// (global=G, local=L) ≡ (L, L) byte-for-byte — the global base never reaches a sliding layer
// — while (·, L) ≠ (·, G) when L ≠ G — the local base genuinely drives the sliding rotation.
//
// It compares the hidden bytes (not greedy tokens): a tiny synthetic model's argmax can be
// stuck regardless of the rotation, but the hidden state shifts with any rope change, so the
// byte comparison is exact in both directions — a leak shows as hGL≠hLL, a no-op as hGL==hLG —
// and works at the real gemma4 thetas (1e6 / 1e4) without needing an exaggerated gap.
func TestDecodeRopePerType(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil { // direct state-build bypasses GenerateBF16's init
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

	// step the prompt through a fresh state at (base, local); return the last hidden bytes.
	lastHidden := func(base, local float32) []byte {
		var h []byte
		withAutoreleasePool(func() {
			lb, moe, _ := buildBF16ArchLayerBufs(layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen, arch.SlidingWindow, nil)
			st := newArchDecodeState(arch.Layer, lb, moe, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, base, local, attnScale, arch.Eps, false, 0)
			for p, id := range prompt {
				embs, err := EmbedTokensBF16(embed, []int32{id}, arch.Vocab, arch.Hidden, embedScale)
				if err != nil {
					t.Fatalf("EmbedTokensBF16: %v", err)
				}
				hh, err := st.stepToken(embs[0], p)
				if err != nil {
					t.Fatalf("stepToken(base=%v,local=%v,pos=%d): %v", base, local, p, err)
				}
				h = hh
			}
		})
		return h
	}
	hGL := lastHidden(1_000_000, 10_000) // all-sliding → uses local 1e4
	hLL := lastHidden(10_000, 10_000)    // uses 1e4
	hLG := lastHidden(10_000, 1_000_000) // all-sliding → uses local 1e6

	if !bytes.Equal(hGL, hLL) {
		t.Fatalf("sliding layers leaked the global base: hidden(1e6,1e4) != hidden(1e4,1e4)")
	}
	if bytes.Equal(hGL, hLG) {
		t.Fatalf("local RoPE base had no effect on the hidden state: hidden(·,1e4) == hidden(·,1e6)")
	}
	t.Logf("per-type RoPE: all-sliding hidden uses the LOCAL theta — hidden(1e6,1e4)≡hidden(1e4,1e4) byte-for-byte, ≠hidden(1e4,1e6); the global base never rotates a sliding layer")
}
