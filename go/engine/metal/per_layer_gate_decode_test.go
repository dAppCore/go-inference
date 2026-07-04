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

// TestGemma4PerLayerGateDecode gates the per-layer-input gate wired into the decode tail: on a
// single-layer model the output WITH the gate must equal PerLayerInputGateQuant applied to the
// un-gated output (byte-exact — stepToken runs the same op host-side), and differ from it. nil
// ple is the no-op (existing gates stay byte-identical).
func TestGemma4PerLayerGateDecode(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const pliDim, gs, bits, maxLen = 32, 32, 4, 16
	const eps = float32(1e-6)
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
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
	layers := []DecodeLayerWeights{forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)}
	embed := toBF16Bytes(mk(vocab*dModel, 11))
	prompt := []int32{1, 5, 3}
	attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
	embedScale := float32(math.Sqrt(float64(dModel)))

	gp, gsc, gb := quantizeProj(t, pliDim, dModel, gs, bits, 5) // gate [pliDim × dModel]
	pp, psc, pb := quantizeProj(t, dModel, pliDim, gs, bits, 7) // proj [dModel × pliDim]
	gate := QuantWeight{Packed: gp, Scales: gsc, Biases: gb}
	proj := QuantWeight{Packed: pp, Scales: psc, Biases: pb}
	postNorm := toBF16Bytes(mk(dModel, 9))
	pli := toBF16Bytes(mk(pliDim, 17))

	lastHidden := func(withPLE bool) []byte {
		var h []byte
		withAutoreleasePool(func() {
			lb, moe, _ := buildBF16ArchLayerBufs(layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen, arch.SlidingWindow, nil)
			st := newArchDecodeState(arch.Layer, lb, moe, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, eps, false, 0)
			if withPLE {
				st.ple = []pleLayer{{gate: gate, proj: proj, postNorm: postNorm, groupSize: gs, bits: bits}}
				st.perLayerInput = pli
				st.pliDim = pliDim
			}
			for p, id := range prompt {
				embs, err := EmbedTokensBF16(embed, []int32{id}, arch.Vocab, arch.Hidden, embedScale)
				if err != nil {
					t.Fatalf("EmbedTokensBF16: %v", err)
				}
				hh, err := st.stepToken(embs[0], p)
				if err != nil {
					t.Fatalf("stepToken: %v", err)
				}
				h = hh
			}
		})
		return h
	}
	hNoPLE := lastHidden(false)
	hPLE := lastHidden(true)

	if bytes.Equal(hNoPLE, hPLE) {
		t.Fatal("the per-layer-input gate had no effect on the decode output")
	}
	want, err := PerLayerInputGateQuant(hNoPLE, gate, pli, proj, postNorm, dModel, pliDim, gs, bits, eps)
	if err != nil {
		t.Fatalf("PerLayerInputGateQuant: %v", err)
	}
	if !bytes.Equal(hPLE, want) {
		t.Fatal("decode-tail gate != PerLayerInputGateQuant(un-gated output)")
	}
	t.Logf("per-layer-input gate wired into the decode tail: single-layer output == PerLayerInputGateQuant(un-gated), byte-for-byte")
}
