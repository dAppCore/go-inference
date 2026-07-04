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

// TestGemma4LayerScalar gates gemma4's per-layer output scalar: on a single-layer model the
// decode hidden with LayerScalarW = 2.0 must be exactly twice the hidden without it (×2 is
// exact in bf16, so element-for-element b == 2·a), confirming the scalar multiplies the layer's
// output. nil LayerScalarW is the no-op (the existing gates stay byte-identical).
func TestGemma4LayerScalar(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const maxLen = 16
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

	lastHidden := func(scalarW []byte) []byte {
		layers[0].LayerScalarW = scalarW
		var h []byte
		withAutoreleasePool(func() {
			lb, moe, _ := buildBF16ArchLayerBufs(layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen, arch.SlidingWindow, nil)
			st := newArchDecodeState(arch.Layer, lb, moe, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, false, 0)
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
	hNone := lastHidden(nil)
	hTwo := lastHidden(toBF16Bytes([]float32{2.0}))

	if bytes.Equal(hNone, hTwo) {
		t.Fatal("layer_scalar = 2 had no effect on the hidden")
	}
	for i := 0; i < dModel; i++ {
		a := bf16ToF32(hNone[i*bf16Size], hNone[i*bf16Size+1])
		b := bf16ToF32(hTwo[i*bf16Size], hTwo[i*bf16Size+1])
		if b != 2*a {
			t.Fatalf("element %d: scalar=2 hidden %v != 2 × no-scalar %v", i, b, 2*a)
		}
	}
	t.Logf("layer_scalar: a single-layer hidden with scalar=2 is exactly 2× the unscaled hidden, element-for-element")
}

func TestLayerScalarBufAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel = 128
	scalarW := toBF16Bytes([]float32{0.75})
	if buf := layerScalarBuf(scalarW, dModel); buf == nil {
		t.Fatal("layerScalarBuf warmup returned nil")
	}
	forceNativeGC()

	allocs := testing.AllocsPerRun(5, func() {
		if buf := layerScalarBuf(scalarW, dModel); buf == nil {
			t.Fatal("layerScalarBuf returned nil")
		}
	})
	if allocs > 6 {
		t.Fatalf("layerScalarBuf allocations = %.0f, want <= 6", allocs)
	}
}

func TestValueNormOnesBufAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const headDim = 256
	if buf := valueNormOnesBuf(true, headDim); buf == nil {
		t.Fatal("valueNormOnesBuf warmup returned nil")
	}
	forceNativeGC()

	allocs := testing.AllocsPerRun(5, func() {
		if buf := valueNormOnesBuf(true, headDim); buf == nil {
			t.Fatal("valueNormOnesBuf returned nil")
		}
	})
	if allocs > 6 {
		t.Fatalf("valueNormOnesBuf allocations = %.0f, want <= 6", allocs)
	}
}
