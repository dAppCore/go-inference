// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"

	core "dappco.re/go"
	g4 "dappco.re/go/inference/model/gemma4"
)

// TestNativeBackend gates the backend seam: NativeBackend.DecodeForward, built from a
// Config-derived Arch + weights, routes to the right arch forward — its output equals
// the direct forward call for every path (bf16/4-bit × re-encode/ICB), and a MoE arch
// asked for ICB falls back to the re-encode path (rather than erroring). The Arch is
// built via Config.Arch() so this also exercises config → arch → backend end-to-end.
func TestNativeBackend(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 4, 64, 1024, 64, 4
	const maxLen, T = 8, 4

	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 3, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: 1000, RMSNormEps: 1e-5, RopeTheta: 10000,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Config.Arch: %v", err)
	}
	base, eps := arch.RopeBase, arch.Eps
	scale := arch.AttnScale // the model-declared SDPA scale (gemma4 1.0), matching NativeBackend.DecodeForward

	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}

	eq := func(name string, got, want [][]byte) {
		for tok := 0; tok < T; tok++ {
			eqBytes(t, core.Sprintf("%s tok%d", name, tok), got[tok], want[tok])
		}
	}

	// bf16: re-encode + ICB.
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	bRe, err := NewBF16Backend(arch, layers, maxLen)
	if err != nil {
		t.Fatalf("NewBF16Backend: %v", err)
	}
	gotRe, err := bRe.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("bf16 re-encode DecodeForward: %v", err)
	}
	wantRe, err := DecodeForwardArch(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, base, scale, eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArch: %v", err)
	}
	eq("bf16-reencode", gotRe, wantRe)

	bICB, _ := NewBF16Backend(arch, layers, maxLen, WithICB())
	gotICB, err := bICB.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("bf16 ICB DecodeForward: %v", err)
	}
	wantICB, err := DecodeForwardArchICB(inputs, layers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, base, scale, eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchICB: %v", err)
	}
	eq("bf16-icb", gotICB, wantICB)

	// 4-bit: re-encode + ICB.
	qlayers := make([]QuantizedLayerWeights, len(arch.Layer))
	for li := range qlayers {
		qlayers[li] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (li+1)*100)
	}
	bQ, _ := NewQuantBackend(arch, qlayers, maxLen)
	gotQ, err := bQ.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("quant re-encode DecodeForward: %v", err)
	}
	wantQ, err := DecodeForwardArchQuant(inputs, qlayers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, base, scale, eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant: %v", err)
	}
	eq("quant-reencode", gotQ, wantQ)

	bQICB, _ := NewQuantBackend(arch, qlayers, maxLen, WithICB())
	gotQICB, err := bQICB.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("quant ICB DecodeForward: %v", err)
	}
	wantQICB, err := DecodeForwardArchICBQuant(inputs, qlayers, arch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, arch.SlidingWindow, base, scale, eps, arch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant: %v", err)
	}
	eq("quant-icb", gotQICB, wantQICB)

	// MoE arch asked for ICB → falls back to the re-encode path (no error).
	const numExperts, topK, expertDFF = 8, 2, 768
	moeCfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: 1000, RMSNormEps: 1e-5, RopeTheta: 10000,
		EnableMoEBlock: true, NumExperts: numExperts, TopKExperts: topK, MoEIntermediateSize: expertDFF,
	}
	moeArch, err := moeCfg.Arch()
	if err != nil {
		t.Fatalf("moe Config.Arch: %v", err)
	}
	moeLayers := make([]DecodeLayerWeights, len(moeArch.Layer))
	for li := range moeLayers {
		moeLayers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*50)
		moeLayers[li].MoE = buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, (li+1)*300)
	}
	bMoE, _ := NewBF16Backend(moeArch, moeLayers, maxLen, WithICB()) // WithICB, but MoE → re-encode
	gotMoE, err := bMoE.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("MoE backend DecodeForward: %v (ICB should have fallen back, not errored)", err)
	}
	wantMoE, err := DecodeForwardArch(inputs, moeLayers, moeArch.Layer, dModel, nHeads, nKV, headDim, maxLen, dFF, moeArch.SlidingWindow, base, scale, eps, moeArch.ValueNorm)
	if err != nil {
		t.Fatalf("DecodeForwardArch (MoE): %v", err)
	}
	eq("moe-fallback", gotMoE, wantMoE)

	// constructor validates the layer count.
	if _, err := NewBF16Backend(arch, layers[:2], maxLen); err == nil {
		t.Fatal("expected NewBF16Backend to reject a layer-count mismatch")
	}

	t.Logf("backend seam: config→arch→NativeBackend routes all four paths (bf16/4-bit × re-encode/ICB) ≡ the direct forward; MoE+ICB falls back to re-encode")
}

func TestNativeBackendDecodeForwardIntoReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers, maxLen = 64, 1, 1, 64, 128, 32, 1, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	backend, err := NewBF16Backend(arch, layers, maxLen)
	if err != nil {
		t.Fatalf("NewBF16Backend: %v", err)
	}
	inputs := decodeInputsFixture(2, dModel)
	want, err := backend.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("DecodeForward reference: %v", err)
	}
	out := [][]byte{
		bytes.Repeat([]byte{0xa5}, dModel*bf16Size),
		bytes.Repeat([]byte{0x5a}, dModel*bf16Size),
	}
	ptrs := []unsafe.Pointer{unsafe.Pointer(&out[0][0]), unsafe.Pointer(&out[1][0])}

	got, err := backend.DecodeForwardInto(out, inputs)
	if err != nil {
		t.Fatalf("DecodeForwardInto: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("DecodeForwardInto returned %d outputs, want %d", len(got), len(want))
	}
	for tok := range want {
		if len(got[tok]) != dModel*bf16Size || unsafe.Pointer(&got[tok][0]) != ptrs[tok] {
			t.Fatalf("DecodeForwardInto token %d did not reuse caller-owned output backing", tok)
		}
		eqBytes(t, "NativeBackend.DecodeForwardInto token", got[tok], want[tok])
	}
}
