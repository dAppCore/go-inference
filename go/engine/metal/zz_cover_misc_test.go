// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// zz_cover_misc_test.go mops up the remaining reachable legs: the untied-LM-head
// branch in loadedToQuant, the encode-step legs in the re-encode measure helpers
// (attentionReEncode / layerReEncode / tokenReEncode), and the composed-gelu
// downstream legs in MoEExperts + PerLayerInputs that the fused-gelu path skips.

// TestCoverLoadedToQuantUntiedHead covers the `m.LMHead != nil` branch in
// loadedToQuant by handing it a model with a SEPARATE (untied) output projection.
// The existing test only exercises the tied (nil LMHead) branch and the nil-model
// guard.
func TestCoverLoadedToQuantUntiedHead(t *testing.T) {
	lin := func(out, in int) *model.Linear {
		return &model.Linear{
			Weight:    make([]byte, out*in/2),
			Scales:    make([]byte, out*(in/64)*2),
			Biases:    make([]byte, out*(in/64)*2),
			GroupSize: 64,
			Bits:      4,
		}
	}
	const dModel, vocab = 64, 128
	m := &model.LoadedModel{
		Arch:      model.Arch{Hidden: dModel, Vocab: vocab},
		Embed:     lin(vocab, dModel),
		LMHead:    lin(vocab, dModel), // untied ⇒ the m.LMHead != nil branch
		FinalNorm: make([]byte, dModel*2),
	}
	q, err := loadedToQuant(m, 64, 4)
	if err != nil {
		t.Fatalf("loadedToQuant untied head: %v", err)
	}
	if q.Tied {
		t.Fatal("expected untied model (Tied=false) when LMHead is separate")
	}
	if string(q.LMHead) != string(m.LMHead.Weight) {
		t.Fatal("untied LMHead weight not taken from m.LMHead")
	}
}

// TestCoverMeasureReEncodeEncodeLegs covers the encode-step error legs in the
// re-encode measure helpers via single-key eviction. The guard suite calls these
// successfully; here a warmed pipeline is evicted so the encode step fails.
func TestCoverMeasureReEncodeEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 4, 2, 64, 4, 256
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-6)
	const offset = 0
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	kCache := make([]byte, nKV*kvLen*headDim*bf16Size)
	vCache := make([]byte, nKV*kvLen*headDim*bf16Size)

	coverEncodeEvictAll(t, func() error {
		return attentionReEncode(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache,
			dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1)
	})
	coverEncodeEvictAll(t, func() error {
		return layerReEncode(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache,
			layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown,
			dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1)
	})
	coverEncodeEvictAll(t, func() error {
		_, e := tokenReEncode(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache,
			layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown,
			dModel, nHeads, nKV, headDim, kvLen, dFF, 1, base, scale, offset, eps, 1)
		return e
	})
}

// TestCoverMoEExpertsComposedEncodeLegs re-covers MoEExperts with the composed
// gelu path so the encGeluGateMul error leg in the expert loop is reachable.
func TestCoverMoEExpertsComposedEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, numExperts, topK = 64, 256, 2, 2
	w := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	idx := []int32{0, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})

	coverEncodeEvictAllComposed(t, func() error {
		_, e := MoEExperts(x, idx, weights, w.ExpGateW, w.ExpUpW, w.ExpDownW, numExperts, topK, dModel, dFF)
		return e
	})

	// quant experts: the encGeluGateMul composed leg in MoEExpertsQuant.
	const gs, bits = 64, 4
	qw := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, dFF, gs, bits)
	coverEncodeEvictAllComposed(t, func() error {
		_, e := MoEExpertsQuant(x, idx, weights, qw.ExpGate, qw.ExpUp, qw.ExpDown, numExperts, topK, dModel, dFF, gs, bits)
		return e
	})
}

// TestCoverPerLayerInputsComposedEncodeLegs re-covers PerLayerInputs with the
// composed gelu path so the gate/projection downstream legs in the per-layer-input
// gate (which the fused path shortcuts) are reachable.
func TestCoverPerLayerInputsComposedEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, pliDim = 64, 64
	const gs, bits = 64, 4
	const eps = float32(1e-5)
	hNext := toBF16Bytes(syntheticFloat32(dModel, 1))
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 3))
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 7))
	projW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 9))
	qGate := quantWeightFixture(t, pliDim, dModel, gs, bits, 11)
	qProj := quantWeightFixture(t, dModel, pliDim, gs, bits, 13)

	coverEncodeEvictAllComposed(t, func() error {
		_, e := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps)
		return e
	})
	coverEncodeEvictAllComposed(t, func() error {
		_, e := PerLayerInputGateQuant(hNext, qGate, perLayerInput, qProj, postNormW, dModel, pliDim, gs, bits, eps)
		return e
	})
}

// TestCoverComposedMLPHalfEncodeLegs re-covers the dense + arch forward paths with
// the composed gelu so the encMLPHalfBF16 / encGeluGateMul error legs in the MLP
// half (which the fused-gelu single dispatch shortcuts) become reachable by
// eviction. Hits chain.go MLPBlock, DecodeForward, DecodeForwardQuant, MoEExperts.
func TestCoverComposedMLPHalfEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 4, 2, 64, 256, 4
	const gs, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	qlayers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 5)}

	// dense bf16 forward (encMLPHalfBF16 composed gelu legs).
	coverEncodeEvictAllComposed(t, func() error {
		_, e := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		return e
	})
	// quant forward.
	coverEncodeEvictAllComposed(t, func() error {
		_, e := DecodeForwardQuant(inputs, qlayers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		return e
	})
	// float32 chain MLPBlock (the composed gelu lives in chain.go's gemv loop).
	x := syntheticFloat32(dModel, 1)
	normW := syntheticFloat32(dModel, 3)
	wGate := syntheticFloat32(dFF*dModel, 5)
	wUp := syntheticFloat32(dFF*dModel, 7)
	wDown := syntheticFloat32(dModel*dFF, 9)
	coverEncodeEvictAllComposed(t, func() error {
		_, e := MLPBlock(x, normW, wGate, wUp, wDown, dModel, dFF, eps)
		return e
	})
}

// TestCoverHeadEncoderEncodeLegs covers the zero-copy head encoder's encode-step
// error legs (head_nocopy.go) by loading a bf16 token model from a directory (so
// the no-copy headEnc is built), warming its Head, then evicting a warmed pipeline
// so the head's rmsnorm / gemv encode fails.
func TestCoverHeadEncoderEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab = 64, 2, 1, 64, 256, 32
	const maxLen = 8
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	dir := t.TempDir()
	writeLocal(t, core.PathJoin(dir, "config.json"), gemma4ConfigJSON(t, cfg))
	writeLocal(t, core.PathJoin(dir, "model.safetensors"), encodedTensors(t, gemma4TensorsMust(t, arch)))

	tm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	if closer, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = closer.Close() }()
	}
	nm, ok := tm.(*NativeTokenModel)
	if !ok {
		t.Fatalf("loaded token model is %T, want *NativeTokenModel", tm)
	}
	if nm.headEnc == nil {
		t.Fatal("directory-loaded bf16 token model has no zero-copy head encoder")
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 1))
	coverEncodeEvictAll(t, func() error {
		_, e := nm.Head(hidden)
		return e
	})
}

// TestCoverRopeCacheHitAndStepValidation mops up two small reachable legs: the
// ropePSOCache HIT branch in ropePipeline (a second RoPE call returns the cached
// pipeline) and the validateStepKV error leg in DecodeStepKV (a wrong-length x
// trips validation before any encode).
func TestCoverRopeCacheHitAndStepValidation(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 2, 1, 64, 4, 0, 256
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-6)

	// (1) ropePipeline cache-hit branch: first call builds + caches, second hits.
	x32 := syntheticFloat32(nHeads*headDim, 1)
	if _, err := RoPE(x32, 1, nHeads, headDim, 10000, 1, 0, false); err != nil {
		t.Fatalf("RoPE warm: %v", err)
	}
	if _, err := RoPE(x32, 1, nHeads, headDim, 10000, 1, 0, false); err != nil {
		t.Fatalf("RoPE cache hit: %v", err)
	}

	// (2) DecodeStepKV validateStepKV error leg via a wrong-length x.
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	shortX := toBF16Bytes(syntheticFloat32(dModel-1, 5))
	kCache := make([]byte, nKV*maxLen*headDim*bf16Size)
	vCache := make([]byte, nKV*maxLen*headDim*bf16Size)
	if _, err := DecodeStepKV(shortX, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache,
		layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown,
		dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps); err == nil {
		t.Fatal("DecodeStepKV: expected validateStepKV error for wrong-length x")
	}
}
