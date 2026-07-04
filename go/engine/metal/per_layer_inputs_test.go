// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"os"
	"testing"
	"unsafe"
)

// TestPerLayerInputs gates the model-level per-layer-input pipeline: the [numLayers·pliDim]
// tensor combining the 4-bit per-layer embedding (×√pliDim) with the bf16 projection of the
// main embedding (×1/√dModel, RMSNorm'd per layer-row), summed and ×1/√2. The projW=0 case is
// an INDEPENDENT anchor — the projection branch vanishes (RMSNorm(0)=0), so the result is
// exactly the embed branch ×1/√2 — and a non-zero projW must differ (the projection is live).
func TestPerLayerInputs(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const vocabPLI, numLayers, pliDim, dModel, gs, bits = 4, 2, 32, 64, 32, 4
	const plDim = numLayers * pliDim // 64, a multiple of the group size
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.03
		}
		return s
	}
	embedPacked, embedScales, embedBiases := quantizeProj(t, vocabPLI, plDim, gs, bits, 3)
	projW := toBF16Bytes(mk(plDim*dModel, 5))
	projNormW := toBF16Bytes(mk(pliDim, 7))
	hidden := toBF16Bytes(mk(dModel, 9))
	const tokenID int32 = 2

	got, err := PerLayerInputs(embedPacked, embedScales, embedBiases, projW, nil, nil, projNormW, tokenID, hidden, vocabPLI, numLayers, pliDim, dModel, gs, bits, 0, 0, eps, bufView{})
	if err != nil {
		t.Fatalf("PerLayerInputs: %v", err)
	}
	if len(got) != plDim*bf16Size {
		t.Fatalf("shape: %d bytes, want %d", len(got), plDim*bf16Size)
	}

	embScale := float32(math.Sqrt(float64(pliDim)))
	perLayer, err := EmbedTokensQuant(embedPacked, embedScales, embedBiases, []int32{tokenID}, vocabPLI, plDim, gs, bits, embScale)
	if err != nil {
		t.Fatalf("EmbedTokensQuant: %v", err)
	}

	// independent anchor: projW=0 → projected=0 → RMSNorm(0)=0 → got == perLayer × 1/√2.
	gotZero, err := PerLayerInputs(embedPacked, embedScales, embedBiases, make([]byte, len(projW)), nil, nil, projNormW, tokenID, hidden, vocabPLI, numLayers, pliDim, dModel, gs, bits, 0, 0, eps, bufView{})
	if err != nil {
		t.Fatalf("PerLayerInputs(projW=0): %v", err)
	}
	wantZero, err := MulBF16(perLayer[0], bf16ConstBytes(plDim, gemma4PerLayerCombineScale))
	if err != nil {
		t.Fatalf("MulBF16: %v", err)
	}
	if !bytes.Equal(gotZero, wantZero) {
		t.Fatal("projW=0: result is not the embed branch × 1/√2")
	}
	if bytes.Equal(got, gotZero) {
		t.Fatal("the per_layer_model_projection branch had no effect")
	}

	// rebuilt reference: the same pipeline, written out — catches a wrong scale/order/norm/combine.
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	pr, err := MatVecBF16(projW, hidden, plDim, dModel)
	if err != nil {
		t.Fatalf("MatVecBF16: %v", err)
	}
	if pr, err = MulBF16(pr, bf16ConstBytes(plDim, projScale)); err != nil {
		t.Fatalf("MulBF16: %v", err)
	}
	if pr, err = RMSNormBF16(pr, projNormW, numLayers, pliDim, eps); err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	comb, err := AddBF16(pr, perLayer[0])
	if err != nil {
		t.Fatalf("AddBF16: %v", err)
	}
	wantRef, err := MulBF16(comb, bf16ConstBytes(plDim, gemma4PerLayerCombineScale))
	if err != nil {
		t.Fatalf("MulBF16: %v", err)
	}
	if !bytes.Equal(got, wantRef) {
		t.Fatal("per-layer-input pipeline diverged from the rebuilt reference (scale/order/norm/combine)")
	}
	t.Logf("per-layer-input tensor [%d×%d]: 4-bit embed (×√pliDim) + bf16 projection (×1/√dModel, normed), ×1/√2; projW=0 ≡ embed-only anchor holds", numLayers, pliDim)
}

func TestPerLayerInputsBF16CachesProjectionWeight(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const vocabPLI, numLayers, pliDim, dModel = 4, 2, 32, 64
	const plDim = numLayers * pliDim
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 5))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))

	if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{}); err != nil {
		t.Fatalf("PerLayerInputs bf16: %v", err)
	}

	residentBufMu.Lock()
	got := len(residentBufs)
	_, ok := residentBufs[uintptr(unsafe.Pointer(&projW[0]))]
	residentBufMu.Unlock()
	if !ok {
		t.Fatalf("PerLayerInputs did not keep bf16 projection resident (resident=%d want>=1)", got)
	}
}

func TestPerLayerInputsBF16FallbackAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const vocabPLI, numLayers, pliDim, dModel = 4, 2, 32, 64
	const plDim = numLayers * pliDim
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 5))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))

	if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{}); err != nil {
		t.Fatalf("PerLayerInputs warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{}); err != nil {
			t.Fatalf("PerLayerInputs: %v", err)
		}
	})
	if allocs > 8 {
		t.Fatalf("PerLayerInputs bf16 fallback allocations = %.0f, want <= 8", allocs)
	}
}

func TestPerLayerInputsBF16ScratchAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const vocabPLI, numLayers, pliDim, dModel = 4, 2, 32, 64
	const plDim = numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 5))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))
	projView := copyView(projW)
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, projView, scratch); err != nil {
		t.Fatalf("PerLayerInputs warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, projView, scratch); err != nil {
			t.Fatalf("PerLayerInputs: %v", err)
		}
	})
	if allocs > 529 {
		t.Fatalf("PerLayerInputs scratch allocations = %.0f, want <= 529", allocs)
	}
}

func TestPerLayerInputsBF16UsesScalarScaleBuffers(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const vocabPLI, numLayers, pliDim, dModel = 4, 2, 32, 64
	const plDim = numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 5))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))

	projKey := bf16ConstKey{n: plDim, v: projScale}
	combineKey := bf16ConstKey{n: plDim, v: gemma4PerLayerCombineScale}
	bf16ConstMu.Lock()
	delete(bf16ConstCache, projKey)
	delete(bf16ConstCache, combineKey)
	bf16ConstMu.Unlock()

	if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{}); err != nil {
		t.Fatalf("PerLayerInputs bf16: %v", err)
	}

	bf16ConstMu.Lock()
	_, projectedScaleCached := bf16ConstCache[projKey]
	_, combineScaleCached := bf16ConstCache[combineKey]
	bf16ConstMu.Unlock()
	if projectedScaleCached || combineScaleCached {
		t.Fatalf("PerLayerInputs materialized plDim-wide scale buffers (projected=%v combine=%v), want scalar-bound BF16 scales", projectedScaleCached, combineScaleCached)
	}
}

func TestPerLayerInputsQuantCachesProjectionWeight(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const vocabPLI, numLayers, pliDim, dModel, groupSize, bits = 4, 2, 32, 64, 32, 4
	const plDim = numLayers * pliDim
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	proj := quantWeightFixture(t, plDim, dModel, groupSize, bits, 5)
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))

	if _, err := PerLayerInputs(embed, nil, nil, proj.Packed, proj.Scales, proj.Biases, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, groupSize, bits, 1e-5, bufView{}); err != nil {
		t.Fatalf("PerLayerInputs quant: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	got := len(residentBufs)
	_, hasPacked := residentBufs[key(proj.Packed)]
	_, hasScales := residentBufs[key(proj.Scales)]
	_, hasBiases := residentBufs[key(proj.Biases)]
	residentBufMu.Unlock()
	if !hasPacked || !hasScales || !hasBiases {
		t.Fatalf("PerLayerInputs did not keep quant projection resident (packed=%v scales=%v biases=%v resident=%d want>=3)", hasPacked, hasScales, hasBiases, got)
	}
}

func TestPerLayerInputsQuantFallbackAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const vocabPLI, numLayers, pliDim, dModel, groupSize, bits = 4, 2, 32, 64, 32, 4
	const plDim = numLayers * pliDim
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	proj := quantWeightFixture(t, plDim, dModel, groupSize, bits, 5)
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))

	if _, err := PerLayerInputs(embed, nil, nil, proj.Packed, proj.Scales, proj.Biases, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, groupSize, bits, 1e-5, bufView{}); err != nil {
		t.Fatalf("PerLayerInputs quant warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, err := PerLayerInputs(embed, nil, nil, proj.Packed, proj.Scales, proj.Biases, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, groupSize, bits, 1e-5, bufView{}); err != nil {
			t.Fatalf("PerLayerInputs quant: %v", err)
		}
	})
	if allocs > 8 {
		t.Fatalf("PerLayerInputs quant fallback allocations = %.0f, want <= 8", allocs)
	}
}

func TestPerLayerInputsQuantScratchAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const vocabPLI, numLayers, pliDim, dModel, groupSize, bits = 4, 2, 32, 64, 32, 4
	const plDim = numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	proj := quantWeightFixture(t, plDim, dModel, groupSize, bits, 5)
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	if _, err := PerLayerInputs(embed, nil, nil, proj.Packed, proj.Scales, proj.Biases, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, groupSize, bits, 1e-5, bufView{}, scratch); err != nil {
		t.Fatalf("PerLayerInputs quant warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, err := PerLayerInputs(embed, nil, nil, proj.Packed, proj.Scales, proj.Biases, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, groupSize, bits, 1e-5, bufView{}, scratch); err != nil {
			t.Fatalf("PerLayerInputs quant: %v", err)
		}
	})
	if allocs > 490 {
		t.Fatalf("PerLayerInputs quant scratch allocations = %.0f, want <= 490", allocs)
	}
}
