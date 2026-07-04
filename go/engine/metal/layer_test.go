// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
)

func TestDecodeLayerMatchesAttentionThenMLP(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 29))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 31))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 37))

	got, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("DecodeLayer: %v", err)
	}
	h, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("AttentionBlock: %v", err)
	}
	want, err := MLPBlockBF16(h, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MLPBlockBF16: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("DecodeLayer = %v, want AttentionBlock+MLPBlockBF16 %v", bf16Floats(got), bf16Floats(want))
	}
}

func TestDecodeLayerIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 4, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	want, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("DecodeLayer reference: %v", err)
	}
	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVBF16Scratch(dModel, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := DecodeLayerInto(out, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("DecodeLayerInto: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("DecodeLayerInto did not reuse caller-owned output backing")
	}
	eqBytes(t, "DecodeLayerInto", got, want)

	scratch, err = getQMVBF16Scratch(dModel, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("DecodeLayerInto wrote through pooled scratch output instead of caller output")
	}
}

func TestDecodeLayerRejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := DecodeLayer(nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 64, 1, 1, 64, 1, 128, 10000, 0.125, 0, 1e-5); err == nil {
		t.Fatal("expected DecodeLayer to reject missing inputs and weights")
	}
}

func TestDecodeLayerKeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 29))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 31))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 37))

	if _, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps); err != nil {
		t.Fatalf("DecodeLayer: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	got := len(residentBufs)
	weights := map[string][]byte{
		"attnNorm": layer.AttnNormW,
		"wQ":       layer.WQ,
		"wO":       layer.WO,
		"mlpNorm":  layer.MLPNormW,
		"wGate":    layer.WGate,
		"wUp":      layer.WUp,
		"wDown":    layer.WDown,
	}
	missing := make([]string, 0)
	for name, weight := range weights {
		if _, ok := residentBufs[key(weight)]; !ok {
			missing = append(missing, name)
		}
	}
	residentBufMu.Unlock()

	if len(missing) != 0 {
		t.Fatalf("DecodeLayer did not keep fixed weights resident (missing=%v resident=%d want>=7)", missing, got)
	}
}

func TestDecodeLayerResidualScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small := getDecodeLayerResidualScratch(96)
	putDecodeLayerResidualScratch(small)
	large := getDecodeLayerResidualScratch(160)
	putDecodeLayerResidualScratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall := getDecodeLayerResidualScratch(96)
	defer putDecodeLayerResidualScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("DecodeLayer residual scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge := getDecodeLayerResidualScratch(160)
	defer putDecodeLayerResidualScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("DecodeLayer residual scratch pool evicted the large scratch after reusing the small scratch")
	}
}

func TestDecodeLayerAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 4, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	if _, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps); err != nil {
		t.Fatalf("DecodeLayer warmup: %v", err)
	}

	var decodeErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, decodeErr = DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	})
	if decodeErr != nil {
		t.Fatalf("DecodeLayer: %v", decodeErr)
	}
	if allocs > 10 {
		t.Fatalf("DecodeLayer allocations = %.0f, want <= 10", allocs)
	}
}
