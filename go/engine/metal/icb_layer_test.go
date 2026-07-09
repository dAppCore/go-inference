// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
)

func TestDecodeLayerICBMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	want, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("DecodeLayer: %v", err)
	}
	got, err := DecodeLayerICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("DecodeLayerICB: %v", err)
	}
	eqBytes(t, "DecodeLayerICB", got, want)
}

func TestDecodeLayerICBIntoUsesCallerBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	want, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("DecodeLayer: %v", err)
	}

	scratch, err := getQMVBF16Scratch(dModel, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := make([]byte, dModel*bf16Size)
	for i := range sentinel {
		sentinel[i] = 0x7f
	}
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	out := make([]byte, dModel*bf16Size)
	got, err := DecodeLayerICBInto(out, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("DecodeLayerICBInto: %v", err)
	}
	if len(got) == 0 || unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("DecodeLayerICBInto did not return the caller output backing")
	}
	eqBytes(t, "DecodeLayerICBInto", got, want)

	reused, err := getQMVBF16Scratch(dModel, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch reused: %v", err)
	}
	defer putQMVBF16Scratch(reused)
	if reused.out != scratch.out {
		t.Fatal("DecodeLayerICBInto did not return the seeded scratch to the pool")
	}
	if !bytes.Equal(reused.out.bytes[:len(sentinel)], sentinel) {
		t.Fatal("DecodeLayerICBInto still staged output through pooled scratch")
	}
}

func TestDecodeLayerICBKeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))

	if _, err := DecodeLayerICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1); err != nil {
		t.Fatalf("DecodeLayerICB: %v", err)
	}

	assertDecodeLayerWeightsResident(t, layer)
}

func TestDecodeLayerICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 4, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	if _, err := DecodeLayerICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1); err != nil {
		t.Fatalf("DecodeLayerICB warmup: %v", err)
	}

	var decodeErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, decodeErr = DecodeLayerICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1)
	})
	if decodeErr != nil {
		t.Fatalf("DecodeLayerICB: %v", decodeErr)
	}
	if allocs > 166 {
		t.Fatalf("DecodeLayerICB allocations = %.0f, want <= 166", allocs)
	}
}

func TestDecodeTokenICBOneLayerMatchesDecodeLayer(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	want, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("DecodeLayer: %v", err)
	}
	got, err := DecodeTokenICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 1, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("DecodeTokenICB: %v", err)
	}
	eqBytes(t, "DecodeTokenICB", got, want)
}

func TestDecodeTokenICBIntoUsesCallerBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	want, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("DecodeLayer: %v", err)
	}

	scratch, err := getQMVBF16Scratch(dModel, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := make([]byte, dModel*bf16Size)
	for i := range sentinel {
		sentinel[i] = 0x7e
	}
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	out := make([]byte, dModel*bf16Size)
	got, err := DecodeTokenICBInto(out, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 1, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("DecodeTokenICBInto: %v", err)
	}
	if len(got) == 0 || unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("DecodeTokenICBInto did not return the caller output backing")
	}
	eqBytes(t, "DecodeTokenICBInto", got, want)

	reused, err := getQMVBF16Scratch(dModel, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch reused: %v", err)
	}
	defer putQMVBF16Scratch(reused)
	if reused.out != scratch.out {
		t.Fatal("DecodeTokenICBInto did not return the seeded scratch to the pool")
	}
	if !bytes.Equal(reused.out.bytes[:len(sentinel)], sentinel) {
		t.Fatal("DecodeTokenICBInto still staged output through pooled scratch")
	}
}

func TestDecodeTokenICBKeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 2, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))

	if _, err := DecodeTokenICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, 1, base, scale, offset, eps, 1); err != nil {
		t.Fatalf("DecodeTokenICB: %v", err)
	}

	assertDecodeLayerWeightsResident(t, layer)
}

func TestDecodeTokenICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers = 64, 1, 1, 64, 4, 128, 1
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	if _, err := DecodeTokenICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, 1); err != nil {
		t.Fatalf("DecodeTokenICB warmup: %v", err)
	}

	var decodeErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, decodeErr = DecodeTokenICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, 1)
	})
	if decodeErr != nil {
		t.Fatalf("DecodeTokenICB: %v", decodeErr)
	}
	if allocs > 166 {
		t.Fatalf("DecodeTokenICB allocations = %.0f, want <= 166", allocs)
	}
}

func assertDecodeLayerWeightsResident(t *testing.T, layer DecodeLayerWeights) {
	t.Helper()

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
		t.Fatalf("ICB decode layer did not keep fixed weights resident (missing=%v resident=%d want>=7)", missing, got)
	}
}
