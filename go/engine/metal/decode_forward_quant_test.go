// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"unsafe"
)

func TestDecodeForwardQuantProducesTokenOutputs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}

	got, err := DecodeForwardQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant: %v", err)
	}
	if len(got) != len(inputs) {
		t.Fatalf("DecodeForwardQuant returned %d tokens, want %d", len(got), len(inputs))
	}
	for i := range got {
		if len(got[i]) != dModel*bf16Size {
			t.Fatalf("DecodeForwardQuant token %d has %d bytes, want %d", i, len(got[i]), dModel*bf16Size)
		}
	}
}

func TestDecodeForwardQuantKeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}

	if _, err := DecodeForwardQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
		t.Fatalf("DecodeForwardQuant: %v", err)
	}

	layer := layers[0]
	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	weights := []struct {
		name string
		buf  []byte
	}{
		{"attnNorm", layer.AttnNormW},
		{"mlpNorm", layer.MLPNormW},
		{"q.packed", layer.Q.Packed}, {"q.scales", layer.Q.Scales}, {"q.biases", layer.Q.Biases},
		{"k.packed", layer.K.Packed}, {"k.scales", layer.K.Scales}, {"k.biases", layer.K.Biases},
		{"v.packed", layer.V.Packed}, {"v.scales", layer.V.Scales}, {"v.biases", layer.V.Biases},
		{"o.packed", layer.O.Packed}, {"o.scales", layer.O.Scales}, {"o.biases", layer.O.Biases},
		{"gate.packed", layer.Gate.Packed}, {"gate.scales", layer.Gate.Scales}, {"gate.biases", layer.Gate.Biases},
		{"up.packed", layer.Up.Packed}, {"up.scales", layer.Up.Scales}, {"up.biases", layer.Up.Biases},
		{"down.packed", layer.Down.Packed}, {"down.scales", layer.Down.Scales}, {"down.biases", layer.Down.Biases},
	}

	residentBufMu.Lock()
	got := len(residentBufs)
	missing := make([]string, 0)
	for _, weight := range weights {
		if _, ok := residentBufs[key(weight.buf)]; !ok {
			missing = append(missing, weight.name)
		}
	}
	residentBufMu.Unlock()

	if len(missing) != 0 {
		t.Fatalf("DecodeForwardQuant did not keep fixed weights resident (missing=%v resident=%d want>=%d)", missing, got, len(weights))
	}
}

func TestDecodeForwardQuantAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	if _, err := DecodeForwardQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
		t.Fatalf("DecodeForwardQuant warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardQuant: %v", forwardErr)
	}
	if allocs > 45 {
		t.Fatalf("DecodeForwardQuant allocations = %.0f, want <= 45", allocs)
	}
}

func TestDecodeForwardQuantHonoursPerWeightGeometry(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const mlpGroupSize, mlpBits = 32, 8
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	layer.Gate = quantWeightFixture(t, dFF, dModel, mlpGroupSize, mlpBits, 20)
	layer.Up = quantWeightFixture(t, dFF, dModel, mlpGroupSize, mlpBits, 22)
	layer.Down = quantWeightFixture(t, dModel, dFF, mlpGroupSize, mlpBits, 26)

	got, err := DecodeForwardQuant(inputs, []QuantizedLayerWeights{layer}, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant with per-weight MLP geometry: %v", err)
	}
	if len(got) != len(inputs) {
		t.Fatalf("DecodeForwardQuant returned %d tokens, want %d", len(got), len(inputs))
	}
	for i := range got {
		if len(got[i]) != dModel*bf16Size {
			t.Fatalf("DecodeForwardQuant token %d has %d bytes, want %d", i, len(got[i]), dModel*bf16Size)
		}
	}
}

func TestDecodeForwardQuantRejectsUnsetQuantGeometry(t *testing.T) {
	requireNativeRuntime(t)

	inputs := decodeInputsFixture(1, 64)
	layers := []QuantizedLayerWeights{{AttnNormW: toBF16Bytes(syntheticFloat32(64, 3)), MLPNormW: toBF16Bytes(syntheticFloat32(64, 5))}}
	if _, err := DecodeForwardQuant(inputs, layers, 64, 1, 1, 64, 1, 128, 10000, 0.125, 1e-5); err == nil {
		t.Fatal("expected DecodeForwardQuant to reject unset GroupSize/Bits")
	}
}
