// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
)

func TestDecodeForwardICBQuantMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}

	want, err := DecodeForwardQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant: %v", err)
	}
	got, err := DecodeForwardICBQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardICBQuant: %v", err)
	}
	for i := range want {
		eqBytes(t, "DecodeForwardICBQuant token", got[i], want[i])
	}
}

func TestDecodeForwardICBQuantIntoReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	want, err := DecodeForwardICBQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardICBQuant reference: %v", err)
	}
	out := [][]byte{
		bytes.Repeat([]byte{0xa5}, dModel*bf16Size),
		bytes.Repeat([]byte{0x5a}, dModel*bf16Size),
	}
	ptrs := []unsafe.Pointer{unsafe.Pointer(&out[0][0]), unsafe.Pointer(&out[1][0])}

	got, err := DecodeForwardICBQuantInto(out, inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardICBQuantInto: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("DecodeForwardICBQuantInto returned %d outputs, want %d", len(got), len(want))
	}
	for tok := range want {
		if len(got[tok]) != dModel*bf16Size || unsafe.Pointer(&got[tok][0]) != ptrs[tok] {
			t.Fatalf("DecodeForwardICBQuantInto token %d did not reuse caller-owned output backing", tok)
		}
		eqBytes(t, "DecodeForwardICBQuantInto token", got[tok], want[tok])
	}
}

func TestDecodeForwardICBQuantAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)}
	if _, err := DecodeForwardICBQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
		t.Fatalf("DecodeForwardICBQuant warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardICBQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardICBQuant: %v", forwardErr)
	}
	if allocs > 255 {
		t.Fatalf("DecodeForwardICBQuant allocations = %.0f, want <= 255", allocs)
	}
}

func TestDecodeForwardICBQuantKeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const groupSize, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layer := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, 3)
	layers := []QuantizedLayerWeights{layer}

	if _, err := DecodeForwardICBQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
		t.Fatalf("DecodeForwardICBQuant: %v", err)
	}

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
		t.Fatalf("DecodeForwardICBQuant did not keep fixed weights resident (missing=%v resident=%d want>=%d)", missing, got, len(weights))
	}
}

func TestDecodeForwardICBQuantHonoursPerWeightGeometry(t *testing.T) {
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
	layers := []QuantizedLayerWeights{layer}

	want, err := DecodeForwardQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant with per-weight MLP geometry: %v", err)
	}
	got, err := DecodeForwardICBQuant(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardICBQuant with per-weight MLP geometry: %v", err)
	}
	for i := range want {
		eqBytes(t, "DecodeForwardICBQuant mixed geometry token", got[i], want[i])
	}
}
