// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// perLayerInputGateRef rebuilds the per-layer-input gate from the parity-proven
// primitives following the metal rule — the oracle for PerLayerInputGateBF16.
func perLayerInputGateRef(t *testing.T, hNext, gateW, perLayerInput, projW, postNormW []byte, dModel, pliDim int, eps float32) []byte {
	t.Helper()
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("perLayerInputGateRef op: %v", err)
		}
		return b
	}
	gate := must(MatVecBF16(gateW, hNext, pliDim, dModel))
	multiplied := must(GeluGateMulBF16(gate, perLayerInput))
	projected := must(MatVecBF16(projW, multiplied, dModel, pliDim))
	projNormed := must(RMSNormBF16(projected, postNormW, 1, dModel, eps))
	return must(AddBF16(hNext, projNormed))
}

func TestPerLayerInputGateScratchOutputViewReusesPinnedOwnerBuffer(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, pliDim = 64, 32
	pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes: %v", err)
	}
	defer pinned.Close()

	scratch := newPerLayerInputGateScratch(dModel, pliDim)
	defer scratch.Close()

	outBuf, ok := scratch.outputView(pinned.bytes)
	if !ok {
		t.Fatal("per-layer input output view did not accept pinned caller bytes")
	}
	if got, want := outBuf.GetID(), pinned.buf.GetID(); got != want {
		t.Fatalf("per-layer input output view buffer id = %d, want pinned owner buffer %d", got, want)
	}
	if got, want := uintptr(outBuf.Contents()), uintptr(unsafe.Pointer(&pinned.bytes[0])); got != want {
		t.Fatalf("per-layer input output view pointer = %#x, want pinned backing %#x", got, want)
	}
}

// TestPerLayerInputGate gates the gemma4 per-layer-input gate. PerLayerInputGateBF16
// is byte-for-byte the independent reference that wires the gate (gate → gelu-mul →
// project → norm → residual) from primitives — proving the WIRING (each dim/op in the
// right place, the residual), since the sub-ops are gated elsewhere. A non-vacuous
// check confirms the gate genuinely modifies the layer output (out ≠ hNext). pliDim ≠
// dModel deliberately, to catch a dimension mixup in the two projections.
func TestPerLayerInputGate(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, pliDim = 256, 128
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	hNext := toBF16Bytes(mk(dModel, 29))
	gateW := toBF16Bytes(mk(pliDim*dModel, 17))
	perLayerInput := toBF16Bytes(mk(pliDim, 7))
	projW := toBF16Bytes(mk(dModel*pliDim, 23))
	postNormW := toBF16Bytes(mk(dModel, 5))

	got, err := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps)
	if err != nil {
		t.Fatalf("PerLayerInputGateBF16: %v", err)
	}
	want := perLayerInputGateRef(t, hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps)
	eqBytes(t, "PerLayerInputGateBF16", got, want)

	// non-vacuous: the gate must change the layer output (the projected, normed
	// per-layer contribution is summed in, not a no-op).
	same := len(got) == len(hNext)
	for i := range got {
		if i < len(hNext) && got[i] != hNext[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("PerLayerInputGateBF16 output equals hNext unchanged — the gate did not contribute")
	}
	t.Logf("per-layer-input gate (dModel %d, pliDim %d): ≡ composed reference and modifies the layer output", dModel, pliDim)
}

func TestPerLayerInputGateBF16CachesWeights(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, pliDim = 64, 32
	const eps = float32(1e-6)
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 17))
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	projW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 23))
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))

	if _, err := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps); err != nil {
		t.Fatalf("PerLayerInputGateBF16: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	got := len(residentBufs)
	_, hasGate := residentBufs[key(gateW)]
	_, hasProj := residentBufs[key(projW)]
	residentBufMu.Unlock()
	if !hasGate || !hasProj {
		t.Fatalf("PerLayerInputGateBF16 did not keep fixed weights resident (gate=%v proj=%v resident=%d want>=2)", hasGate, hasProj, got)
	}
}

func TestPerLayerInputGateQuantCachesWeights(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, pliDim, groupSize, bits = 64, 32, 32, 4
	const eps = float32(1e-6)
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gate := quantWeightFixture(t, pliDim, dModel, groupSize, bits, 17)
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	proj := quantWeightFixture(t, dModel, pliDim, groupSize, bits, 23)
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))

	if _, err := PerLayerInputGateQuant(hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, eps); err != nil {
		t.Fatalf("PerLayerInputGateQuant: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	weights := []struct {
		name string
		buf  []byte
	}{
		{"gate packed", gate.Packed},
		{"gate scales", gate.Scales},
		{"gate biases", gate.Biases},
		{"proj packed", proj.Packed},
		{"proj scales", proj.Scales},
		{"proj biases", proj.Biases},
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
		t.Fatalf("PerLayerInputGateQuant did not keep fixed quant weights resident (missing %v resident=%d want>=6)", missing, got)
	}
}

func TestPerLayerInputGateAllocationBudgets(t *testing.T) {
	requireNativeRuntime(t)

	t.Run("bf16", func(t *testing.T) {
		const dModel, pliDim = 64, 32
		const eps = float32(1e-6)
		hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
		gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 17))
		perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
		projW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 23))
		postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
		if _, err := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps); err != nil {
			t.Fatalf("PerLayerInputGateBF16 warmup: %v", err)
		}
		allocs := testing.AllocsPerRun(5, func() {
			if _, err := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps); err != nil {
				t.Fatalf("PerLayerInputGateBF16: %v", err)
			}
		})
		if allocs > 20 {
			t.Fatalf("PerLayerInputGateBF16 allocations = %.0f, want <= 20", allocs)
		}
	})

	t.Run("quant", func(t *testing.T) {
		const dModel, pliDim, groupSize, bits = 64, 32, 32, 4
		const eps = float32(1e-6)
		hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
		gate := quantWeightFixture(t, pliDim, dModel, groupSize, bits, 17)
		perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
		proj := quantWeightFixture(t, dModel, pliDim, groupSize, bits, 23)
		postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
		if _, err := PerLayerInputGateQuant(hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, eps); err != nil {
			t.Fatalf("PerLayerInputGateQuant warmup: %v", err)
		}
		allocs := testing.AllocsPerRun(5, func() {
			if _, err := PerLayerInputGateQuant(hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, eps); err != nil {
				t.Fatalf("PerLayerInputGateQuant: %v", err)
			}
		})
		if allocs > 20 {
			t.Fatalf("PerLayerInputGateQuant allocations = %.0f, want <= 20", allocs)
		}
	})
}

func TestPerLayerInputGateScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small := getPerLayerInputGateScratch(64, 32)
	putPerLayerInputGateScratch(small)
	large := getPerLayerInputGateScratch(128, 64)
	putPerLayerInputGateScratch(large)
	forceNativeGC()
	forceNativeGC()
	gotSmall := getPerLayerInputGateScratch(64, 32)
	defer putPerLayerInputGateScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("gate scratch pool evicted the 64x32 scratch after using a 128x64 scratch")
	}
	gotLarge := getPerLayerInputGateScratch(128, 64)
	defer putPerLayerInputGateScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("gate scratch pool evicted the 128x64 scratch after reusing the 64x32 scratch")
	}
}

func TestPerLayerInputGateScratchInputBuffersUseCallerBackingAfterWarmup(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, pliDim = 64, 32
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	scratch := getPerLayerInputGateScratch(dModel, pliDim)
	defer scratch.Close()

	var hBuf, perLayerBuf metal.MTLBuffer
	for i := range 3 {
		var err error
		hBuf, perLayerBuf, err = scratch.inputBuffers(hNext, perLayerInput)
		if err != nil {
			t.Fatalf("scratch.inputBuffers warmup %d: %v", i, err)
		}
	}
	if got, want := uintptr(hBuf.Contents()), uintptr(unsafe.Pointer(&hNext[0])); got != want {
		t.Fatalf("hNext buffer pointer = %#x, want caller backing %#x", got, want)
	}
	if got, want := uintptr(perLayerBuf.Contents()), uintptr(unsafe.Pointer(&perLayerInput[0])); got != want {
		t.Fatalf("perLayerInput buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reusedH, reusedPerLayer, err := scratch.inputBuffers(hNext, perLayerInput)
	if err != nil {
		t.Fatalf("scratch.inputBuffers reused: %v", err)
	}
	if reusedH.GetID() != hBuf.GetID() || reusedPerLayer.GetID() != perLayerBuf.GetID() {
		t.Fatal("inputBuffers did not reuse cached no-copy input views")
	}
}

func TestPerLayerInputGateBF16EncodedWritesDirectlyToOutput(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, pliDim = 64, 32
	const eps = float32(1e-6)
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 17))
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	projW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 23))
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	scratch := getPerLayerInputGateScratch(dModel, pliDim)
	defer putPerLayerInputGateScratch(scratch)
	hBuf, perLayerBuf, err := scratch.inputBuffers(hNext, perLayerInput)
	if err != nil {
		t.Fatalf("scratch.inputBuffers: %v", err)
	}
	scratchOut := unsafe.Slice((*byte)(scratch.out.Contents()), dModel*bf16Size)
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratchOut))
	copy(scratchOut, sentinel)

	out := make([]byte, dModel*bf16Size)
	err = perLayerInputGateBF16EncodedInto(
		scratch, out, hBuf, residentBytes(gateW), perLayerBuf, residentBytes(projW), residentBytes(postNormW),
		dModel, pliDim, eps,
	)
	if err != nil {
		t.Fatalf("perLayerInputGateBF16Encoded: %v", err)
	}
	want := perLayerInputGateRef(t, hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps)
	eqBytes(t, "perLayerInputGateBF16Encoded direct output", out, want)
	if !bytes.Equal(scratchOut, sentinel) {
		t.Fatal("perLayerInputGateBF16Encoded wrote through pooled scratch output instead of caller output")
	}
}

func TestPerLayerInputGateQuantEncodedWritesDirectlyToOutput(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, pliDim, groupSize, bits = 64, 32, 32, 4
	const eps = float32(1e-6)
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gate := quantWeightFixture(t, pliDim, dModel, groupSize, bits, 17)
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	proj := quantWeightFixture(t, dModel, pliDim, groupSize, bits, 23)
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	scratch := getPerLayerInputGateScratch(dModel, pliDim)
	defer putPerLayerInputGateScratch(scratch)
	hBuf, perLayerBuf, err := scratch.inputBuffers(hNext, perLayerInput)
	if err != nil {
		t.Fatalf("scratch.inputBuffers: %v", err)
	}
	scratchOut := unsafe.Slice((*byte)(scratch.out.Contents()), dModel*bf16Size)
	sentinel := bytes.Repeat([]byte{0x5a}, len(scratchOut))
	copy(scratchOut, sentinel)
	gatePacked, gateScales, gateBiases := quantWeightViews(gate)
	projPacked, projScales, projBiases := quantWeightViews(proj)

	out := make([]byte, dModel*bf16Size)
	err = perLayerInputGateQuantEncodedInto(
		scratch, out, hBuf, gatePacked, gateScales, gateBiases, perLayerBuf, projPacked, projScales, projBiases,
		residentBytes(postNormW), dModel, pliDim, groupSize, bits, groupSize, bits, eps,
	)
	if err != nil {
		t.Fatalf("perLayerInputGateQuantEncoded: %v", err)
	}
	if len(out) != dModel*bf16Size {
		t.Fatalf("perLayerInputGateQuantEncoded length = %d, want %d", len(out), dModel*bf16Size)
	}
	if !bytes.Equal(scratchOut, sentinel) {
		t.Fatal("perLayerInputGateQuantEncoded wrote through pooled scratch output instead of caller output")
	}
}

func TestPerLayerInputGateBF16IntoReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, pliDim = 64, 32
	const eps = float32(1e-6)
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 17))
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	projW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 23))
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])

	got, err := perLayerInputGateBF16Into(out, hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps)
	if err != nil {
		t.Fatalf("perLayerInputGateBF16Into: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("perLayerInputGateBF16Into did not reuse caller-owned output backing")
	}
	want, err := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps)
	if err != nil {
		t.Fatalf("PerLayerInputGateBF16 reference: %v", err)
	}
	eqBytes(t, "perLayerInputGateBF16Into", got, want)
}

func TestPerLayerInputGateQuantIntoReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, pliDim, groupSize, bits = 64, 32, 32, 4
	const eps = float32(1e-6)
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gate := quantWeightFixture(t, pliDim, dModel, groupSize, bits, 17)
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	proj := quantWeightFixture(t, dModel, pliDim, groupSize, bits, 23)
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])

	got, err := perLayerInputGateQuantInto(out, hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, eps)
	if err != nil {
		t.Fatalf("perLayerInputGateQuantInto: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("perLayerInputGateQuantInto did not reuse caller-owned output backing")
	}
	want, err := PerLayerInputGateQuant(hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, eps)
	if err != nil {
		t.Fatalf("PerLayerInputGateQuant reference: %v", err)
	}
	eqBytes(t, "perLayerInputGateQuantInto", got, want)
}

func TestPerLayerInputGateIntoEdgeCases(t *testing.T) {
	requireNativeRuntime(t)

	t.Run("bf16 zero PLI copies hNext into caller output", func(t *testing.T) {
		const dModel, pliDim = 16, 0
		hNext := toBF16Bytes(syntheticFloat32(dModel, 41))
		out := make([]byte, dModel*bf16Size)
		got, err := perLayerInputGateBF16Into(out, hNext, nil, nil, nil, make([]byte, dModel*bf16Size), dModel, pliDim, 1e-6)
		if err != nil {
			t.Fatalf("perLayerInputGateBF16Into: %v", err)
		}
		if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
			t.Fatal("zero-PLI BF16 gate did not reuse caller output")
		}
		eqBytes(t, "zero-PLI BF16 gate", got, hNext)
	})

	t.Run("quant zero PLI copies hNext into caller output", func(t *testing.T) {
		const dModel, pliDim, groupSize, bits = 16, 0, 32, 4
		hNext := toBF16Bytes(syntheticFloat32(dModel, 43))
		out := make([]byte, dModel*bf16Size)
		got, err := perLayerInputGateQuantInto(out, hNext, QuantWeight{}, nil, QuantWeight{}, make([]byte, dModel*bf16Size), dModel, pliDim, groupSize, bits, 1e-6)
		if err != nil {
			t.Fatalf("perLayerInputGateQuantInto: %v", err)
		}
		if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
			t.Fatal("zero-PLI quant gate did not reuse caller output")
		}
		eqBytes(t, "zero-PLI quant gate", got, hNext)
	})

	t.Run("bf16 rejects bad hNext length", func(t *testing.T) {
		if _, err := perLayerInputGateBF16Into(nil, []byte{1}, nil, nil, nil, nil, 16, 0, 1e-6); err == nil {
			t.Fatal("perLayerInputGateBF16Into accepted bad hNext length")
		}
	})

	t.Run("quant rejects bad hNext length", func(t *testing.T) {
		if _, err := perLayerInputGateQuantInto(nil, []byte{1}, QuantWeight{}, nil, QuantWeight{}, nil, 16, 0, 32, 4, 1e-6); err == nil {
			t.Fatal("perLayerInputGateQuantInto accepted bad hNext length")
		}
	})
}
