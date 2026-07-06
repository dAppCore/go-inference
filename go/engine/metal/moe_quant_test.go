// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"
)

func quantMoEExpertsFixture(tb testing.TB, numExperts, dModel, dFF, groupSize, bits int) (QuantWeight, QuantWeight, QuantWeight) {
	tb.Helper()
	buildBatched := func(outDim, inDim, saltBase int) QuantWeight {
		var packed, scales, biases []byte
		for e := range numExperts {
			p, s, b := quantizeProj(tb, outDim, inDim, groupSize, bits, saltBase+e*7)
			packed, scales, biases = append(packed, p...), append(scales, s...), append(biases, b...)
		}
		return QuantWeight{Packed: packed, Scales: scales, Biases: biases, GroupSize: groupSize, Bits: bits}
	}
	return buildBatched(dFF, dModel, 3), buildBatched(dFF, dModel, 51), buildBatched(dModel, dFF, 91)
}

// TestMoEExpertsQuant gates the 4-bit batched experts: MoEExpertsQuant over a SwitchGLU-style
// batched quant tensor must equal a composed reference (per selected expert: QMVBF16 gate/up →
// GeluGateMulBF16 → QMVBF16 down, weighted-summed) byte-for-byte, and differ for a different
// expert selection (the routing is genuinely consumed).
func TestMoEExpertsQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, dFF, gs, bits = 4, 2, 64, 128, 32, 4
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.02
		}
		return s
	}
	// batch each expert's [outDim × inDim] quant weight into one tensor (the SwitchGLU layout).
	buildBatched := func(outDim, inDim, saltBase int) QuantWeight {
		var p, s, b []byte
		for e := range numExperts {
			pe, se, be := quantizeProj(t, outDim, inDim, gs, bits, saltBase+e*7)
			p, s, b = append(p, pe...), append(s, se...), append(b, be...)
		}
		return QuantWeight{Packed: p, Scales: s, Biases: b}
	}
	gate := buildBatched(dFF, dModel, 3)
	up := buildBatched(dFF, dModel, 51)
	down := buildBatched(dModel, dFF, 91)
	x := toBF16Bytes(mk(dModel, 5))
	idx := []int32{2, 0}
	weights := toBF16Bytes([]float32{0.7, 0.3})

	got, err := MoEExpertsQuant(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, gs, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuant: %v", err)
	}

	gp, gsz := dFF*dModel*bits/8, dFF*(dModel/gs)*bf16Size
	dp, dsz := dModel*dFF*bits/8, dModel*(dFF/gs)*bf16Size
	must := func(b []byte, e error) []byte {
		t.Helper()
		if e != nil {
			t.Fatalf("ref op: %v", e)
		}
		return b
	}
	var acc []byte
	for i, e := range idx {
		ee := int(e)
		ge := must(QMVBF16(x, gate.Packed[ee*gp:(ee+1)*gp], gate.Scales[ee*gsz:(ee+1)*gsz], gate.Biases[ee*gsz:(ee+1)*gsz], dFF, dModel, gs, bits))
		ue := must(QMVBF16(x, up.Packed[ee*gp:(ee+1)*gp], up.Scales[ee*gsz:(ee+1)*gsz], up.Biases[ee*gsz:(ee+1)*gsz], dFF, dModel, gs, bits))
		gg := must(GeluGateMulBF16(ge, ue))
		de := must(QMVBF16(gg, down.Packed[ee*dp:(ee+1)*dp], down.Scales[ee*dsz:(ee+1)*dsz], down.Biases[ee*dsz:(ee+1)*dsz], dModel, dFF, gs, bits))
		scaled := must(MulBF16(de, scalarFillBF16(weights[i*bf16Size:(i+1)*bf16Size], dModel)))
		if i == 0 {
			acc = scaled
		} else {
			acc = must(AddBF16(acc, scaled))
		}
	}
	if !bytes.Equal(got, acc) {
		t.Fatal("MoEExpertsQuant != composed quant reference")
	}
	// non-vacuous: a different expert selection changes the result.
	other, err := MoEExpertsQuant(x, []int32{1, 3}, weights, gate, up, down, numExperts, topK, dModel, dFF, gs, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuant(other): %v", err)
	}
	if bytes.Equal(got, other) {
		t.Fatal("different expert selection produced the same output (routing not consumed)")
	}
	t.Logf("4-bit batched experts: topK SwiGLU over the SwitchGLU tensor ≡ composed QMV reference, selection-sensitive")
}

func TestMoEExpertsQuantBindsWholeBatchedExpertTensors(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, gs, bits = 4, 2, 64, 128, 32, 4
	buildBatched := func(outDim, inDim, saltBase int) QuantWeight {
		var p, s, b []byte
		for e := range numExperts {
			pe, se, be := quantizeProj(t, outDim, inDim, gs, bits, saltBase+e*7)
			p, s, b = append(p, pe...), append(s, se...), append(b, be...)
		}
		return QuantWeight{Packed: p, Scales: s, Biases: b}
	}
	gate := buildBatched(dFF, dModel, 3)
	up := buildBatched(dFF, dModel, 51)
	down := buildBatched(dModel, dFF, 91)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	idx := []int32{1, 3}
	weights := toBF16Bytes([]float32{0.7, 0.3})

	if _, err := MoEExpertsQuant(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, gs, bits); err != nil {
		t.Fatalf("MoEExpertsQuant: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	whole := []struct {
		name string
		buf  []byte
	}{
		{"gate packed", gate.Packed}, {"gate scales", gate.Scales}, {"gate biases", gate.Biases},
		{"up packed", up.Packed}, {"up scales", up.Scales}, {"up biases", up.Biases},
		{"down packed", down.Packed}, {"down scales", down.Scales}, {"down biases", down.Biases},
	}
	gp, gsz := dFF*dModel*bits/8, dFF*(dModel/gs)*bf16Size
	dp, dsz := dModel*dFF*bits/8, dModel*(dFF/gs)*bf16Size
	selectedSlices := make([]uintptr, 0, len(idx)*len(whole))
	for _, e := range idx {
		ee := int(e)
		selectedSlices = append(selectedSlices,
			key(gate.Packed[ee*gp:(ee+1)*gp]), key(gate.Scales[ee*gsz:(ee+1)*gsz]), key(gate.Biases[ee*gsz:(ee+1)*gsz]),
			key(up.Packed[ee*gp:(ee+1)*gp]), key(up.Scales[ee*gsz:(ee+1)*gsz]), key(up.Biases[ee*gsz:(ee+1)*gsz]),
			key(down.Packed[ee*dp:(ee+1)*dp]), key(down.Scales[ee*dsz:(ee+1)*dsz]), key(down.Biases[ee*dsz:(ee+1)*dsz]),
		)
	}

	residentBufMu.Lock()
	got := len(residentBufs)
	missingWhole := make([]string, 0)
	for _, item := range whole {
		if _, ok := residentBufs[key(item.buf)]; !ok {
			missingWhole = append(missingWhole, item.name)
		}
	}
	sliceHits := 0
	for _, k := range selectedSlices {
		if _, ok := residentBufs[k]; ok {
			sliceHits++
		}
	}
	residentBufMu.Unlock()

	if len(missingWhole) != 0 {
		t.Fatalf("MoEExpertsQuant did not keep whole batched expert tensors resident (missing=%v resident=%d)", missingWhole, got)
	}
	if sliceHits != 0 {
		t.Fatalf("MoEExpertsQuant kept %d selected expert slices resident; want whole batched tensors with qmv offsets", sliceHits)
	}
	if got > len(whole) {
		t.Fatalf("resident quant expert buffers = %d, want at most %d whole tensors", got, len(whole))
	}
}

func TestMoEExpertsQuantAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, groupSize, bits = 4, 2, 64, 128, 32, 4
	gate, up, down := quantMoEExpertsFixture(t, numExperts, dModel, dFF, groupSize, bits)
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	if _, err := MoEExpertsQuant(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, groupSize, bits); err != nil {
		t.Fatalf("MoEExpertsQuant warmup: %v", err)
	}

	var expertsErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, expertsErr = MoEExpertsQuant(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, groupSize, bits)
	})
	if expertsErr != nil {
		t.Fatalf("MoEExpertsQuant: %v", expertsErr)
	}
	if allocs > 30 {
		t.Fatalf("MoEExpertsQuant allocations = %.0f, want <= 30", allocs)
	}
}

func TestMoEExpertsQuantIntoWritesDirectlyToCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, groupSize, bits = 4, 2, 64, 128, 32, 4
	gate, up, down := quantMoEExpertsFixture(t, numExperts, dModel, dFF, groupSize, bits)
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	want, err := MoEExpertsQuant(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, groupSize, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuant: %v", err)
	}

	scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
	if err != nil {
		t.Fatalf("getMoEExpertsScratch: %v", err)
	}
	accBytes := unsafe.Slice((*byte)(scratch.acc.Contents()), dModel*bf16Size)
	sentinel := bytes.Repeat([]byte{0x4e}, len(accBytes))
	copy(accBytes, sentinel)
	putMoEExpertsScratch(scratch)

	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	got, err := MoEExpertsQuantInto(out, x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, groupSize, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuantInto: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MoEExpertsQuantInto did not reuse caller-owned output backing")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("MoEExpertsQuantInto != default split quant expert path")
	}

	scratch, err = getMoEExpertsScratch(dModel, dFF, topK)
	if err != nil {
		t.Fatalf("getMoEExpertsScratch after call: %v", err)
	}
	defer putMoEExpertsScratch(scratch)
	accBytes = unsafe.Slice((*byte)(scratch.acc.Contents()), dModel*bf16Size)
	if !bytes.Equal(accBytes, sentinel) {
		t.Fatal("MoEExpertsQuantInto wrote through pooled accumulator instead of caller output")
	}
}

func TestMoEExpertsQuantFusedGateUpMatchesSplitExperts(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, gs, bits = 4, 2, 64, 128, 32, 4
	buildBatched := func(outDim, inDim, saltBase int) QuantWeight {
		var p, s, b []byte
		for e := range numExperts {
			pe, se, be := quantizeProj(t, outDim, inDim, gs, bits, saltBase+e*7)
			p, s, b = append(p, pe...), append(s, se...), append(b, be...)
		}
		return QuantWeight{Packed: p, Scales: s, Biases: b, GroupSize: gs, Bits: bits}
	}
	gate := buildBatched(dFF, dModel, 3)
	up := buildBatched(dFF, dModel, 51)
	down := buildBatched(dModel, dFF, 91)
	gateUp := fusedGateUpQuantForBench(gate, up, numExperts, dFF, dModel, gs, bits)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	idx := []int32{1, 3}
	weights := toBF16Bytes([]float32{0.7, 0.3})

	want, err := MoEExpertsQuant(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, gs, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuant: %v", err)
	}
	resetResidentBufsForTest()
	got, err := MoEExpertsQuantFusedGateUp(x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, gs, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuantFusedGateUp: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("MoEExpertsQuantFusedGateUp != split gate/up expert path")
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	whole := []struct {
		name string
		buf  []byte
	}{
		{"gate_up packed", gateUp.Packed}, {"gate_up scales", gateUp.Scales}, {"gate_up biases", gateUp.Biases},
		{"down packed", down.Packed}, {"down scales", down.Scales}, {"down biases", down.Biases},
	}
	residentBufMu.Lock()
	gotResident := len(residentBufs)
	missing := []string{}
	for _, item := range whole {
		if _, ok := residentBufs[key(item.buf)]; !ok {
			missing = append(missing, item.name)
		}
	}
	residentBufMu.Unlock()
	if len(missing) != 0 {
		t.Fatalf("fused gate_up resident tensors missing %v (resident=%d)", missing, gotResident)
	}
	if gotResident > len(whole) {
		t.Fatalf("resident fused expert buffers = %d, want at most %d whole tensors", gotResident, len(whole))
	}
}

func TestMoEExpertsQuantFusedGateUpAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, groupSize, bits = 4, 2, 64, 128, 32, 4
	gate, up, down := quantMoEExpertsFixture(t, numExperts, dModel, dFF, groupSize, bits)
	gateUp := fusedGateUpQuantForBench(gate, up, numExperts, dFF, dModel, groupSize, bits)
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	if _, err := MoEExpertsQuantFusedGateUp(x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, groupSize, bits); err != nil {
		t.Fatalf("MoEExpertsQuantFusedGateUp warmup: %v", err)
	}

	var expertsErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, expertsErr = MoEExpertsQuantFusedGateUp(x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, groupSize, bits)
	})
	if expertsErr != nil {
		t.Fatalf("MoEExpertsQuantFusedGateUp: %v", expertsErr)
	}
	if allocs > 30 {
		t.Fatalf("MoEExpertsQuantFusedGateUp allocations = %.0f, want <= 30", allocs)
	}
}

func TestMoEExpertsQuantFusedGateUpIntoWritesDirectlyToCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, groupSize, bits = 4, 2, 64, 128, 32, 4
	gate, up, down := quantMoEExpertsFixture(t, numExperts, dModel, dFF, groupSize, bits)
	gateUp := fusedGateUpQuantForBench(gate, up, numExperts, dFF, dModel, groupSize, bits)
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	want, err := MoEExpertsQuantFusedGateUp(x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, groupSize, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuantFusedGateUp: %v", err)
	}

	scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
	if err != nil {
		t.Fatalf("getMoEExpertsScratch: %v", err)
	}
	accBytes := unsafe.Slice((*byte)(scratch.acc.Contents()), dModel*bf16Size)
	sentinel := bytes.Repeat([]byte{0x6d}, len(accBytes))
	copy(accBytes, sentinel)
	putMoEExpertsScratch(scratch)

	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	got, err := MoEExpertsQuantFusedGateUpInto(out, x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, groupSize, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuantFusedGateUpInto: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MoEExpertsQuantFusedGateUpInto did not reuse caller-owned output backing")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("MoEExpertsQuantFusedGateUpInto != default fused gate/up path")
	}

	scratch, err = getMoEExpertsScratch(dModel, dFF, topK)
	if err != nil {
		t.Fatalf("getMoEExpertsScratch after call: %v", err)
	}
	defer putMoEExpertsScratch(scratch)
	accBytes = unsafe.Slice((*byte)(scratch.acc.Contents()), dModel*bf16Size)
	if !bytes.Equal(accBytes, sentinel) {
		t.Fatal("MoEExpertsQuantFusedGateUpInto wrote through pooled accumulator instead of caller output")
	}
}

func TestMLPTransformQuantAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF, groupSize, bits = 64, 128, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gate := quantWeightFixture(t, dFF, dModel, groupSize, bits, 3)
	up := quantWeightFixture(t, dFF, dModel, groupSize, bits, 31)
	down := quantWeightFixture(t, dModel, dFF, groupSize, bits, 37)
	if _, err := mlpTransformQuant(x, gate, up, down, dModel, dFF, groupSize, bits); err != nil {
		t.Fatalf("mlpTransformQuant warmup: %v", err)
	}

	var transformErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, transformErr = mlpTransformQuant(x, gate, up, down, dModel, dFF, groupSize, bits)
	})
	if transformErr != nil {
		t.Fatalf("mlpTransformQuant: %v", transformErr)
	}
	if allocs > 17 {
		t.Fatalf("mlpTransformQuant allocations = %.0f, want <= 17", allocs)
	}
}

func TestMLPTransformQuantComposedWritesDirectlyToReturnedOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF, groupSize, bits = 64, 128, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gate := quantWeightFixture(t, dFF, dModel, groupSize, bits, 3)
	up := quantWeightFixture(t, dFF, dModel, groupSize, bits, 31)
	down := quantWeightFixture(t, dModel, dFF, groupSize, bits, 37)

	want, err := mlpTransformQuantComposed(x, gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		t.Fatalf("mlpTransformQuantComposed: %v", err)
	}
	scratch, err := getMLPTransformScratch(dModel, dFF)
	if err != nil {
		t.Fatalf("getMLPTransformScratch: %v", err)
	}
	scratchOut := unsafe.Slice((*byte)(scratch.mlp.down.Contents()), dModel*bf16Size)
	sentinel := bytes.Repeat([]byte{0x9c}, len(scratchOut))
	copy(scratchOut, sentinel)
	putMLPTransformScratch(scratch)

	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	got, err := mlpTransformQuantComposedInto(out, x, gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		t.Fatalf("mlpTransformQuantComposedInto: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("mlpTransformQuantComposedInto did not reuse caller-owned output backing")
	}
	eqBytes(t, "mlpTransformQuantComposed direct output", got, want)

	scratch, err = getMLPTransformScratch(dModel, dFF)
	if err != nil {
		t.Fatalf("getMLPTransformScratch after call: %v", err)
	}
	defer putMLPTransformScratch(scratch)
	scratchOut = unsafe.Slice((*byte)(scratch.mlp.down.Contents()), dModel*bf16Size)
	if !bytes.Equal(scratchOut, sentinel) {
		t.Fatal("mlpTransformQuantComposed wrote through pooled scratch output instead of returned output")
	}
}

func TestMLPTransformQuantMegaMatchesTransform(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := ffnMegaPipeline(); err != nil {
		t.Skipf("ffn megakernel unavailable: %v", err)
	}

	const dModel, dFF, groupSize, bits = 256, 512, 64, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gate := quantWeightFixture(t, dFF, dModel, groupSize, bits, 3)
	up := quantWeightFixture(t, dFF, dModel, groupSize, bits, 31)
	down := quantWeightFixture(t, dModel, dFF, groupSize, bits, 37)

	want, err := mlpTransformQuantComposed(x, gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		t.Fatalf("mlpTransformQuantComposed: %v", err)
	}
	got, err := mlpTransformQuantMega(x, gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		t.Fatalf("mlpTransformQuantMega: %v", err)
	}
	if cosineBF16(got, want) < 0.9999 {
		t.Fatalf("mlpTransformQuantMega != composed quant path: cosine %.6f", cosineBF16(got, want))
	}
}

func TestMLPTransformQuantMegaAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := ffnMegaPipeline(); err != nil {
		t.Skipf("ffn megakernel unavailable: %v", err)
	}

	const dModel, dFF, groupSize, bits = 256, 512, 64, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gate := quantWeightFixture(t, dFF, dModel, groupSize, bits, 3)
	up := quantWeightFixture(t, dFF, dModel, groupSize, bits, 31)
	down := quantWeightFixture(t, dModel, dFF, groupSize, bits, 37)
	if _, err := mlpTransformQuantMega(x, gate, up, down, dModel, dFF, groupSize, bits); err != nil {
		t.Fatalf("mlpTransformQuantMega warmup: %v", err)
	}

	var transformErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, transformErr = mlpTransformQuantMega(x, gate, up, down, dModel, dFF, groupSize, bits)
	})
	if transformErr != nil {
		t.Fatalf("mlpTransformQuantMega: %v", transformErr)
	}
	if allocs > 10 {
		t.Fatalf("mlpTransformQuantMega allocations = %.0f, want <= 10", allocs)
	}
}

func TestMLPTransformQuantMegaWritesDirectlyToCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := ffnMegaPipeline(); err != nil {
		t.Skipf("ffn megakernel unavailable: %v", err)
	}

	const dModel, dFF, groupSize, bits = 256, 512, 64, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gate := quantWeightFixture(t, dFF, dModel, groupSize, bits, 3)
	up := quantWeightFixture(t, dFF, dModel, groupSize, bits, 31)
	down := quantWeightFixture(t, dModel, dFF, groupSize, bits, 37)

	want, err := mlpTransformQuantMega(x, gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		t.Fatalf("mlpTransformQuantMega: %v", err)
	}
	scratch, err := getMLPTransformMegaScratch(dModel, dFF)
	if err != nil {
		t.Fatalf("getMLPTransformMegaScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x42}, len(scratch.outBytes))
	copy(scratch.outBytes, sentinel)
	putMLPTransformMegaScratch(scratch)

	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	got, err := mlpTransformQuantMegaInto(out, x, gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		t.Fatalf("mlpTransformQuantMegaInto: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("mlpTransformQuantMegaInto did not reuse caller-owned output backing")
	}
	if cosineBF16(got, want) < 0.9999 {
		t.Fatalf("mlpTransformQuantMegaInto != default mega path: cosine %.6f", cosineBF16(got, want))
	}

	scratch, err = getMLPTransformMegaScratch(dModel, dFF)
	if err != nil {
		t.Fatalf("getMLPTransformMegaScratch after call: %v", err)
	}
	defer putMLPTransformMegaScratch(scratch)
	if !bytes.Equal(scratch.outBytes, sentinel) {
		t.Fatal("mlpTransformQuantMegaInto wrote through pooled megakernel output instead of caller output")
	}
}

func TestMLPTransformQuantLargeAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := ffnMegaPipeline(); err != nil {
		t.Skipf("ffn megakernel unavailable: %v", err)
	}

	const dModel, dFF, groupSize, bits = 256, 512, 64, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gate := quantWeightFixture(t, dFF, dModel, groupSize, bits, 3)
	up := quantWeightFixture(t, dFF, dModel, groupSize, bits, 31)
	down := quantWeightFixture(t, dModel, dFF, groupSize, bits, 37)
	if _, err := mlpTransformQuant(x, gate, up, down, dModel, dFF, groupSize, bits); err != nil {
		t.Fatalf("mlpTransformQuant warmup: %v", err)
	}

	var transformErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, transformErr = mlpTransformQuant(x, gate, up, down, dModel, dFF, groupSize, bits)
	})
	if transformErr != nil {
		t.Fatalf("mlpTransformQuant: %v", transformErr)
	}
	if allocs > 8 {
		t.Fatalf("mlpTransformQuant large allocations = %.0f, want <= 8", allocs)
	}
}

func TestMLPTransformQuantRejectsInvalidInputs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, groupSize, bits = 64, 128, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gate := quantWeightFixture(t, dFF, dModel, groupSize, bits, 3)
	up := quantWeightFixture(t, dFF, dModel, groupSize, bits, 31)
	down := quantWeightFixture(t, dModel, dFF, groupSize, bits, 37)

	if _, err := mlpTransformQuant(x[:len(x)-bf16Size], gate, up, down, dModel, dFF, groupSize, bits); err == nil {
		t.Fatal("expected mlpTransformQuant to reject short input")
	}
	badGate := gate
	badGate.Packed = badGate.Packed[:len(badGate.Packed)-1]
	if _, err := mlpTransformQuant(x, badGate, up, down, dModel, dFF, groupSize, bits); err == nil {
		t.Fatal("expected mlpTransformQuant to reject mismatched gate weight")
	}
	if _, _, _, _, _, err := quantWeightViewsForShape("test.quant", QuantWeight{}, dFF, dModel, 0, bits); err == nil {
		t.Fatal("expected quantWeightViewsForShape to reject invalid geometry")
	}
	zero, err := mlpTransformQuant(nil, QuantWeight{}, QuantWeight{}, QuantWeight{}, 0, 0, groupSize, bits)
	if err != nil {
		t.Fatalf("mlpTransformQuant zero dimensions: %v", err)
	}
	if len(zero) != 0 {
		t.Fatalf("mlpTransformQuant zero dimensions len = %d, want 0", len(zero))
	}
}

func TestMLPTransformScratchClose(t *testing.T) {
	requireNativeRuntime(t)

	s, err := newMLPTransformScratch(64, 128)
	if err != nil {
		t.Fatalf("newMLPTransformScratch: %v", err)
	}
	if s.x == nil || s.x.buf == nil {
		t.Fatal("newMLPTransformScratch did not allocate pinned input")
	}
	s.Close()
	if s.x != nil || s.dModel != 0 || s.dFF != 0 {
		t.Fatal("Close did not clear pinned input and dimensions")
	}
	s.Close()
}

func TestMLPTransformScratchPoolKeepsShapesResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getMLPTransformScratch(64, 128)
	if err != nil {
		t.Fatalf("get small MLPTransform scratch: %v", err)
	}
	putMLPTransformScratch(small)
	large, err := getMLPTransformScratch(96, 192)
	if err != nil {
		t.Fatalf("get large MLPTransform scratch: %v", err)
	}
	putMLPTransformScratch(large)

	gotSmall, err := getMLPTransformScratch(64, 128)
	if err != nil {
		t.Fatalf("get small MLPTransform scratch again: %v", err)
	}
	defer putMLPTransformScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("MLPTransform scratch pool evicted the small shape after using a larger shape")
	}
	gotLarge, err := getMLPTransformScratch(96, 192)
	if err != nil {
		t.Fatalf("get large MLPTransform scratch again: %v", err)
	}
	defer putMLPTransformScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("MLPTransform scratch pool evicted the large shape after reusing the small shape")
	}
}

func TestMLPTransformMegaScratchPoolKeepsShapesResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getMLPTransformMegaScratch(256, 512)
	if err != nil {
		t.Fatalf("get small MLPTransformMega scratch: %v", err)
	}
	putMLPTransformMegaScratch(small)
	large, err := getMLPTransformMegaScratch(384, 768)
	if err != nil {
		t.Fatalf("get large MLPTransformMega scratch: %v", err)
	}
	putMLPTransformMegaScratch(large)

	gotSmall, err := getMLPTransformMegaScratch(256, 512)
	if err != nil {
		t.Fatalf("get small MLPTransformMega scratch again: %v", err)
	}
	defer putMLPTransformMegaScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("MLPTransformMega scratch pool evicted the small shape after using a larger shape")
	}
	gotLarge, err := getMLPTransformMegaScratch(384, 768)
	if err != nil {
		t.Fatalf("get large MLPTransformMega scratch again: %v", err)
	}
	defer putMLPTransformMegaScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("MLPTransformMega scratch pool evicted the large shape after reusing the small shape")
	}
}

func TestMLPTransformScratchInputViewsUseCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))

	composed, err := getMLPTransformScratch(dModel, dFF)
	if err != nil {
		t.Fatalf("getMLPTransformScratch: %v", err)
	}
	defer composed.Close()
	buf, ok := composed.inputView(x)
	if !ok {
		t.Fatal("composed inputView ok = false")
	}
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&x[0])); got != want {
		t.Fatalf("composed inputView buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reused, ok := composed.inputView(x)
	if !ok {
		t.Fatal("reused composed inputView ok = false")
	}
	if reused.GetID() != buf.GetID() {
		t.Fatal("composed inputView did not reuse the cached no-copy buffer")
	}

	mega, err := getMLPTransformMegaScratch(256, 512)
	if err != nil {
		t.Fatalf("getMLPTransformMegaScratch: %v", err)
	}
	defer mega.Close()
	megaX := toBF16Bytes(syntheticFloat32(256, 41))
	megaBuf, ok := mega.inputView(megaX)
	if !ok {
		t.Fatal("mega inputView ok = false")
	}
	if got, want := uintptr(megaBuf.Contents()), uintptr(unsafe.Pointer(&megaX[0])); got != want {
		t.Fatalf("mega inputView buffer pointer = %#x, want caller backing %#x", got, want)
	}
	megaReused, ok := mega.inputView(megaX)
	if !ok {
		t.Fatal("reused mega inputView ok = false")
	}
	if megaReused.GetID() != megaBuf.GetID() {
		t.Fatal("mega inputView did not reuse the cached no-copy buffer")
	}
}

func TestMoEBlockScratchIndexViewUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	s, err := newMoEBlockBF16Scratch(64, 128, 96, 2)
	if err != nil {
		t.Fatalf("newMoEBlockBF16Scratch: %v", err)
	}
	defer s.Close()

	idx := []int32{2, 0}
	buf, ok := s.indexView(idx)
	if !ok {
		t.Fatal("indexView ok = false")
	}
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&idx[0])); got != want {
		t.Fatalf("indexView buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reused, ok := s.indexView(idx)
	if !ok {
		t.Fatal("reused indexView ok = false")
	}
	if reused.GetID() != buf.GetID() {
		t.Fatal("indexView did not reuse the cached no-copy buffer")
	}
}

func TestMoEBlockQuantAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	if _, err := MoEBlockQuant(h, w, dModel, dFF, eps); err != nil {
		t.Fatalf("MoEBlockQuant warmup: %v", err)
	}

	var blockErr error
	allocs := testing.AllocsPerRun(3, func() {
		_, blockErr = MoEBlockQuant(h, w, dModel, dFF, eps)
	})
	if blockErr != nil {
		t.Fatalf("MoEBlockQuant: %v", blockErr)
	}
	if allocs > 10 {
		t.Fatalf("MoEBlockQuant allocations = %.0f, want <= 10", allocs)
	}
}

func TestMoEBlockQuantIntoWritesDirectlyToCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	want, err := MoEBlockQuant(h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockQuant: %v", err)
	}

	scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
	if err != nil {
		t.Fatalf("getMoEBlockBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x5a}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	seededScratch := scratch
	putMoEBlockBF16Scratch(scratch)

	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	got, err := MoEBlockQuantInto(out, h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockQuantInto: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MoEBlockQuantInto did not reuse caller-owned output backing")
	}
	if cosineBF16(got, want) < 0.9999 {
		t.Fatalf("MoEBlockQuantInto != default quant block path: cosine %.6f", cosineBF16(got, want))
	}

	if !bytes.Equal(seededScratch.out.bytes, sentinel) {
		t.Fatal("MoEBlockQuantInto wrote through pooled block output instead of caller output")
	}
}

func TestMoEBlockQuantWithBufferOutputWritesDirectlyToProvidedBuffer(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	want, err := MoEBlockQuant(h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockQuant: %v", err)
	}

	scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
	if err != nil {
		t.Fatalf("getMoEBlockBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xc3}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	seededScratch := scratch
	putMoEBlockBF16Scratch(scratch)

	input, err := newPinnedNoCopyBytes(len(h))
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes input: %v", err)
	}
	defer input.Close()
	hBuf, err := input.copyBuffer(h)
	if err != nil {
		t.Fatalf("copy input buffer: %v", err)
	}
	out, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes output: %v", err)
	}
	defer out.Close()

	if err := moeBlockQuantWithBufferOutputInPool(h, hBuf, out.buf, w, dModel, dFF, eps); err != nil {
		t.Fatalf("moeBlockQuantWithBufferOutputInPool: %v", err)
	}
	if cosineBF16(out.bytes, want) < 0.9999 {
		t.Fatalf("MoEBlockQuant direct Metal output != default quant block path: cosine %.6f", cosineBF16(out.bytes, want))
	}

	if !bytes.Equal(seededScratch.out.bytes, sentinel) {
		t.Fatal("moeBlockQuantWithBufferOutputInPool wrote through pooled block output")
	}
}

func TestMoEBlockQuantAfterRouterLargeLocalAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 256, 512, 128, 2, 1, 64, 4
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	idx := []int32{0}
	weights := toBF16Bytes([]float32{1})
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	if _, err := moeBlockQuantAfterRouter(h, idx, weights, nil, w, dModel, dFF, eps); err != nil {
		t.Fatalf("moeBlockQuantAfterRouter warmup: %v", err)
	}

	var blockErr error
	allocs := testing.AllocsPerRun(3, func() {
		_, blockErr = moeBlockQuantAfterRouter(h, idx, weights, nil, w, dModel, dFF, eps)
	})
	if blockErr != nil {
		t.Fatalf("moeBlockQuantAfterRouter: %v", blockErr)
	}
	if allocs > 8 {
		t.Fatalf("moeBlockQuantAfterRouter large local allocations = %.0f, want <= 8", allocs)
	}
}

func TestMoEBlockQuantAfterRouterLargeLocalMatchesComposed(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := ffnMegaPipeline(); err != nil {
		t.Skipf("ffn megakernel unavailable: %v", err)
	}

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 256, 512, 128, 2, 1, 64, 4
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	idx := []int32{0}
	weights := toBF16Bytes([]float32{1})
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)

	got, err := moeBlockQuantAfterRouter(h, idx, weights, nil, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("moeBlockQuantAfterRouter: %v", err)
	}
	must := func(b []byte, err error) []byte {
		t.Helper()
		if err != nil {
			t.Fatalf("reference op: %v", err)
		}
		return b
	}
	local := must(mlpTransformQuantComposed(
		must(RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps)),
		w.LocalGate, w.LocalUp, w.LocalDown, dModel, dFF, groupSize, bits,
	))
	expert := must(MoEExpertsQuant(
		must(RMSNormBF16(h, w.PreFFNorm2W, 1, dModel, eps)),
		idx, weights, w.ExpGate, w.ExpUp, w.ExpDown, numExperts, topK, dModel, expertDFF, groupSize, bits,
	))
	combined := must(AddBF16(
		must(RMSNormBF16(local, w.PostFFNorm1W, 1, dModel, eps)),
		must(RMSNormBF16(expert, w.PostFFNorm2W, 1, dModel, eps)),
	))
	want := must(AddBF16(h, must(RMSNormBF16(combined, w.PostFFNormW, 1, dModel, eps))))
	if cosineBF16(got, want) < 0.9999 {
		t.Fatalf("moeBlockQuantAfterRouter large local != composed reference: cosine %.6f", cosineBF16(got, want))
	}
	if bytes.Equal(got, h) {
		t.Fatal("moeBlockQuantAfterRouter large local did not transform the residual")
	}
}

func TestGatherQMVBF16ByExpertIndexMatchesSlicedQMV(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, outDim, inDim, groupSize, bits = 4, 3, 64, 96, 32, 4
	if _, err := gatherQMVBF16SteelPipeline(outDim, inDim, groupSize, bits); err != nil {
		t.Skipf("gather qmv kernel unavailable: %v", err)
	}
	idx := []int32{2, 0, 3}
	w := quantMoELayerWeightsGuard(t, numExperts, 1, inDim, 128, outDim, groupSize, bits).ExpGate
	x := toBF16Bytes(syntheticFloat32(inDim, 37))

	got, err := gatherQMVBF16ByExpertIndex(x, idx, w, numExperts, topK, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("gatherQMVBF16ByExpertIndex: %v", err)
	}
	want := make([]byte, 0, topK*outDim*bf16Size)
	expertPacked := outDim * inDim * bits / 8
	expertSB := outDim * (inDim / groupSize) * bf16Size
	for _, expert := range idx {
		e := int(expert)
		ref, err := QMVBF16(
			x,
			w.Packed[e*expertPacked:(e+1)*expertPacked],
			w.Scales[e*expertSB:(e+1)*expertSB],
			w.Biases[e*expertSB:(e+1)*expertSB],
			outDim,
			inDim,
			groupSize,
			bits,
		)
		if err != nil {
			t.Fatalf("sliced QMVBF16 expert %d: %v", e, err)
		}
		want = append(want, ref...)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("gathered qmv bytes != sliced QMVBF16")
	}
}

func TestGatherQMVBF16ByExpertIndexIntoUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, outDim, inDim, groupSize, bits = 4, 2, 64, 96, 32, 4
	if _, err := gatherQMVBF16SteelPipeline(outDim, inDim, groupSize, bits); err != nil {
		t.Skipf("gather qmv kernel unavailable: %v", err)
	}
	idx := []int32{1, 3}
	w := quantMoELayerWeightsGuard(t, numExperts, 1, inDim, 128, outDim, groupSize, bits).ExpGate
	x := toBF16Bytes(syntheticFloat32(inDim, 41))
	want, err := gatherQMVBF16ByExpertIndex(x, idx, w, numExperts, topK, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("gatherQMVBF16ByExpertIndex reference: %v", err)
	}

	out := bytes.Repeat([]byte{0xa5}, topK*outDim*bf16Size)
	got, err := gatherQMVBF16ByExpertIndexInto(out, x, idx, w, numExperts, topK, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("gatherQMVBF16ByExpertIndexInto: %v", err)
	}
	if len(got) != len(out) {
		t.Fatalf("gatherQMVBF16ByExpertIndexInto len = %d, want %d", len(got), len(out))
	}
	if len(got) > 0 && &got[0] != &out[0] {
		t.Fatal("gatherQMVBF16ByExpertIndexInto did not return caller-owned output backing")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("gatherQMVBF16ByExpertIndexInto output differs from allocating wrapper")
	}
}

func TestGatherQMVBF16ByExpertIndexRejectsInvalidExpertIndexBeforeDispatch(t *testing.T) {
	requireNativeRuntime(t)

	_, err := gatherQMVBF16ByExpertIndex(nil, []int32{4}, QuantWeight{}, 4, 1, 64, 96, 32, 4)
	if err == nil || !bytes.Contains([]byte(err.Error()), []byte("expert index")) {
		t.Fatalf("gatherQMVBF16ByExpertIndex invalid index error = %v, want expert index rejection", err)
	}
}

func TestMoEBlockQuantAfterRouterDeviceIndexBufferMatchesHostIndex(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-5)
	if _, err := gatherQMVBF16SteelPipeline(expertDFF, dModel, groupSize, bits); err != nil {
		t.Skipf("gather qmv kernel unavailable: %v", err)
	}
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	idx := []int32{2, 0}
	weights := toBF16Bytes([]float32{0.625, 0.375})
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)

	want, err := moeBlockQuantAfterRouter(h, idx, weights, nil, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("moeBlockQuantAfterRouter host index: %v", err)
	}
	idxBytes := unsafe.Slice((*byte)(unsafe.Pointer(&idx[0])), len(idx)*4)
	idxBuf := sharedBytes(idxBytes)
	got, err := moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, nil, nil, nil, idx, idxBuf, weights, nil, w, dModel, dFF, eps, false, false)
	if err != nil {
		t.Fatalf("moeBlockQuantAfterRouter device index: %v", err)
	}
	if cos := cosineBF16(got, want); cos < 0.9999 {
		t.Fatalf("device-index MoE block cosine=%.6f vs host-index path", cos)
	}
}

func TestMoEBlockQuantAfterRouterDeviceBuffersDoNotNeedHostRouterViews(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-5)
	if _, err := gatherQMVBF16SteelPipeline(expertDFF, dModel, groupSize, bits); err != nil {
		t.Skipf("gather qmv kernel unavailable: %v", err)
	}
	h := toBF16Bytes(syntheticFloat32(dModel, 31))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.55, 0.45})
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)

	want, err := moeBlockQuantAfterRouter(h, idx, weights, nil, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("moeBlockQuantAfterRouter host views: %v", err)
	}
	idxBytes := unsafe.Slice((*byte)(unsafe.Pointer(&idx[0])), len(idx)*4)
	idxBuf := sharedBytes(idxBytes)
	weightBuf := sharedBytes(weights)
	got, err := moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, nil, nil, nil, nil, idxBuf, nil, weightBuf, w, dModel, dFF, eps, false, false)
	if err != nil {
		t.Fatalf("moeBlockQuantAfterRouter device buffers without host views: %v", err)
	}
	if cos := cosineBF16(got, want); cos < 0.9999 {
		t.Fatalf("device-buffer-only MoE block cosine=%.6f vs host-view path", cos)
	}
}

func TestMoEBlockQuantDeviceRouterBuffersChainWithoutHostViews(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-5)
	if _, err := gatherQMVBF16SteelPipeline(expertDFF, dModel, groupSize, bits); err != nil {
		t.Skipf("gather qmv kernel unavailable: %v", err)
	}
	h := toBF16Bytes(syntheticFloat32(dModel, 37))
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)

	idx, weights, err := moeRouterQuantWithViews(h, w.RouterNormWScaled, w.routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps)
	if err != nil {
		t.Fatalf("moeRouterQuantWithViews: %v", err)
	}
	want, err := moeBlockQuantAfterRouter(h, idx, weights, nil, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("moeBlockQuantAfterRouter host route: %v", err)
	}

	weightBuf, routerScratch, ok, err := moeRouterQuantDeviceTopKBuffersWithBufferInPool(h, nil, w.RouterNormWScaled, w.routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps)
	if err != nil {
		t.Fatalf("moeRouterQuantDeviceTopKBuffersWithBufferInPool: %v", err)
	}
	if !ok {
		t.Fatal("expected device top-k router to be usable")
	}
	defer putRouterDeviceScratch(routerScratch)
	got, err := moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, nil, nil, nil, nil, routerScratch.idxBuf, nil, weightBuf, w, dModel, dFF, eps, false, false)
	if err != nil {
		t.Fatalf("moeBlockQuantAfterRouter device route buffers: %v", err)
	}
	if cos := cosineBF16(got, want); cos < 0.9999 {
		t.Fatalf("device-router-buffer chain cosine=%.6f vs host route", cos)
	}
}

func TestMoEBlockQuantAfterRouterRejectsInvalidInputs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	idx := []int32{0, 1}
	weights := toBF16Bytes([]float32{0.75, 0.25})
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	if _, err := moeBlockQuantAfterRouter(h[:len(h)-bf16Size], idx, weights, nil, w, dModel, dFF, eps); err == nil {
		t.Fatal("expected moeBlockQuantAfterRouter to reject short residual")
	}
	bad := w
	bad.LocalGate.Packed = bad.LocalGate.Packed[:len(bad.LocalGate.Packed)-1]
	if _, err := moeBlockQuantAfterRouter(h, idx, weights, nil, bad, dModel, dFF, eps); err == nil {
		t.Fatal("expected moeBlockQuantAfterRouter to reject short local gate weight")
	}
}

func TestMoEBlockQuantAfterRouterUsesProvidedHiddenBuffer(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-5)
	hostH := toBF16Bytes(syntheticFloat32(dModel, 7))
	bufferH := toBF16Bytes(syntheticFloat32(dModel, 29))
	idx := []int32{0, 1}
	weights := toBF16Bytes([]float32{0.75, 0.25})
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)

	pinned, err := newPinnedNoCopyBytes(len(bufferH))
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes: %v", err)
	}
	defer pinned.Close()
	hBuf, err := pinned.copyBuffer(bufferH)
	if err != nil {
		t.Fatalf("copyBuffer: %v", err)
	}

	want, err := moeBlockQuantAfterRouter(bufferH, idx, weights, nil, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("moeBlockQuantAfterRouter: %v", err)
	}
	got, err := moeBlockQuantAfterRouterWithBuffer(hostH, hBuf, idx, weights, nil, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("moeBlockQuantAfterRouterWithBuffer: %v", err)
	}
	eqBytes(t, "MoEBlockQuant provided hidden buffer", got, want)
}

// TestMoERouterQuant gates the 4-bit router: MoERouterQuant ≡ the manual RMSNorm → QMVBF16 →
// routerSelect composition, and its selected expert SET matches an independent max-scan top-k
// over the same scores (non-circular on the selection).
func TestMoERouterQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, gs, bits = 8, 3, 64, 32, 4
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.02
		}
		return s
	}
	pp, ps, pb := quantizeProj(t, numExperts, dModel, gs, bits, 13)
	proj := QuantWeight{Packed: pp, Scales: ps, Biases: pb}
	x := toBF16Bytes(mk(dModel, 7))
	norm := toBF16Bytes(mk(dModel, 9))
	scale := toBF16Bytes(mk(numExperts, 5))

	idx, weights, err := MoERouterQuant(x, norm, proj, scale, numExperts, topK, dModel, gs, bits, eps)
	if err != nil {
		t.Fatalf("MoERouterQuant: %v", err)
	}
	if len(idx) != topK || len(weights) != topK*bf16Size {
		t.Fatalf("idx %d / weights %d, want topK %d", len(idx), len(weights)/bf16Size, topK)
	}

	// wiring: ≡ manual RMSNorm → QMVBF16 → routerSelect.
	normed, err := RMSNormBF16(x, norm, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	scoresB, err := QMVBF16(normed, proj.Packed, proj.Scales, proj.Biases, numExperts, dModel, gs, bits)
	if err != nil {
		t.Fatalf("QMVBF16: %v", err)
	}
	wantIdx, wantW := routerSelect(scoresB, scale, numExperts, topK)
	if !bytes.Equal(weights, wantW) {
		t.Fatal("MoERouterQuant weights != manual routerSelect")
	}
	set := func(ids []int32) map[int32]bool {
		m := map[int32]bool{}
		for _, e := range ids {
			m[e] = true
		}
		return m
	}
	got, want := set(idx), set(wantIdx)
	if len(got) != len(want) {
		t.Fatal("idx set size mismatch vs manual")
	}
	for e := range want {
		if !got[e] {
			t.Fatalf("idx set != manual (missing %d)", e)
		}
	}

	// independent selection: the topK highest scores by max-scan must equal idx as a set.
	sc := make([]float32, numExperts)
	for e := range numExperts {
		sc[e] = bf16ToF32(scoresB[e*bf16Size], scoresB[e*bf16Size+1])
	}
	used := map[int32]bool{}
	for range topK {
		best, bv := int32(-1), float32(-1e30)
		for e := range numExperts {
			if !used[int32(e)] && sc[e] > bv {
				best, bv = int32(e), sc[e]
			}
		}
		used[best] = true
	}
	for e := range got {
		if !used[e] {
			t.Fatalf("selected expert %d not in the independent max-scan top-k", e)
		}
	}
	t.Logf("4-bit router: MoERouterQuant ≡ RMSNorm→QMV→routerSelect, expert set ≡ independent max-scan top-k %v", idx)
}

// TestMoEBlockQuant gates the 4-bit dual-branch MoE block: MoEBlockQuant ≡ the composed
// reference (router → local quant MLP + quant experts, each normed, summed, post-normed,
// residual) byte-for-byte, and transforms the residual (non-vacuous).
func TestMoEBlockQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, dFF, expertDFF, numExperts, topK, gs, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.02
		}
		return s
	}
	qw := func(outDim, inDim, salt int) QuantWeight {
		p, s, b := quantizeProj(t, outDim, inDim, gs, bits, salt)
		return QuantWeight{Packed: p, Scales: s, Biases: b}
	}
	batched := func(outDim, inDim, saltBase int) QuantWeight {
		var p, s, b []byte
		for e := range numExperts {
			pe, se, be := quantizeProj(t, outDim, inDim, gs, bits, saltBase+e*7)
			p, s, b = append(p, pe...), append(s, se...), append(b, be...)
		}
		return QuantWeight{Packed: p, Scales: s, Biases: b}
	}
	nrm := func(salt int) []byte { return toBF16Bytes(mk(dModel, salt)) }
	w := MoEQuantLayerWeights{
		NumExperts: numExperts, TopK: topK, ExpertDFF: expertDFF,
		ExpertGroupSize: gs, ExpertBits: bits, LocalGroupSize: gs, LocalBits: bits, RouterGroupSize: gs, RouterBits: bits,
		PreFFNormW: nrm(13), PreFFNorm2W: nrm(17), PostFFNorm1W: nrm(19), PostFFNorm2W: nrm(23), PostFFNormW: nrm(29),
		LocalGate: qw(dFF, dModel, 3), LocalUp: qw(dFF, dModel, 31), LocalDown: qw(dModel, dFF, 37),
		RouterNormWScaled: nrm(41), Router: qw(numExperts, dModel, 43), PerExpertScale: toBF16Bytes(mk(numExperts, 47)),
		ExpGate: batched(expertDFF, dModel, 53), ExpUp: batched(expertDFF, dModel, 101), ExpDown: batched(dModel, expertDFF, 149),
	}
	h := toBF16Bytes(mk(dModel, 5))

	got, err := MoEBlockQuant(h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockQuant: %v", err)
	}

	must := func(b []byte, e error) []byte {
		t.Helper()
		if e != nil {
			t.Fatalf("ref op: %v", e)
		}
		return b
	}
	idx, weights, err := MoERouterQuant(h, w.RouterNormWScaled, w.Router, w.PerExpertScale, numExperts, topK, dModel, gs, bits, eps)
	if err != nil {
		t.Fatalf("MoERouterQuant: %v", err)
	}
	h1 := must(mlpTransformQuant(must(RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps)), w.LocalGate, w.LocalUp, w.LocalDown, dModel, dFF, gs, bits))
	h2 := must(MoEExpertsQuant(must(RMSNormBF16(h, w.PreFFNorm2W, 1, dModel, eps)), idx, weights, w.ExpGate, w.ExpUp, w.ExpDown, numExperts, topK, dModel, expertDFF, gs, bits))
	combined := must(AddBF16(must(RMSNormBF16(h1, w.PostFFNorm1W, 1, dModel, eps)), must(RMSNormBF16(h2, w.PostFFNorm2W, 1, dModel, eps))))
	want := must(AddBF16(h, must(RMSNormBF16(combined, w.PostFFNormW, 1, dModel, eps))))
	if !bytes.Equal(got, want) {
		t.Fatal("MoEBlockQuant != composed dual-branch reference")
	}
	if bytes.Equal(got, h) {
		t.Fatal("MoEBlockQuant did not transform the residual")
	}
	t.Logf("4-bit MoE block: dual-branch (quant local MLP + quant experts, router-gated) ≡ composed reference, byte-for-byte")
}

func TestMoEBlockQuantCachesLocalDenseWeightsWithExperts(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF, expertDFF, numExperts, topK, gs, bits = 64, 128, 96, 4, 2, 32, 4
	const eps = float32(1e-6)
	h := toBF16Bytes(syntheticFloat32(dModel, 5))
	w := quantMoELayerWeightsGuard(t, numExperts, topK, dModel, dFF, expertDFF, gs, bits)

	if _, err := MoEBlockQuant(h, w, dModel, dFF, eps); err != nil {
		t.Fatalf("MoEBlockQuant: %v", err)
	}

	key := func(b []byte) uintptr {
		return uintptr(unsafe.Pointer(&b[0]))
	}
	local := []struct {
		name string
		buf  []byte
	}{
		{"local gate packed", w.LocalGate.Packed},
		{"local gate scales", w.LocalGate.Scales},
		{"local gate biases", w.LocalGate.Biases},
		{"local up packed", w.LocalUp.Packed},
		{"local up scales", w.LocalUp.Scales},
		{"local up biases", w.LocalUp.Biases},
		{"local down packed", w.LocalDown.Packed},
		{"local down scales", w.LocalDown.Scales},
		{"local down biases", w.LocalDown.Biases},
		{"expert gate packed", w.ExpGate.Packed},
		{"expert gate scales", w.ExpGate.Scales},
		{"expert gate biases", w.ExpGate.Biases},
		{"expert up packed", w.ExpUp.Packed},
		{"expert up scales", w.ExpUp.Scales},
		{"expert up biases", w.ExpUp.Biases},
		{"expert down packed", w.ExpDown.Packed},
		{"expert down scales", w.ExpDown.Scales},
		{"expert down biases", w.ExpDown.Biases},
	}

	residentBufMu.Lock()
	got := len(residentBufs)
	missing := make([]string, 0)
	for _, item := range local {
		if _, ok := residentBufs[key(item.buf)]; !ok {
			missing = append(missing, item.name)
		}
	}
	residentBufMu.Unlock()

	wantAtLeast := len(local)
	if len(missing) != 0 {
		t.Fatalf("MoEBlockQuant did not keep quant weights resident (missing %v, resident=%d want>=%d)", missing, got, wantAtLeast)
	}
	if got < wantAtLeast {
		t.Fatalf("resident quant weights = %d, want at least %d local dense + whole expert tensors", got, wantAtLeast)
	}
}
