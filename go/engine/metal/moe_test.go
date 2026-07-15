// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"
)

// moeExpertsRef is the oracle for MoEExperts, composed from the parity-proven
// standalone ops: per selected expert, MatVec gate/up, GeluGateMul, MatVec down,
// scale by the router weight, accumulate. Mirrors MoEExperts op-for-op.
func moeExpertsRef(t *testing.T, x []byte, idx []int32, weights, gateW, upW, downW []byte, numExperts, topK, dModel, dFF int) []byte {
	t.Helper()
	gateSz, downSz := dFF*dModel*bf16Size, dModel*dFF*bf16Size
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("moeExpertsRef op: %v", err)
		}
		return b
	}
	var acc []byte
	for i := range topK {
		e := int(idx[i])
		gate := must(MatVecBF16(gateW[e*gateSz:(e+1)*gateSz], x, dFF, dModel))
		up := must(MatVecBF16(upW[e*gateSz:(e+1)*gateSz], x, dFF, dModel))
		act := must(GeluGateMulBF16(gate, up))
		downE := must(MatVecBF16(downW[e*downSz:(e+1)*downSz], act, dModel, dFF))
		scaled := must(MulBF16(downE, scalarFillBF16(weights[i*bf16Size:(i+1)*bf16Size], dModel)))
		if i == 0 {
			acc = scaled
		} else {
			acc = must(AddBF16(acc, scaled))
		}
	}
	return acc
}

// TestMoEExperts gates the MoE expert branch: the chained on-device MoEExperts (top-k
// experts' SwiGLU + router-weighted combine) is byte-for-byte the composed reference
// of proven standalone ops, on the same routing + expert weights. The routing
// (top-k + softmax) and the dual-branch composition are separate sub-slices.
func TestMoEExperts(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, dFF = 8, 2, 256, 512
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+7)%101-50) * 0.02
		}
		return s
	}
	x := toBF16Bytes(mk(dModel, 37))
	gateW := toBF16Bytes(mk(numExperts*dFF*dModel, 53))
	upW := toBF16Bytes(mk(numExperts*dFF*dModel, 71))
	downW := toBF16Bytes(mk(numExperts*dModel*dFF, 47))
	idx := []int32{5, 2} // an arbitrary top-2 selection
	weights := toBF16Bytes([]float32{0.6, 0.4})

	got, err := MoEExperts(x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF)
	if err != nil {
		t.Fatalf("MoEExperts: %v", err)
	}
	want := moeExpertsRef(t, x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF)
	eqBytes(t, "MoEExperts", got, want)
	t.Logf("MoEExperts(%d experts, top-%d): chained expert branch ≡ composed reference (per-expert SwiGLU + weighted combine)", numExperts, topK)
}

func TestMoEExpertsBindsWholeBatchedExpertMatrices(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF = 4, 2, 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gateW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 53))
	upW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 71))
	downW := toBF16Bytes(syntheticFloat32(numExperts*dModel*dFF, 47))
	idx := []int32{1, 3}
	weights := toBF16Bytes([]float32{0.6, 0.4})

	if _, err := MoEExperts(x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF); err != nil {
		t.Fatalf("MoEExperts: %v", err)
	}

	key := func(b []byte) uintptr {
		return uintptr(unsafe.Pointer(&b[0]))
	}
	gateSz, downSz := dFF*dModel*bf16Size, dModel*dFF*bf16Size
	wholeKeys := map[uintptr]string{
		key(gateW): "gate",
		key(upW):   "up",
		key(downW): "down",
	}
	selectedSliceKeys := map[uintptr]string{}
	for _, e32 := range idx {
		e := int(e32)
		selectedSliceKeys[key(gateW[e*gateSz:(e+1)*gateSz])] = "gate"
		selectedSliceKeys[key(upW[e*gateSz:(e+1)*gateSz])] = "up"
		selectedSliceKeys[key(downW[e*downSz:(e+1)*downSz])] = "down"
	}

	residentBufMu.Lock()
	got := len(residentBufs)
	missingWhole := []string{}
	for k, name := range wholeKeys {
		if _, ok := residentBufs[k]; !ok {
			missingWhole = append(missingWhole, name)
		}
	}
	sliceHits := 0
	for k := range selectedSliceKeys {
		if _, ok := residentBufs[k]; ok {
			sliceHits++
		}
	}
	residentBufMu.Unlock()

	if len(missingWhole) > 0 || sliceHits > 0 || got != len(wholeKeys) {
		t.Fatalf("MoEExperts resident tensors mismatch: missing whole=%v selected-slice hits=%d resident=%d want exactly %d whole batched tensors", missingWhole, sliceHits, got, len(wholeKeys))
	}
}

func TestMoEExpertsAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const numExperts, topK, dModel, dFF = 4, 2, 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	gateW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 53))
	upW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 71))
	downW := toBF16Bytes(syntheticFloat32(numExperts*dModel*dFF, 47))
	if _, err := MoEExperts(x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF); err != nil {
		t.Fatalf("MoEExperts warmup: %v", err)
	}

	var expertsErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, expertsErr = MoEExperts(x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF)
	})
	if expertsErr != nil {
		t.Fatalf("MoEExperts: %v", expertsErr)
	}
	if allocs > 30 {
		t.Fatalf("MoEExperts allocations = %.0f, want <= 30", allocs)
	}
}

func TestMoEExpertsScratchPoolKeepsShapesResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getMoEExpertsScratch(64, 128, 2)
	if err != nil {
		t.Fatalf("get small MoEExperts scratch: %v", err)
	}
	putMoEExpertsScratch(small)
	large, err := getMoEExpertsScratch(96, 192, 3)
	if err != nil {
		t.Fatalf("get large MoEExperts scratch: %v", err)
	}
	putMoEExpertsScratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getMoEExpertsScratch(64, 128, 2)
	if err != nil {
		t.Fatalf("get small MoEExperts scratch again: %v", err)
	}
	defer putMoEExpertsScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("MoEExperts scratch pool evicted the small shape after using a larger shape")
	}
	gotLarge, err := getMoEExpertsScratch(96, 192, 3)
	if err != nil {
		t.Fatalf("get large MoEExperts scratch again: %v", err)
	}
	defer putMoEExpertsScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("MoEExperts scratch pool evicted the large shape after reusing the small shape")
	}
}

func TestMoEExpertsIntoWritesDirectlyToCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF = 4, 2, 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	gateW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 53))
	upW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 71))
	downW := toBF16Bytes(syntheticFloat32(numExperts*dModel*dFF, 47))
	want, err := MoEExperts(x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF)
	if err != nil {
		t.Fatalf("MoEExperts: %v", err)
	}

	scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
	if err != nil {
		t.Fatalf("getMoEExpertsScratch: %v", err)
	}
	accBytes := unsafe.Slice((*byte)(scratch.acc.Contents()), dModel*bf16Size)
	sentinel := bytes.Repeat([]byte{0xb6}, len(accBytes))
	copy(accBytes, sentinel)
	putMoEExpertsScratch(scratch)

	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	got, err := MoEExpertsInto(out, x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF)
	if err != nil {
		t.Fatalf("MoEExpertsInto: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MoEExpertsInto did not reuse caller-owned output backing")
	}
	eqBytes(t, "MoEExpertsInto direct output", got, want)

	scratch, err = getMoEExpertsScratch(dModel, dFF, topK)
	if err != nil {
		t.Fatalf("getMoEExpertsScratch after call: %v", err)
	}
	defer putMoEExpertsScratch(scratch)
	accBytes = unsafe.Slice((*byte)(scratch.acc.Contents()), dModel*bf16Size)
	if !bytes.Equal(accBytes, sentinel) {
		t.Fatal("MoEExpertsInto wrote through pooled accumulator instead of caller output")
	}
}

func TestMoEExpertsScratchInputViewUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const topK, dModel, dFF = 2, 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
	if err != nil {
		t.Fatalf("getMoEExpertsScratch: %v", err)
	}
	defer scratch.Close()

	buf, ok := scratch.inputView(x)
	if !ok {
		t.Fatal("inputView ok = false")
	}
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&x[0])); got != want {
		t.Fatalf("inputView buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reused, ok := scratch.inputView(x)
	if !ok {
		t.Fatal("reused inputView ok = false")
	}
	if reused.GetID() != buf.GetID() {
		t.Fatal("inputView did not reuse the cached no-copy buffer for the same backing")
	}
}

func TestMoEExpertsScratchWeightsViewUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const topK, dModel, dFF = 2, 64, 128
	weights := toBF16Bytes([]float32{0.6, 0.4})
	scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
	if err != nil {
		t.Fatalf("getMoEExpertsScratch: %v", err)
	}
	defer scratch.Close()

	buf, ok := scratch.weightsView(weights)
	if !ok {
		t.Fatal("weightsView ok = false")
	}
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&weights[0])); got != want {
		t.Fatalf("weightsView buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reused, ok := scratch.weightsView(weights)
	if !ok {
		t.Fatal("reused weightsView ok = false")
	}
	if reused.GetID() != buf.GetID() {
		t.Fatal("weightsView did not reuse the cached no-copy buffer for the same backing")
	}
}
