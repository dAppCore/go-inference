// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
)

func TestMatVecAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim = 128, 256
	mat := syntheticFloat32(outDim*inDim, 3)
	vec := syntheticFloat32(inDim, 5)
	if _, err := MatVec(mat, vec, outDim, inDim); err != nil {
		t.Fatalf("MatVec warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MatVec(mat, vec, outDim, inDim); err != nil {
			t.Fatalf("MatVec: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MatVec allocations = %.0f, want <= 10", allocs)
	}
}

func TestMatVecComputesRowMajorProjection(t *testing.T) {
	requireNativeRuntime(t)

	mat := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	vec := []float32{1, -1, 0.5, 2}
	got, err := MatVec(mat, vec, 2, 4)
	if err != nil {
		t.Fatalf("MatVec: %v", err)
	}
	assertFloat32Near(t, "MatVec", got, []float32{8.5, 18.5}, 1e-5)
}

func TestMatVecIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim = 128, 256
	mat := syntheticFloat32(outDim*inDim, 3)
	vec := syntheticFloat32(inDim, 5)
	want, err := MatVec(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVec reference: %v", err)
	}
	out := syntheticFloat32(outDim, 11)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVFloatScratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVFloatScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x5a}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVFloatScratch(scratch)

	got, err := MatVecInto(out, mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVecInto: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MatVecInto did not reuse caller-owned output backing")
	}
	assertFloat32Near(t, "MatVecInto", got, want, 1e-5)

	scratch, err = getQMVFloatScratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVFloatScratch after call: %v", err)
	}
	defer putQMVFloatScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("MatVecInto wrote through pooled scratch output instead of caller output")
	}
}

func TestMatVecRejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := MatVec([]float32{1, 2, 3}, []float32{1, 2}, 2, 2); err == nil {
		t.Fatal("expected MatVec to reject matrix length mismatch")
	}
}
