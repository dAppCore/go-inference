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

// hostMatVecF32 computes out = mat @ vec on the host in float64: mat is
// row-major (outDim x inDim), vec has length inDim.
func hostMatVecF32(mat, vec []float32, outDim, inDim int) []float32 {
	out := make([]float32, outDim)
	for r := range outDim {
		var sum float64
		row := mat[r*inDim : (r+1)*inDim]
		for i, v := range row {
			sum += float64(v) * float64(vec[i])
		}
		out[r] = float32(sum)
	}
	return out
}

// TestGemv_MatVec_Good pins the mathematical contract on a hand-checkable
// shape: out = mat @ vec for a row-major (2x4) matrix.
func TestGemv_MatVec_Good(t *testing.T) {
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

// TestGemv_MatVec_Bad pins the shape validation: mat must be outDim*inDim and
// vec must be inDim, each rejected with an error rather than a mis-shaped
// dispatch.
func TestGemv_MatVec_Bad(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := MatVec([]float32{1, 2, 3}, []float32{1, 2}, 2, 2); err == nil {
		t.Fatal("expected MatVec to reject matrix length mismatch")
	}
	if _, err := MatVec([]float32{1, 2, 3, 4}, []float32{1}, 2, 2); err == nil {
		t.Fatal("expected MatVec to reject vector length mismatch")
	}
}

// TestGemv_MatVec_Ugly pins the tile-selection lanes of gemvTiles — each picks
// a DIFFERENT template-specialised kernel variant, so a wrong name is a wrong
// dispatch, not a slow one: the short-k lane (k<=64), the tall-output lane
// (outDim>=4096, bm=8), the wide-k lane (k>=16*outDim, bn=8) and the tiny
// output (outDim<tm, tm=1) — plus the zero-dim no-dispatch contract (an empty
// reduction is a zero vector).
func TestGemv_MatVec_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	lanes := []struct {
		name          string
		outDim, inDim int
	}{
		{"short-k", 8, 32},
		{"tall-output", 4096, 64},
		{"wide-k", 8, 2048},
		{"tiny-output", 2, 128},
	}
	for _, lane := range lanes {
		mat := syntheticFloat32(lane.outDim*lane.inDim, 3)
		vec := syntheticFloat32(lane.inDim, 5)
		got, err := MatVec(mat, vec, lane.outDim, lane.inDim)
		if err != nil {
			t.Fatalf("MatVec %s lane: %v", lane.name, err)
		}
		assertFloat32Near(t, "MatVec "+lane.name, got, hostMatVecF32(mat, vec, lane.outDim, lane.inDim), 2e-4)
	}

	zeroOut, err := MatVec(nil, syntheticFloat32(4, 5), 0, 4)
	if err != nil {
		t.Fatalf("MatVec outDim 0: %v", err)
	}
	if len(zeroOut) != 0 {
		t.Fatalf("MatVec outDim 0 returned %d values, want none", len(zeroOut))
	}
	zeroIn, err := MatVec(nil, nil, 3, 0)
	if err != nil {
		t.Fatalf("MatVec inDim 0: %v", err)
	}
	if len(zeroIn) != 3 {
		t.Fatalf("MatVec inDim 0 returned %d values, want 3 zeros", len(zeroIn))
	}
	for i, v := range zeroIn {
		if v != 0 {
			t.Fatalf("MatVec inDim 0 value %d = %v, want 0 (empty reduction)", i, v)
		}
	}
}

// TestGemv_MatVecInto_Good pins the Into contract: the caller-owned output
// backing is used directly (no pooled-scratch write-through) and the values
// match the allocating wrapper.
func TestGemv_MatVecInto_Good(t *testing.T) {
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

// TestGemv_MatVecInto_Bad pins the shape validation through the Into surface:
// the same mat/vec length checks fire regardless of the output the caller
// offers.
func TestGemv_MatVecInto_Bad(t *testing.T) {
	requireNativeRuntime(t)

	out := make([]float32, 2)
	if _, err := MatVecInto(out, []float32{1, 2, 3}, []float32{1, 2}, 2, 2); err == nil {
		t.Fatal("expected MatVecInto to reject matrix length mismatch")
	}
	if _, err := MatVecInto(out, []float32{1, 2, 3, 4}, []float32{1}, 2, 2); err == nil {
		t.Fatal("expected MatVecInto to reject vector length mismatch")
	}
}

// TestGemv_MatVecInto_Ugly pins the fallback allocation contract: a nil or
// under-capacity out cannot be reused, so the call allocates a fresh result —
// still matching the allocating wrapper — and a zero outDim with a caller
// slice returns its empty prefix without dispatching.
func TestGemv_MatVecInto_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim = 16, 64
	mat := syntheticFloat32(outDim*inDim, 3)
	vec := syntheticFloat32(inDim, 5)
	want, err := MatVec(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVec reference: %v", err)
	}

	fromNil, err := MatVecInto(nil, mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVecInto(nil): %v", err)
	}
	assertFloat32Near(t, "MatVecInto(nil)", fromNil, want, 1e-5)

	short := make([]float32, 0, outDim-1)
	fromShort, err := MatVecInto(short, mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVecInto(short): %v", err)
	}
	if len(fromShort) != outDim {
		t.Fatalf("MatVecInto(short) returned %d values, want %d", len(fromShort), outDim)
	}
	assertFloat32Near(t, "MatVecInto(short)", fromShort, want, 1e-5)

	caller := make([]float32, 4)
	empty, err := MatVecInto(caller, nil, vec[:4], 0, 4)
	if err != nil {
		t.Fatalf("MatVecInto outDim 0: %v", err)
	}
	if len(empty) != 0 || unsafe.Pointer(unsafe.SliceData(empty)) != unsafe.Pointer(unsafe.SliceData(caller)) {
		t.Fatal("MatVecInto outDim 0 must return the caller slice's empty prefix")
	}
}
