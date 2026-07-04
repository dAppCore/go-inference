// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"
)

func rmsNormFixture(rows, axisSize int) ([]float32, []float32) {
	x := syntheticFloat32(rows*axisSize, axisSize+1)
	w := syntheticFloat32(axisSize, axisSize+7)
	return x, w
}

func TestRMSNormAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 4, 1024
	x, w := rmsNormFixture(rows, axisSize)
	if _, err := RMSNorm(x, w, rows, axisSize, 1e-5); err != nil {
		t.Fatalf("RMSNorm warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := RMSNorm(x, w, rows, axisSize, 1e-5); err != nil {
			t.Fatalf("RMSNorm: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("RMSNorm allocations = %.0f, want <= 10", allocs)
	}
}

func TestRMSNormComputesScaledRows(t *testing.T) {
	requireNativeRuntime(t)

	x := []float32{3, 4}
	weight := []float32{2, 4}
	got, err := RMSNorm(x, weight, 1, 2, 0)
	if err != nil {
		t.Fatalf("RMSNorm: %v", err)
	}
	rms := float32(math.Sqrt((9 + 16) / 2.0))
	want := []float32{3 / rms * 2, 4 / rms * 4}
	assertFloat32Near(t, "RMSNorm", got, want, 1e-5)
}

func TestRMSNormIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 4, 1024
	x, w := rmsNormFixture(rows, axisSize)
	want, err := RMSNorm(x, w, rows, axisSize, 1e-5)
	if err != nil {
		t.Fatalf("RMSNorm reference: %v", err)
	}

	out := make([]float32, len(x))
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVFloatScratch(len(x), len(x))
	if err != nil {
		t.Fatalf("getQMVFloatScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVFloatScratch(scratch)

	got, err := RMSNormInto(out, x, w, rows, axisSize, 1e-5)
	if err != nil {
		t.Fatalf("RMSNormInto: %v", err)
	}
	if len(got) != len(out) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("RMSNormInto did not reuse caller-owned output backing")
	}
	if !bytes.Equal(float32Bytes(got), float32Bytes(want)) {
		t.Fatal("RMSNormInto output differs from allocating wrapper")
	}

	scratch, err = getQMVFloatScratch(len(x), len(x))
	if err != nil {
		t.Fatalf("getQMVFloatScratch after call: %v", err)
	}
	defer putQMVFloatScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("RMSNormInto wrote through pooled scratch output instead of caller output")
	}
}

func TestRMSNormRejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := RMSNorm([]float32{1, 2, 3}, []float32{1, 2}, 2, 2, 1e-5); err == nil {
		t.Fatal("expected RMSNorm to reject x length mismatch")
	}
}
