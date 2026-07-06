// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"
)

func TestSoftmaxF32AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 8, 512
	x := syntheticFloat32(rows*axisSize, 5)
	if _, err := SoftmaxF32(x, axisSize); err != nil {
		t.Fatalf("SoftmaxF32 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := SoftmaxF32(x, axisSize); err != nil {
			t.Fatalf("SoftmaxF32: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("SoftmaxF32 allocations = %.0f, want <= 10", allocs)
	}
}

// TestSoftmaxF32 (BYTE-IDENTICAL to pkg/metal.Softmax) lives in softmax_metal_test.go —
// it needs the real cgo metal package as its oracle, so it's gated behind metal_runtime.

func TestSoftmaxF32IntoReusesOutputBackingAndMatchesSoftmaxF32(t *testing.T) {
	requireNativeRuntime(t)

	const rows, ax = 8, 512
	x := syntheticFloat32(rows*ax, 5)
	want, err := SoftmaxF32(x, ax)
	if err != nil {
		t.Fatalf("SoftmaxF32 reference: %v", err)
	}
	out := syntheticFloat32(rows*ax, 11)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVFloatScratch(len(x), len(x))
	if err != nil {
		t.Fatalf("getQMVFloatScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVFloatScratch(scratch)

	got, err := SoftmaxF32Into(out, x, ax)
	if err != nil {
		t.Fatalf("SoftmaxF32Into: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("SoftmaxF32Into did not reuse caller-owned output backing")
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("SoftmaxF32Into differs at %d: %v vs %v", i, got[i], want[i])
		}
	}

	scratch, err = getQMVFloatScratch(len(x), len(x))
	if err != nil {
		t.Fatalf("getQMVFloatScratch after call: %v", err)
	}
	defer putQMVFloatScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("SoftmaxF32Into wrote through pooled scratch output instead of caller output")
	}
}

func TestSoftmaxF32LoopedAxis(t *testing.T) {
	requireNativeRuntime(t)
	const rows, ax = 2, 5000
	x := syntheticFloat32(rows*ax, 17)

	got, err := SoftmaxF32(x, ax)
	if err != nil {
		t.Fatalf("SoftmaxF32 looped axis: %v", err)
	}
	want := hostSoftmaxF32(x, rows, ax)
	assertFloat32Near(t, "SoftmaxF32 looped axis", got, want, 1e-5)

	for r := range rows {
		sum := float32(0)
		for _, v := range got[r*ax : (r+1)*ax] {
			sum += v
		}
		if d := math.Abs(float64(sum - 1)); d > 2e-4 {
			t.Fatalf("SoftmaxF32 looped row %d sum = %.8f, want 1", r, sum)
		}
	}
}

func hostSoftmaxF32(in []float32, rows, axisSize int) []float32 {
	out := make([]float32, len(in))
	for r := range rows {
		row := in[r*axisSize : (r+1)*axisSize]
		maxV := row[0]
		for _, v := range row[1:] {
			if v > maxV {
				maxV = v
			}
		}
		var denom float64
		for _, v := range row {
			denom += math.Exp(float64(v - maxV))
		}
		dst := out[r*axisSize : (r+1)*axisSize]
		for i, v := range row {
			dst[i] = float32(math.Exp(float64(v-maxV)) / denom)
		}
	}
	return out
}
