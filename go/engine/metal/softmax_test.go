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

// The BYTE-IDENTICAL oracle test against pkg/metal.Softmax lives in
// softmax_metal_test.go — it needs the real cgo metal package as its oracle,
// so it's gated behind metal_runtime.

// TestSoftmax_SoftmaxF32_Good pins the block-kernel lane's mathematical
// contract on the row-major [rows,axis] shape: each row matches the float64
// host softmax and sums to 1.
func TestSoftmax_SoftmaxF32_Good(t *testing.T) {
	requireNativeRuntime(t)

	const rows, ax = 3, 8
	x := syntheticFloat32(rows*ax, 5)
	got, err := SoftmaxF32(x, ax)
	if err != nil {
		t.Fatalf("SoftmaxF32: %v", err)
	}
	assertFloat32Near(t, "SoftmaxF32", got, hostSoftmaxF32(x, rows, ax), 1e-6)
	for r := range rows {
		sum := float32(0)
		for _, v := range got[r*ax : (r+1)*ax] {
			sum += v
		}
		if d := math.Abs(float64(sum - 1)); d > 1e-5 {
			t.Fatalf("SoftmaxF32 row %d sum = %.8f, want 1", r, sum)
		}
	}
}

// TestSoftmax_SoftmaxF32_Bad pins the shape validation: a zero axis and an
// input that is not a whole number of rows are each rejected with an error
// rather than a mis-shaped dispatch.
func TestSoftmax_SoftmaxF32_Bad(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := SoftmaxF32([]float32{1, 2}, 0); err == nil {
		t.Fatal("expected SoftmaxF32 to reject axisSize 0")
	}
	if _, err := SoftmaxF32([]float32{1, 2, 3}, 2); err == nil {
		t.Fatal("expected SoftmaxF32 to reject len(in) not a multiple of axisSize")
	}
}

// TestSoftmax_SoftmaxF32_Ugly pins the corners: an axis above
// softmaxLoopedLimit must route to the LOOPED kernel and stay numerically
// correct (rows still sum to 1), and an empty input is a valid no-op.
func TestSoftmax_SoftmaxF32_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	const rows, ax = 2, softmaxLoopedLimit + 904 // 5000: looped lane
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

	empty, err := SoftmaxF32(nil, 4)
	if err != nil {
		t.Fatalf("SoftmaxF32 zero rows: %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("SoftmaxF32 zero rows returned %d values, want none", len(empty))
	}
}

// TestSoftmax_SoftmaxF32Into_Good pins the Into contract: the caller-owned
// output backing is used directly (no pooled-scratch write-through) and the
// values are bit-identical to the allocating wrapper.
func TestSoftmax_SoftmaxF32Into_Good(t *testing.T) {
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

// TestSoftmax_SoftmaxF32Into_Bad pins the shape validation through the Into
// surface: the same axis/length checks fire regardless of the output the
// caller offers.
func TestSoftmax_SoftmaxF32Into_Bad(t *testing.T) {
	requireNativeRuntime(t)

	out := make([]float32, 4)
	if _, err := SoftmaxF32Into(out, []float32{1, 2}, 0); err == nil {
		t.Fatal("expected SoftmaxF32Into to reject axisSize 0")
	}
	if _, err := SoftmaxF32Into(out, []float32{1, 2, 3}, 2); err == nil {
		t.Fatal("expected SoftmaxF32Into to reject len(in) not a multiple of axisSize")
	}
}

// TestSoftmax_SoftmaxF32Into_Ugly pins the fallback allocation contract: a nil
// or under-capacity out cannot be reused, so the call allocates a fresh result
// — still bit-identical to the allocating wrapper.
func TestSoftmax_SoftmaxF32Into_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const rows, ax = 2, 64
	x := syntheticFloat32(rows*ax, 7)
	want, err := SoftmaxF32(x, ax)
	if err != nil {
		t.Fatalf("SoftmaxF32 reference: %v", err)
	}

	fromNil, err := SoftmaxF32Into(nil, x, ax)
	if err != nil {
		t.Fatalf("SoftmaxF32Into(nil): %v", err)
	}
	if !bytes.Equal(float32Bytes(fromNil), float32Bytes(want)) {
		t.Fatal("SoftmaxF32Into(nil) output differs from allocating wrapper")
	}

	short := make([]float32, 0, len(x)-1)
	fromShort, err := SoftmaxF32Into(short, x, ax)
	if err != nil {
		t.Fatalf("SoftmaxF32Into(short): %v", err)
	}
	if len(fromShort) != len(x) {
		t.Fatalf("SoftmaxF32Into(short) returned %d values, want %d", len(fromShort), len(x))
	}
	if !bytes.Equal(float32Bytes(fromShort), float32Bytes(want)) {
		t.Fatal("SoftmaxF32Into(short) output differs from allocating wrapper")
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
