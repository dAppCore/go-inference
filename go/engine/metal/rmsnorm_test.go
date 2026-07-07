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

// rmsNormHostReference computes the RMSNorm contract on the host in float64:
// out[r,i] = x[r,i] * rsqrt(mean_i(x[r,:]²) + eps) * w[i].
func rmsNormHostReference(x, w []float32, rows, axisSize int, eps float32) []float32 {
	out := make([]float32, len(x))
	for r := range rows {
		row := x[r*axisSize : (r+1)*axisSize]
		var sum float64
		for _, v := range row {
			sum += float64(v) * float64(v)
		}
		inv := 1 / math.Sqrt(sum/float64(axisSize)+float64(eps))
		for i, v := range row {
			out[r*axisSize+i] = float32(float64(v) * inv * float64(w[i]))
		}
	}
	return out
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

// TestRmsnorm_RMSNorm_Good pins the mathematical contract on a hand-checkable
// row: out[r,i] = x[r,i] * rsqrt(mean(x[r,:]²) + eps) * weight[i].
func TestRmsnorm_RMSNorm_Good(t *testing.T) {
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

// TestRmsnorm_RMSNorm_Bad pins the shape validation: x must be rows*axisSize
// and weight must be axisSize, each rejected with an error rather than a
// mis-shaped dispatch.
func TestRmsnorm_RMSNorm_Bad(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := RMSNorm([]float32{1, 2, 3}, []float32{1, 2}, 2, 2, 1e-5); err == nil {
		t.Fatal("expected RMSNorm to reject x length mismatch")
	}
	if _, err := RMSNorm([]float32{1, 2, 3, 4}, []float32{1}, 2, 2, 1e-5); err == nil {
		t.Fatal("expected RMSNorm to reject weight length mismatch")
	}
}

// TestRmsnorm_RMSNorm_Ugly pins the corners: an axis above rmsLoopedLimit must
// route to the LOOPED kernel and stay numerically correct (the single-row
// kernel's threadgroup would exceed Metal's 1024-thread cap there — the 31B
// hidden_size=5376 invalid-dispatch scar), and zero rows is a valid no-op.
func TestRmsnorm_RMSNorm_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 2, rmsLoopedLimit + 1280 // 5376: gemma4-31B hidden, looped lane
	x, w := rmsNormFixture(rows, axisSize)
	got, err := RMSNorm(x, w, rows, axisSize, 1e-5)
	if err != nil {
		t.Fatalf("RMSNorm looped lane: %v", err)
	}
	assertFloat32Near(t, "RMSNorm looped", got, rmsNormHostReference(x, w, rows, axisSize, 1e-5), 1e-4)

	empty, err := RMSNorm(nil, nil, 0, 0, 1e-5)
	if err != nil {
		t.Fatalf("RMSNorm zero rows: %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("RMSNorm zero rows returned %d values, want none", len(empty))
	}
}

// TestRmsnorm_RMSNormInto_Good pins the Into contract: the caller-owned output
// backing is used directly (no pooled-scratch write-through) and the values
// match the allocating wrapper byte-for-byte.
func TestRmsnorm_RMSNormInto_Good(t *testing.T) {
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

// TestRmsnorm_RMSNormInto_Bad pins the shape validation through the Into
// surface: the same x/weight length checks fire regardless of the output the
// caller offers.
func TestRmsnorm_RMSNormInto_Bad(t *testing.T) {
	requireNativeRuntime(t)

	out := make([]float32, 4)
	if _, err := RMSNormInto(out, []float32{1, 2, 3}, []float32{1, 2}, 2, 2, 1e-5); err == nil {
		t.Fatal("expected RMSNormInto to reject x length mismatch")
	}
	if _, err := RMSNormInto(out, []float32{1, 2, 3, 4}, []float32{1}, 2, 2, 1e-5); err == nil {
		t.Fatal("expected RMSNormInto to reject weight length mismatch")
	}
}

// TestRmsnorm_RMSNormInto_Ugly pins the fallback allocation contract: a nil or
// under-capacity out cannot be reused, so the call allocates a fresh result —
// still numerically identical to the allocating wrapper.
func TestRmsnorm_RMSNormInto_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 2, 64
	x, w := rmsNormFixture(rows, axisSize)
	want, err := RMSNorm(x, w, rows, axisSize, 1e-5)
	if err != nil {
		t.Fatalf("RMSNorm reference: %v", err)
	}

	fromNil, err := RMSNormInto(nil, x, w, rows, axisSize, 1e-5)
	if err != nil {
		t.Fatalf("RMSNormInto(nil): %v", err)
	}
	if !bytes.Equal(float32Bytes(fromNil), float32Bytes(want)) {
		t.Fatal("RMSNormInto(nil) output differs from allocating wrapper")
	}

	short := make([]float32, 0, len(x)-1)
	fromShort, err := RMSNormInto(short, x, w, rows, axisSize, 1e-5)
	if err != nil {
		t.Fatalf("RMSNormInto(short): %v", err)
	}
	if len(fromShort) != len(x) {
		t.Fatalf("RMSNormInto(short) returned %d values, want %d", len(fromShort), len(x))
	}
	if !bytes.Equal(float32Bytes(fromShort), float32Bytes(want)) {
		t.Fatal("RMSNormInto(short) output differs from allocating wrapper")
	}
}
