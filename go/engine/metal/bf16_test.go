// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
)

func rmsNormBF16Fixture(rows, axisSize int) ([]byte, []byte) {
	x := toBF16Bytes(syntheticFloat32(rows*axisSize, axisSize+1))
	w := toBF16Bytes(syntheticFloat32(axisSize, axisSize+7))
	return x, w
}

func matVecBF16Fixture(outDim, inDim int) ([]byte, []byte) {
	mat := toBF16Bytes(syntheticFloat32(outDim*inDim, outDim+3))
	vec := toBF16Bytes(syntheticFloat32(inDim, inDim+5))
	return mat, vec
}

func TestMatVecBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const outDim, inDim = 128, 256
	mat, vec := matVecBF16Fixture(outDim, inDim)
	if _, err := MatVecBF16(mat, vec, outDim, inDim); err != nil {
		t.Fatalf("MatVecBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MatVecBF16(mat, vec, outDim, inDim); err != nil {
			t.Fatalf("MatVecBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MatVecBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestRMSNormBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 4, 512
	const eps = float32(1e-6)
	x, w := rmsNormBF16Fixture(rows, axisSize)
	if _, err := RMSNormBF16(x, w, rows, axisSize, eps); err != nil {
		t.Fatalf("RMSNormBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := RMSNormBF16(x, w, rows, axisSize, eps); err != nil {
			t.Fatalf("RMSNormBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("RMSNormBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestRMSNormBF16IntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 4, 512
	const eps = float32(1e-6)
	x, w := rmsNormBF16Fixture(rows, axisSize)
	want, err := RMSNormBF16(x, w, rows, axisSize, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 reference: %v", err)
	}
	out := make([]byte, len(want))
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVBF16Scratch(rows*axisSize, rows*axisSize)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x6d}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := RMSNormBF16Into(out, x, w, rows, axisSize, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16Into: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("RMSNormBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "RMSNormBF16Into", got, want)

	scratch, err = getQMVBF16Scratch(rows*axisSize, rows*axisSize)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("RMSNormBF16Into wrote through pooled scratch output instead of caller output")
	}
}

func TestRMSNormBF16ViewAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 4, 512
	const eps = float32(1e-6)
	x, w := rmsNormBF16Fixture(rows, axisSize)
	view := bufView{buf: residentBytes(w)}
	if _, err := rmsNormBF16View(x, w, view, rows, axisSize, eps); err != nil {
		t.Fatalf("rmsNormBF16View warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := rmsNormBF16View(x, w, view, rows, axisSize, eps); err != nil {
			t.Fatalf("rmsNormBF16View: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("rmsNormBF16View allocations = %.0f, want <= 10", allocs)
	}
}

func TestRMSNormBF16ViewIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 4, 512
	const eps = float32(1e-6)
	x, w := rmsNormBF16Fixture(rows, axisSize)
	view := bufView{buf: residentBytes(w)}
	want, err := rmsNormBF16View(x, w, view, rows, axisSize, eps)
	if err != nil {
		t.Fatalf("rmsNormBF16View reference: %v", err)
	}
	out := make([]byte, len(want))
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVBF16Scratch(rows*axisSize, rows*axisSize)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x9b}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := rmsNormBF16ViewInto(out, x, w, view, rows, axisSize, eps)
	if err != nil {
		t.Fatalf("rmsNormBF16ViewInto: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("rmsNormBF16ViewInto did not reuse caller-owned output backing")
	}
	eqBytes(t, "rmsNormBF16ViewInto", got, want)

	scratch, err = getQMVBF16Scratch(rows*axisSize, rows*axisSize)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("rmsNormBF16ViewInto wrote through pooled scratch output instead of caller output")
	}
}

func TestAddBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	a := toBF16Bytes(syntheticFloat32(1024, 3))
	b := toBF16Bytes(syntheticFloat32(1024, 5))
	if _, err := AddBF16(a, b); err != nil {
		t.Fatalf("AddBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := AddBF16(a, b); err != nil {
			t.Fatalf("AddBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("AddBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestAddBF16IntoUsesCallerOutput(t *testing.T) {
	requireNativeRuntime(t)

	a := toBF16Bytes(syntheticFloat32(1024, 3))
	b := toBF16Bytes(syntheticFloat32(1024, 5))
	out := make([]byte, len(a))
	for i := range out {
		out[i] = 0xA5
	}

	if err := AddBF16Into(out, a, b); err != nil {
		t.Fatalf("AddBF16Into: %v", err)
	}
	want, err := AddBF16(a, b)
	if err != nil {
		t.Fatalf("AddBF16 reference: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatal("AddBF16Into output differs from allocating wrapper")
	}
}

func TestAddBF16ComputesResidualBytes(t *testing.T) {
	requireNativeRuntime(t)

	a := toBF16Bytes([]float32{1, -2, 0.5})
	b := toBF16Bytes([]float32{3, 2, -0.25})
	got, err := AddBF16(a, b)
	if err != nil {
		t.Fatalf("AddBF16: %v", err)
	}
	want := toBF16Bytes([]float32{4, 0, 0.25})
	if !bytes.Equal(got, want) {
		t.Fatalf("AddBF16 bytes = %v (%v), want %v (%v)", got, bf16Floats(got), want, bf16Floats(want))
	}
}

func TestBF16ShapeContracts(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := AddBF16([]byte{0}, []byte{0}); err == nil {
		t.Fatal("expected AddBF16 to reject odd byte length")
	}
	if _, err := MatVecBF16(toBF16Bytes([]float32{1, 2, 3}), toBF16Bytes([]float32{1, 2}), 2, 2); err == nil {
		t.Fatal("expected MatVecBF16 to reject matrix byte length mismatch")
	}
	if _, err := RoPEDimsBF16(toBF16Bytes([]float32{1, 2, 3, 4}), 1, 1, 4, 3, 10000, 1, 0, false); err == nil {
		t.Fatal("expected RoPEDimsBF16 to reject odd rotaryDim")
	}
}

func TestBF16IdentityKernels(t *testing.T) {
	requireNativeRuntime(t)

	x := toBF16Bytes([]float32{1, -2, 3, -4})
	rope, err := RoPEBF16(x, 1, 1, 4, 10000, 1, 0, false)
	if err != nil {
		t.Fatalf("RoPEBF16: %v", err)
	}
	if !bytes.Equal(rope, x) {
		t.Fatalf("RoPEBF16 offset zero changed values: got %v want %v", bf16Floats(rope), bf16Floats(x))
	}

	normInput := toBF16Bytes([]float32{1, 1})
	normWeight := toBF16Bytes([]float32{1, 1})
	norm, err := RMSNormBF16(normInput, normWeight, 1, 2, 0)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	if !bytes.Equal(norm, normInput) {
		t.Fatalf("RMSNormBF16 unit vector = %v, want %v", bf16Floats(norm), bf16Floats(normInput))
	}
}
