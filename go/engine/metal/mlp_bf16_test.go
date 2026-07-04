// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
)

func TestMLPBF16PrimitiveKernels(t *testing.T) {
	requireNativeRuntime(t)

	a := toBF16Bytes([]float32{2, -3, 0.5})
	b := toBF16Bytes([]float32{3, -2, -1})
	mul, err := MulBF16(a, b)
	if err != nil {
		t.Fatalf("MulBF16: %v", err)
	}
	wantMul := toBF16Bytes([]float32{6, 6, -0.5})
	if !bytes.Equal(mul, wantMul) {
		t.Fatalf("MulBF16 = %v, want %v", bf16Floats(mul), bf16Floats(wantMul))
	}

	zeros := toBF16Bytes([]float32{0, 0, 0})
	for name, fn := range map[string]func([]byte) ([]byte, error){
		"TanhBF16": TanhBF16,
		"GeluBF16": GeluBF16,
	} {
		got, err := fn(zeros)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !bytes.Equal(got, zeros) {
			t.Fatalf("%s zeros = %v, want zeros", name, bf16Floats(got))
		}
	}

	gated, err := GeluGateMulBF16(zeros, b)
	if err != nil {
		t.Fatalf("GeluGateMulBF16: %v", err)
	}
	assertFloat32Near(t, "GeluGateMulBF16 zero gate", bf16Floats(gated), []float32{0, 0, 0}, 0)
}

func TestGeluGateMulBF16RejectsLengthMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := GeluGateMulBF16(toBF16Bytes([]float32{1, 2}), toBF16Bytes([]float32{1})); err == nil {
		t.Fatal("expected GeluGateMulBF16 to reject mismatched lengths")
	}
}

func TestGeluGateMulBF16ComposedRejectsOddByteLength(t *testing.T) {
	requireNativeRuntime(t)
	withComposedGELU(t)

	if _, err := GeluGateMulBF16([]byte{1}, []byte{1}); err == nil {
		t.Fatal("expected GeluGateMulBF16 composed path to reject odd byte length")
	}
}

func TestGeluGateMulBF16EmptyInput(t *testing.T) {
	requireNativeRuntime(t)

	out, err := GeluGateMulBF16(nil, nil)
	if err != nil {
		t.Fatalf("GeluGateMulBF16 empty: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("GeluGateMulBF16 empty len = %d, want 0", len(out))
	}
}

func TestGeluGateMulBF16FusedAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom GELU kernel unavailable")
	}

	const n = 1024
	gate := toBF16Bytes(syntheticFloat32(n, 3))
	up := toBF16Bytes(syntheticFloat32(n, 5))
	if _, err := GeluGateMulBF16(gate, up); err != nil {
		t.Fatalf("GeluGateMulBF16 fused warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := GeluGateMulBF16(gate, up); err != nil {
			t.Fatalf("GeluGateMulBF16 fused: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("GeluGateMulBF16 fused allocations = %.0f, want <= 10", allocs)
	}
}

func TestMulBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const n = 1024
	a := toBF16Bytes(syntheticFloat32(n, 3))
	b := toBF16Bytes(syntheticFloat32(n, 5))
	if _, err := MulBF16(a, b); err != nil {
		t.Fatalf("MulBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MulBF16(a, b); err != nil {
			t.Fatalf("MulBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MulBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestMulBF16IntoUsesCallerOutput(t *testing.T) {
	requireNativeRuntime(t)

	const n = 1024
	a := toBF16Bytes(syntheticFloat32(n, 3))
	b := toBF16Bytes(syntheticFloat32(n, 5))
	out := make([]byte, len(a))
	for i := range out {
		out[i] = 0xA5
	}

	if err := MulBF16Into(out, a, b); err != nil {
		t.Fatalf("MulBF16Into: %v", err)
	}
	want, err := MulBF16(a, b)
	if err != nil {
		t.Fatalf("MulBF16 reference: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatal("MulBF16Into output differs from allocating wrapper")
	}
}

func TestGeluGateMulBF16ComposedAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	withComposedGELU(t)

	const n = 1024
	gate := toBF16Bytes(syntheticFloat32(n, 3))
	up := toBF16Bytes(syntheticFloat32(n, 5))
	if _, err := GeluGateMulBF16(gate, up); err != nil {
		t.Fatalf("GeluGateMulBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := GeluGateMulBF16(gate, up); err != nil {
			t.Fatalf("GeluGateMulBF16: %v", err)
		}
	})
	if allocs > 55 {
		t.Fatalf("GeluGateMulBF16 allocations = %.0f, want <= 55", allocs)
	}
}

func TestGeluGateMulBF16IntoComposedUsesCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	withComposedGELU(t)

	const n = 1024
	gate := toBF16Bytes(syntheticFloat32(n, 3))
	up := toBF16Bytes(syntheticFloat32(n, 5))
	out := make([]byte, len(gate))
	for i := range out {
		out[i] = 0xA5
	}
	want, err := GeluGateMulBF16(gate, up)
	if err != nil {
		t.Fatalf("GeluGateMulBF16 reference: %v", err)
	}

	scratch, err := getBinaryByteScratch(len(gate))
	if err != nil {
		t.Fatalf("getBinaryByteScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x9B}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putBinaryByteScratch(scratch)

	if err := GeluGateMulBF16Into(out, gate, up); err != nil {
		t.Fatalf("GeluGateMulBF16Into: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatal("GeluGateMulBF16Into output differs from allocating wrapper")
	}

	scratch, err = getBinaryByteScratch(len(gate))
	if err != nil {
		t.Fatalf("getBinaryByteScratch after call: %v", err)
	}
	defer putBinaryByteScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("GeluGateMulBF16Into wrote through pooled scratch output instead of caller output")
	}
}

func TestTanhBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const n = 1024
	x := toBF16Bytes(syntheticFloat32(n, 3))
	if _, err := TanhBF16(x); err != nil {
		t.Fatalf("TanhBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := TanhBF16(x); err != nil {
			t.Fatalf("TanhBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("TanhBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestTanhBF16IntoUsesCallerOutput(t *testing.T) {
	requireNativeRuntime(t)

	const n = 1024
	x := toBF16Bytes(syntheticFloat32(n, 3))
	out := make([]byte, len(x))
	for i := range out {
		out[i] = 0xA5
	}

	if err := TanhBF16Into(out, x); err != nil {
		t.Fatalf("TanhBF16Into: %v", err)
	}
	want, err := TanhBF16(x)
	if err != nil {
		t.Fatalf("TanhBF16 reference: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatal("TanhBF16Into output differs from allocating wrapper")
	}
}

func TestMulBF16ConstAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const n = 1024
	x := toBF16Bytes(syntheticFloat32(n, 3))
	if _, err := mulBF16Const(x, n, 0.375); err != nil {
		t.Fatalf("mulBF16Const warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := mulBF16Const(x, n, 0.375); err != nil {
			t.Fatalf("mulBF16Const: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("mulBF16Const allocations = %.0f, want <= 10", allocs)
	}
}

func TestMulBF16ConstIntoUsesCallerOutput(t *testing.T) {
	requireNativeRuntime(t)

	const n = 1024
	x := toBF16Bytes(syntheticFloat32(n, 3))
	out := make([]byte, len(x))
	for i := range out {
		out[i] = 0xA5
	}
	want, err := mulBF16Const(x, n, 0.375)
	if err != nil {
		t.Fatalf("mulBF16Const reference: %v", err)
	}

	scratch, err := getQMVBF16Scratch(n, n)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x9B}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	if err := mulBF16ConstInto(x, n, 0.375, out); err != nil {
		t.Fatalf("mulBF16ConstInto: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatal("mulBF16ConstInto output differs from allocating wrapper")
	}

	scratch, err = getQMVBF16Scratch(n, n)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("mulBF16ConstInto wrote through pooled scratch output instead of caller output")
	}
}

func TestGeluBF16SingleCommandAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const n = 1024
	x := toBF16Bytes(syntheticFloat32(n, 3))
	if _, err := GeluBF16(x); err != nil {
		t.Fatalf("GeluBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := GeluBF16(x); err != nil {
			t.Fatalf("GeluBF16: %v", err)
		}
	})
	if allocs > 40 {
		t.Fatalf("GeluBF16 allocations = %.0f, want <= 40", allocs)
	}
}

func TestGeluBF16IntoUsesCallerOutput(t *testing.T) {
	requireNativeRuntime(t)

	const n = 1024
	x := toBF16Bytes(syntheticFloat32(n, 3))
	out := make([]byte, len(x))
	for i := range out {
		out[i] = 0xA5
	}

	if err := GeluBF16Into(out, x); err != nil {
		t.Fatalf("GeluBF16Into: %v", err)
	}
	want, err := GeluBF16(x)
	if err != nil {
		t.Fatalf("GeluBF16 reference: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatal("GeluBF16Into output differs from allocating wrapper")
	}
}

func TestGeluBF16KeepsConstantsResident(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const n = 16
	x := toBF16Bytes(syntheticFloat32(n, 3))
	if _, err := GeluBF16(x); err != nil {
		t.Fatalf("GeluBF16: %v", err)
	}

	consts := []struct {
		name string
		buf  []byte
	}{
		{"c044", bf16ConstBytes(n, 0.044715)},
		{"c079", bf16ConstBytes(n, 0.7978845608028654)},
		{"c1", bf16ConstBytes(n, 1.0)},
		{"c05", bf16ConstBytes(n, 0.5)},
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	defer residentBufMu.Unlock()
	for _, c := range consts {
		if _, ok := residentBufs[key(c.buf)]; !ok {
			t.Fatalf("GeluBF16 constant %s was not resident", c.name)
		}
	}
}
