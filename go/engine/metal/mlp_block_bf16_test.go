// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
)

func TestMLPBlockBF16MatchesComposedPrimitives(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 4, 4
	x := toBF16Bytes([]float32{1, -2, 3, -4})
	normW := toBF16Bytes([]float32{1, 1, 1, 1})
	wGate := toBF16Bytes([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	})
	wUp := toBF16Bytes([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	})
	wDown := wUp

	got, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 0)
	if err != nil {
		t.Fatalf("MLPBlockBF16: %v", err)
	}
	normed, err := RMSNormBF16(x, normW, 1, dModel, 0)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	gate, err := MatVecBF16(wGate, normed, dFF, dModel)
	if err != nil {
		t.Fatalf("gate MatVecBF16: %v", err)
	}
	up, err := MatVecBF16(wUp, normed, dFF, dModel)
	if err != nil {
		t.Fatalf("up MatVecBF16: %v", err)
	}
	gated, err := GeluGateMulBF16(gate, up)
	if err != nil {
		t.Fatalf("GeluGateMulBF16: %v", err)
	}
	down, err := MatVecBF16(wDown, gated, dModel, dFF)
	if err != nil {
		t.Fatalf("down MatVecBF16: %v", err)
	}
	want, err := AddBF16(x, down)
	if err != nil {
		t.Fatalf("AddBF16: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("MLPBlockBF16 = %v, want composed primitives %v", bf16Floats(got), bf16Floats(want))
	}
}

func TestMLPBlockBF16IntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 13))
	want, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5)
	if err != nil {
		t.Fatalf("MLPBlockBF16 reference: %v", err)
	}
	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVBF16Scratch(dModel, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := MLPBlockBF16Into(out, x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5)
	if err != nil {
		t.Fatalf("MLPBlockBF16Into: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MLPBlockBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "MLPBlockBF16Into", got, want)

	scratch, err = getQMVBF16Scratch(dModel, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("MLPBlockBF16Into wrote through pooled scratch output instead of caller output")
	}
}

func TestMLPBlockBF16RejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := MLPBlockBF16(toBF16Bytes([]float32{1}), toBF16Bytes([]float32{1}), nil, nil, nil, 2, 2, 1e-5); err == nil {
		t.Fatal("expected MLPBlockBF16 to reject x/normWeight shape mismatch")
	}
}

func TestMLPBlockBF16KeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF = 8, 16
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 13))

	if _, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
		t.Fatalf("MLPBlockBF16: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	got := len(residentBufs)
	_, hasNorm := residentBufs[key(normW)]
	_, hasGate := residentBufs[key(wGate)]
	_, hasUp := residentBufs[key(wUp)]
	_, hasDown := residentBufs[key(wDown)]
	residentBufMu.Unlock()

	if !hasNorm || !hasGate || !hasUp || !hasDown {
		t.Fatalf("MLPBlockBF16 did not keep fixed weights resident (norm=%v gate=%v up=%v down=%v resident=%d want>=4)", hasNorm, hasGate, hasUp, hasDown, got)
	}
}

func TestMLPBlockBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 13))
	if _, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
		t.Fatalf("MLPBlockBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
			t.Fatalf("MLPBlockBF16: %v", err)
		}
	})
	if allocs > 15 {
		t.Fatalf("MLPBlockBF16 allocations = %.0f, want <= 15", allocs)
	}
}

func TestMLPBlockBF16ComposedKeepsGELUConstantsResident(t *testing.T) {
	requireNativeRuntime(t)
	old := customLibraryLoaded
	customLibraryLoaded = false
	t.Cleanup(func() { customLibraryLoaded = old })

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF = 8, 16
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 13))

	if _, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
		t.Fatalf("MLPBlockBF16 composed: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	consts := []struct {
		name string
		buf  []byte
	}{
		{"c044", bf16ConstBytes(dFF, 0.044715)},
		{"c079", bf16ConstBytes(dFF, 0.7978845608028654)},
		{"c1", bf16ConstBytes(dFF, 1.0)},
		{"c05", bf16ConstBytes(dFF, 0.5)},
	}

	residentBufMu.Lock()
	defer residentBufMu.Unlock()
	for _, c := range consts {
		if _, ok := residentBufs[key(c.buf)]; !ok {
			t.Fatalf("MLPBlockBF16 composed GELU constant %s was not resident", c.name)
		}
	}
}

func TestMLPBlockBF16ComposedAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	withComposedGELU(t)

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 13))
	if _, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
		t.Fatalf("MLPBlockBF16 composed warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, 1e-5); err != nil {
			t.Fatalf("MLPBlockBF16 composed: %v", err)
		}
	})
	if allocs > 1535 {
		t.Fatalf("MLPBlockBF16 composed allocations = %.0f, want <= 1535", allocs)
	}
}
