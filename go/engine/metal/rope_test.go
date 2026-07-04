// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
)

func TestRoPEOffsetZeroIsIdentity(t *testing.T) {
	requireNativeRuntime(t)

	x := []float32{1, 2, 3, 4, -1, -2, -3, -4}
	got, err := RoPE(x, 1, 2, 4, 10000, 1, 0, false)
	if err != nil {
		t.Fatalf("RoPE: %v", err)
	}
	assertFloat32Near(t, "RoPE offset zero", got, x, 0)
}

func TestRoPERejectsShapeMismatch(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := RoPE([]float32{1, 2, 3}, 1, 2, 4, 10000, 1, 0, false); err == nil {
		t.Fatal("expected RoPE to reject input length mismatch")
	}
}

func TestRoPEIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const batch, nHeads, headDim = 1, 8, 64
	x := syntheticFloat32(batch*nHeads*headDim, 3)
	want, err := RoPE(x, batch, nHeads, headDim, 10000, 1, 17, false)
	if err != nil {
		t.Fatalf("RoPE reference: %v", err)
	}
	out := syntheticFloat32(len(x), 11)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVFloatScratch(len(x), len(x))
	if err != nil {
		t.Fatalf("getQMVFloatScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x8e}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVFloatScratch(scratch)

	got, err := RoPEInto(out, x, batch, nHeads, headDim, 10000, 1, 17, false)
	if err != nil {
		t.Fatalf("RoPEInto: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("RoPEInto did not reuse caller-owned output backing")
	}
	if !bytes.Equal(float32Bytes(got), float32Bytes(want)) {
		t.Fatal("RoPEInto output differs from allocating wrapper")
	}

	scratch, err = getQMVFloatScratch(len(x), len(x))
	if err != nil {
		t.Fatalf("getQMVFloatScratch after call: %v", err)
	}
	defer putQMVFloatScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("RoPEInto wrote through pooled scratch output instead of caller output")
	}
}

func TestRoPEAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	x := syntheticFloat32(8*64, 3)
	if _, err := RoPE(x, 1, 8, 64, 10000, 1, 17, false); err != nil {
		t.Fatalf("RoPE warmup: %v", err)
	}

	var ropeErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, ropeErr = RoPE(x, 1, 8, 64, 10000, 1, 17, false)
	})
	if ropeErr != nil {
		t.Fatalf("RoPE: %v", ropeErr)
	}
	if allocs > 10 {
		t.Fatalf("RoPE allocations = %.0f, want <= 10", allocs)
	}
}
