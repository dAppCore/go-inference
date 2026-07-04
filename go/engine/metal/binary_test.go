// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

func TestRunBinaryAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	a := syntheticFloat32(1024, 3)
	b := syntheticFloat32(1024, 5)
	if _, err := Add(a, b); err != nil {
		t.Fatalf("Add warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := Add(a, b); err != nil {
			t.Fatalf("Add: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("Add allocations = %.0f, want <= 10", allocs)
	}
}

func TestBinaryByteScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getBinaryByteScratch(128)
	if err != nil {
		t.Fatalf("get small binary scratch: %v", err)
	}
	putBinaryByteScratch(small)

	large, err := getBinaryByteScratch(256)
	if err != nil {
		t.Fatalf("get large binary scratch: %v", err)
	}
	putBinaryByteScratch(large)

	gotSmall, err := getBinaryByteScratch(128)
	if err != nil {
		t.Fatalf("get small binary scratch again: %v", err)
	}
	defer putBinaryByteScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("binary scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge, err := getBinaryByteScratch(256)
	if err != nil {
		t.Fatalf("get large binary scratch again: %v", err)
	}
	defer putBinaryByteScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("binary scratch pool evicted the large scratch after reusing the small scratch")
	}
}

func TestBinaryByteScratchBuffersUseCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	a := toBF16Bytes(syntheticFloat32(1024, 3))
	b := toBF16Bytes(syntheticFloat32(1024, 5))
	scratch, err := getBinaryByteScratch(len(a))
	if err != nil {
		t.Fatalf("getBinaryByteScratch: %v", err)
	}
	defer scratch.Close()

	var aBuf, bBuf metal.MTLBuffer
	for i := 0; i < 3; i++ {
		aBuf, bBuf, _, err = scratch.buffers(a, b)
		if err != nil {
			t.Fatalf("scratch.buffers warmup %d: %v", i, err)
		}
	}
	if got, want := uintptr(aBuf.Contents()), uintptr(unsafe.Pointer(&a[0])); got != want {
		t.Fatalf("a buffer pointer = %#x, want caller backing %#x", got, want)
	}
	if got, want := uintptr(bBuf.Contents()), uintptr(unsafe.Pointer(&b[0])); got != want {
		t.Fatalf("b buffer pointer = %#x, want caller backing %#x", got, want)
	}
	reusedA, reusedB, _, err := scratch.buffers(a, b)
	if err != nil {
		t.Fatalf("scratch.buffers reused: %v", err)
	}
	if reusedA.GetID() != aBuf.GetID() || reusedB.GetID() != bBuf.GetID() {
		t.Fatal("scratch.buffers did not reuse cached no-copy input views")
	}
}

func TestBinaryByteScratchOutputViewReusesPinnedOwnerBuffer(t *testing.T) {
	requireNativeRuntime(t)

	pinned, err := newPinnedNoCopyBytes(1024 * bf16Size)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes: %v", err)
	}
	defer pinned.Close()

	scratch, err := getBinaryByteScratch(len(pinned.bytes))
	if err != nil {
		t.Fatalf("getBinaryByteScratch: %v", err)
	}
	defer scratch.Close()

	outBuf, ok := scratch.outputView(pinned.bytes)
	if !ok {
		t.Fatal("binary output view did not accept pinned caller bytes")
	}
	if got, want := outBuf.GetID(), pinned.buf.GetID(); got != want {
		t.Fatalf("binary output view buffer id = %d, want pinned owner buffer %d", got, want)
	}
	if got, want := uintptr(outBuf.Contents()), uintptr(unsafe.Pointer(&pinned.bytes[0])); got != want {
		t.Fatalf("binary output view pointer = %#x, want pinned backing %#x", got, want)
	}
}

func TestBinaryFloat32Kernels(t *testing.T) {
	requireNativeRuntime(t)

	a := []float32{-3, -2, 0, 4}
	b := []float32{10, -2, 5, 0.25}
	tests := []struct {
		name string
		fn   func([]float32, []float32) ([]float32, error)
		want []float32
	}{
		{name: "Add", fn: Add, want: []float32{7, -4, 5, 4.25}},
		{name: "Mul", fn: Mul, want: []float32{-30, 4, 0, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.fn(a, b)
			if err != nil {
				t.Fatalf("%s: %v", tt.name, err)
			}
			assertFloat32Near(t, tt.name, got, tt.want, 0)
		})
	}
}

func TestRunBinaryIntoBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	a := syntheticFloat32(1024, 3)
	b := syntheticFloat32(1024, 5)
	want, err := Add(a, b)
	if err != nil {
		t.Fatalf("Add reference: %v", err)
	}

	out := make([]float32, len(a))
	scratch, err := getBinaryByteScratch(len(a) * 4)
	if err != nil {
		t.Fatalf("getBinaryByteScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putBinaryByteScratch(scratch)

	if err := RunBinaryInto("vv_Addfloat32", a, b, out); err != nil {
		t.Fatalf("RunBinaryInto: %v", err)
	}
	if !bytes.Equal(float32Bytes(out), float32Bytes(want)) {
		t.Fatal("RunBinaryInto output differs from allocating wrapper")
	}

	scratch, err = getBinaryByteScratch(len(a) * 4)
	if err != nil {
		t.Fatalf("getBinaryByteScratch after call: %v", err)
	}
	defer putBinaryByteScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("RunBinaryInto wrote through pooled scratch output instead of caller output")
	}
}

func TestRunBinaryRejectsMismatchedLengths(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := Add([]float32{1, 2}, []float32{1}); err == nil {
		t.Fatal("expected Add to reject mismatched input lengths")
	}
}
