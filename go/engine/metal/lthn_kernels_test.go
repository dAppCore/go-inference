// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
)

func TestGeluKernelCapabilityReflectsLoadedFlag(t *testing.T) {
	old := customLibraryLoaded
	defer func() { customLibraryLoaded = old }()

	customLibraryLoaded = false
	if gpuHasGeluKernel() {
		t.Fatal("gpuHasGeluKernel true when custom library flag is false")
	}
	customLibraryLoaded = true
	if !gpuHasGeluKernel() {
		t.Fatal("gpuHasGeluKernel false when custom library flag is true")
	}
}

func TestMulScalarBF16MatchesBroadcastMultiply(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}

	in := toBF16Bytes([]float32{-2, -0.5, 0, 0.25, 1.5, 3})
	scalar := toBF16Bytes([]float32{0.375})
	got, err := MulScalarBF16(in, scalar)
	if err != nil {
		t.Fatalf("MulScalarBF16: %v", err)
	}
	want, err := MulBF16(in, scalarFillBF16(scalar, len(in)/bf16Size))
	if err != nil {
		t.Fatalf("broadcast MulBF16: %v", err)
	}
	eqBytes(t, "MulScalarBF16", got, want)
}

func TestMulScalarBF16IntoUsesCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}

	in := toBF16Bytes(syntheticFloat32(1024, 17))
	scalar := toBF16Bytes([]float32{0.375})
	out := make([]byte, len(in))
	for i := range out {
		out[i] = 0xA5
	}

	if err := MulScalarBF16Into(out, in, scalar); err != nil {
		t.Fatalf("MulScalarBF16Into: %v", err)
	}
	want, err := MulBF16(in, scalarFillBF16(scalar, len(in)/bf16Size))
	if err != nil {
		t.Fatalf("broadcast MulBF16: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatal("MulScalarBF16Into output differs from broadcast multiply")
	}
}

func TestMulScalarBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}

	in := toBF16Bytes(syntheticFloat32(1024, 17))
	scalar := toBF16Bytes([]float32{0.375})
	if _, err := MulScalarBF16(in, scalar); err != nil {
		t.Fatalf("MulScalarBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MulScalarBF16(in, scalar); err != nil {
			t.Fatalf("MulScalarBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MulScalarBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestMulScalarBF16KeepsScalarBufferResident(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const scalarValue = float32(0.375)
	key := bf16ConstKey{n: 1, v: scalarValue}
	bf16ConstMu.Lock()
	delete(bf16ConstCache, key)
	bf16ConstMu.Unlock()

	in := toBF16Bytes([]float32{-2, -0.5, 0, 0.25, 1.5, 3})
	scalar := toBF16Bytes([]float32{scalarValue})
	if _, err := MulScalarBF16(in, scalar); err != nil {
		t.Fatalf("MulScalarBF16: %v", err)
	}

	bf16ConstMu.Lock()
	_, cached := bf16ConstCache[key]
	bf16ConstMu.Unlock()
	if !cached {
		t.Fatal("MulScalarBF16 did not cache its one-element BF16 scalar buffer")
	}
}
