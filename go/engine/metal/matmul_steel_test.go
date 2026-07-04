// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"
)

func matMulF32NTFixture(M, K, N int) ([]float32, []float32) {
	a := syntheticFloat32(M*K, M+3)
	b := syntheticFloat32(N*K, N+5)
	return a, b
}

func TestMatMulF32NTAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const M, K, N = 16, 64, 137
	a, b := matMulF32NTFixture(M, K, N)
	if _, err := MatMulF32NT(a, b, M, K, N); err != nil {
		t.Fatalf("MatMulF32NT warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MatMulF32NT(a, b, M, K, N); err != nil {
			t.Fatalf("MatMulF32NT: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MatMulF32NT allocations = %.0f, want <= 10", allocs)
	}
}

func TestMatMulF32SteelScratchBuffersUseCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const M, K, N = 16, 64, 137
	a, _ := matMulF32NTFixture(M, K, N)
	scratch, err := getMatMulF32SteelScratch(M, K, N, K, steelNT)
	if err != nil {
		t.Fatalf("get MatMulF32 steel scratch: %v", err)
	}
	defer putMatMulF32SteelScratch(scratch)
	aBuf, _, _, err := scratch.buffers(a)
	if err != nil {
		t.Fatalf("MatMulF32 steel scratch buffers: %v", err)
	}
	if got, want := uintptr(aBuf.Contents()), uintptr(unsafe.Pointer(&a[0])); got != want {
		t.Fatalf("A buffer pointer = %#x, want caller backing %#x", got, want)
	}
}

func TestSteelGemmPipelineWarmedLookupAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := steelGemmPipeline(steelNT.name, false, false, false, false, false, true); err != nil {
		t.Fatalf("steelGemmPipeline warmup: %v", err)
	}

	var pipeErr error
	allocs := testing.AllocsPerRun(10, func() {
		_, pipeErr = steelGemmPipeline(steelNT.name, false, false, false, false, false, true)
	})
	if pipeErr != nil {
		t.Fatalf("steelGemmPipeline: %v", pipeErr)
	}
	if allocs > 0 {
		t.Fatalf("steelGemmPipeline warmed lookup allocations = %.0f, want 0", allocs)
	}
}

func TestMatMulF32NTSplitKAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const M, K, N = 3, 128, 128
	a, b := matMulF32NTFixture(M, K, N)
	if _, err := MatMulF32NT(a, b, M, K, N); err != nil {
		t.Fatalf("MatMulF32NT split-K warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MatMulF32NT(a, b, M, K, N); err != nil {
			t.Fatalf("MatMulF32NT split-K: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MatMulF32NT split-K allocations = %.0f, want <= 10", allocs)
	}
}

func TestMatMulF32SplitKParamsScratchPoolKeepsShapesResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getMatMulF32SplitKParamsScratch(3, 128, 128, 4, 1, 2, 384, 64, 4)
	if err != nil {
		t.Fatalf("get small MatMulF32 split-K params scratch: %v", err)
	}
	putMatMulF32SplitKParamsScratch(small)
	large, err := getMatMulF32SplitKParamsScratch(5, 256, 160, 5, 1, 4, 800, 64, 4)
	if err != nil {
		t.Fatalf("get large MatMulF32 split-K params scratch: %v", err)
	}
	putMatMulF32SplitKParamsScratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getMatMulF32SplitKParamsScratch(3, 128, 128, 4, 1, 2, 384, 64, 4)
	if err != nil {
		t.Fatalf("get small MatMulF32 split-K params scratch again: %v", err)
	}
	defer putMatMulF32SplitKParamsScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("MatMulF32 split-K params scratch pool evicted the small shape after using a larger shape")
	}
	gotLarge, err := getMatMulF32SplitKParamsScratch(5, 256, 160, 5, 1, 4, 800, 64, 4)
	if err != nil {
		t.Fatalf("get large MatMulF32 split-K params scratch again: %v", err)
	}
	defer putMatMulF32SplitKParamsScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("MatMulF32 split-K params scratch pool evicted the large shape after reusing the small shape")
	}
}

func TestMatMulF32SplitKAccumScratchPoolKeepsShapesResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getMatMulF32SplitKAccumScratch(3, 128, 2)
	if err != nil {
		t.Fatalf("get small MatMulF32 split-K accum scratch: %v", err)
	}
	putMatMulF32SplitKAccumScratch(small)
	large, err := getMatMulF32SplitKAccumScratch(5, 160, 4)
	if err != nil {
		t.Fatalf("get large MatMulF32 split-K accum scratch: %v", err)
	}
	putMatMulF32SplitKAccumScratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getMatMulF32SplitKAccumScratch(3, 128, 2)
	if err != nil {
		t.Fatalf("get small MatMulF32 split-K accum scratch again: %v", err)
	}
	defer putMatMulF32SplitKAccumScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("MatMulF32 split-K accum scratch pool evicted the small shape after using a larger shape")
	}
	gotLarge, err := getMatMulF32SplitKAccumScratch(5, 160, 4)
	if err != nil {
		t.Fatalf("get large MatMulF32 split-K accum scratch again: %v", err)
	}
	defer putMatMulF32SplitKAccumScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("MatMulF32 split-K accum scratch pool evicted the large shape after reusing the small shape")
	}
}

func TestMatMulF32IntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const M, K, N = 16, 64, 137
	a := syntheticFloat32(M*K, 3)
	b := syntheticFloat32(K*N, 4)
	want, err := MatMulF32(a, b, M, K, N)
	if err != nil {
		t.Fatalf("MatMulF32 reference: %v", err)
	}
	out := syntheticFloat32(M*N, 11)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getMatMulF32SteelScratch(M, K, N, N, steelNN)
	if err != nil {
		t.Fatalf("getMatMulF32SteelScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xc7}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putMatMulF32SteelScratch(scratch)

	got, err := MatMulF32Into(out, a, b, M, K, N)
	if err != nil {
		t.Fatalf("MatMulF32Into: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MatMulF32Into did not reuse caller-owned output backing")
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("MatMulF32Into differs at %d: %v vs %v", i, got[i], want[i])
		}
	}

	scratch, err = getMatMulF32SteelScratch(M, K, N, N, steelNN)
	if err != nil {
		t.Fatalf("getMatMulF32SteelScratch after call: %v", err)
	}
	defer putMatMulF32SteelScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("MatMulF32Into wrote through pooled scratch output instead of caller output")
	}
}

func TestMatMulF32NTIntoReusesOutputBackingAndBypassesScratchOutputForSplitK(t *testing.T) {
	requireNativeRuntime(t)

	const M, K, N = 3, 128, 128
	a, b := matMulF32NTFixture(M, K, N)
	want, err := MatMulF32NT(a, b, M, K, N)
	if err != nil {
		t.Fatalf("MatMulF32NT reference: %v", err)
	}
	out := syntheticFloat32(M*N, 13)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVFloatScratch(M*N, M*K)
	if err != nil {
		t.Fatalf("getQMVFloatScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x2b}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVFloatScratch(scratch)

	got, err := MatMulF32NTInto(out, a, b, M, K, N)
	if err != nil {
		t.Fatalf("MatMulF32NTInto: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MatMulF32NTInto did not reuse caller-owned output backing")
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("MatMulF32NTInto differs at %d: %v vs %v", i, got[i], want[i])
		}
	}

	scratch, err = getQMVFloatScratch(M*N, M*K)
	if err != nil {
		t.Fatalf("getQMVFloatScratch after call: %v", err)
	}
	defer putQMVFloatScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("MatMulF32NTInto split-K wrote through pooled scratch output instead of caller output")
	}
}

func TestMatMulF32NTIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const M, K, N = 16, 64, 137
	a, b := matMulF32NTFixture(M, K, N)
	want, err := MatMulF32NT(a, b, M, K, N)
	if err != nil {
		t.Fatalf("MatMulF32NT reference: %v", err)
	}
	out := syntheticFloat32(M*N, 17)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getMatMulF32SteelScratch(M, K, N, K, steelNT)
	if err != nil {
		t.Fatalf("getMatMulF32SteelScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x9d}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putMatMulF32SteelScratch(scratch)

	got, err := MatMulF32NTInto(out, a, b, M, K, N)
	if err != nil {
		t.Fatalf("MatMulF32NTInto: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MatMulF32NTInto did not reuse caller-owned output backing")
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("MatMulF32NTInto differs at %d: %v vs %v", i, got[i], want[i])
		}
	}

	scratch, err = getMatMulF32SteelScratch(M, K, N, K, steelNT)
	if err != nil {
		t.Fatalf("getMatMulF32SteelScratch after call: %v", err)
	}
	defer putMatMulF32SteelScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("MatMulF32NTInto wrote through pooled scratch output instead of caller output")
	}
}

// TestMatMulF32 (BYTE-IDENTICAL to pkg/metal.Matmul) lives in matmul_steel_metal_test.go — it needs
// the real cgo metal package as its oracle, so it's gated behind metal_runtime.

func BenchmarkMatMulF32NTSplitK3x128x128(b *testing.B) {
	requireNativeRuntime(b)

	const M, K, N = 3, 128, 128
	a, w := matMulF32NTFixture(M, K, N)
	b.SetBytes(int64((len(a) + len(w)) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatMulF32NT(a, w, M, K, N); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulF32Into16x64x137(b *testing.B) {
	requireNativeRuntime(b)

	const M, K, N = 16, 64, 137
	a := syntheticFloat32(M*K, 3)
	w := syntheticFloat32(K*N, 4)
	out := make([]float32, M*N)
	b.SetBytes(int64((len(a) + len(w)) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = MatMulF32Into(out, a, w, M, K, N)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulF3216x64x137(b *testing.B) {
	requireNativeRuntime(b)

	const M, K, N = 16, 64, 137
	a := syntheticFloat32(M*K, 3)
	w := syntheticFloat32(K*N, 4)
	b.SetBytes(int64((len(a) + len(w)) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatMulF32(a, w, M, K, N); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulF32NTIntoSplitK3x128x128(b *testing.B) {
	requireNativeRuntime(b)

	const M, K, N = 3, 128, 128
	a, w := matMulF32NTFixture(M, K, N)
	out := make([]float32, M*N)
	b.SetBytes(int64((len(a) + len(w)) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = MatMulF32NTInto(out, a, w, M, K, N)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulF32NT16x64x137(b *testing.B) {
	requireNativeRuntime(b)

	const M, K, N = 16, 64, 137
	a, w := matMulF32NTFixture(M, K, N)
	b.SetBytes(int64((len(a) + len(w)) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatMulF32NT(a, w, M, K, N); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulF32NTInto16x64x137(b *testing.B) {
	requireNativeRuntime(b)

	const M, K, N = 16, 64, 137
	a, w := matMulF32NTFixture(M, K, N)
	out := make([]float32, M*N)
	b.SetBytes(int64((len(a) + len(w)) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = MatMulF32NTInto(out, a, w, M, K, N)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// TestMatMulF32NT (BYTE-IDENTICAL to metal.Matmul(a, Transpose(b))) lives in
// matmul_steel_metal_test.go — same reason as TestMatMulF32 above.
