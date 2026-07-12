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

func TestFFNMegaPipelineBits_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if _, err := ffnMegaPipelineBits(8); err != nil {
		t.Fatalf("ffnMegaPipelineBits(8): %v", err)
	}
}

func TestFFNMegaPipelineBits_Bad(t *testing.T) {
	if _, err := ffnMegaPipelineBits(3); err == nil {
		t.Fatal("ffnMegaPipelineBits(3) accepted an unsupported quant width")
	}
}

func TestFFNMegaPipelineBits_Ugly(t *testing.T) {
	if _, err := ffnMegaPipelineBits(0); err == nil {
		t.Fatal("ffnMegaPipelineBits(0) accepted an empty quant width")
	}
}

func TestEncMulScalarBF16_Good(t *testing.T) {
	if err := encMulScalarBF16(nil, nil, nil, nil, 13, 0); err != nil {
		t.Fatalf("encMulScalarBF16 zero-length operation: %v", err)
	}
}

func TestEncMulScalarBF16_Bad(t *testing.T) {
	if err := encMulScalarBF16(nil, nil, nil, nil, 7, -1); err == nil {
		t.Fatal("encMulScalarBF16 accepted a negative element count")
	}
}

func TestEncMulScalarBF16_Ugly(t *testing.T) {
	if err := encMulScalarBF16(nil, nil, nil, nil, ^uint(0), 0); err != nil {
		t.Fatalf("encMulScalarBF16 empty operation with boundary offset: %v", err)
	}
}

func TestBF16ScalarBytes_Good(t *testing.T) {
	want := toBF16Bytes([]float32{-3.625})
	got := bf16ScalarBytes(-3.625)
	if !bytes.Equal(got[:], want) {
		t.Fatalf("bf16ScalarBytes(-3.625) = %v, want %v", got, want)
	}
}

func TestBF16ScalarBytes_Bad(t *testing.T) {
	want := toBF16Bytes([]float32{0.33325195})
	got := bf16ScalarBytes(0.33325195)
	if !bytes.Equal(got[:], want) {
		t.Fatalf("bf16ScalarBytes rounded value = %v, want %v", got, want)
	}
}

func TestBF16ScalarBytes_Ugly(t *testing.T) {
	want := toBF16Bytes([]float32{-0.0})
	got := bf16ScalarBytes(float32(-0.0))
	if !bytes.Equal(got[:], want) {
		t.Fatalf("bf16ScalarBytes negative zero = %v, want %v", got, want)
	}
}

func TestQMVLogitsArgmaxUsable_Bad(t *testing.T) {
	cases := [][4]int{{0, 97, 32, 4}, {96, 0, 32, 4}, {96, 97, 48, 4}, {95, 97, 32, 4}, {96, 97, 32, 3}}
	for _, tc := range cases {
		if qmvLogitsArgmaxUsable(tc[0], tc[1], tc[2], tc[3]) {
			t.Fatalf("qmvLogitsArgmaxUsable%v accepted invalid geometry", tc)
		}
	}
}

func TestQMVLogitsTopKUsable_Bad(t *testing.T) {
	cases := [][5]int{{96, 97, 32, 4, 0}, {96, 17, 32, 4, 18}, {96, 97, 48, 4, 7}, {95, 97, 32, 4, 7}, {96, 97, 32, 3, 7}}
	for _, tc := range cases {
		if qmvLogitsTopKUsable(tc[0], tc[1], tc[2], tc[3], tc[4]) {
			t.Fatalf("qmvLogitsTopKUsable%v accepted invalid geometry", tc)
		}
	}
}

func TestQ4LMHeadTopKUsable_Bad(t *testing.T) {
	cases := [][5]int{{512, 97, 32, 8, 7}, {512, 97, 48, 4, 7}, {544, 97, 32, 4, 7}, {512, 7, 32, 4, 8}}
	for _, tc := range cases {
		if q4LMHeadTopKUsable(tc[0], tc[1], tc[2], tc[3], tc[4]) {
			t.Fatalf("q4LMHeadTopKUsable%v accepted invalid geometry", tc)
		}
	}
}

func TestQ4LMHeadTopKCandidateCount_Good(t *testing.T) {
	if got := q4LMHeadTopKCandidateCount(257, 11); got != 33 {
		t.Fatalf("q4LMHeadTopKCandidateCount(257, 11) = %d, want 33", got)
	}
}

func TestQ4LMHeadTopKCandidateCount_Ugly(t *testing.T) {
	if got := q4LMHeadTopKCandidateCount(128, 191); got != 128 {
		t.Fatalf("q4LMHeadTopKCandidateCount(128, 191) = %d, want 128", got)
	}
}

func TestQ4LMHeadTopKCandidatesPerTile_Good(t *testing.T) {
	if got := q4LMHeadTopKCandidatesPerTile(23); got != 23 {
		t.Fatalf("q4LMHeadTopKCandidatesPerTile(23) = %d, want 23", got)
	}
}

func TestQ4LMHeadTopKCandidatesPerTile_Ugly(t *testing.T) {
	if got := q4LMHeadTopKCandidatesPerTile(191); got != q4LMHeadTopKRowsPerTile {
		t.Fatalf("q4LMHeadTopKCandidatesPerTile(191) = %d, want %d", got, q4LMHeadTopKRowsPerTile)
	}
}

func TestEncBF16LogitsArgmaxTilesBF16_Bad(t *testing.T) {
	if err := encBF16LogitsArgmaxTilesBF16(nil, nil, nil, nil, nil, 0, 3); err == nil {
		t.Fatal("encBF16LogitsArgmaxTilesBF16 accepted an empty vocabulary")
	}
}

func TestEncBF16LMHeadArgmaxTilesBF16_Bad(t *testing.T) {
	if err := encBF16LMHeadArgmaxTilesBF16(nil, nil, nil, nil, nil, nil, 0, 0, 0, 31, 3); err == nil {
		t.Fatal("encBF16LMHeadArgmaxTilesBF16 accepted an empty model width")
	}
}

func TestEncArgmaxMergeF32_Bad(t *testing.T) {
	if err := encArgmaxMergeF32(nil, nil, nil, nil, 0); err == nil {
		t.Fatal("encArgmaxMergeF32 accepted an empty candidate set")
	}
}

func TestEncBF16LMHeadArgmaxTilesRowsBF16_Bad(t *testing.T) {
	if err := encBF16LMHeadArgmaxTilesRowsBF16(nil, nil, nil, nil, nil, nil, 0, 0, 64, 31, 0, 9, 1); err == nil {
		t.Fatal("encBF16LMHeadArgmaxTilesRowsBF16 accepted nine rows")
	}
}

func TestEncArgmaxMergeRowsF32_Bad(t *testing.T) {
	if err := encArgmaxMergeRowsF32(nil, nil, nil, nil, 7, 0); err == nil {
		t.Fatal("encArgmaxMergeRowsF32 accepted an empty row count")
	}
}

func TestEncBF16LMHeadCandidatesBF16_Bad(t *testing.T) {
	if err := encBF16LMHeadCandidatesBF16(nil, nil, nil, nil, nil, nil, nil, 0, 0, 0, 31, 0, 0, 1.125, 7.5); err == nil {
		t.Fatal("encBF16LMHeadCandidatesBF16 accepted an empty model width")
	}
}

func TestEncBF16LogitsCandidatesBF16_Bad(t *testing.T) {
	if err := encBF16LogitsCandidatesBF16(nil, nil, nil, nil, nil, 0, 3, 7.5); err == nil {
		t.Fatal("encBF16LogitsCandidatesBF16 accepted an empty vocabulary")
	}
}

func TestEncBF16LogitsTopKTilesBF16_Bad(t *testing.T) {
	if err := encBF16LogitsTopKTilesBF16(nil, nil, nil, nil, nil, nil, 31, 0, 0, 65, 1.125, 7.5); err == nil {
		t.Fatal("encBF16LogitsTopKTilesBF16 accepted top-k above 64")
	}
}

func TestEncTopKMergeF32_Bad(t *testing.T) {
	if err := encTopKMergeF32(nil, nil, nil, nil, nil, 7, 0); err == nil {
		t.Fatal("encTopKMergeF32 accepted an empty top-k")
	}
}

func TestEncTopKMergeSampleF32_Bad(t *testing.T) {
	if err := encTopKMergeSampleF32(nil, nil, nil, nil, nil); err == nil {
		t.Fatal("encTopKMergeSampleF32 accepted a missing parameter buffer")
	}
}

func TestEncLogitsSampleBF16_Bad(t *testing.T) {
	if err := encLogitsSampleBF16(nil, nil, nil, nil, nil, nil); err == nil {
		t.Fatal("encLogitsSampleBF16 accepted a missing parameter buffer")
	}
}
