// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// gemma4_12b_mtp_shapes_test.go — #352 instrument. The 12B MTP drafter's
// pre_projection (M=1, K=2·3840=7680, N=1024) returns all-NaN from clean
// inputs on both the steel-GEMM (MatMulBF16NT) and gemv lanes, while the E2B
// drafter's K=4096 is healthy. This sweeps the projection shapes GPU-vs-CPU to
// pin whether the defect is shape-dependent inside the kernels or contextual
// to the live pair. K values: E2B (4096), 12B (7680), 31B (10752), plus
// bisection probes between.

func mtpShapeRandBF16(rng *rand.Rand, n int) []byte {
	out := make([]byte, n*bf16Size)
	for i := 0; i < n; i++ {
		bits := f32ToBF16((rng.Float32() - 0.5) * 0.1)
		out[2*i] = byte(bits)
		out[2*i+1] = byte(bits >> 8)
	}
	return out
}

func mtpShapeCPUDotNT(a, w []byte, K, n int) float32 {
	var acc float32
	for k := 0; k < K; k++ {
		av := bf16ToF32(a[2*k], a[2*k+1])
		wv := bf16ToF32(w[(n*K+k)*2], w[(n*K+k)*2+1])
		acc += av * wv
	}
	return acc
}

// TestNoCopyOffsetAlignmentRule probes the RAW no-copy wrap (bypassing
// residentBytes' gate) at interior offsets of varying alignment to pin the
// driver's real base-alignment requirement: the 12B assistant broke at +9423
// (odd) while E2B works at +416 (16-aligned) with the same wrap.
func TestNoCopyOffsetAlignmentRule(t *testing.T) {
	requireNativeRuntime(t)
	K, N := 1024, 512
	rng := rand.New(rand.NewSource(416))
	vec := mtpShapeRandBF16(rng, K)
	for _, off := range []int{0, 16384, 9423, 9425, 9426, 9428, 9424, 9432, 416, 4096} {
		blob := make([]byte, off+N*K*bf16Size+16384)
		w := blob[off : off+N*K*bf16Size]
		for i := 0; i < N*K; i++ {
			bits := f32ToBF16((rng.Float32() - 0.5) * 0.1)
			w[2*i] = byte(bits)
			w[2*i+1] = byte(bits >> 8)
		}
		var nan int
		var maxDiff float64
		var noCopy bool
		var encErr error
		withAutoreleasePool(func() {
			wBuf, pinner, nc := residentNoCopyBytes(w)
			noCopy = nc
			if pinner != nil {
				defer pinner.Unpin()
			}
			vecBuf := sharedBytes(vec)
			outBuf := scratchBF16(N)
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			encErr = encGemvBF16(enc, wBuf, vecBuf, outBuf, N, K)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			got := unsafe.Slice((*byte)(outBuf.Contents()), N*bf16Size)
			nan, _ = bf16NaNScanBytes(got)
			for _, n := range []int{0, N / 2, N - 1} {
				want := float64(mtpShapeCPUDotNT(vec, w, K, n))
				gg := float64(bf16ToF32(got[2*n], got[2*n+1]))
				if d := math.Abs(gg - want); d > maxDiff {
					maxDiff = d
				}
			}
		})
		if encErr != nil {
			t.Fatalf("off=%d enc: %v", off, encErr)
		}
		t.Logf("off=%5d (mod16=%2d mod4=%d) noCopy=%v nan=%d maxDiff=%.4f", off, off%16, off%4, noCopy, nan, maxDiff)
	}
}

// TestUnalignedNoCopyGPURead pins the #352 fix end-to-end: a weight slice at
// an ODD interior offset (the 12B assistant's pre_projection.weight sits at
// blob+9423) must read byte-exactly through the resident-weight wrap. Before
// residentBytes' odd-base copy gate, the no-copy wrap looked perfect from the
// CPU (Contents()==base, full Length()) while the GPU's element reads were
// sheared — MatMulBF16NT returned all-NaN from clean operands and every 12B
// draft token was 0.
func TestUnalignedNoCopyGPURead(t *testing.T) {
	const off = 9423
	K, N := 7680, 1024
	rng := rand.New(rand.NewSource(9423))
	blob := make([]byte, off+N*K*bf16Size+16384)
	for i := range blob {
		blob[i] = byte(rng.Intn(256))
	}
	w := blob[off : off+N*K*bf16Size]
	// overwrite with valid small bf16 so CPU reference is finite
	for i := 0; i < N*K; i++ {
		bits := f32ToBF16((rng.Float32() - 0.5) * 0.1)
		w[2*i] = byte(bits)
		w[2*i+1] = byte(bits >> 8)
	}
	a := mtpShapeRandBF16(rng, K)

	buf, pinner, noCopy := residentNoCopyBytes(w)
	if pinner != nil {
		defer pinner.Unpin()
	}
	t.Logf("wrap: noCopy=%v ptr%%16384=%d contentsDelta=%d len=%d",
		noCopy, uintptr(unsafe.Pointer(&w[0]))%16384, int64(uintptr(buf.Contents()))-int64(uintptr(unsafe.Pointer(&w[0]))), buf.Length())

	got, err := MatMulBF16NT(a, w, 1, K, N)
	if err != nil {
		t.Fatalf("MatMulBF16NT: %v", err)
	}
	nan, _ := bf16NaNScanBytes(got)
	var maxDiff float64
	for _, n := range []int{0, 1, N / 2, N - 1} {
		want := float64(mtpShapeCPUDotNT(a, w, K, n))
		gg := float64(bf16ToF32(got[2*n], got[2*n+1]))
		if d := math.Abs(gg - want); d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("unaligned-wrap gemm: nan=%d maxDiff=%.4f", nan, maxDiff)
	if nan > 0 || maxDiff > 0.5 {
		t.Errorf("GPU read through unaligned no-copy wrap diverges from CPU bytes: nan=%d maxDiff=%.4f", nan, maxDiff)
	}
}

func TestMTPProjectionShapesGPUvsCPU(t *testing.T) {
	rng := rand.New(rand.NewSource(352))
	N := 1024
	for _, K := range []int{4096, 4608, 5120, 6144, 7680, 8192, 10752} {
		a := mtpShapeRandBF16(rng, K)
		w := mtpShapeRandBF16(rng, N*K)

		got, err := MatMulBF16NT(a, w, 1, K, N)
		if err != nil {
			t.Fatalf("K=%d MatMulBF16NT: %v", K, err)
		}
		nanGemm, _ := bf16NaNScanBytes(got)

		gv, err := MatVecBF16(w, a, N, K)
		if err != nil {
			t.Fatalf("K=%d MatVecBF16: %v", K, err)
		}
		nanGemv, _ := bf16NaNScanBytes(gv)

		var maxDiffGemm, maxDiffGemv float64
		for _, n := range []int{0, 1, N/2, N - 1} {
			want := float64(mtpShapeCPUDotNT(a, w, K, n))
			gg := float64(bf16ToF32(got[2*n], got[2*n+1]))
			gm := float64(bf16ToF32(gv[2*n], gv[2*n+1]))
			if d := math.Abs(gg - want); d > maxDiffGemm {
				maxDiffGemm = d
			}
			if d := math.Abs(gm - want); d > maxDiffGemv {
				maxDiffGemv = d
			}
		}
		t.Logf("K=%5d gemm{nan=%d maxDiff=%.4f} gemv{nan=%d maxDiff=%.4f}", K, nanGemm, maxDiffGemm, nanGemv, maxDiffGemv)
		if nanGemm > 0 || nanGemv > 0 {
			t.Errorf("K=%d produced NaN: gemm=%d gemv=%d", K, nanGemm, nanGemv)
		}
	}
}
