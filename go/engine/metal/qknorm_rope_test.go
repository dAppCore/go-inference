// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"os"
	"testing"
	"unsafe"
)

func qkNormRopePeriods(rotaryDim int, log2Theta float32) []float32 {
	periods := make([]float32, rotaryDim/2)
	for i := range periods {
		invFreq := float32(math.Exp2(-float64(i) / float64(rotaryDim/2) * float64(log2Theta)))
		periods[i] = 1.0 / invFreq
	}
	return periods
}

// TestQknormRope_QKNormRopeBF16_Bad exercises QKNormRopeBF16's shape and headDim-cap guards.
func TestQknormRope_QKNormRopeBF16_Bad(t *testing.T) {
	requireNativeRuntime(t)

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, log2Theta = float32(1e-6), float32(1.0), float32(13.2877)
	validX := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	validW := toBF16Bytes(syntheticFloat32(headDim, headDim+7))

	t.Run("x length mismatch", func(t *testing.T) {
		if _, err := QKNormRopeBF16(validX[:len(validX)-2], validW, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil); err == nil {
			t.Fatal("expected QKNormRopeBF16 to reject an x length mismatch")
		}
	})
	t.Run("weight length mismatch", func(t *testing.T) {
		if _, err := QKNormRopeBF16(validX, validW[:len(validW)-2], nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil); err == nil {
			t.Fatal("expected QKNormRopeBF16 to reject a weight length mismatch")
		}
	})
	t.Run("headDim exceeds 512 threadgroup cap", func(t *testing.T) {
		const bigHeadDim = 640
		x := toBF16Bytes(syntheticFloat32(nHeads*bigHeadDim, 3))
		w := toBF16Bytes(syntheticFloat32(bigHeadDim, 5))
		if _, err := QKNormRopeBF16(x, w, nHeads, bigHeadDim, bigHeadDim, 7, scale, eps, log2Theta, nil); err == nil {
			t.Fatal("expected QKNormRopeBF16 to reject headDim > 512")
		}
	})
}

// TestQknormRope_QKNormRopeBF16_Ugly pins the headDim==512 boundary itself (the cap QKNormRopeBF16
// rejects ABOVE, this proves it still runs correctly AT).
func TestQknormRope_QKNormRopeBF16_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const nHeads, headDim = 4, 512
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, 3))
	w := toBF16Bytes(syntheticFloat32(headDim, 5))
	log2Theta := float32(math.Log2(float64(theta)))

	got, err := QKNormRopeBF16(x, w, nHeads, headDim, headDim, 7, scale, eps, log2Theta, nil)
	if err != nil {
		t.Fatalf("QKNormRopeBF16 at headDim=512: %v", err)
	}
	normed, err := RMSNormBF16(x, w, nHeads, headDim, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	want, err := RoPEDimsBF16(normed, 1, nHeads, headDim, headDim, theta, scale, 7, false)
	if err != nil {
		t.Fatalf("RoPEDimsBF16: %v", err)
	}
	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("QKNormRopeBF16 at headDim=512 cosine=%.6f vs composed RMSNorm+RoPE, want ~1", cos)
	}
}

func TestQKNormRopeBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	log2Theta := float32(math.Log2(float64(theta)))
	if _, err := QKNormRopeBF16(x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil); err != nil {
		t.Fatalf("QKNormRopeBF16 warmup: %v", err)
	}

	var qkErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, qkErr = QKNormRopeBF16(x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil)
	})
	if qkErr != nil {
		t.Fatalf("QKNormRopeBF16: %v", qkErr)
	}
	if allocs > 10 {
		t.Fatalf("QKNormRopeBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestQKNormRopeBF16FreqsAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	log2Theta := float32(math.Log2(float64(theta)))
	periods := qkNormRopePeriods(rotaryDim, log2Theta)
	if _, err := QKNormRopeBF16(x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, periods); err != nil {
		t.Fatalf("QKNormRopeBF16 freqs warmup: %v", err)
	}

	var qkErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, qkErr = QKNormRopeBF16(x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, periods)
	})
	if qkErr != nil {
		t.Fatalf("QKNormRopeBF16 freqs: %v", qkErr)
	}
	if allocs > 10 {
		t.Fatalf("QKNormRopeBF16 freqs allocations = %.0f, want <= 10", allocs)
	}
}

// TestQknormRope_QKNormRopeBF16Into_Good proves QKNormRopeBF16Into returns caller-owned output
// backing and matches the allocating wrapper byte-for-byte, both on the base-rope and the
// freqs/YaRN path.
func TestQknormRope_QKNormRopeBF16Into_Good(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	log2Theta := float32(math.Log2(float64(theta)))
	cases := []struct {
		name    string
		periods []float32
	}{
		{name: "base"},
		{name: "freqs", periods: qkNormRopePeriods(rotaryDim, log2Theta)},
	}
	for _, c := range cases {
		out := make([]byte, len(x))
		for i := range out {
			out[i] = 0xA5
		}

		got, err := QKNormRopeBF16Into(out, x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, c.periods)
		if err != nil {
			t.Fatalf("%s: QKNormRopeBF16Into: %v", c.name, err)
		}
		if len(got) != len(out) {
			t.Fatalf("%s: QKNormRopeBF16Into len = %d, want %d", c.name, len(got), len(out))
		}
		if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
			t.Fatalf("%s: QKNormRopeBF16Into did not return caller-owned output backing", c.name)
		}
		want, err := QKNormRopeBF16(x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, c.periods)
		if err != nil {
			t.Fatalf("%s: QKNormRopeBF16 reference: %v", c.name, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("%s: QKNormRopeBF16Into output differs from allocating wrapper", c.name)
		}
	}
}

// TestQknormRope_QKNormRopeBF16Into_Bad mirrors QKNormRopeBF16's shape guard through the
// caller-output entry point.
func TestQknormRope_QKNormRopeBF16Into_Bad(t *testing.T) {
	requireNativeRuntime(t)

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, log2Theta = float32(1e-6), float32(1.0), float32(13.2877)
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	out := make([]byte, nHeads*headDim*bf16Size)

	badX := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))[:nHeads*headDim*bf16Size-2]
	if _, err := QKNormRopeBF16Into(out, badX, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil); err == nil {
		t.Fatal("expected QKNormRopeBF16Into to reject an x length mismatch")
	}
}

// TestQknormRope_QKNormRopeBF16Into_Ugly proves the too-small-capacity path: when cap(out) is
// smaller than len(x), QKNormRopeBF16Into must allocate fresh storage rather than write out of
// bounds, and still return the correct fused result.
func TestQknormRope_QKNormRopeBF16Into_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	log2Theta := float32(math.Log2(float64(theta)))
	want, err := QKNormRopeBF16(x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil)
	if err != nil {
		t.Fatalf("QKNormRopeBF16 reference: %v", err)
	}

	tooSmall := make([]byte, 1)
	got, err := QKNormRopeBF16Into(tooSmall, x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil)
	if err != nil {
		t.Fatalf("QKNormRopeBF16Into (undersized out): %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("QKNormRopeBF16Into (undersized out) output differs from allocating wrapper")
	}
}

func TestQKNormRopeBF16WithBufferOutputWritesDirectlyToProvidedBuffer(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	log2Theta := float32(math.Log2(float64(theta)))
	cases := []struct {
		name    string
		periods []float32
	}{
		{name: "base"},
		{name: "freqs", periods: qkNormRopePeriods(rotaryDim, log2Theta)},
	}
	for _, c := range cases {
		want, err := QKNormRopeBF16(x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, c.periods)
		if err != nil {
			t.Fatalf("%s: QKNormRopeBF16: %v", c.name, err)
		}

		dim := len(x) / bf16Size
		scratch, err := getQMVBF16Scratch(dim, dim)
		if err != nil {
			t.Fatalf("%s: getQMVBF16Scratch: %v", c.name, err)
		}
		sentinel := bytes.Repeat([]byte{0x9d}, len(scratch.out.bytes))
		copy(scratch.out.bytes, sentinel)
		putQMVBF16Scratch(scratch)

		input, err := newPinnedNoCopyBytes(len(x))
		if err != nil {
			t.Fatalf("%s: newPinnedNoCopyBytes input: %v", c.name, err)
		}
		defer input.Close()
		xBuf, err := input.copyBuffer(x)
		if err != nil {
			t.Fatalf("%s: copy input buffer: %v", c.name, err)
		}
		out, err := newPinnedNoCopyBytes(len(x))
		if err != nil {
			t.Fatalf("%s: newPinnedNoCopyBytes output: %v", c.name, err)
		}
		defer out.Close()

		if err := qkNormRopeBF16WithBufferOutputInPool(x, xBuf, out.buf, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, c.periods); err != nil {
			t.Fatalf("%s: qkNormRopeBF16WithBufferOutputInPool: %v", c.name, err)
		}
		if !bytes.Equal(out.bytes, want) {
			t.Fatalf("%s: QKNormRopeBF16 direct Metal output differs from allocating wrapper", c.name)
		}

		scratch, err = getQMVBF16Scratch(dim, dim)
		if err != nil {
			t.Fatalf("%s: getQMVBF16Scratch after call: %v", c.name, err)
		}
		defer putQMVBF16Scratch(scratch)
		if !bytes.Equal(scratch.out.bytes, sentinel) {
			t.Fatalf("%s: qkNormRopeBF16WithBufferOutputInPool wrote through pooled scratch output", c.name)
		}
	}
}

// TestQknormRope_QKNormRopeBF16_Good is the NUMERICAL gate for the fused per-head QK-norm + RoPE
// kernel: QKNormRopeBF16(x, w) must track the composed RoPE(RMSNormBF16(x, w, nHeads, headDim)) —
// across full rotary, partial rotary, and the freqs/YaRN path — at cosine ~1.0. Not bit-exact (the
// ~1 ULP native bfloat vs MLX bfloat16_t gap, lockstep numerics); a real rope bug (wrong pairing /
// freq / offset) collapses the cosine, so this proves the rotation math in isolation before any
// decode wiring.
func TestQknormRope_QKNormRopeBF16_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded — run `task build:kernels`")
	}
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	log2Theta := float32(math.Log2(float64(theta)))

	cases := []struct {
		name                               string
		nHeads, headDim, rotaryDim, offset int
		freqs                              bool
	}{
		{"base full rotary (e2b sliding)", 8, 256, 256, 5, false},
		{"base partial rotary", 8, 256, 128, 7, false},
		{"freqs partial (global proportional)", 8, 512, 128, 3, true},
	}
	for _, c := range cases {
		x := toBF16Bytes(syntheticFloat32(c.nHeads*c.headDim, c.headDim+1))
		w := toBF16Bytes(syntheticFloat32(c.headDim, c.headDim+7))

		normed, err := RMSNormBF16(x, w, c.nHeads, c.headDim, eps)
		if err != nil {
			t.Fatalf("%s: RMSNormBF16: %v", c.name, err)
		}

		var ref, got []byte
		if c.freqs {
			invFreqs := make([]float32, c.rotaryDim/2)
			for i := range invFreqs { // proportional inverse frequencies (any positive set proves parity)
				invFreqs[i] = float32(math.Exp2(-float64(i) / float64(c.rotaryDim/2) * float64(log2Theta)))
			}
			ref, err = RoPEFreqsBF16(normed, 1, c.nHeads, c.headDim, c.rotaryDim, invFreqs, scale, c.offset, false)
			if err != nil {
				t.Fatalf("%s: RoPEFreqsBF16: %v", c.name, err)
			}
			periods := make([]float32, len(invFreqs))
			for i, f := range invFreqs {
				periods[i] = 1.0 / f
			}
			got, err = QKNormRopeBF16(x, w, c.nHeads, c.headDim, c.rotaryDim, c.offset, scale, eps, log2Theta, periods)
		} else {
			ref, err = RoPEDimsBF16(normed, 1, c.nHeads, c.headDim, c.rotaryDim, theta, scale, c.offset, false)
			if err != nil {
				t.Fatalf("%s: RoPEDimsBF16: %v", c.name, err)
			}
			got, err = QKNormRopeBF16(x, w, c.nHeads, c.headDim, c.rotaryDim, c.offset, scale, eps, log2Theta, nil)
		}
		if err != nil {
			t.Fatalf("%s: QKNormRopeBF16: %v", c.name, err)
		}

		cos := cosineBF16(got, ref)
		t.Logf("%-38s cosine=%.7f", c.name, cos)
		if cos < 0.999 {
			t.Fatalf("%s: fused qk-norm+rope cosine=%.7f < 0.999 — rotation math wrong, not just bf16 rounding", c.name, cos)
		}
	}
}
