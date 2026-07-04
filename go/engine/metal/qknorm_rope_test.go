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

func TestQKNormRopeBF16IntoUsesCallerBacking(t *testing.T) {
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

// TestQKNormRopeBF16ParityComposed is the NUMERICAL gate for the fused per-head QK-norm + RoPE kernel:
// QKNormRopeBF16(x, w) must track the composed RoPE(RMSNormBF16(x, w, nHeads, headDim)) — across full
// rotary, partial rotary, and the freqs/YaRN path — at cosine ~1.0. Not bit-exact (the ~1 ULP native
// bfloat vs MLX bfloat16_t gap, lockstep numerics); a real rope bug (wrong pairing / freq / offset)
// collapses the cosine, so this proves the rotation math in isolation before any decode wiring.
func TestQKNormRopeBF16ParityComposed(t *testing.T) {
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
