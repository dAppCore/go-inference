// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"
)

func rmsQMVFixture(tb testing.TB, outDim, inDim, groupSize, bits int) ([]byte, []byte, QuantWeight) {
	tb.Helper()
	x := toBF16Bytes(syntheticFloat32(inDim, inDim+1))
	normW := toBF16Bytes(syntheticFloat32(inDim, inDim+7))
	qw := quantWeightFixture(tb, outDim, inDim, groupSize, bits, groupSize+3)
	return x, normW, qw
}

func TestRMSQMVKernelNameCachesGeometryString(t *testing.T) {
	names := []string{
		rmsQMVKernelName(64, 4),
		rmsQMVKernelName(64, 4),
	}
	if names[0] != names[1] {
		t.Fatalf("rms qmv kernel names differ: %q vs %q", names[0], names[1])
	}
	if unsafe.StringData(names[0]) != unsafe.StringData(names[1]) {
		t.Fatalf("rms qmv kernel name backing was not cached for repeated geometry")
	}
}

func TestRMSQMVFastBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const outDim, inDim, groupSize, bits = 64, 512, 64, 4
	const eps = float32(1e-6)
	x, normW, qw := rmsQMVFixture(t, outDim, inDim, groupSize, bits)
	if _, err := RMSQMVFastBF16(x, normW, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits, eps); err != nil {
		t.Fatalf("RMSQMVFastBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := RMSQMVFastBF16(x, normW, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits, eps); err != nil {
			t.Fatalf("RMSQMVFastBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("RMSQMVFastBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestRMSQMVFastBF16IntoUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const outDim, inDim, groupSize, bits = 256, 1536, 64, 4
	const eps = float32(1e-6)
	x, normW, qw := rmsQMVFixture(t, outDim, inDim, groupSize, bits)
	out := make([]byte, outDim*bf16Size)
	for i := range out {
		out[i] = 0xA5
	}

	got, err := RMSQMVFastBF16Into(out, x, normW, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits, eps)
	if err != nil {
		t.Fatalf("RMSQMVFastBF16Into: %v", err)
	}
	if len(got) != len(out) {
		t.Fatalf("RMSQMVFastBF16Into len = %d, want %d", len(got), len(out))
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("RMSQMVFastBF16Into did not return caller-owned output backing")
	}
	want, err := RMSQMVFastBF16(x, normW, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits, eps)
	if err != nil {
		t.Fatalf("RMSQMVFastBF16 reference: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("RMSQMVFastBF16Into output differs from allocating wrapper")
	}
}

func TestRMSQMVFastBF16WithBufferOutputWritesDirectlyToProvidedBuffer(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const outDim, inDim, groupSize, bits = 256, 1536, 64, 4
	const eps = float32(1e-6)
	x, normW, qw := rmsQMVFixture(t, outDim, inDim, groupSize, bits)
	want, err := RMSQMVFastBF16(x, normW, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits, eps)
	if err != nil {
		t.Fatalf("RMSQMVFastBF16: %v", err)
	}

	scratch, err := getQMVBF16Scratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x5a}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	input, err := newPinnedNoCopyBytes(len(x))
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes input: %v", err)
	}
	defer input.Close()
	xBuf, err := input.copyBuffer(x)
	if err != nil {
		t.Fatalf("copy input buffer: %v", err)
	}
	out, err := newPinnedNoCopyBytes(outDim * bf16Size)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes output: %v", err)
	}
	defer out.Close()

	if err := rmsQMVFastBF16WithBufferOutputInPool(x, xBuf, out.buf, normW, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits, eps); err != nil {
		t.Fatalf("rmsQMVFastBF16WithBufferOutputInPool: %v", err)
	}
	if !bytes.Equal(out.bytes, want) {
		t.Fatal("RMSQMVFastBF16 direct Metal output differs from allocating wrapper")
	}

	scratch, err = getQMVBF16Scratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("rmsQMVFastBF16WithBufferOutputInPool wrote through pooled scratch output")
	}
}

// TestRMSQMVFastBF16ParityComposed is the NUMERICAL gate for the fused rms-norm + affine_qmv_fast
// kernel — the matmul-fusion tier. RMSQMVFastBF16(x, normW, W) must track the composed
// QMVBF16(RMSNormBF16(x, normW), W) at cosine ~1.0. The qmv arithmetic is byte-identical (bfloat16_t ==
// native bfloat); only the rms reduction order differs (~1 ULP). A real bug in the rms prologue or the
// per-element normalise collapses the cosine, so this proves the fused matmul in isolation before any
// decode wiring. Random quant weights (both paths share them, so cosine isolates the fusion).
func TestRMSQMVFastBF16ParityComposed(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded — run `task build:kernels`")
	}
	const eps = float32(1e-6)
	cases := []struct{ inDim, outDim, gs int }{
		{1536, 256, 64},  // e2b dModel → a KV-ish out, gs 64
		{1536, 2048, 32}, // e2b dModel → Q-proj width, gs 32
		{512, 1024, 64},  // smaller, single block
	}
	const bits = 4
	for _, c := range cases {
		x := toBF16Bytes(syntheticFloat32(c.inDim, c.inDim+1))
		normW := toBF16Bytes(syntheticFloat32(c.inDim, c.inDim+7))
		wq := make([]byte, c.outDim*c.inDim*bits/8)
		for i := range wq {
			wq[i] = byte((i*131 + 17) % 256) // deterministic packed 4-bit weights
		}
		nSB := c.outDim * (c.inDim / c.gs)
		scales := toBF16Bytes(syntheticFloat32(nSB, c.gs+3))
		biases := toBF16Bytes(syntheticFloat32(nSB, c.gs+5))

		normed, err := RMSNormBF16(x, normW, 1, c.inDim, eps)
		if err != nil {
			t.Fatalf("in=%d out=%d: RMSNormBF16: %v", c.inDim, c.outDim, err)
		}
		ref, err := QMVBF16(normed, wq, scales, biases, c.outDim, c.inDim, c.gs, bits)
		if err != nil {
			t.Fatalf("in=%d out=%d: QMVBF16: %v", c.inDim, c.outDim, err)
		}
		got, err := RMSQMVFastBF16(x, normW, wq, scales, biases, c.outDim, c.inDim, c.gs, bits, eps)
		if err != nil {
			t.Fatalf("in=%d out=%d: RMSQMVFastBF16: %v", c.inDim, c.outDim, err)
		}

		cos := cosineBF16(got, ref)
		t.Logf("in=%-4d out=%-4d gs=%-3d  cosine=%.7f", c.inDim, c.outDim, c.gs, cos)
		if cos < 0.999 {
			t.Fatalf("in=%d out=%d: fused rms+qmv cosine=%.7f < 0.999 — rms prologue / normalise wrong", c.inDim, c.outDim, cos)
		}
	}
}
