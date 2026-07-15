// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"
)

type vProjHeadRMSFixture struct {
	x       []byte
	inNormW []byte
	wq      []byte
	scales  []byte
	biases  []byte
}

func newVProjHeadRMSFixture(nKVHeads, headDim, inDim, groupSize, bits int) vProjHeadRMSFixture {
	outDim := nKVHeads * headDim
	x := toBF16Bytes(syntheticFloat32(inDim, inDim+1))
	inNormW := toBF16Bytes(syntheticFloat32(inDim, inDim+7))
	wq := make([]byte, outDim*inDim*bits/8)
	for i := range wq {
		wq[i] = byte((i*131 + 17) % 256)
	}
	nSB := outDim * (inDim / groupSize)
	return vProjHeadRMSFixture{
		x:       x,
		inNormW: inNormW,
		wq:      wq,
		scales:  toBF16Bytes(syntheticFloat32(nSB, groupSize+3)),
		biases:  toBF16Bytes(syntheticFloat32(nSB, groupSize+5)),
	}
}

func TestVProjHeadRMSKernelNameCachesGeometryString(t *testing.T) {
	names := []string{
		vprojHeadRMSKernelName(64, 4),
		vprojHeadRMSKernelName(64, 4),
	}
	if names[0] != names[1] {
		t.Fatalf("vproj head rms kernel names differ: %q vs %q", names[0], names[1])
	}
	if unsafe.StringData(names[0]) != unsafe.StringData(names[1]) {
		t.Fatalf("vproj head rms kernel name backing was not cached for repeated geometry")
	}
}

func TestVProjHeadRMSBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded — run `task build:kernels`")
	}
	const nKVHeads, headDim, inDim, groupSize, bits = 1, 256, 1536, 64, 4
	const eps = float32(1e-6)
	fx := newVProjHeadRMSFixture(nKVHeads, headDim, inDim, groupSize, bits)
	if _, err := VProjHeadRMSBF16(fx.x, fx.inNormW, fx.wq, fx.scales, fx.biases, nKVHeads, headDim, inDim, groupSize, bits, eps); err != nil {
		t.Fatalf("VProjHeadRMSBF16 warmup: %v", err)
	}

	var vprojErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, vprojErr = VProjHeadRMSBF16(fx.x, fx.inNormW, fx.wq, fx.scales, fx.biases, nKVHeads, headDim, inDim, groupSize, bits, eps)
	})
	if vprojErr != nil {
		t.Fatalf("VProjHeadRMSBF16: %v", vprojErr)
	}
	if allocs > 10 {
		t.Fatalf("VProjHeadRMSBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestVProjHeadRMSBF16IntoUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded — run `task build:kernels`")
	}
	const nKVHeads, headDim, inDim, groupSize, bits = 1, 256, 1536, 64, 4
	const eps = float32(1e-6)
	fx := newVProjHeadRMSFixture(nKVHeads, headDim, inDim, groupSize, bits)
	out := make([]byte, nKVHeads*headDim*bf16Size)
	for i := range out {
		out[i] = 0xA5
	}

	got, err := VProjHeadRMSBF16Into(out, fx.x, fx.inNormW, fx.wq, fx.scales, fx.biases, nKVHeads, headDim, inDim, groupSize, bits, eps)
	if err != nil {
		t.Fatalf("VProjHeadRMSBF16Into: %v", err)
	}
	if len(got) != len(out) {
		t.Fatalf("VProjHeadRMSBF16Into len = %d, want %d", len(got), len(out))
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("VProjHeadRMSBF16Into did not return caller-owned output backing")
	}
	want, err := VProjHeadRMSBF16(fx.x, fx.inNormW, fx.wq, fx.scales, fx.biases, nKVHeads, headDim, inDim, groupSize, bits, eps)
	if err != nil {
		t.Fatalf("VProjHeadRMSBF16 reference: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("VProjHeadRMSBF16Into output differs from allocating wrapper")
	}
}

func TestVProjHeadRMSBF16WithBufferOutputWritesDirectlyToProvidedBuffer(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded — run `task build:kernels`")
	}

	const nKVHeads, headDim, inDim, groupSize, bits = 1, 256, 1536, 64, 4
	const eps = float32(1e-6)
	fx := newVProjHeadRMSFixture(nKVHeads, headDim, inDim, groupSize, bits)
	want, err := VProjHeadRMSBF16(fx.x, fx.inNormW, fx.wq, fx.scales, fx.biases, nKVHeads, headDim, inDim, groupSize, bits, eps)
	if err != nil {
		t.Fatalf("VProjHeadRMSBF16: %v", err)
	}

	outDim := nKVHeads * headDim
	scratch, err := getQMVBF16Scratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xc7}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	input, err := newPinnedNoCopyBytes(len(fx.x))
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes input: %v", err)
	}
	defer input.Close()
	xBuf, err := input.copyBuffer(fx.x)
	if err != nil {
		t.Fatalf("copy input buffer: %v", err)
	}
	out, err := newPinnedNoCopyBytes(outDim * bf16Size)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes output: %v", err)
	}
	defer out.Close()

	if err := vProjHeadRMSBF16WithBufferOutputInPool(fx.x, xBuf, out.buf, fx.inNormW, fx.wq, fx.scales, fx.biases, nKVHeads, headDim, inDim, groupSize, bits, eps); err != nil {
		t.Fatalf("vProjHeadRMSBF16WithBufferOutputInPool: %v", err)
	}
	if !bytes.Equal(out.bytes, want) {
		t.Fatal("VProjHeadRMSBF16 direct Metal output differs from allocating wrapper")
	}

	scratch, err = getQMVBF16Scratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("vProjHeadRMSBF16WithBufferOutputInPool wrote through pooled scratch output")
	}
}

// TestVProjHeadRMSBF16ParityComposed gates the fused V-path kernel (input-rms → V-proj → value-norm)
// against the composed RMSNormBF16(QMVBF16(RMSNormBF16(x, inNormW)), ones, nKVHeads, headDim). Cosine
// ~1.0 (lockstep: the per-thread dot + the two rms reductions differ in summation order, ~1 ULP). A
// real wiring bug — wrong head slicing, missing a norm — collapses the cosine.
func TestVProjHeadRMSBF16ParityComposed(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded — run `task build:kernels`")
	}
	const eps = float32(1e-6)
	const bits = 4
	cases := []struct{ nKVHeads, headDim, inDim, gs int }{
		{1, 256, 1536, 64}, // e2b: 1 KV head, headDim 256
		{2, 128, 1536, 64}, // 2 KV heads, headDim 128 (outDim 256)
		{1, 512, 1536, 32}, // wider head
	}
	for _, c := range cases {
		outDim := c.nKVHeads * c.headDim
		fx := newVProjHeadRMSFixture(c.nKVHeads, c.headDim, c.inDim, c.gs, bits)
		onesF := make([]float32, c.headDim)
		for i := range onesF {
			onesF[i] = 1
		}
		ones := toBF16Bytes(onesF)

		normed, err := RMSNormBF16(fx.x, fx.inNormW, 1, c.inDim, eps)
		if err != nil {
			t.Fatalf("nkv=%d hd=%d: input RMSNorm: %v", c.nKVHeads, c.headDim, err)
		}
		vproj, err := QMVBF16(normed, fx.wq, fx.scales, fx.biases, outDim, c.inDim, c.gs, bits)
		if err != nil {
			t.Fatalf("nkv=%d hd=%d: QMV: %v", c.nKVHeads, c.headDim, err)
		}
		ref, err := RMSNormBF16(vproj, ones, c.nKVHeads, c.headDim, eps)
		if err != nil {
			t.Fatalf("nkv=%d hd=%d: value-norm: %v", c.nKVHeads, c.headDim, err)
		}
		got, err := VProjHeadRMSBF16(fx.x, fx.inNormW, fx.wq, fx.scales, fx.biases, c.nKVHeads, c.headDim, c.inDim, c.gs, bits, eps)
		if err != nil {
			t.Fatalf("nkv=%d hd=%d: VProjHeadRMSBF16: %v", c.nKVHeads, c.headDim, err)
		}

		cos := cosineBF16(got, ref)
		t.Logf("nkv=%-2d hd=%-4d inDim=%-4d gs=%-3d  cosine=%.7f", c.nKVHeads, c.headDim, c.inDim, c.gs, cos)
		if cos < 0.999 {
			t.Fatalf("nkv=%d hd=%d: fused V-path cosine=%.7f < 0.999 — wrong", c.nKVHeads, c.headDim, cos)
		}
	}
}
