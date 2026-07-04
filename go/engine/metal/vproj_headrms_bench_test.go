// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkVProjHeadRMSBF16E2BShape(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library not loaded")
	}
	const nKVHeads, headDim, inDim, groupSize, bits = 1, 256, 1536, 64, 4
	const eps = float32(1e-6)
	fx := newVProjHeadRMSFixture(nKVHeads, headDim, inDim, groupSize, bits)

	b.SetBytes(int64(len(fx.x) + len(fx.inNormW) + len(fx.wq) + len(fx.scales) + len(fx.biases)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := VProjHeadRMSBF16(fx.x, fx.inNormW, fx.wq, fx.scales, fx.biases, nKVHeads, headDim, inDim, groupSize, bits, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVProjHeadRMSBF16IntoE2BShape(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library not loaded")
	}
	const nKVHeads, headDim, inDim, groupSize, bits = 1, 256, 1536, 64, 4
	const eps = float32(1e-6)
	fx := newVProjHeadRMSFixture(nKVHeads, headDim, inDim, groupSize, bits)
	out := make([]byte, nKVHeads*headDim*bf16Size)

	b.SetBytes(int64(len(fx.x) + len(fx.inNormW) + len(fx.wq) + len(fx.scales) + len(fx.biases)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := VProjHeadRMSBF16Into(out, fx.x, fx.inNormW, fx.wq, fx.scales, fx.biases, nKVHeads, headDim, inDim, groupSize, bits, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVProjHeadRMSBF16BufferOutputE2BShape(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library not loaded")
	}
	const nKVHeads, headDim, inDim, groupSize, bits = 1, 256, 1536, 64, 4
	const eps = float32(1e-6)
	fx := newVProjHeadRMSFixture(nKVHeads, headDim, inDim, groupSize, bits)
	input, err := newPinnedNoCopyBytes(len(fx.x))
	if err != nil {
		b.Fatal(err)
	}
	defer input.Close()
	xBuf, err := input.copyBuffer(fx.x)
	if err != nil {
		b.Fatal(err)
	}
	out, err := newPinnedNoCopyBytes(nKVHeads * headDim * bf16Size)
	if err != nil {
		b.Fatal(err)
	}
	defer out.Close()

	b.SetBytes(int64(len(fx.x) + len(fx.inNormW) + len(fx.wq) + len(fx.scales) + len(fx.biases)))
	if err := vProjHeadRMSBF16WithBufferOutputInPool(fx.x, xBuf, out.buf, fx.inNormW, fx.wq, fx.scales, fx.biases, nKVHeads, headDim, inDim, groupSize, bits, eps); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := vProjHeadRMSBF16WithBufferOutputInPool(fx.x, xBuf, out.buf, fx.inNormW, fx.wq, fx.scales, fx.biases, nKVHeads, headDim, inDim, groupSize, bits, eps); err != nil {
			b.Fatal(err)
		}
	}
}
