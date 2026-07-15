// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkRMSQMVFastBF1664x512(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const outDim, inDim, groupSize, bits = 64, 512, 64, 4
	const eps = float32(1e-6)
	x, normW, qw := rmsQMVFixture(b, outDim, inDim, groupSize, bits)
	b.SetBytes(int64(len(qw.Packed) + len(qw.Scales) + len(qw.Biases) + len(x) + len(normW)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RMSQMVFastBF16(x, normW, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRMSQMVFastBF16Into64x512(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const outDim, inDim, groupSize, bits = 64, 512, 64, 4
	const eps = float32(1e-6)
	x, normW, qw := rmsQMVFixture(b, outDim, inDim, groupSize, bits)
	out := make([]byte, outDim*bf16Size)
	b.SetBytes(int64(len(qw.Packed) + len(qw.Scales) + len(qw.Biases) + len(x) + len(normW)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RMSQMVFastBF16Into(out, x, normW, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRMSQMVFastBF16BufferOutput64x512(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const outDim, inDim, groupSize, bits = 64, 512, 64, 4
	const eps = float32(1e-6)
	x, normW, qw := rmsQMVFixture(b, outDim, inDim, groupSize, bits)
	input, err := newPinnedNoCopyBytes(len(x))
	if err != nil {
		b.Fatal(err)
	}
	defer input.Close()
	xBuf, err := input.copyBuffer(x)
	if err != nil {
		b.Fatal(err)
	}
	out, err := newPinnedNoCopyBytes(outDim * bf16Size)
	if err != nil {
		b.Fatal(err)
	}
	defer out.Close()
	b.SetBytes(int64(len(qw.Packed) + len(qw.Scales) + len(qw.Biases) + len(x) + len(normW)))
	if err := rmsQMVFastBF16WithBufferOutputInPool(x, xBuf, out.buf, normW, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits, eps); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := rmsQMVFastBF16WithBufferOutputInPool(x, xBuf, out.buf, normW, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits, eps); err != nil {
			b.Fatal(err)
		}
	}
}
