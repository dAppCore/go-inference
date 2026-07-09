// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

func BenchmarkQKNormRopeBF16Heads8Dim256(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	log2Theta := float32(math.Log2(float64(theta)))

	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := QKNormRopeBF16(x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQKNormRopeBF16IntoHeads8Dim256(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	out := make([]byte, len(x))
	log2Theta := float32(math.Log2(float64(theta)))

	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := QKNormRopeBF16Into(out, x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQKNormRopeBF16BufferOutputHeads8Dim256(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	input, err := newPinnedNoCopyBytes(len(x))
	if err != nil {
		b.Fatal(err)
	}
	defer input.Close()
	xBuf, err := input.copyBuffer(x)
	if err != nil {
		b.Fatal(err)
	}
	out, err := newPinnedNoCopyBytes(len(x))
	if err != nil {
		b.Fatal(err)
	}
	defer out.Close()
	log2Theta := float32(math.Log2(float64(theta)))

	b.SetBytes(int64(len(x)))
	if err := qkNormRopeBF16WithBufferOutputInPool(x, xBuf, out.buf, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := qkNormRopeBF16WithBufferOutputInPool(x, xBuf, out.buf, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQKNormRopeBF16FreqsHeads8Dim256(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const nHeads, headDim, rotaryDim = 8, 256, 128
	const eps, scale, theta = float32(1e-6), float32(1.0), float32(10000)
	x := toBF16Bytes(syntheticFloat32(nHeads*headDim, headDim+1))
	w := toBF16Bytes(syntheticFloat32(headDim, headDim+7))
	log2Theta := float32(math.Log2(float64(theta)))
	periods := make([]float32, rotaryDim/2)
	for i := range periods {
		invFreq := float32(math.Exp2(-float64(i) / float64(rotaryDim/2) * float64(log2Theta)))
		periods[i] = 1.0 / invFreq
	}

	b.SetBytes(int64(len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := QKNormRopeBF16(x, w, nHeads, headDim, rotaryDim, 7, scale, eps, log2Theta, periods); err != nil {
			b.Fatal(err)
		}
	}
}
