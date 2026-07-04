// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkQMV64x128(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(b, outDim, inDim, groupSize, bits, 3)
	x := syntheticFloat32(inDim, 5)
	b.SetBytes(int64(len(qw.Packed) + len(qw.Scales) + len(qw.Biases) + len(x)*4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := QMV(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQMVInto64x128(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(b, outDim, inDim, groupSize, bits, 3)
	x := syntheticFloat32(inDim, 5)
	out := make([]float32, outDim)
	b.SetBytes(int64(len(qw.Packed) + len(qw.Scales) + len(qw.Biases) + len(x)*4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = QMVInto(out, x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQMVBF1664x128(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(b, outDim, inDim, groupSize, bits, 3)
	x := toBF16Bytes(syntheticFloat32(inDim, 5))
	b.SetBytes(int64(len(qw.Packed) + len(qw.Scales) + len(qw.Biases) + len(x)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := QMVBF16(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQMVBF16Into64x128(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(b, outDim, inDim, groupSize, bits, 3)
	x := toBF16Bytes(syntheticFloat32(inDim, 5))
	out := make([]byte, outDim*bf16Size)
	b.SetBytes(int64(len(qw.Packed) + len(qw.Scales) + len(qw.Biases) + len(x)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = QMVBF16Into(out, x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQMVBF16Resident64x128(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(b, outDim, inDim, groupSize, bits, 3)
	x := toBF16Bytes(syntheticFloat32(inDim, 5))
	b.SetBytes(int64(len(qw.Packed) + len(qw.Scales) + len(qw.Biases) + len(x)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := qmvBF16Resident(x, qw, outDim, inDim, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQMVBF16ResidentInto64x128(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim, groupSize, bits = 64, 128, 64, 4
	qw := quantWeightFixture(b, outDim, inDim, groupSize, bits, 3)
	x := toBF16Bytes(syntheticFloat32(inDim, 5))
	out := make([]byte, outDim*bf16Size)
	b.SetBytes(int64(len(qw.Packed) + len(qw.Scales) + len(qw.Biases) + len(x)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = qmvBF16ResidentInto(out, x, qw, outDim, inDim, groupSize, bits)
		if err != nil {
			b.Fatal(err)
		}
	}
}
