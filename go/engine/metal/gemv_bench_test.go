// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMatVec128x256(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim = 128, 256
	mat := syntheticFloat32(outDim*inDim, 3)
	vec := syntheticFloat32(inDim, 5)
	b.SetBytes(int64((len(mat) + len(vec)) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatVec(mat, vec, outDim, inDim); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecInto128x256(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim = 128, 256
	mat := syntheticFloat32(outDim*inDim, 3)
	vec := syntheticFloat32(inDim, 5)
	out := make([]float32, outDim)
	b.SetBytes(int64((len(mat) + len(vec)) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = MatVecInto(out, mat, vec, outDim, inDim)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecBF16128x256(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim = 128, 256
	mat, vec := matVecBF16Fixture(outDim, inDim)
	b.SetBytes(int64(len(mat) + len(vec)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatVecBF16(mat, vec, outDim, inDim); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecBF16Into128x256(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim = 128, 256
	mat, vec := matVecBF16Fixture(outDim, inDim)
	out := make([]byte, outDim*bf16Size)
	b.SetBytes(int64(len(mat) + len(vec)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = MatVecBF16Into(out, mat, vec, outDim, inDim)
		if err != nil {
			b.Fatal(err)
		}
	}
}
