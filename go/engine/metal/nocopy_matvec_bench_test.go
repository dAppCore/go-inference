// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMatVecBF16BufResident(b *testing.B) {
	requireNativeRuntime(b)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const outDim, inDim = 128, 256
	mat := toBF16Bytes(syntheticFloat32(outDim*inDim, 3))
	vec := toBF16Bytes(syntheticFloat32(inDim, 5))
	matView := bufView{buf: residentBytes(mat), off: 0}
	b.SetBytes(int64(len(mat) + len(vec)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatVecBF16Buf(matView, vec, outDim, inDim); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecBF16BufResidentInto(b *testing.B) {
	requireNativeRuntime(b)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const outDim, inDim = 128, 256
	mat := toBF16Bytes(syntheticFloat32(outDim*inDim, 3))
	vec := toBF16Bytes(syntheticFloat32(inDim, 5))
	matView := bufView{buf: residentBytes(mat), off: 0}
	out := make([]byte, outDim*bf16Size)
	b.SetBytes(int64(len(mat) + len(vec)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = MatVecBF16BufInto(out, matView, vec, outDim, inDim)
		if err != nil {
			b.Fatal(err)
		}
	}
}
