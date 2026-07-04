// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkBF16Add1024(b *testing.B) {
	requireNativeRuntime(b)

	a := toBF16Bytes(syntheticFloat32(1024, 3))
	c := toBF16Bytes(syntheticFloat32(1024, 5))
	b.SetBytes(int64(len(a)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AddBF16(a, c); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBF16AddInto1024(b *testing.B) {
	requireNativeRuntime(b)

	a := toBF16Bytes(syntheticFloat32(1024, 3))
	c := toBF16Bytes(syntheticFloat32(1024, 5))
	out := make([]byte, len(a))
	b.ReportAllocs()
	b.SetBytes(int64(len(a)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := AddBF16Into(out, a, c); err != nil {
			b.Fatal(err)
		}
	}
}
