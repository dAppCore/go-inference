// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkSDPACausalBF16Heads2KV1Len4Dim64(b *testing.B) {
	requireNativeRuntime(b)

	const H, Hkv, qL, kL, D = 2, 1, 4, 4, 64
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	b.SetBytes(int64(len(q) + len(k) + len(v)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSDPACausalBF16IntoHeads2KV1Len4Dim64(b *testing.B) {
	requireNativeRuntime(b)

	const H, Hkv, qL, kL, D = 2, 1, 4, 4, 64
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	out := make([]byte, H*qL*D*bf16Size)
	b.SetBytes(int64(len(q) + len(k) + len(v)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = SDPACausalBF16Into(out, q, k, v, H, Hkv, qL, kL, D, scale)
		if err != nil {
			b.Fatal(err)
		}
	}
}
