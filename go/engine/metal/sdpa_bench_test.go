// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkSDPAHeads8KV4Dim64Len16(b *testing.B) {
	requireNativeRuntime(b)

	const batch, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 16
	q := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 7))
	b.SetBytes(int64(len(q) + len(k) + len(v)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SDPA(q, k, v, batch, nHeads, nKV, headDim, kvLen, 0.125); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSDPAIntoHeads8KV4Dim64Len16(b *testing.B) {
	requireNativeRuntime(b)

	const batch, nHeads, nKV, headDim, kvLen = 1, 8, 4, 64, 16
	q := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 7))
	out := make([]byte, batch*nHeads*headDim*bf16Size)
	b.SetBytes(int64(len(q) + len(k) + len(v)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SDPAInto(out, q, k, v, batch, nHeads, nKV, headDim, kvLen, 0.125); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSDPA2PassHeads4KV2Dim64Len2048(b *testing.B) {
	requireNativeRuntime(b)

	const batch, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 2048
	q := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 7))
	b.SetBytes(int64(len(q) + len(k) + len(v)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SDPA2Pass(q, k, v, batch, nHeads, nKV, headDim, kvLen, 0.125); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSDPA2PassIntoHeads4KV2Dim64Len2048(b *testing.B) {
	requireNativeRuntime(b)

	const batch, nHeads, nKV, headDim, kvLen = 1, 4, 2, 64, 2048
	q := toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, 3))
	k := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 5))
	v := toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, 7))
	out := make([]byte, batch*nHeads*headDim*bf16Size)
	b.SetBytes(int64(len(q) + len(k) + len(v)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SDPA2PassInto(out, q, k, v, batch, nHeads, nKV, headDim, kvLen, 0.125); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSDPAAlternatingShapes(b *testing.B) {
	requireNativeRuntime(b)

	type fixture struct {
		batch, nHeads, nKV, headDim, kvLen int
		q, k, v                            []byte
	}
	makeFixture := func(batch, nHeads, nKV, headDim, kvLen, salt int) fixture {
		return fixture{
			batch: batch, nHeads: nHeads, nKV: nKV, headDim: headDim, kvLen: kvLen,
			q: toBF16Bytes(syntheticFloat32(batch*nHeads*headDim, salt+2)),
			k: toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, salt+4)),
			v: toBF16Bytes(syntheticFloat32(batch*nKV*kvLen*headDim, salt+8)),
		}
	}
	fixtures := []fixture{
		makeFixture(1, 8, 4, 64, 16, 3),
		makeFixture(1, 4, 2, 64, 32, 11),
	}
	perCallBytes := 0
	for _, f := range fixtures {
		perCallBytes += len(f.q) + len(f.k) + len(f.v)
		if _, err := SDPA(f.q, f.k, f.v, f.batch, f.nHeads, f.nKV, f.headDim, f.kvLen, 0.125); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(perCallBytes / len(fixtures)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := fixtures[i&1]
		if _, err := SDPA(f.q, f.k, f.v, f.batch, f.nHeads, f.nKV, f.headDim, f.kvLen, 0.125); err != nil {
			b.Fatal(err)
		}
	}
}
