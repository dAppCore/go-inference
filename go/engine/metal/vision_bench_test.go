// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMatRowsBF16_4x128x256(b *testing.B) {
	requireNativeRuntime(b)

	const L, outDim, inDim = 4, 128, 256
	w, in := matRowsBF16Fixture(L, outDim, inDim)
	b.SetBytes(int64(len(w) + len(in)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatRowsBF16(w, in, L, outDim, inDim); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatRowsF32_4x128x256(b *testing.B) {
	requireNativeRuntime(b)

	const L, outDim, inDim = 4, 128, 256
	w := syntheticFloat32(outDim*inDim, outDim+7)
	in := syntheticFloat32(L*inDim, inDim+5)
	b.SetBytes(int64((len(w) + len(in)) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := matRowsF32(w, in, L, outDim, inDim); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVisionSDPAHeads4KV2Len64Dim64(b *testing.B) {
	requireNativeRuntime(b)

	const L, nHeads, nKVHeads, headDim = 64, 4, 2, 64
	q := toBF16Bytes(bf16Round(syntheticFloat32(nHeads*L*headDim, 31)))
	k := toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*L*headDim, 37)))
	v := toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*L*headDim, 41)))
	b.SetBytes(int64(len(q) + len(k) + len(v)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := VisionSDPA(q, k, v, L, nHeads, nKVHeads, headDim, 0.125); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVisionSDPAAlternatingShapes(b *testing.B) {
	requireNativeRuntime(b)

	type fixture struct {
		L, nHeads, nKVHeads, headDim int
		q, k, v                      []byte
	}
	makeFixture := func(L, nHeads, nKVHeads, headDim, salt int) fixture {
		return fixture{
			L: L, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
			q: toBF16Bytes(bf16Round(syntheticFloat32(nHeads*L*headDim, salt+2))),
			k: toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*L*headDim, salt+4))),
			v: toBF16Bytes(bf16Round(syntheticFloat32(nKVHeads*L*headDim, salt+8))),
		}
	}
	fixtures := []fixture{
		makeFixture(32, 4, 2, 64, 3),
		makeFixture(64, 4, 2, 64, 11),
	}
	perCallBytes := 0
	for _, f := range fixtures {
		perCallBytes += len(f.q) + len(f.k) + len(f.v)
		if _, err := VisionSDPA(f.q, f.k, f.v, f.L, f.nHeads, f.nKVHeads, f.headDim, 0.125); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(perCallBytes / len(fixtures)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := fixtures[i&1]
		if _, err := VisionSDPA(f.q, f.k, f.v, f.L, f.nHeads, f.nKVHeads, f.headDim, 0.125); err != nil {
			b.Fatal(err)
		}
	}
}
