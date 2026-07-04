// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkAttentionBlock64(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAttentionBlockInto64(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AttentionBlockInto(out, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAttentionBlockAlternatingKVShapes(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim = 64, 1, 1, 64
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	type fixture struct {
		kvLen          int
		kCache, vCache []byte
	}
	fixtures := []fixture{
		{
			kvLen:  4,
			kCache: toBF16Bytes(syntheticFloat32(nKV*4*headDim, 7)),
			vCache: toBF16Bytes(syntheticFloat32(nKV*4*headDim, 11)),
		},
		{
			kvLen:  8,
			kCache: toBF16Bytes(syntheticFloat32(nKV*8*headDim, 13)),
			vCache: toBF16Bytes(syntheticFloat32(nKV*8*headDim, 17)),
		},
	}
	perCallBytes := 0
	for _, f := range fixtures {
		perCallBytes += len(x) + len(f.kCache) + len(f.vCache)
		if _, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, f.kCache, f.vCache, dModel, nHeads, nKV, headDim, f.kvLen, base, scale, offset, eps); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(perCallBytes / len(fixtures)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := fixtures[i&1]
		if _, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, f.kCache, f.vCache, dModel, nHeads, nKV, headDim, f.kvLen, base, scale, offset, eps); err != nil {
			b.Fatal(err)
		}
	}
}
