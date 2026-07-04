// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkDecodeLayerICB64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 4, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeLayerICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeLayerICBInto64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 4, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	out := make([]byte, dModel*bf16Size)
	if _, err := DecodeLayerICBInto(out, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeLayerICBInto(out, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeTokenICB64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers = 64, 1, 1, 64, 4, 128, 1
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeTokenICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeTokenICBInto64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers = 64, 1, 1, 64, 4, 128, 1
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	out := make([]byte, dModel*bf16Size)
	if _, err := DecodeTokenICBInto(out, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, 1); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeTokenICBInto(out, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, base, scale, offset, eps, 1); err != nil {
			b.Fatal(err)
		}
	}
}
