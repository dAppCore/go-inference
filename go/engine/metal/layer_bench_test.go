// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkDecodeLayer64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 4, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeLayerInto64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 1, 1, 64, 4, 128
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeLayerInto(out, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeLayerAlternatingShapes(b *testing.B) {
	requireNativeRuntime(b)

	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	type fixture struct {
		dModel, nHeads, nKV, headDim, kvLen, dFF int
		layer                                    DecodeLayerWeights
		x, kCache, vCache                        []byte
	}
	makeFixture := func(dModel, nHeads, nKV, headDim, kvLen, dFF, salt int) fixture {
		return fixture{
			dModel: dModel, nHeads: nHeads, nKV: nKV, headDim: headDim, kvLen: kvLen, dFF: dFF,
			layer:  decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, salt),
			x:      toBF16Bytes(syntheticFloat32(dModel, salt+2)),
			kCache: toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, salt+4)),
			vCache: toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, salt+8)),
		}
	}
	fixtures := []fixture{
		makeFixture(64, 1, 1, 64, 4, 128, 3),
		makeFixture(128, 2, 1, 64, 8, 256, 11),
	}
	perCallBytes := 0
	for _, f := range fixtures {
		perCallBytes += len(f.x) + len(f.kCache) + len(f.vCache)
		if _, err := DecodeLayer(f.x, f.layer.AttnNormW, f.layer.WQ, f.layer.WO, f.kCache, f.vCache, f.layer.MLPNormW, f.layer.WGate, f.layer.WUp, f.layer.WDown, f.dModel, f.nHeads, f.nKV, f.headDim, f.kvLen, f.dFF, base, scale, int(offset), eps); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(perCallBytes / len(fixtures)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := fixtures[i&1]
		if _, err := DecodeLayer(f.x, f.layer.AttnNormW, f.layer.WQ, f.layer.WO, f.kCache, f.vCache, f.layer.MLPNormW, f.layer.WGate, f.layer.WUp, f.layer.WDown, f.dModel, f.nHeads, f.nKV, f.headDim, f.kvLen, f.dFF, base, scale, int(offset), eps); err != nil {
			b.Fatal(err)
		}
	}
}
