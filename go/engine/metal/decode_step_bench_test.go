// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkDecodeStepKV64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 1, 1, 64, 4, 1, 128
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	qDim, kvDim := nHeads*headDim, nKV*headDim
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 11))
	_ = qDim
	kc := append([]byte(nil), kCache...)
	vc := append([]byte(nil), vCache...)
	if _, err := DecodeStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeStepKVInto64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 1, 1, 64, 4, 1, 128
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	kvDim := nKV * headDim
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 11))
	out := make([]byte, dModel*bf16Size)
	kc := append([]byte(nil), kCache...)
	vc := append([]byte(nil), vCache...)
	if _, err := DecodeStepKVInto(out, x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeStepKVInto(out, x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAttentionStepKV64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 1, 1, 64, 4, 1, 128
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	kvDim := nKV * headDim
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 11))
	kc := append([]byte(nil), kCache...)
	vc := append([]byte(nil), vCache...)
	if _, err := AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, dModel, nHeads, nKV, headDim, maxLen, pos, base, scale, eps); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, dModel, nHeads, nKV, headDim, maxLen, pos, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAttentionStepKVInto64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 1, 1, 64, 4, 1, 128
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	kvDim := nKV * headDim
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(maxLen*kvDim, 11))
	out := make([]byte, dModel*bf16Size)
	kc := append([]byte(nil), kCache...)
	vc := append([]byte(nil), vCache...)
	if _, err := AttentionStepKVInto(out, x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, dModel, nHeads, nKV, headDim, maxLen, pos, base, scale, eps); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AttentionStepKVInto(out, x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kc, vc, dModel, nHeads, nKV, headDim, maxLen, pos, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeStepKVAlternatingShapes(b *testing.B) {
	requireNativeRuntime(b)

	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	type fixture struct {
		dModel, nHeads, nKV, headDim, maxLen, pos, dFF int
		layer                                          DecodeLayerWeights
		x, kCache, vCache                              []byte
	}
	makeFixture := func(dModel, nHeads, nKV, headDim, maxLen, pos, dFF, salt int) fixture {
		kvDim := nKV * headDim
		return fixture{
			dModel: dModel, nHeads: nHeads, nKV: nKV, headDim: headDim, maxLen: maxLen, pos: pos, dFF: dFF,
			layer:  decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, salt),
			x:      toBF16Bytes(syntheticFloat32(dModel, salt+2)),
			kCache: toBF16Bytes(syntheticFloat32(maxLen*kvDim, salt+4)),
			vCache: toBF16Bytes(syntheticFloat32(maxLen*kvDim, salt+8)),
		}
	}
	fixtures := []fixture{
		makeFixture(64, 1, 1, 64, 4, 1, 128, 3),
		makeFixture(128, 2, 1, 64, 8, 2, 256, 11),
	}
	perCallBytes := 0
	for _, f := range fixtures {
		perCallBytes += len(f.x) + len(f.kCache) + len(f.vCache)
		kc := append([]byte(nil), f.kCache...)
		vc := append([]byte(nil), f.vCache...)
		if _, err := DecodeStepKV(f.x, f.layer.AttnNormW, f.layer.WQ, f.layer.WK, f.layer.WV, f.layer.WO, kc, vc, f.layer.MLPNormW, f.layer.WGate, f.layer.WUp, f.layer.WDown, f.dModel, f.nHeads, f.nKV, f.headDim, f.maxLen, f.dFF, f.pos, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(perCallBytes / len(fixtures)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := fixtures[i&1]
		kc := append([]byte(nil), f.kCache...)
		vc := append([]byte(nil), f.vCache...)
		if _, err := DecodeStepKV(f.x, f.layer.AttnNormW, f.layer.WQ, f.layer.WK, f.layer.WV, f.layer.WO, kc, vc, f.layer.MLPNormW, f.layer.WGate, f.layer.WUp, f.layer.WDown, f.dModel, f.nHeads, f.nKV, f.headDim, f.maxLen, f.dFF, f.pos, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}
