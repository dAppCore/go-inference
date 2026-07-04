// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkSquareICB64(b *testing.B) {
	requireNativeRuntime(b)

	in := syntheticFloat32(64, 19)
	b.SetBytes(int64(len(in) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := squareICB(in); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGemvICB16x64(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim = 16, 64
	mat := syntheticFloat32(outDim*inDim, 37)
	vec := syntheticFloat32(inDim, 53)
	b.SetBytes(int64((len(mat) + len(vec)) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := gemvICB(mat, vec, outDim, inDim); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRebindProbeICB3x16x64(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim, nRows = 16, 64, 3
	mat := syntheticFloat32(outDim*inDim, 37)
	vec := syntheticFloat32(inDim, 53)
	b.SetBytes(int64((len(mat) + len(vec)) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := rebindProbeICB(mat, vec, outDim, inDim, nRows); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNormProjectICB128x256(b *testing.B) {
	requireNativeRuntime(b)

	const dIn, dOut = 128, 256
	x := syntheticFloat32(dIn, 3)
	normW := syntheticFloat32(dIn, 5)
	projW := syntheticFloat32(dOut*dIn, 7)
	b.SetBytes(int64((len(x) + len(normW) + len(projW)) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := NormProjectICB(x, normW, projW, dIn, dOut, 1e-5, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQMVICB16x64(b *testing.B) {
	requireNativeRuntime(b)

	const outDim, inDim, groupSize, bits = 16, 64, 32, 4
	qw := quantWeightFixture(b, outDim, inDim, groupSize, bits, 37)
	x := toBF16Bytes(syntheticFloat32(inDim, 53))
	b.SetBytes(int64(len(x) + len(qw.Packed) + len(qw.Scales) + len(qw.Biases)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := qmvICB(x, qw.Packed, qw.Scales, qw.Biases, outDim, inDim, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAttentionBlockICB64(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	b.SetBytes(int64(len(x) + len(kCache) + len(vCache)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAttentionBlockICBAlternatingShape(b *testing.B) {
	requireNativeRuntime(b)

	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	cases := []struct {
		dModel, nHeads, nKV, headDim, kvLen, dFF int
		layer                                    DecodeLayerWeights
		x, kCache, vCache                        []byte
	}{
		{dModel: 64, nHeads: 1, nKV: 1, headDim: 64, kvLen: 2, dFF: 128},
		{dModel: 128, nHeads: 2, nKV: 1, headDim: 64, kvLen: 4, dFF: 256},
	}
	var totalBytes int64
	for i := range cases {
		cases[i].layer = decodeLayerFixture(cases[i].dModel, cases[i].nHeads, cases[i].nKV, cases[i].headDim, cases[i].dFF, 3)
		cases[i].x = toBF16Bytes(syntheticFloat32(cases[i].dModel, 5))
		cases[i].kCache = toBF16Bytes(syntheticFloat32(cases[i].nKV*cases[i].kvLen*cases[i].headDim, 7))
		cases[i].vCache = toBF16Bytes(syntheticFloat32(cases[i].nKV*cases[i].kvLen*cases[i].headDim, 11))
		totalBytes += int64(len(cases[i].x) + len(cases[i].kCache) + len(cases[i].vCache))
		if _, err := AttentionBlockICB(cases[i].x, cases[i].layer.AttnNormW, cases[i].layer.WQ, cases[i].layer.WO, cases[i].kCache, cases[i].vCache, cases[i].dModel, cases[i].nHeads, cases[i].nKV, cases[i].headDim, cases[i].kvLen, base, scale, offset, eps, 1); err != nil {
			b.Fatalf("warmup dModel %d: %v", cases[i].dModel, err)
		}
	}
	b.SetBytes(totalBytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := cases[i&1]
		if _, err := AttentionBlockICB(c.x, c.layer.AttnNormW, c.layer.WQ, c.layer.WO, c.kCache, c.vCache, c.dModel, c.nHeads, c.nKV, c.headDim, c.kvLen, base, scale, offset, eps, 1); err != nil {
			b.Fatal(err)
		}
	}
}
