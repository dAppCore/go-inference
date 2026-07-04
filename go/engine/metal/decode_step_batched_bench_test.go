// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkDecodeLayerBatchedKV4x256(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K = 256, 4, 2, 64, 32, 512, 5, 4
	const base, scale, eps = float32(10000), float32(1.0 / 8.0), float32(1e-6)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	attnNormW := toBF16Bytes(syntheticFloat32(dModel, 1))
	mlpNormW := toBF16Bytes(syntheticFloat32(dModel, 2))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 3))
	wK := toBF16Bytes(syntheticFloat32(kvDim*dModel, 4))
	wV := toBF16Bytes(syntheticFloat32(kvDim*dModel, 5))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 6))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 8))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 9))
	kCache := make([]byte, maxLen*kvDim*bf16Size)
	vCache := make([]byte, maxLen*kvDim*bf16Size)
	copy(kCache, toBF16Bytes(syntheticFloat32(basePos*kvDim, 10)))
	copy(vCache, toBF16Bytes(syntheticFloat32(basePos*kvDim, 11)))
	xs := toBF16Bytes(syntheticFloat32(K*dModel, 12))
	kc := make([]byte, len(kCache))
	vc := make([]byte, len(vCache))
	copy(kc, kCache)
	copy(vc, vCache)
	if _, err := DecodeLayerBatchedKV(xs, attnNormW, wQ, wK, wV, wO, kc, vc, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps); err != nil {
		b.Fatal(err)
	}

	b.SetBytes(int64(len(xs) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(kc, kCache)
		copy(vc, vCache)
		if _, err := DecodeLayerBatchedKV(xs, attnNormW, wQ, wK, wV, wO, kc, vc, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeLayerBatchedKVInto4x256(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K = 256, 4, 2, 64, 32, 512, 5, 4
	const base, scale, eps = float32(10000), float32(1.0 / 8.0), float32(1e-6)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	attnNormW := toBF16Bytes(syntheticFloat32(dModel, 1))
	mlpNormW := toBF16Bytes(syntheticFloat32(dModel, 2))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 3))
	wK := toBF16Bytes(syntheticFloat32(kvDim*dModel, 4))
	wV := toBF16Bytes(syntheticFloat32(kvDim*dModel, 5))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 6))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 8))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 9))
	kCache := make([]byte, maxLen*kvDim*bf16Size)
	vCache := make([]byte, maxLen*kvDim*bf16Size)
	copy(kCache, toBF16Bytes(syntheticFloat32(basePos*kvDim, 10)))
	copy(vCache, toBF16Bytes(syntheticFloat32(basePos*kvDim, 11)))
	xs := toBF16Bytes(syntheticFloat32(K*dModel, 12))
	out := make([]byte, K*dModel*bf16Size)
	kc := make([]byte, len(kCache))
	vc := make([]byte, len(vCache))
	copy(kc, kCache)
	copy(vc, vCache)
	if _, err := DecodeLayerBatchedKVInto(out, xs, attnNormW, wQ, wK, wV, wO, kc, vc, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps); err != nil {
		b.Fatal(err)
	}

	b.SetBytes(int64(len(xs) + len(kCache) + len(vCache)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(kc, kCache)
		copy(vc, vCache)
		if _, err := DecodeLayerBatchedKVInto(out, xs, attnNormW, wQ, wK, wV, wO, kc, vc, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeLayerBatchedKVAlternatingShape(b *testing.B) {
	requireNativeRuntime(b)

	const base, eps = float32(10000), float32(1e-6)
	cases := []struct {
		dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K int
		scale                                                      float32
		xs, attnNormW, wQ, wK, wV, wO, kCache, vCache              []byte
		mlpNormW, wGate, wUp, wDown                                []byte
	}{
		{dModel: 128, nHeads: 2, nKVHeads: 1, headDim: 64, maxLen: 16, dFF: 256, basePos: 3, K: 2, scale: float32(1.0 / 8.0)},
		{dModel: 256, nHeads: 4, nKVHeads: 2, headDim: 64, maxLen: 32, dFF: 512, basePos: 5, K: 4, scale: float32(1.0 / 8.0)},
	}
	var totalBytes int64
	for i := range cases {
		c := &cases[i]
		qDim, kvDim := c.nHeads*c.headDim, c.nKVHeads*c.headDim
		c.attnNormW = toBF16Bytes(syntheticFloat32(c.dModel, 1))
		c.mlpNormW = toBF16Bytes(syntheticFloat32(c.dModel, 2))
		c.wQ = toBF16Bytes(syntheticFloat32(qDim*c.dModel, 3))
		c.wK = toBF16Bytes(syntheticFloat32(kvDim*c.dModel, 4))
		c.wV = toBF16Bytes(syntheticFloat32(kvDim*c.dModel, 5))
		c.wO = toBF16Bytes(syntheticFloat32(c.dModel*qDim, 6))
		c.wGate = toBF16Bytes(syntheticFloat32(c.dFF*c.dModel, 7))
		c.wUp = toBF16Bytes(syntheticFloat32(c.dFF*c.dModel, 8))
		c.wDown = toBF16Bytes(syntheticFloat32(c.dModel*c.dFF, 9))
		c.kCache = make([]byte, c.maxLen*kvDim*bf16Size)
		c.vCache = make([]byte, c.maxLen*kvDim*bf16Size)
		copy(c.kCache, toBF16Bytes(syntheticFloat32(c.basePos*kvDim, 10)))
		copy(c.vCache, toBF16Bytes(syntheticFloat32(c.basePos*kvDim, 11)))
		c.xs = toBF16Bytes(syntheticFloat32(c.K*c.dModel, 12))
		totalBytes += int64(len(c.xs) + len(c.kCache) + len(c.vCache))
		kc := append([]byte(nil), c.kCache...)
		vc := append([]byte(nil), c.vCache...)
		if _, err := DecodeLayerBatchedKV(c.xs, c.attnNormW, c.wQ, c.wK, c.wV, c.wO, kc, vc, c.mlpNormW, c.wGate, c.wUp, c.wDown, c.dModel, c.nHeads, c.nKVHeads, c.headDim, c.maxLen, c.dFF, c.basePos, c.K, base, c.scale, eps); err != nil {
			b.Fatalf("warmup dModel %d K %d: %v", c.dModel, c.K, err)
		}
	}
	b.SetBytes(totalBytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := cases[i&1]
		kc := append([]byte(nil), c.kCache...)
		vc := append([]byte(nil), c.vCache...)
		if _, err := DecodeLayerBatchedKV(c.xs, c.attnNormW, c.wQ, c.wK, c.wV, c.wO, kc, vc, c.mlpNormW, c.wGate, c.wUp, c.wDown, c.dModel, c.nHeads, c.nKVHeads, c.headDim, c.maxLen, c.dFF, c.basePos, c.K, base, c.scale, eps); err != nil {
			b.Fatal(err)
		}
	}
}
