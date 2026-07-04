// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMoEBlockBF16Top2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, 3)
	b.SetBytes(int64(len(h) + len(w.WGate) + len(w.ExpGateW)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEBlockBF16(h, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockBF16(h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockBF16IntoTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, 3)
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(h) + len(w.WGate) + len(w.ExpGateW)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEBlockBF16Into(out, h, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockBF16Into(out, h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockBF16PinnedInputTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, 3)
	pinned, err := newPinnedNoCopyBytes(len(h))
	if err != nil {
		b.Fatal(err)
	}
	defer pinned.Close()
	hBuf, err := pinned.copyBuffer(h)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(h) + len(w.WGate) + len(w.ExpGateW)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := moeBlockBF16WithBuffer(h, hBuf, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := moeBlockBF16WithBuffer(h, hBuf, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockBF16BufferOutputTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, 3)
	input, err := newPinnedNoCopyBytes(len(h))
	if err != nil {
		b.Fatal(err)
	}
	defer input.Close()
	hBuf, err := input.copyBuffer(h)
	if err != nil {
		b.Fatal(err)
	}
	out, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		b.Fatal(err)
	}
	defer out.Close()
	b.SetBytes(int64(len(h) + len(w.WGate) + len(w.ExpGateW)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if err := moeBlockBF16WithBufferOutputInPool(h, hBuf, out.buf, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := moeBlockBF16WithBufferOutputInPool(h, hBuf, out.buf, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLPTransformBF1664x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 17))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 19))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 23))
	b.SetBytes(int64(len(x) + len(wGate) + len(wUp) + len(wDown)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := mlpTransformBF16(x, wGate, wUp, wDown, dModel, dFF); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := mlpTransformBF16(x, wGate, wUp, wDown, dModel, dFF); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLPTransformQuant64x128(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, groupSize, bits = 64, 128, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gate := quantWeightFixture(b, dFF, dModel, groupSize, bits, 3)
	up := quantWeightFixture(b, dFF, dModel, groupSize, bits, 31)
	down := quantWeightFixture(b, dModel, dFF, groupSize, bits, 37)
	b.SetBytes(int64(len(x) + len(gate.Packed) + len(up.Packed) + len(down.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := mlpTransformQuant(x, gate, up, down, dModel, dFF, groupSize, bits); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := mlpTransformQuant(x, gate, up, down, dModel, dFF, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkMLPTransformQuantVariant(b *testing.B, dModel, dFF, groupSize, bits int, fn func([]byte, QuantWeight, QuantWeight, QuantWeight, int, int, int, int) ([]byte, error)) {
	b.Helper()
	requireNativeRuntime(b)

	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gate := quantWeightFixture(b, dFF, dModel, groupSize, bits, 3)
	up := quantWeightFixture(b, dFF, dModel, groupSize, bits, 31)
	down := quantWeightFixture(b, dModel, dFF, groupSize, bits, 37)
	b.SetBytes(int64(len(x) + len(gate.Packed) + len(up.Packed) + len(down.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := fn(x, gate, up, down, dModel, dFF, groupSize, bits); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := fn(x, gate, up, down, dModel, dFF, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLPTransformQuantComposed256x512(b *testing.B) {
	benchmarkMLPTransformQuantVariant(b, 256, 512, 64, 4, mlpTransformQuantComposed)
}

func BenchmarkMLPTransformQuantDefault256x512(b *testing.B) {
	benchmarkMLPTransformQuantVariant(b, 256, 512, 64, 4, mlpTransformQuant)
}

func BenchmarkMLPTransformQuantMega256x512(b *testing.B) {
	requireNativeRuntime(b)
	if _, err := ffnMegaPipeline(); err != nil {
		b.Skipf("ffn megakernel unavailable: %v", err)
	}
	benchmarkMLPTransformQuantVariant(b, 256, 512, 64, 4, mlpTransformQuantMega)
}

func BenchmarkMLPTransformQuantMegaInto256x512(b *testing.B) {
	requireNativeRuntime(b)
	if _, err := ffnMegaPipeline(); err != nil {
		b.Skipf("ffn megakernel unavailable: %v", err)
	}

	const dModel, dFF, groupSize, bits = 256, 512, 64, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	gate := quantWeightFixture(b, dFF, dModel, groupSize, bits, 3)
	up := quantWeightFixture(b, dFF, dModel, groupSize, bits, 31)
	down := quantWeightFixture(b, dModel, dFF, groupSize, bits, 37)
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(x) + len(gate.Packed) + len(up.Packed) + len(down.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := mlpTransformQuantMegaInto(out, x, gate, up, down, dModel, dFF, groupSize, bits); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := mlpTransformQuantMegaInto(out, x, gate, up, down, dModel, dFF, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockQuantTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	b.SetBytes(int64(len(h) + len(w.LocalGate.Packed) + len(w.ExpGate.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockQuantIntoTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(h) + len(w.LocalGate.Packed) + len(w.ExpGate.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEBlockQuantInto(out, h, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockQuantInto(out, h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockQuantPinnedInputTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	pinned, err := newPinnedNoCopyBytes(len(h))
	if err != nil {
		b.Fatal(err)
	}
	defer pinned.Close()
	hBuf, err := pinned.copyBuffer(h)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(h) + len(w.LocalGate.Packed) + len(w.ExpGate.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := moeBlockQuantWithBuffer(h, hBuf, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := moeBlockQuantWithBuffer(h, hBuf, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockQuantBufferOutputTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	input, err := newPinnedNoCopyBytes(len(h))
	if err != nil {
		b.Fatal(err)
	}
	defer input.Close()
	hBuf, err := input.copyBuffer(h)
	if err != nil {
		b.Fatal(err)
	}
	out, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		b.Fatal(err)
	}
	defer out.Close()
	b.SetBytes(int64(len(h) + len(w.LocalGate.Packed) + len(w.ExpGate.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if err := moeBlockQuantWithBufferOutputInPool(h, hBuf, out.buf, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := moeBlockQuantWithBufferOutputInPool(h, hBuf, out.buf, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockQuantFusedGateUpTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	w.ExpGateUp = fusedGateUpQuantForBench(w.ExpGate, w.ExpUp, numExperts, expertDFF, dModel, groupSize, bits)
	w.ExpGate, w.ExpUp = QuantWeight{}, QuantWeight{}
	b.SetBytes(int64(len(h) + len(w.LocalGate.Packed) + len(w.ExpGateUp.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockQuantViewBackedTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 64, 128, 96, 4, 2, 32, 4
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := withQuantViewsForBench(quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits))
	b.SetBytes(int64(len(h) + len(w.LocalGate.Packed) + len(w.ExpGate.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEBlockQuant(h, w, dModel, dFF, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEBlockQuantAfterRouterLargeLocalTop1Of2(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, dFF, expertDFF, numExperts, topK, groupSize, bits = 256, 512, 128, 2, 1, 64, 4
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	idx := []int32{0}
	weights := toBF16Bytes([]float32{1})
	w := quantMoELayerWeightsGuard(b, numExperts, topK, dModel, dFF, expertDFF, groupSize, bits)
	b.SetBytes(int64(len(h) + len(w.LocalGate.Packed) + len(w.LocalUp.Packed) + len(w.LocalDown.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := moeBlockQuantAfterRouter(h, idx, weights, nil, w, dModel, dFF, eps); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := moeBlockQuantAfterRouter(h, idx, weights, nil, w, dModel, dFF, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func fusedGateUpQuantForBench(gate, up QuantWeight, numExperts, expertDFF, dModel, groupSize, bits int) QuantWeight {
	gatePacked := expertDFF * dModel * bits / 8
	gateScale := expertDFF * (dModel / groupSize) * bf16Size
	fuse := func(a, b []byte, perExpert int) []byte {
		out := make([]byte, 0, len(a)+len(b))
		for e := 0; e < numExperts; e++ {
			start := e * perExpert
			end := start + perExpert
			out = append(out, a[start:end]...)
			out = append(out, b[start:end]...)
		}
		return out
	}
	return QuantWeight{
		Packed:    fuse(gate.Packed, up.Packed, gatePacked),
		Scales:    fuse(gate.Scales, up.Scales, gateScale),
		Biases:    fuse(gate.Biases, up.Biases, gateScale),
		GroupSize: groupSize,
		Bits:      bits,
	}
}

func withQuantViewsForBench(w MoEQuantLayerWeights) MoEQuantLayerWeights {
	view := func(q QuantWeight) QuantWeight {
		if len(q.Packed) == 0 {
			return q
		}
		q.packedView = bufView{buf: sharedBytes(q.Packed)}
		q.scalesView = bufView{buf: sharedBytes(q.Scales)}
		q.biasesView = bufView{buf: sharedBytes(q.Biases)}
		return q
	}
	w.LocalGate = view(w.LocalGate)
	w.LocalUp = view(w.LocalUp)
	w.LocalDown = view(w.LocalDown)
	w.Router = view(w.Router)
	w.ExpGate = view(w.ExpGate)
	w.ExpUp = view(w.ExpUp)
	w.ExpGateUp = view(w.ExpGateUp)
	w.ExpDown = view(w.ExpDown)
	w.preFFNormView = bufView{buf: sharedBytes(w.PreFFNormW)}
	w.preFFNorm2View = bufView{buf: sharedBytes(w.PreFFNorm2W)}
	w.postFFNorm1View = bufView{buf: sharedBytes(w.PostFFNorm1W)}
	w.postFFNorm2View = bufView{buf: sharedBytes(w.PostFFNorm2W)}
	w.postFFNormView = bufView{buf: sharedBytes(w.PostFFNormW)}
	w.routerNormView = bufView{buf: sharedBytes(w.RouterNormWScaled)}
	w.perExpertScaleView = bufView{buf: sharedBytes(w.PerExpertScale)}
	return w
}
