// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMoEExpertsTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF = 4, 2, 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	gateW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 53))
	upW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 71))
	downW := toBF16Bytes(syntheticFloat32(numExperts*dModel*dFF, 47))
	b.SetBytes(int64(len(x) + len(gateW) + len(upW) + len(downW)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEExperts(x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEExpertsIntoTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF = 4, 2, 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	gateW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 53))
	upW := toBF16Bytes(syntheticFloat32(numExperts*dFF*dModel, 71))
	downW := toBF16Bytes(syntheticFloat32(numExperts*dModel*dFF, 47))
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(x) + len(gateW) + len(upW) + len(downW)))
	if _, err := MoEExpertsInto(out, x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEExpertsInto(out, x, idx, weights, gateW, upW, downW, numExperts, topK, dModel, dFF); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEExpertsQuantTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF, groupSize, bits = 4, 2, 64, 128, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	gate, up, down := quantMoEExpertsFixture(b, numExperts, dModel, dFF, groupSize, bits)
	b.SetBytes(int64(len(x) + len(gate.Packed) + len(up.Packed) + len(down.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEExpertsQuant(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, groupSize, bits); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEExpertsQuant(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEExpertsQuantIntoTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF, groupSize, bits = 4, 2, 64, 128, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	gate, up, down := quantMoEExpertsFixture(b, numExperts, dModel, dFF, groupSize, bits)
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(x) + len(gate.Packed) + len(up.Packed) + len(down.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEExpertsQuantInto(out, x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, groupSize, bits); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEExpertsQuantInto(out, x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEExpertsQuantFusedGateUpTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF, groupSize, bits = 4, 2, 64, 128, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	gate, up, down := quantMoEExpertsFixture(b, numExperts, dModel, dFF, groupSize, bits)
	gateUp := fusedGateUpQuantForBench(gate, up, numExperts, dFF, dModel, groupSize, bits)
	b.SetBytes(int64(len(x) + len(gateUp.Packed) + len(down.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEExpertsQuantFusedGateUp(x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, groupSize, bits); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEExpertsQuantFusedGateUp(x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEExpertsQuantFusedGateUpIntoTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, dModel, dFF, groupSize, bits = 4, 2, 64, 128, 32, 4
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	idx := []int32{3, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})
	gate, up, down := quantMoEExpertsFixture(b, numExperts, dModel, dFF, groupSize, bits)
	gateUp := fusedGateUpQuantForBench(gate, up, numExperts, dFF, dModel, groupSize, bits)
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(x) + len(gateUp.Packed) + len(down.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := MoEExpertsQuantFusedGateUpInto(out, x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, groupSize, bits); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MoEExpertsQuantFusedGateUpInto(out, x, idx, weights, gateUp, down, numExperts, topK, dModel, dFF, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGatherQMVBF16ByExpertIndexTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, outDim, inDim, groupSize, bits = 4, 2, 64, 96, 32, 4
	if _, err := gatherQMVBF16SteelPipeline(outDim, inDim, groupSize, bits); err != nil {
		b.Skipf("gather qmv kernel unavailable: %v", err)
	}
	idx := []int32{3, 1}
	w := quantMoELayerWeightsGuard(b, numExperts, 1, inDim, 128, outDim, groupSize, bits).ExpGate
	x := toBF16Bytes(syntheticFloat32(inDim, 37))
	b.SetBytes(int64(len(x) + len(w.Packed) + len(w.Scales) + len(w.Biases)))
	b.ReportAllocs()
	if _, err := gatherQMVBF16ByExpertIndex(x, idx, w, numExperts, topK, outDim, inDim, groupSize, bits); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := gatherQMVBF16ByExpertIndex(x, idx, w, numExperts, topK, outDim, inDim, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGatherQMVBF16ByExpertIndexIntoTop2Of4(b *testing.B) {
	requireNativeRuntime(b)

	const numExperts, topK, outDim, inDim, groupSize, bits = 4, 2, 64, 96, 32, 4
	if _, err := gatherQMVBF16SteelPipeline(outDim, inDim, groupSize, bits); err != nil {
		b.Skipf("gather qmv kernel unavailable: %v", err)
	}
	idx := []int32{3, 1}
	w := quantMoELayerWeightsGuard(b, numExperts, 1, inDim, 128, outDim, groupSize, bits).ExpGate
	x := toBF16Bytes(syntheticFloat32(inDim, 37))
	out := make([]byte, topK*outDim*bf16Size)
	b.SetBytes(int64(len(x) + len(w.Packed) + len(w.Scales) + len(w.Biases)))
	b.ReportAllocs()
	if _, err := gatherQMVBF16ByExpertIndexInto(out, x, idx, w, numExperts, topK, outDim, inDim, groupSize, bits); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := gatherQMVBF16ByExpertIndexInto(out, x, idx, w, numExperts, topK, outDim, inDim, groupSize, bits); err != nil {
			b.Fatal(err)
		}
	}
}
