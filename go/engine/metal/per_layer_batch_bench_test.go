// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

func BenchmarkPerLayerProjBatched(b *testing.B) {
	requireNativeRuntime(b)
	const numLayers, pliDim, dModel = 2, 64, 128
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 11))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 12))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 13))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 14))
	projView := copyView(projW)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerProjBatchedScratch(b *testing.B) {
	requireNativeRuntime(b)
	const numLayers, pliDim, dModel = 2, 64, 128
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 11))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 12))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 13))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 14))
	projView := copyView(projW)
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		b.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerProjBatchedResidentScratch(b *testing.B) {
	requireNativeRuntime(b)
	const numLayers, pliDim, dModel = 2, 64, 128
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 11))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 12))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 13))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 14))
	projView := copyView(projW)
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		b.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := perLayerProjBatchedResident(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerProjQuantBatchedScratch(b *testing.B) {
	requireNativeRuntime(b)
	const numLayers, pliDim, dModel = 2, 64, 128
	const groupSize, bits = 32, 4
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 11))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 12))
	proj := quantWeightFixture(b, plDim, dModel, groupSize, bits, 13)
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 14))
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		b.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := perLayerProjQuantBatched(proj, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, groupSize, bits, eps, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerProjUnbatchedReference(b *testing.B) {
	requireNativeRuntime(b)
	const numLayers, pliDim, dModel = 2, 64, 128
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 11))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 12))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 13))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 14))

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = perLayerProjUnbatchedRef(b, projW, hidden, perLayer, projScale, projNormW, numLayers, pliDim, dModel, eps)
	}
}

// BenchmarkPerLayerInputsBatchQuantIntoSlab measures the quant PLE slab builder at #381's
// "plain" e2b/e4b profile (quantised table, bf16-resident tower projection) at a
// steelGEMMMinRows-width chunk — the floor prefill width the batched lane engages at.
func BenchmarkPerLayerInputsBatchQuantIntoSlab(b *testing.B) {
	requireNativeRuntime(b)
	const numLayers, pliDim, dModel, vocabPLI = 2, 32, 256, 64
	const tableGS, tableBits = 64, 4
	const eps = float32(1e-5)
	const k = steelGEMMMinRows
	plDim := numLayers * pliDim
	tPacked, tScales, tBiases := packAffineQuant(syntheticFloat32(vocabPLI*plDim, 9), vocabPLI, plDim, tableGS, tableBits)
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))
	projView := copyView(projW)
	ids := make([]int32, k)
	embs := make([][]byte, k)
	for i := range ids {
		ids[i] = int32((i*11 + 5) % vocabPLI)
		embs[i] = toBF16Bytes(syntheticFloat32(dModel, i+1))
	}
	tablePacked, tableScales, tableBiases := residentBytes(tPacked), residentBytes(tScales), residentBytes(tBiases)
	sc := &pleBatchScratch{}
	slab := make([]byte, k*plDim*bf16Size)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := perLayerInputsBatchQuantIntoSlab(sc, tablePacked, tableScales, tableBiases, tableGS, tableBits,
			projView.buf, projView.off, nil, nil, nil, 0, 0,
			projNormW, ids, embs, slab, numLayers, pliDim, dModel, eps); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPerLayerInputsBatchQuantDevice measures the device-resident quant PLE builder
// (#381's device-first prefill seam): gather K main-embed rows AND the layer-major PLE
// tensor in one committed-not-waited command buffer, no host round-trip for either.
func BenchmarkPerLayerInputsBatchQuantDevice(b *testing.B) {
	requireNativeRuntime(b)
	const numLayers, pliDim, dModel, vocabPLI = 2, 32, 256, 64
	const tableGS, tableBits = 64, 4
	const embVocab, embGS, embBits = 128, 64, 4
	const eps = float32(1e-5)
	const k = steelGEMMMinRows
	plDim := numLayers * pliDim
	tPacked, tScales, tBiases := packAffineQuant(syntheticFloat32(vocabPLI*plDim, 9), vocabPLI, plDim, tableGS, tableBits)
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))
	projView := copyView(projW)
	ePacked, eScales, eBiases := embedGatherQuantFixture(embVocab, dModel, embGS, embBits)
	mainEmb := &mainEmbedGather{
		packed: residentBytes(ePacked), scales: residentBytes(eScales), biases: residentBytes(eBiases),
		gs: embGS, bits: embBits, scale: float32(math.Sqrt(float64(dModel))),
	}
	ids := make([]int32, k)
	for i := range ids {
		ids[i] = int32((i*11 + 5) % min(vocabPLI, embVocab))
	}
	tablePacked, tableScales, tableBiases := residentBytes(tPacked), residentBytes(tScales), residentBytes(tBiases)
	sc := &pleBatchScratch{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, ok, err := perLayerInputsBatchQuantDevice(sc, tablePacked, tableScales, tableBiases, tableGS, tableBits,
			projView.buf, projView.off, nil, nil, nil, 0, 0,
			projNormW, ids, mainEmb, numLayers, numLayers, pliDim, dModel, eps); err != nil || !ok {
			b.Fatalf("ok=%v err=%v", ok, err)
		}
	}
}
