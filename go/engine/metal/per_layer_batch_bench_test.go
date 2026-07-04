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
