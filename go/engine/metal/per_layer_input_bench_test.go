// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

func BenchmarkPerLayerInputsBF16(b *testing.B) {
	requireNativeRuntime(b)

	const vocabPLI, numLayers, pliDim, dModel = 8, 2, 32, 64
	const plDim = numLayers * pliDim
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 5))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))
	b.SetBytes(int64(len(embed) + len(projW) + len(hidden)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, bufView{}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerInputsBF16Scratch(b *testing.B) {
	requireNativeRuntime(b)

	const vocabPLI, numLayers, pliDim, dModel = 8, 2, 32, 64
	const plDim = numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 5))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))
	projView := copyView(projW)
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		b.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	b.SetBytes(int64(len(embed) + len(projW) + len(hidden)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PerLayerInputs(embed, nil, nil, projW, nil, nil, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, 1e-5, projView, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerInputsQuant(b *testing.B) {
	requireNativeRuntime(b)

	const vocabPLI, numLayers, pliDim, dModel, groupSize, bits = 8, 2, 32, 64, 32, 4
	const plDim = numLayers * pliDim
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	proj := quantWeightFixture(b, plDim, dModel, groupSize, bits, 5)
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))
	b.SetBytes(int64(len(embed) + len(proj.Packed) + len(hidden)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PerLayerInputs(embed, nil, nil, proj.Packed, proj.Scales, proj.Biases, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, groupSize, bits, 1e-5, bufView{}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerInputsQuantScratch(b *testing.B) {
	requireNativeRuntime(b)

	const vocabPLI, numLayers, pliDim, dModel, groupSize, bits = 8, 2, 32, 64, 32, 4
	const plDim = numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	embed := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	proj := quantWeightFixture(b, plDim, dModel, groupSize, bits, 5)
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		b.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	b.SetBytes(int64(len(embed) + len(proj.Packed) + len(hidden)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PerLayerInputs(embed, nil, nil, proj.Packed, proj.Scales, proj.Biases, projNormW, 2, hidden, vocabPLI, numLayers, pliDim, dModel, 0, 0, groupSize, bits, 1e-5, bufView{}, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerInputGateBF16(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, pliDim = 64, 32
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 17))
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	projW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 23))
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	b.SetBytes(int64(len(hNext) + len(gateW) + len(projW)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerInputGateBF16Into(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, pliDim = 64, 32
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 17))
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	projW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 23))
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(hNext) + len(gateW) + len(projW)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := perLayerInputGateBF16Into(out, hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = perLayerInputGateBF16Into(out, hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, 1e-5)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerInputGateBF16IntoAlternatingShapes(b *testing.B) {
	requireNativeRuntime(b)

	type fixture struct {
		dModel, pliDim                     int
		hNext, gateW, perLayerInput, projW []byte
		postNormW, out                     []byte
	}
	fixtures := []fixture{
		{
			dModel: 64, pliDim: 32,
			hNext: toBF16Bytes(syntheticFloat32(64, 29)), gateW: toBF16Bytes(syntheticFloat32(32*64, 17)),
			perLayerInput: toBF16Bytes(syntheticFloat32(32, 7)), projW: toBF16Bytes(syntheticFloat32(64*32, 23)),
			postNormW: toBF16Bytes(syntheticFloat32(64, 5)), out: make([]byte, 64*bf16Size),
		},
		{
			dModel: 128, pliDim: 64,
			hNext: toBF16Bytes(syntheticFloat32(128, 31)), gateW: toBF16Bytes(syntheticFloat32(64*128, 19)),
			perLayerInput: toBF16Bytes(syntheticFloat32(64, 11)), projW: toBF16Bytes(syntheticFloat32(128*64, 27)),
			postNormW: toBF16Bytes(syntheticFloat32(128, 13)), out: make([]byte, 128*bf16Size),
		},
	}
	for i := range fixtures {
		f := &fixtures[i]
		if _, err := perLayerInputGateBF16Into(f.out, f.hNext, f.gateW, f.perLayerInput, f.projW, f.postNormW, f.dModel, f.pliDim, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := &fixtures[i&1]
		var err error
		f.out, err = perLayerInputGateBF16Into(f.out, f.hNext, f.gateW, f.perLayerInput, f.projW, f.postNormW, f.dModel, f.pliDim, 1e-5)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerInputGateQuant(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, pliDim, groupSize, bits = 64, 32, 32, 4
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gate := quantWeightFixture(b, pliDim, dModel, groupSize, bits, 17)
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	proj := quantWeightFixture(b, dModel, pliDim, groupSize, bits, 23)
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	b.SetBytes(int64(len(hNext) + len(gate.Packed) + len(proj.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := PerLayerInputGateQuant(hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PerLayerInputGateQuant(hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, 1e-5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPerLayerInputGateQuantInto(b *testing.B) {
	requireNativeRuntime(b)

	const dModel, pliDim, groupSize, bits = 64, 32, 32, 4
	hNext := toBF16Bytes(syntheticFloat32(dModel, 29))
	gate := quantWeightFixture(b, pliDim, dModel, groupSize, bits, 17)
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 7))
	proj := quantWeightFixture(b, dModel, pliDim, groupSize, bits, 23)
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	out := make([]byte, dModel*bf16Size)
	b.SetBytes(int64(len(hNext) + len(gate.Packed) + len(proj.Packed)))
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()
	if _, err := perLayerInputGateQuantInto(out, hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, 1e-5); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = perLayerInputGateQuantInto(out, hNext, gate, perLayerInput, proj, postNormW, dModel, pliDim, groupSize, bits, 1e-5)
		if err != nil {
			b.Fatal(err)
		}
	}
}
