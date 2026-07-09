// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkPerLayerInputsGPU(b *testing.B) {
	requireNativeRuntime(b)
	const vocabPLI, numLayers, pliDim, dModel = 32, 4, 64, 128
	const embGS, embBits = 32, 4
	const eps = float32(1e-6)
	fx := newPerLayerInputsGPUFixture(b, vocabPLI, numLayers, pliDim, dModel, embGS, embBits)
	tokens := []int32{0, 5, 17, 31}
	if _, err := PerLayerInputsGPU(tokens[0], fx.emb, fx.embedPacked, fx.embedScales, fx.embedBiases, fx.projW, fx.projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps); err != nil {
		b.Fatalf("PerLayerInputsGPU warmup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PerLayerInputsGPU(tokens[i%len(tokens)], fx.emb, fx.embedPacked, fx.embedScales, fx.embedBiases, fx.projW, fx.projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps); err != nil {
			b.Fatalf("PerLayerInputsGPU: %v", err)
		}
	}
}

func BenchmarkPerLayerInputsGPUInto(b *testing.B) {
	requireNativeRuntime(b)
	const vocabPLI, numLayers, pliDim, dModel = 32, 4, 64, 128
	const embGS, embBits = 32, 4
	const eps = float32(1e-6)
	fx := newPerLayerInputsGPUFixture(b, vocabPLI, numLayers, pliDim, dModel, embGS, embBits)
	tokens := []int32{0, 5, 17, 31}
	out := make([]byte, numLayers*pliDim*bf16Size)
	if _, err := perLayerInputsGPUInto(out, tokens[0], fx.emb, fx.embedPacked, fx.embedScales, fx.embedBiases, fx.projW, fx.projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps); err != nil {
		b.Fatalf("perLayerInputsGPUInto warmup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		out, err = perLayerInputsGPUInto(out, tokens[i%len(tokens)], fx.emb, fx.embedPacked, fx.embedScales, fx.embedBiases, fx.projW, fx.projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps)
		if err != nil {
			b.Fatalf("perLayerInputsGPUInto: %v", err)
		}
	}
}

func BenchmarkPerLayerInputsGPUIntoAlternatingShapes(b *testing.B) {
	requireNativeRuntime(b)

	type fixture struct {
		vocabPLI, numLayers, pliDim, dModel, embGS, embBits int
		fx                                                  perLayerInputsGPUFixture
		out                                                 []byte
	}
	fixtures := []fixture{
		{
			vocabPLI: 32, numLayers: 2, pliDim: 32, dModel: 64, embGS: 32, embBits: 4,
		},
		{
			vocabPLI: 32, numLayers: 4, pliDim: 64, dModel: 128, embGS: 32, embBits: 4,
		},
	}
	for i := range fixtures {
		f := &fixtures[i]
		f.fx = newPerLayerInputsGPUFixture(b, f.vocabPLI, f.numLayers, f.pliDim, f.dModel, f.embGS, f.embBits)
		f.out = make([]byte, f.numLayers*f.pliDim*bf16Size)
		if _, err := perLayerInputsGPUInto(f.out, int32(i+1), f.fx.emb, f.fx.embedPacked, f.fx.embedScales, f.fx.embedBiases, f.fx.projW, f.fx.projNormW, f.vocabPLI, f.numLayers, f.pliDim, f.dModel, f.embGS, f.embBits, 1e-6); err != nil {
			b.Fatalf("perLayerInputsGPUInto warmup: %v", err)
		}
	}

	tokens := []int32{1, 5, 17, 31}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := &fixtures[i&1]
		var err error
		f.out, err = perLayerInputsGPUInto(f.out, tokens[i%len(tokens)], f.fx.emb, f.fx.embedPacked, f.fx.embedScales, f.fx.embedBiases, f.fx.projW, f.fx.projNormW, f.vocabPLI, f.numLayers, f.pliDim, f.dModel, f.embGS, f.embBits, 1e-6)
		if err != nil {
			b.Fatalf("perLayerInputsGPUInto: %v", err)
		}
	}
}

func BenchmarkSessionNextInputsGPU(b *testing.B) {
	requireNativeRuntime(b)
	g, arch, maxLen := icbSessionStateFixture(b)
	sess := newICBSessionStateFixture(b, g, arch, maxLen)
	if sess.encNextInputsGPU == nil {
		b.Fatal("fixture did not wire GPU next-inputs seam")
	}
	tokens := []int32{1, 5, 17, 31}
	if _, _, ok, err := sess.nextInputsGPU(tokens[0]); err != nil || !ok {
		b.Fatalf("nextInputsGPU warmup ok=%v err=%v", ok, err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, ok, err := sess.nextInputsGPU(tokens[i%len(tokens)]); err != nil || !ok {
			b.Fatalf("nextInputsGPU ok=%v err=%v", ok, err)
		}
	}
}
