// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

func TestAudioPositionTable(t *testing.T) {
	got := AudioPositionTable(2, 4)
	if len(got) != 8 {
		t.Fatalf("position table len = %d, want 8", len(got))
	}
	if got[4] != 0 || got[5] != 0 || got[6] != 1 || got[7] != 1 {
		t.Fatalf("zero-position row = %v, want [0 0 1 1]", got[4:])
	}
	if maxInt(2, 1) != 2 || maxInt(1, 2) != 2 {
		t.Fatal("maxInt did not return the larger value")
	}
}

func TestReLUF32(t *testing.T) {
	got := reluF32([]float32{-3, 0, 2.5})
	want := []float32{0, 0, 2.5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("reluF32[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestAudioF32HelperInputGuards(t *testing.T) {
	in := []float32{-2, 0.5, 3}
	noOpClamp := clampF32(in, 0, 0)
	if &noOpClamp[0] != &in[0] {
		t.Fatal("clampF32 no-op should return the original slice")
	}
	if allocs := testing.AllocsPerRun(100, func() { _ = clampF32(in, 0, 0) }); allocs != 0 {
		t.Fatalf("clampF32 no-op allocations = %v, want 0", allocs)
	}

	bin := toBF16Bytes([]float32{-2, 0.5, 3})
	noOpBF16 := clampBF16(bin, 0, 0)
	if &noOpBF16[0] != &bin[0] {
		t.Fatal("clampBF16 no-op should return the original slice")
	}
	if allocs := testing.AllocsPerRun(100, func() { _ = clampBF16(bin, 0, 0) }); allocs != 0 {
		t.Fatalf("clampBF16 no-op allocations = %v, want 0", allocs)
	}

	if got := (ClipBound{}).applyF32(in); &got[0] != &in[0] {
		t.Fatal("ClipBound.applyF32 without Present should return the original slice")
	}
	clipped := (ClipBound{Present: true, Min: -1, Max: 1}).applyF32(in)
	wantClip := []float32{-1, 0.5, 1}
	for i := range wantClip {
		if clipped[i] != wantClip[i] {
			t.Fatalf("applyF32 clipped[%d] = %v, want %v", i, clipped[i], wantClip[i])
		}
	}

	if _, err := matF32MixedNT([]float32{1}, toBF16Bytes([]float32{1, 2}), 1, 1, 2); err == nil {
		t.Fatal("matF32MixedNT(short input) error = nil")
	}
	if _, err := matF32MixedNT([]float32{1, 2}, toBF16Bytes([]float32{1}), 1, 1, 2); err == nil {
		t.Fatal("matF32MixedNT(short weight) error = nil")
	}
	if _, err := clippedMatF32([]float32{1}, toBF16Bytes([]float32{1, 2}), 1, 1, 2, ClipPair{}); err == nil {
		t.Fatal("clippedMatF32(mat error) error = nil")
	}

	reluF, err := audioActivateF32([]float32{-1, 0, 2}, "relu")
	if err != nil {
		t.Fatalf("audioActivateF32(relu): %v", err)
	}
	if reluF[0] != 0 || reluF[1] != 0 || reluF[2] != 2 {
		t.Fatalf("audioActivateF32(relu) = %v, want [0 0 2]", reluF)
	}

	reluB, err := audioActivateBF16(toBF16Bytes([]float32{-1, 0, 2}), "relu")
	if err != nil {
		t.Fatalf("audioActivateBF16(relu): %v", err)
	}
	reluBF := bf16Floats(reluB)
	if reluBF[0] != 0 || reluBF[1] != 0 || reluBF[2] != 2 {
		t.Fatalf("audioActivateBF16(relu) = %v, want [0 0 2]", reluBF)
	}

	requireNativeRuntime(t)
	geluF, err := audioActivateF32([]float32{-0.5, 0.5}, "gelu")
	if err != nil {
		t.Fatalf("audioActivateF32(gelu): %v", err)
	}
	for i, x := range []float32{-0.5, 0.5} {
		if diff := math.Abs(float64(geluF[i] - geluTanhScalar(x))); diff > 1e-5 {
			t.Fatalf("audioActivateF32(gelu)[%d] diff = %.3g", i, diff)
		}
	}
}

func TestAudioEncodeAndSubsampleF32InputGuards(t *testing.T) {
	requireNativeRuntime(t)
	if _, _, err := AudioSubsampleF32([]byte{1}, &AudioSubsampleWeights{}, AudioSubsampleConfig{Frames: 1, MelBins: 1}, nil); err == nil {
		t.Fatal("AudioSubsampleF32(short features) error = nil")
	}
	if _, err := AudioEncode([]byte{1}, &AudioEncoderWeights{}, AudioConfig{}, nil); err == nil {
		t.Fatal("AudioEncode(short features) error = nil")
	}
}

func TestAudioBlockInputGuards(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := AudioFeedForward(toBF16Bytes([]float32{1, 2}), &AudioFeedForwardWeights{}, AudioConfig{}); err == nil {
		t.Fatal("AudioFeedForward(zero geometry) error = nil")
	}
	if _, err := AudioFeedForwardF32([]float32{1, 2}, &AudioFeedForwardWeights{}, AudioConfig{}); err == nil {
		t.Fatal("AudioFeedForwardF32(zero geometry) error = nil")
	}
	if _, err := AudioLightConv(toBF16Bytes([]float32{1, 2}), &AudioLightConvWeights{}, AudioConfig{}); err == nil {
		t.Fatal("AudioLightConv(zero geometry) error = nil")
	}
	if _, err := AudioLightConvF32([]float32{1, 2}, &AudioLightConvWeights{}, AudioConfig{}); err == nil {
		t.Fatal("AudioLightConvF32(zero geometry) error = nil")
	}
}

func TestAudioFeedForwardActivationModes(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, inter, rows = 2, 3, 2
	weights := audioGuardFeedForwardWeights(hidden, inter)
	xBF16 := toBF16Bytes(syntheticFloat32(rows*hidden, 77))
	xF32 := syntheticFloat32(rows*hidden, 79)
	for _, act := range []string{"relu", "gelu", "gelu_pytorch_tanh"} {
		cfg := AudioConfig{
			Hidden: hidden, FFInter: inter, Eps: 1e-5, Act: act,
			FFResidual: 0.5, ClipMin: -6, ClipMax: 6,
		}
		gotF32, err := AudioFeedForwardF32(xF32, weights, cfg)
		if err != nil {
			t.Fatalf("AudioFeedForwardF32 act=%s: %v", act, err)
		}
		if len(gotF32) != len(xF32) {
			t.Fatalf("AudioFeedForwardF32 act=%s len = %d, want %d", act, len(gotF32), len(xF32))
		}

		gotBF16, err := AudioFeedForward(xBF16, weights, cfg)
		if err != nil {
			t.Fatalf("AudioFeedForward act=%s: %v", act, err)
		}
		if len(gotBF16) != len(xBF16) {
			t.Fatalf("AudioFeedForward act=%s len = %d, want %d", act, len(gotBF16), len(xBF16))
		}
	}
}

func TestAudioBlockKernelFailureGuards(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, inter, channels, kernel = 2, 3, 2, 1
	ffWeights := audioGuardFeedForwardWeights(hidden, inter)
	lcWeights := audioGuardLightConvWeights(hidden, channels, kernel)
	cfg := AudioConfig{
		Hidden: hidden, FFInter: inter, Channels: channels, KernelSize: kernel,
		Eps: 1e-5, Act: "silu", FFResidual: 0.5, ClipMin: -6, ClipMax: 6,
	}
	xBF16 := toBF16Bytes(syntheticFloat32(2*hidden, 31))
	xF32 := syntheticFloat32(2*hidden, 33)
	subWeights, subCfg, features := audioGuardSubsampleWeights(2, 2, 1, 1, hidden)

	withWrongMainLibrary(t, func() {
		if _, err := AudioFeedForward(xBF16, ffWeights, cfg); err == nil {
			t.Fatal("AudioFeedForward(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := AudioFeedForwardF32(xF32, ffWeights, cfg); err == nil {
			t.Fatal("AudioFeedForwardF32(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := AudioLightConv(xBF16, lcWeights, cfg); err == nil {
			t.Fatal("AudioLightConv(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := AudioLightConvF32(xF32, lcWeights, cfg); err == nil {
			t.Fatal("AudioLightConvF32(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := AudioSubsample(features, subWeights, subCfg); err == nil {
			t.Fatal("AudioSubsample(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, _, err := AudioSubsampleF32(features, subWeights, subCfg, nil); err == nil {
			t.Fatal("AudioSubsampleF32(wrong library) error = nil")
		}
	})
}

func audioGuardFeedForwardWeights(hidden, inter int) *AudioFeedForwardWeights {
	return &AudioFeedForwardWeights{
		PreNorm:  toBF16Bytes(syntheticFloat32(hidden, 41)),
		PostNorm: toBF16Bytes(syntheticFloat32(hidden, 43)),
		FFW1:     toBF16Bytes(syntheticFloat32(inter*hidden, 45)),
		FFW2:     toBF16Bytes(syntheticFloat32(hidden*inter, 47)),
	}
}

func audioGuardLightConvWeights(hidden, channels, kernel int) *AudioLightConvWeights {
	return &AudioLightConvWeights{
		PreNorm:         toBF16Bytes(syntheticFloat32(hidden, 51)),
		ConvNorm:        toBF16Bytes(syntheticFloat32(channels, 53)),
		LinearStart:     toBF16Bytes(syntheticFloat32(2*channels*hidden, 55)),
		LinearEnd:       toBF16Bytes(syntheticFloat32(hidden*channels, 57)),
		DepthwiseWeight: toBF16Bytes(syntheticFloat32(channels*kernel, 59)),
	}
}

func audioGuardSubsampleWeights(frames, melBins, outC0, outC1, hidden int) (*AudioSubsampleWeights, AudioSubsampleConfig, []byte) {
	t0, f0 := convOut(frames), convOut(melBins)
	_, f1 := convOut(t0), convOut(f0)
	weights := &AudioSubsampleWeights{
		Conv0:     toBF16Bytes(syntheticFloat32(outC0*9, 61)),
		Norm0W:    toBF16Bytes(syntheticFloat32(outC0, 63)),
		Norm0B:    toBF16Bytes(syntheticFloat32(outC0, 65)),
		Conv1:     toBF16Bytes(syntheticFloat32(outC1*9*outC0, 67)),
		Norm1W:    toBF16Bytes(syntheticFloat32(outC1, 69)),
		Norm1B:    toBF16Bytes(syntheticFloat32(outC1, 71)),
		InputProj: toBF16Bytes(syntheticFloat32(hidden*f1*outC1, 73)),
	}
	cfg := AudioSubsampleConfig{Frames: frames, MelBins: melBins, OutC0: outC0, OutC1: outC1, Hidden: hidden, Eps: 1e-5}
	return weights, cfg, toBF16Bytes(syntheticFloat32(frames*melBins, 75))
}
