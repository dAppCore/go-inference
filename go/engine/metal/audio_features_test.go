// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"

	core "dappco.re/go"
)

func TestAudioFeatureConfigLoadAndNormalize(t *testing.T) {
	dir := t.TempDir()
	data := []byte(`{
		"audio_ms_per_token": 160,
		"audio_seq_length": 188,
		"feature_extractor": {
			"num_mel_filters": 24,
			"sampling_rate": 8000,
			"frame_length_ms": 2,
			"hop_length_ms": 1,
			"max_frequency": 4000
		}
	}`)
	if result := core.WriteFile(core.PathJoin(dir, "processor_config.json"), data, 0o644); !result.OK {
		t.Fatalf("write processor config: %v", result.Value)
	}
	cfg, err := LoadAudioFeatureConfig(dir)
	if err != nil {
		t.Fatalf("LoadAudioFeatureConfig: %v", err)
	}
	if cfg == nil {
		t.Fatal("LoadAudioFeatureConfig returned nil config")
	}
	if cfg.FeatureSize != 0 || cfg.NumMelFilters != 24 {
		t.Fatalf("raw config feature fields = (%d, %d), want (0, 24)", cfg.FeatureSize, cfg.NumMelFilters)
	}
	normalizeAudioFeatureConfig(cfg)
	if cfg.FeatureSize != 24 {
		t.Fatalf("normalised FeatureSize = %d, want 24", cfg.FeatureSize)
	}
	if cfg.FrameLength != 16 || cfg.HopLength != 8 {
		t.Fatalf("normalised frame/hop = (%d, %d), want (16, 8)", cfg.FrameLength, cfg.HopLength)
	}

	missing, err := LoadAudioFeatureConfig(t.TempDir())
	if err != nil {
		t.Fatalf("LoadAudioFeatureConfig(missing): %v", err)
	}
	if missing != nil {
		t.Fatalf("missing processor config = %+v, want nil", missing)
	}
}

func TestAudioFeatureExtractorExtractMasksPaddedFrames(t *testing.T) {
	extractor, err := NewAudioFeatureExtractor(&AudioFeatureConfig{
		NumMelFilters:    4,
		SamplingRate:     16_000,
		FrameLength:      4,
		HopLength:        2,
		FFTOverdrive:     true,
		MaxFrequency:     8000,
		MelFloor:         1e-3,
		InputScaleFactor: 2,
		PadToMultiple:    8,
	})
	if err != nil {
		t.Fatalf("NewAudioFeatureExtractor: %v", err)
	}
	if extractor.SamplingRate() != 16_000 {
		t.Fatalf("SamplingRate = %d, want 16000", extractor.SamplingRate())
	}
	if got := (*AudioFeatureExtractor)(nil).SamplingRate(); got != 0 {
		t.Fatalf("nil SamplingRate = %d, want 0", got)
	}
	features, mask, frames, err := extractor.Extract([]float32{0.1, -0.2, 0.3, -0.4})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if frames != 3 {
		t.Fatalf("frames = %d, want 3", frames)
	}
	if len(mask) != 3 || !mask[0] || mask[1] || mask[2] {
		t.Fatalf("mask = %v, want [true false false]", mask)
	}
	if len(features) != frames*4 {
		t.Fatalf("features len = %d, want %d", len(features), frames*4)
	}
	nonZero := false
	for _, v := range features[:4] {
		nonZero = nonZero || v != 0
	}
	if !nonZero {
		t.Fatal("first real frame was fully zero")
	}
	for i, v := range features[4:] {
		if v != 0 {
			t.Fatalf("padded feature %d = %v, want zero", i, v)
		}
	}
}

func TestAudioInputFeatures_Good(t *testing.T) {
	extractor, err := NewAudioFeatureExtractor(&AudioFeatureConfig{
		NumMelFilters: 4,
		SamplingRate:  16_000,
		FrameLength:   4,
		HopLength:     2,
		FFTLength:     4,
		MaxFrequency:  8000,
		PadToMultiple: 8,
	})
	if err != nil {
		t.Fatalf("NewAudioFeatureExtractor: %v", err)
	}
	samples := []float32{0.1, -0.2, 0.3, -0.4}
	wantF32, _, wantFrames, err := extractor.Extract(samples)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	got, frames, melBins, err := AudioInputFeatures(samples, extractor)
	if err != nil {
		t.Fatalf("AudioInputFeatures: %v", err)
	}
	if frames != wantFrames {
		t.Fatalf("frames = %d, want %d", frames, wantFrames)
	}
	if melBins != 4 {
		t.Fatalf("melBins = %d, want 4", melBins)
	}
	if len(got) != wantFrames*melBins*bf16Size {
		t.Fatalf("feature bytes = %d, want %d", len(got), wantFrames*melBins*bf16Size)
	}
	if !slices.Equal(got, f32ToBf16Slice(wantF32)) {
		t.Fatal("AudioInputFeatures did not return bf16-converted extractor rows")
	}
}

func TestAudioFeatureExtractorErrorBranches(t *testing.T) {
	if _, err := NewAudioFeatureExtractor(nil); err == nil {
		t.Fatal("NewAudioFeatureExtractor(nil) error = nil")
	}
	if _, err := NewAudioFeatureExtractor(&AudioFeatureConfig{
		FeatureSize:  1,
		SamplingRate: 16_000,
		FrameLength:  8,
		HopLength:    2,
		FFTLength:    6,
		MaxFrequency: 8000,
	}); err == nil {
		t.Fatal("NewAudioFeatureExtractor(non-power-of-two FFT) error = nil")
	}
	if _, err := NewAudioFeatureExtractor(&AudioFeatureConfig{
		FeatureSize:  1,
		SamplingRate: 16_000,
		FrameLength:  4,
		HopLength:    2,
		FFTLength:    4,
		MinFrequency: 1000,
		MaxFrequency: 1000,
	}); err == nil {
		t.Fatal("NewAudioFeatureExtractor(empty mel band) error = nil")
	}

	extractor, err := NewAudioFeatureExtractor(&AudioFeatureConfig{
		FeatureSize:  2,
		SamplingRate: 16_000,
		FrameLength:  4,
		HopLength:    2,
		FFTLength:    4,
		MaxFrequency: 8000,
		Preemphasis:  0.97,
	})
	if err != nil {
		t.Fatalf("NewAudioFeatureExtractor(preemphasis config): %v", err)
	}
	if _, _, _, err := extractor.Extract([]float32{0.1, 0.2, 0.3, 0.4}); err == nil {
		t.Fatal("Extract(preemphasis) error = nil")
	}
	if _, _, _, err := (*AudioFeatureExtractor)(nil).Extract([]float32{0.1}); err == nil {
		t.Fatal("Extract(nil extractor) error = nil")
	}
	if _, _, _, err := extractor.Extract(nil); err == nil {
		t.Fatal("Extract(empty waveform) error = nil")
	}
}

func TestAudioRFFTMatchesNaiveDFT(t *testing.T) {
	frame := []float64{1, -2, 3, 0.5, -1.5, 2.5, 0, -0.25}
	got := make([]complex128, len(frame))
	audioRFFT(frame, got)

	for k := range frame {
		var want complex128
		for n, x := range frame {
			angle := -2 * math.Pi * float64(k*n) / float64(len(frame))
			want += complex(x, 0) * cmplx.Rect(1, angle)
		}
		if diff := cmplx.Abs(got[k] - want); diff > 1e-9 {
			t.Fatalf("bin %d diff = %.3g, got %v want %v", k, diff, got[k], want)
		}
	}
}

func TestHTKMelFilterBankShapeAndSupport(t *testing.T) {
	filters := htkMelFilterBank(9, 4, 0, 8000, 16000)
	if len(filters) != 4 {
		t.Fatalf("filters = %d, want 4", len(filters))
	}
	for i, row := range filters {
		if len(row) != 9 {
			t.Fatalf("filter %d bins = %d, want 9", i, len(row))
		}
		nonZero := 0
		for _, v := range row {
			if v < 0 || v > 1 {
				t.Fatalf("filter %d value = %v, want triangular weight in [0,1]", i, v)
			}
			if v > 0 {
				nonZero++
			}
		}
		if nonZero == 0 {
			t.Fatalf("filter %d has no support", i)
		}
	}
}
