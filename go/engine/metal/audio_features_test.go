// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"slices"
	"testing"
)

// audio_features_test.go covers the native-specific glue over the shared model/gemma4/audio front-end
// (#44 convergence): AudioFeatureConfig/AudioFeatureExtractor are now type aliases, and
// LoadAudioFeatureConfig/NewAudioFeatureExtractor are one-line delegates, so their behaviour (config
// normalisation, rfft, HTK mel filterbank, extractor error branches) is pinned once in
// model/gemma4/audio's own test suite (features_test.go), not duplicated here. This file keeps only
// AudioInputFeatures — the mask-returning bf16-packing wrapper that has no shared-home equivalent.

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
	wantF32, wantMask, wantFrames, err := extractor.Extract(samples)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	got, mask, frames, melBins, err := AudioInputFeatures(samples, extractor)
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
	if !slices.Equal(mask, wantMask) {
		t.Fatalf("AudioInputFeatures mask = %v, want %v", mask, wantMask)
	}
}
