// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/gemma4/audio"
)

// audio_features.go binds the no-cgo native path to the engine-neutral mel front-end living in
// model/gemma4/audio (Mantis #1839, #44 convergence): raw 16 kHz waveform → the log-mel input_features
// the Conformer encoder consumes. The extractor's config, normalisation, rfft and HTK mel filterbank are
// pure HOST DSP — no GPU dispatch — so metal no longer keeps a duplicate copy; AudioFeatureConfig and
// AudioFeatureExtractor are type aliases onto the shared home, and only the native-specific bf16 packing
// (metal's own f32ToBf16Slice, used engine-wide for byte-parity) plus the mask-returning wrapper stay
// local. The metal/GPU AudioInputFeatures wrapper composes on top; native consumers thread the resulting
// bf16 rows + validity mask through the native tower (audio_encoder.go).

// AudioFeatureConfig mirrors the feature_extractor section of the model's processor_config.json — a
// type alias onto the shared, engine-neutral audio.FeatureConfig.
type AudioFeatureConfig = audio.FeatureConfig

// AudioFeatureExtractor converts waveforms to log-mel features — a type alias onto the shared,
// engine-neutral audio.FeatureExtractor.
type AudioFeatureExtractor = audio.FeatureExtractor

// LoadAudioFeatureConfig reads the audio feature_extractor section from the model directory's
// processor_config.json. Returns (nil, nil) when the model ships no processor config (text-only
// checkpoints). Delegates to the shared audio.LoadFeatureConfig.
func LoadAudioFeatureConfig(modelPath string) (*AudioFeatureConfig, error) {
	return audio.LoadFeatureConfig(modelPath)
}

// NewAudioFeatureExtractor builds the extractor from the model's declared feature config (absent
// fields take the HF constructor defaults via normalisation). Delegates to the shared
// audio.NewFeatureExtractor.
func NewAudioFeatureExtractor(cfg *AudioFeatureConfig) (*AudioFeatureExtractor, error) {
	return audio.NewFeatureExtractor(cfg)
}

// AudioInputFeatures converts one mono waveform through the shared host extractor and returns the bf16
// [frames, melBins] rows consumed by the native audio encoder plus the per-frame validity mask (false =
// padding). The mask flows through AudioEncode into the Conformer attention so padded frames are never
// attended; a fully-valid clip yields an all-true mask (byte-identical to the mask-free path). This is
// the one piece of native-specific glue over the shared DSP: the bf16 packing reuses metal's own
// f32ToBf16Slice (byte-parity verified engine-wide, not the shared package's private conversion).
func AudioInputFeatures(samples []float32, extractor *AudioFeatureExtractor) ([]byte, []bool, int, int, error) {
	features, mask, frames, err := extractor.Extract(samples)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	if frames <= 0 || len(features)%frames != 0 {
		return nil, nil, 0, 0, core.NewError("native.AudioInputFeatures: invalid audio feature geometry")
	}
	melBins := len(features) / frames
	return f32ToBf16Slice(features), mask, frames, melBins, nil
}
