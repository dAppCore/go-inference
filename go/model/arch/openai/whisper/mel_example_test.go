// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

func ExampleLoadFeatureConfig() {
	cfg, err := LoadFeatureConfig("testdata")
	core.Println(err == nil, cfg.NFFT, cfg.HopLength, cfg.NumMelBins)
	// Output: true 400 160 80
}

func ExampleFeatureConfig_ExtractLogMel() {
	cfg, err := LoadFeatureConfig("testdata")
	if err != nil {
		core.Println(err)
		return
	}
	samples := make([]float32, 1600) // 0.1s of silence at 16kHz
	melFrames, err := cfg.ExtractLogMel(samples)
	core.Println(err == nil, len(melFrames))
	// Output: true 80
}
