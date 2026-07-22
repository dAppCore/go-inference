// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

func ExampleEncodeAudio() {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		core.Println(err)
		return
	}
	// A fixed [NumMelBins][MaxSourcePositions*2] mel input — EncodeAudio always runs the FULL
	// position table (real callers get this shape from FeatureConfig.ExtractLogMel over a
	// pad-to-30s waveform; see Model.Transcribe).
	mel := make([][]float32, cfg.NumMelBins)
	for m := range mel {
		mel[m] = seqVals(cfg.MaxSourcePositions * 2)
	}
	encOut, err := EncodeAudio(mel, w, cfg)
	core.Println(err == nil, len(encOut) == cfg.MaxSourcePositions*cfg.DModel)
	// Output: true true
}
