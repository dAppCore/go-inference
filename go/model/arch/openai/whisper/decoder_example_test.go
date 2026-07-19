// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

func ExamplePrecomputeCrossKV() {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		core.Println(err)
		return
	}
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	core.Println(len(crossKV) == len(w.DecoderLayers))
	// Output: true
}

func ExampleDecodeLogits() {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		core.Println(err)
		return
	}
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	logits, err := DecodeLogits([]int32{0, 1, 2}, crossKV, cfg.MaxSourcePositions, w, cfg)
	core.Println(err == nil, len(logits) == cfg.VocabSize)
	// Output: true true
}
