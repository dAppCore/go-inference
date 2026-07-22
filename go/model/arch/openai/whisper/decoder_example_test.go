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

// ExampleNewSelfAttnCache shows DecodeLogitsStep's cache lifecycle: one fresh cache per request/detection
// pass, one entry per decoder layer.
func ExampleNewSelfAttnCache() {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		core.Println(err)
		return
	}
	cache := NewSelfAttnCache(len(w.DecoderLayers))
	core.Println(len(cache) == len(w.DecoderLayers))
	// Output: true
}

// ExampleDecodeLogitsStep shows the incremental call shape GreedyDecode actually drives: a fresh cache,
// fed the new token(s) plus their starting position — see GreedyDecode's doc comment for the full
// prefill-then-single-token loop this powers.
func ExampleDecodeLogitsStep() {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		core.Println(err)
		return
	}
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	cache := NewSelfAttnCache(len(w.DecoderLayers))
	logits, err := DecodeLogitsStep([]int32{0, 1}, 0, cache, crossKV, cfg.MaxSourcePositions, w, cfg)
	core.Println(err == nil, len(logits) == cfg.VocabSize)
	// Output: true true
}
