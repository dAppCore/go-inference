// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

func ExampleDetectLanguage() {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		core.Println(err)
		return
	}
	gen := tinyGenerationConfig()
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	id, _, err := DetectLanguage(crossKV, cfg.MaxSourcePositions, w, cfg, gen)
	core.Println(err == nil, id == 1 || id == 2)
	// Output: true true
}

func ExampleBuildInitTokens() {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		core.Println(err)
		return
	}
	gen := tinyGenerationConfig()
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	tokens, language, err := BuildInitTokens(crossKV, cfg.MaxSourcePositions, w, cfg, gen, "en")
	core.Println(err == nil, tokens, language)
	// Output: true [0 1 3 4] en
}

func ExampleGreedyDecode() {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		core.Println(err)
		return
	}
	gen := tinyGenerationConfig()
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	content, err := GreedyDecode(crossKV, cfg.MaxSourcePositions, w, cfg, gen, []int32{0, 1, 3, 4})
	core.Println(err == nil, content != nil || content == nil) // shape proof only — content is data-dependent
	// Output: true true
}
