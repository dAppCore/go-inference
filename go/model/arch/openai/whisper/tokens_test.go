// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import "testing"

// tinyGenerationConfig builds a GenerationConfig matching tinyWhisperTensors' geometry (vocab=5):
// decoder_start=0, two language tokens {1,2}, task transcribe=3, no_timestamps=4, eos=0 (reusing 0 is
// fine — EOS is only ever compared against GENERATED ids, never the fixed start token literal).
func tinyGenerationConfig() *GenerationConfig {
	return &GenerationConfig{
		DecoderStartTokenID: 0,
		EOSTokenID:          4, // reuse no_timestamps' id as a reachable "stop" for the greedy-loop tests below
		NoTimestampsTokenID: 4,
		MaxLength:           0, // absent ⇒ maxDecodeLength falls back to cfg.MaxTargetPositions alone
		LangToID:            map[string]int32{"<|en|>": 1, "<|fr|>": 2},
		TaskToID:            map[string]int32{"transcribe": 3},
		SuppressTokens:      nil,
		BeginSuppressTokens: nil,
	}
}

func TestDetectLanguage_Good(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	gen := tinyGenerationConfig()
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	id, code, err := DetectLanguage(crossKV, cfg.MaxSourcePositions, w, cfg, gen)
	if err != nil {
		t.Fatalf("DetectLanguage: %v", err)
	}
	if id != 1 && id != 2 {
		t.Fatalf("DetectLanguage id = %d, want 1 (<|en|>) or 2 (<|fr|>) — the only two language ids in scope", id)
	}
	if code != "en" && code != "fr" {
		t.Fatalf("DetectLanguage code = %q, want \"en\" or \"fr\"", code)
	}
}

// TestDetectLanguage_Bad proves an argmax restricted to {1,2} never returns a language id outside that
// set, even though the tiny model's vocab (5) has other ids that could otherwise win an unrestricted
// argmax — the whole point of the masked search.
func TestDetectLanguage_Bad(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	gen := tinyGenerationConfig()
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	for range 5 { // repeat: a masking bug that only sometimes wins would still be caught
		id, _, err := DetectLanguage(crossKV, cfg.MaxSourcePositions, w, cfg, gen)
		if err != nil {
			t.Fatalf("DetectLanguage: %v", err)
		}
		if id != 1 && id != 2 {
			t.Fatalf("DetectLanguage id = %d, want restricted to {1,2}", id)
		}
	}
}

func TestDetectLanguage_Ugly(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	if _, _, err := DetectLanguage(crossKV, cfg.MaxSourcePositions, w, cfg, &GenerationConfig{}); err == nil {
		t.Fatal("DetectLanguage accepted a generation config with no language tokens")
	}
}

func TestBuildInitTokens_Good(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	gen := tinyGenerationConfig()
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	tokens, code, err := BuildInitTokens(crossKV, cfg.MaxSourcePositions, w, cfg, gen, "fr")
	if err != nil {
		t.Fatalf("BuildInitTokens: %v", err)
	}
	want := []int32{0, 2, 3, 4} // start, <|fr|>, transcribe, notimestamps
	for i := range want {
		if tokens[i] != want[i] {
			t.Fatalf("BuildInitTokens(--language fr) = %v, want %v", tokens, want)
		}
	}
	if code != "fr" {
		t.Fatalf("BuildInitTokens resolved language = %q, want \"fr\" (the forced override)", code)
	}
}

// TestBuildInitTokens_Bad proves an unknown --language override is refused rather than silently falling
// back to auto-detect.
func TestBuildInitTokens_Bad(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	gen := tinyGenerationConfig()
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	if _, _, err := BuildInitTokens(crossKV, cfg.MaxSourcePositions, w, cfg, gen, "klingon"); err == nil {
		t.Fatal("BuildInitTokens accepted an unknown --language override")
	}
}

// TestBuildInitTokens_Ugly proves language="" auto-detects (goes through DetectLanguage) instead of
// erroring or defaulting to a fixed language.
func TestBuildInitTokens_Ugly(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	gen := tinyGenerationConfig()
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	tokens, code, err := BuildInitTokens(crossKV, cfg.MaxSourcePositions, w, cfg, gen, "")
	if err != nil {
		t.Fatalf("BuildInitTokens(auto-detect): %v", err)
	}
	if tokens[1] != 1 && tokens[1] != 2 {
		t.Fatalf("BuildInitTokens(\"\") language slot = %d, want auto-detected 1 or 2", tokens[1])
	}
	if code != "en" && code != "fr" {
		t.Fatalf("BuildInitTokens(\"\") resolved code = %q, want auto-detected en/fr", code)
	}
}

func TestMaxDecodeLength_Good(t *testing.T) {
	cfg := &Config{MaxTargetPositions: 448}
	if got := maxDecodeLength(cfg, &GenerationConfig{MaxLength: 448}); got != 448 {
		t.Fatalf("maxDecodeLength = %d, want 448", got)
	}
}

// TestMaxDecodeLength_Bad proves the MIN of the two bounds wins when generation policy caps shorter than
// the architecture's position table (see GenerationConfig.MaxLength's doc comment).
func TestMaxDecodeLength_Bad(t *testing.T) {
	cfg := &Config{MaxTargetPositions: 448}
	if got := maxDecodeLength(cfg, &GenerationConfig{MaxLength: 100}); got != 100 {
		t.Fatalf("maxDecodeLength = %d, want 100 (the shorter generation-policy cap)", got)
	}
}

func TestMaxDecodeLength_Ugly(t *testing.T) {
	cfg := &Config{MaxTargetPositions: 448}
	if got := maxDecodeLength(cfg, &GenerationConfig{MaxLength: 0}); got != 448 {
		t.Fatalf("maxDecodeLength(MaxLength absent) = %d, want 448 (falls back to MaxTargetPositions alone)", got)
	}
}

func TestGreedyDecode_Good(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	gen := tinyGenerationConfig()
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	content, err := GreedyDecode(crossKV, cfg.MaxSourcePositions, w, cfg, gen, []int32{0, 1, 3, 4})
	if err != nil {
		t.Fatalf("GreedyDecode: %v", err)
	}
	// every generated id must be a valid vocab id (0..VocabSize-1) — proves the argmax + append loop
	// never fabricates an out-of-range id.
	for _, id := range content {
		if int(id) < 0 || int(id) >= cfg.VocabSize {
			t.Fatalf("GreedyDecode produced out-of-vocab id %d", id)
		}
	}
}

func TestGreedyDecode_Bad(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	gen := tinyGenerationConfig()
	if _, err := GreedyDecode(nil, cfg.MaxSourcePositions, w, cfg, gen, nil); err == nil {
		t.Fatal("GreedyDecode accepted an empty init prompt")
	}
}

// TestGreedyDecode_Ugly proves the loop actually STOPS at the resolved length bound rather than running
// away — MaxLength=len(initTokens) means zero room for content, so GreedyDecode must return immediately
// with no generated tokens (not panic on a should-never-happen zero-iteration loop).
func TestGreedyDecode_Ugly(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	gen := tinyGenerationConfig()
	gen.MaxLength = 4 // exactly len(initTokens) below — no budget for any content token
	encOut := seqVals(cfg.MaxSourcePositions * cfg.DModel)
	crossKV := PrecomputeCrossKV(encOut, cfg.MaxSourcePositions, w)
	content, err := GreedyDecode(crossKV, cfg.MaxSourcePositions, w, cfg, gen, []int32{0, 1, 3, 4})
	if err != nil {
		t.Fatalf("GreedyDecode: %v", err)
	}
	if len(content) != 0 {
		t.Fatalf("GreedyDecode(no length budget) = %v, want empty", content)
	}
}

func TestArgmaxF32_Good(t *testing.T) {
	if got := argmaxF32([]float32{1, 5, 3}); got != 1 {
		t.Fatalf("argmaxF32 = %d, want 1", got)
	}
}

// TestArgmaxF32_Bad proves ties keep the FIRST index (matching torch.argmax's documented tie-breaking),
// not the last.
func TestArgmaxF32_Bad(t *testing.T) {
	if got := argmaxF32([]float32{5, 5, 5}); got != 0 {
		t.Fatalf("argmaxF32(tie) = %d, want 0 (first index)", got)
	}
}

func TestArgmaxF32_Ugly(t *testing.T) {
	if got := argmaxF32([]float32{-1}); got != 0 {
		t.Fatalf("argmaxF32(single negative) = %d, want 0", got)
	}
}
