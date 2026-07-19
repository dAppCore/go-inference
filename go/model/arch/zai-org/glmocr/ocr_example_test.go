// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import core "dappco.re/go"

// ExampleLoad shows the shape of pointing Load at a real checkpoint directory (config.json,
// preprocessor_config.json, generation_config.json, tokenizer.json, model.safetensors — the
// standard zai-org/GLM-OCR layout). This package's testdata/ deliberately does NOT carry a bare
// "config.json", so this Example's Load call fails predictably — a safe, dependency-free way to
// demonstrate the call shape without needing the ~2.7 GB real checkpoint on disk (that live gate
// is live_test.go's TestLive_RealCheckpoint_Load_Good/_OCR_Good).
func ExampleLoad() {
	m, err := Load("testdata")
	core.Println(m == nil, err != nil)
	// Output: true true
}

// ExampleModel_OCR documents Model.OCR's call shape against a loaded checkpoint. No "Output:"
// comment — a real checkpoint dir + image bytes are needed to execute this meaningfully (per
// Go's testing convention this Example is compiled and type-checked but not run); see
// live_test.go's TestLive_RealCheckpoint_OCR_Good for a real, executed OCR run reproducing the
// reference implementation's exact greedy transcript.
func ExampleModel_OCR() {
	m, err := Load("/path/to/GLM-OCR")
	if err != nil {
		core.Println("load:", err)
		return
	}
	imageBytes := []byte{} // a real PNG/JPEG's bytes, already smart_resize-stable — see DecodeAndPatchify
	text, err := m.OCR(imageBytes, "Text Recognition:")
	if err != nil {
		core.Println("ocr:", err)
		return
	}
	core.Println(text)
}

// ExampleModel_OCRWithOptions documents the MaxNewTokens override for a longer document
// transcription. No "Output:" comment — same reasoning as ExampleModel_OCR.
func ExampleModel_OCRWithOptions() {
	m, err := Load("/path/to/GLM-OCR")
	if err != nil {
		core.Println("load:", err)
		return
	}
	imageBytes := []byte{}
	text, err := m.OCRWithOptions(imageBytes, "Text Recognition:", GenerateOptions{MaxNewTokens: 1024})
	if err != nil {
		core.Println("ocr:", err)
		return
	}
	core.Println(text)
}
