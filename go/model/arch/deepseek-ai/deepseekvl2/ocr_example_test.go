// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import core "dappco.re/go"

// ExampleLoad shows the shape of pointing Load at a real checkpoint directory (config.json,
// tokenizer.json, model.safetensors — the standard deepseek-ai/DeepSeek-OCR layout). This
// package's testdata/ deliberately does NOT carry a bare "config.json" (only the fixture image,
// its generating script, and the golden JSON fixtures — see golden_test.go), so this Example's
// Load call fails predictably — a safe, dependency-free way to demonstrate the call shape without
// needing the real ~6.7 GB checkpoint on disk (that live gate is live_test.go's
// TestLive_RealCheckpoint_Load), mirroring whisper.ExampleLoad's identical precedent
// (arch/openai/whisper/transcribe_example_test.go).
func ExampleLoad() {
	m, err := Load("testdata")
	core.Println(m == nil, err != nil)
	// Output: true true
}

// ExampleModel_OCR documents Model.OCR's call shape against a loaded checkpoint. No "Output:"
// comment — a real checkpoint dir + image bytes are needed to execute this meaningfully, so (per
// Go's testing convention) this Example is compiled and type-checked but not run; see
// live_test.go's TestLive_RealCheckpoint_OCR for a real, executed OCR run (the exact reference
// greedy transcript on the committed testdata/fixture.png).
func ExampleModel_OCR() {
	m, err := Load("/path/to/deepseek-ocr")
	if err != nil {
		core.Println("load:", err)
		return
	}
	image := []byte{} // a real PNG/JPEG file's bytes (v1: exactly 1024x1024 — see DecodeAndNormaliseImage)
	result, err := m.OCR(image, Options{})
	if err != nil {
		core.Println("ocr:", err)
		return
	}
	core.Println(result.Text)
}

// ExampleModel_OCRImage documents the inference.OCRModel adapter's call shape — the primitive-
// string reshape serve/CLI capability discovery calls through a plain interface value, never
// importing this package directly. No "Output:" comment, same reasoning as ExampleModel_OCR; see
// live_test.go's TestLive_RealCheckpoint_OCRImage_Good for a real, executed run.
func ExampleModel_OCRImage() {
	m, err := Load("/path/to/deepseek-ocr")
	if err != nil {
		core.Println("load:", err)
		return
	}
	image := []byte{} // a real PNG/JPEG file's bytes
	text, err := m.OCRImage(image, DefaultPrompt)
	if err != nil {
		core.Println("ocr:", err)
		return
	}
	core.Println(text)
}
