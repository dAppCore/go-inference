// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import core "dappco.re/go"

// ExampleLoad shows the shape of pointing Load at a real checkpoint directory (config.json,
// generation_config.json, preprocessor_config.json, tokenizer.json, model.safetensors — the standard
// openai/whisper-* layout). This package's testdata/ deliberately does NOT carry a bare "config.json"
// (see config_test.go's fixture naming), so this Example's Load call fails predictably — a safe,
// dependency-free way to demonstrate the call shape without needing the ~39 MB real checkpoint on disk
// (that live gate is live_test.go's TestLive_RealCheckpoint_Load/_Transcribe).
func ExampleLoad() {
	m, err := Load("testdata")
	core.Println(m == nil, err != nil)
	// Output: true true
}

// ExampleModel_Transcribe documents Model.Transcribe's call shape against a loaded checkpoint. No
// "Output:" comment — a real checkpoint dir + WAV bytes are needed to execute this meaningfully, so (per
// Go's testing convention) this Example is compiled and type-checked but not run; see live_test.go's
// TestLive_RealCheckpoint_Transcribe for a real, executed transcription (openai/whisper-tiny → the exact
// reference greedy transcript).
func ExampleModel_Transcribe() {
	m, err := Load("/path/to/whisper-tiny")
	if err != nil {
		core.Println("load:", err)
		return
	}
	wav := []byte{} // a real 16-bit PCM mono 16 kHz WAV file's bytes
	result, err := m.Transcribe(wav, Options{Language: "en"})
	if err != nil {
		core.Println("transcribe:", err)
		return
	}
	core.Println(result.Text, result.Language)
}

// ExampleModel_TranscribeAudio documents the inference.Transcriber adapter's call shape — the primitive-
// string reshape serve's POST /v1/audio/transcriptions handler calls through a plain interface value,
// never importing this package directly. No "Output:" comment, same reasoning as ExampleModel_Transcribe;
// see live_test.go's TestLive_RealCheckpoint_TranscribeAudio_Good for a real, executed run.
func ExampleModel_TranscribeAudio() {
	m, err := Load("/path/to/whisper-tiny")
	if err != nil {
		core.Println("load:", err)
		return
	}
	wav := []byte{} // a real 16-bit PCM mono 16 kHz WAV file's bytes
	text, language, err := m.TranscribeAudio(wav, "en")
	if err != nil {
		core.Println("transcribe:", err)
		return
	}
	core.Println(text, language)
}
