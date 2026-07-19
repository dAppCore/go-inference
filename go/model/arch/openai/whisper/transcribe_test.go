// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"testing"

	core "dappco.re/go"
)

// TestLoad_Bad proves a directory with no config.json at all is refused (Load's real-checkpoint Good
// path is live_test.go's TestLive_RealCheckpoint_Load — no hermetic Good case here needs a whole
// synthetic checkpoint directory, since that would just re-stage tinyWhisperTensors' fixture as files).
func TestLoad_Bad(t *testing.T) {
	if _, err := Load(t.TempDir()); err == nil {
		t.Fatal("Load accepted a directory with no config.json")
	}
}

// TestLoad_Ugly proves a directory whose config.json declares a DIFFERENT model_type gets the clean
// capability refusal (deliverable #3's "a non-whisper model dir gets a clean capability refusal") —
// checked BEFORE any generation_config.json/preprocessor_config.json/tokenizer.json/safetensors read,
// so this hermetic case needs nothing but the one file.
func TestLoad_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "config.json", `{"model_type":"llama","hidden_size":4096}`)
	_, err := Load(dir)
	if err == nil {
		t.Fatal("Load accepted a non-whisper checkpoint directory")
	}
	if !core.Contains(err.Error(), "not a Whisper checkpoint") {
		t.Fatalf("refusal %q must say clearly this is not a Whisper checkpoint", err.Error())
	}
}

// TestTranscribe_NilModel_Ugly proves a nil *Model refuses cleanly rather than panicking on a nil
// dereference.
func TestTranscribe_NilModel_Ugly(t *testing.T) {
	var m *Model
	if _, err := m.Transcribe(nil, Options{}); err == nil {
		t.Fatal("Transcribe accepted a nil model")
	}
}

// TestTranscribe_BadWAV_Bad proves a malformed WAV is refused before ANY model field (Features/Weights/
// Tokenizer) is ever touched — the zero-value Model{} below would panic immediately if Transcribe read
// past DecodeWAV16Mono without erroring first.
func TestTranscribe_BadWAV_Bad(t *testing.T) {
	m := &Model{}
	if _, err := m.Transcribe([]byte("not a wav"), Options{}); err == nil {
		t.Fatal("Transcribe accepted malformed WAV bytes")
	}
}

// TestTranscribe_TooLong_Bad proves audio exceeding the checkpoint's fixed window is a NAMED refusal
// (states the bound) rather than silent truncation — the v1 scope's documented behaviour. Only
// Features needs to be populated: the length check runs before EncodeAudio/Weights/Tokenizer are ever
// touched.
func TestTranscribe_TooLong_Bad(t *testing.T) {
	m := &Model{Features: &FeatureConfig{NFFT: 400, HopLength: 160, NSamples: 480000, SamplingRate: 16000, NumMelBins: 80, MelFilters: make([][]float64, 80)}}
	for i := range m.Features.MelFilters {
		m.Features.MelFilters[i] = make([]float64, 201)
	}
	raw := buildWAV(1, 16000, 16, int16LEBytes(make([]int16, 480001))) // one sample over the 30s bound
	_, err := m.Transcribe(raw, Options{})
	if err == nil {
		t.Fatal("Transcribe accepted audio longer than the 30s window")
	}
	if !core.Contains(err.Error(), "30") {
		t.Fatalf("refusal %q must name the bound (30s)", err.Error())
	}
}

// TestTranscribeAudio_Bad proves the inference.Transcriber adapter forwards Transcribe's error AND
// reshapes it to the (empty, empty, err) contract rather than a partial/zero Result being papered over —
// mirrors TestTranscribe_BadWAV_Bad's malformed-WAV case through the adapter. The real-checkpoint success
// reshape (text/language correctly unwrapped from Result) is live_test.go's
// TestLive_RealCheckpoint_TranscribeAudio_Good — this package carries no hermetic fixture that gets
// Transcribe all the way to a successful Result (see ExampleModel_Transcribe's doc comment).
func TestTranscribeAudio_Bad(t *testing.T) {
	m := &Model{}
	text, lang, err := m.TranscribeAudio([]byte("not a wav"), "")
	if err == nil {
		t.Fatal("TranscribeAudio accepted malformed WAV bytes")
	}
	if text != "" || lang != "" {
		t.Fatalf("TranscribeAudio on error = (%q, %q), want (\"\", \"\") — no partial result on failure", text, lang)
	}
}

// TestTranscribeAudio_Ugly proves a nil *Model refuses cleanly through the adapter too, rather than
// panicking on the nil dereference Transcribe itself guards against (TestTranscribe_NilModel_Ugly).
func TestTranscribeAudio_Ugly(t *testing.T) {
	var m *Model
	text, lang, err := m.TranscribeAudio(nil, "en")
	if err == nil {
		t.Fatal("TranscribeAudio accepted a nil model")
	}
	if text != "" || lang != "" {
		t.Fatalf("TranscribeAudio(nil model) = (%q, %q), want (\"\", \"\")", text, lang)
	}
}
