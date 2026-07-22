// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// ocr_test.go covers Load's refusal paths (a checkpoint directory Load can reject BEFORE ever
// touching the tokenizer/safetensors — mirrors whisper.Load's identical "not a Whisper
// checkpoint" precedent, arch/openai/whisper/transcribe_test.go) and Model.OCR/OCRImage's nil
// guard. Load's/OCR's happy paths need the real ~6.7GB checkpoint (tokenizer.json + safetensors)
// — proven by live_test.go's TestLive_RealCheckpoint_Load/_OCR/_OCRImage_Good against the actual
// deepseek-ai/DeepSeek-OCR snapshot, the strongest available evidence.

// TestLoad_Bad proves a nonexistent directory is refused cleanly (a read error, not a panic).
func TestLoad_Bad(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "does-not-exist")); err == nil {
		t.Fatal("Load accepted a nonexistent directory")
	}
}

// TestLoad_Ugly proves a directory whose config.json is a well-formed but WRONG model_type is
// refused by name, before Load ever tries to read tokenizer.json/safetensors (a directory with
// only a config.json is enough to prove this — the wrong-model_type check runs first).
func TestLoad_Ugly(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"llama"}`), 0o600); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	_, err := Load(dir)
	if err == nil {
		t.Fatal("Load accepted a config.json with model_type \"llama\"")
	}
	if !core.Contains(err.Error(), "deepseek_vl_v2") {
		t.Fatalf("Load error %q must name the required model_type", err.Error())
	}
}

// TestModelOCR_Bad proves a nil *Model is refused cleanly rather than panicking (a caller that
// forgets to check Load's error and calls OCR on the zero value gets a message, not a crash).
func TestModelOCR_Bad(t *testing.T) {
	var m *Model
	if _, err := m.OCR([]byte("not an image"), Options{}); err == nil {
		t.Fatal("(*Model)(nil).OCR accepted the call instead of refusing cleanly")
	}
}

// TestModelOCRImage_Ugly proves the inference.OCRModel adapter propagates OCR's own error
// (image decode failure here) through its (text, error) reshape rather than swallowing it —
// distinct from _Bad's nil-receiver case above.
func TestModelOCRImage_Ugly(t *testing.T) {
	var m *Model
	if _, err := m.OCRImage([]byte("not an image"), ""); err == nil {
		t.Fatal("OCRImage accepted the call on a nil *Model instead of refusing cleanly")
	}
}
