// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"testing"
)

// Load's SUCCESS path is proven against the REAL checkpoint in live_test.go
// (TestLive_RealCheckpoint_Load_Good) — a stronger proof than a hand-built synthetic checkpoint
// directory would be. This file covers Load's refusal paths (hermetic, no checkpoint needed)
// and Model.OCR's nil-receiver guard.

func TestLoad_Bad(t *testing.T) {
	dir := t.TempDir() // no config.json at all
	if _, err := Load(dir); err == nil {
		t.Fatal("Load accepted a directory with no config.json")
	}
}

func TestLoad_Ugly(t *testing.T) {
	dir := t.TempDir()
	// well-formed config.json, but the WRONG architecture — must be refused by name, not
	// silently accepted or failed deeper with a confusing tensor-name error.
	writeTestFile(t, dir, "config.json", `{"model_type":"glm4"}`)
	_, err := Load(dir)
	if err == nil {
		t.Fatal("Load accepted a non-glm_ocr config.json")
	}
}

func TestModel_OCR_Bad(t *testing.T) {
	var m *Model
	if _, err := m.OCR([]byte{}, "Text Recognition:"); err == nil {
		t.Fatal("Model.OCR accepted a nil receiver")
	}
}

func TestModel_OCR_Ugly(t *testing.T) {
	var m *Model
	if _, err := m.OCRWithOptions([]byte{}, "", GenerateOptions{MaxNewTokens: 5}); err == nil {
		t.Fatal("Model.OCRWithOptions accepted a nil receiver")
	}
}

func TestLoadGenerationConfig_Good(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, dir, "generation_config.json", `{"eos_token_id":[59246,59253],"pad_token_id":59246}`)
	g, err := loadGenerationConfig(dir)
	if err != nil {
		t.Fatalf("loadGenerationConfig: %v", err)
	}
	if len(g.EOSTokenIDs) != 2 || g.EOSTokenIDs[0] != 59246 || g.EOSTokenIDs[1] != 59253 {
		t.Fatalf("loadGenerationConfig EOSTokenIDs = %v, want [59246 59253]", g.EOSTokenIDs)
	}
}

func TestLoadGenerationConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	if _, err := loadGenerationConfig(dir); err == nil {
		t.Fatal("loadGenerationConfig accepted a directory with no generation_config.json")
	}
}

func TestLoadGenerationConfig_Ugly(t *testing.T) {
	dir := t.TempDir()
	// well-formed JSON, but missing eos_token_id
	writeTestFile(t, dir, "generation_config.json", `{"do_sample":false}`)
	if _, err := loadGenerationConfig(dir); err == nil {
		t.Fatal("loadGenerationConfig accepted a document with no eos_token_id")
	}
}
