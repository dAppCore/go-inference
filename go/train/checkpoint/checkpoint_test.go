// SPDX-Licence-Identifier: EUPL-1.2

package checkpoint

import (
	"testing"

	core "dappco.re/go"
)

// sample is a minimal JSON-shaped fixture standing in for a domain
// package's own CheckpointMetadata — checkpoint is domain-agnostic (see
// the package doc), so its tests exercise the engine against a throwaway
// type rather than importing distill/grpo/train.
type sample struct {
	Version int    `json:"version"`
	Step    int    `json:"step"`
	Name    string `json:"name,omitempty"`
}

// --- Save ---

// Good: Save writes a readable, round-trippable JSON sidecar and creates
// any missing parent directory.
func TestCheckpoint_Save_Good(t *testing.T) {
	dir := t.TempDir()
	sidecarPath := core.PathJoin(dir, "nested", "deeper", "meta.json")
	if err := Save(sidecarPath, sample{Version: 1, Step: 5, Name: "x"}); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	read := core.ReadFile(sidecarPath)
	if !read.OK {
		t.Fatalf("sidecar not written at %s", sidecarPath)
	}
}

// Bad: a value encoding/json cannot marshal (a bare channel) surfaces as
// an error instead of writing a corrupt sidecar.
func TestCheckpoint_Save_Bad(t *testing.T) {
	dir := t.TempDir()
	sidecarPath := core.PathJoin(dir, "meta.json")
	if err := Save(sidecarPath, make(chan int)); err == nil {
		t.Fatal("Save(unmarshalable value): expected error, got nil")
	}
	if core.ReadFile(sidecarPath).OK {
		t.Fatal("Save(unmarshalable value): sidecar should not have been written")
	}
}

// Ugly: re-saving to the same path overwrites the previous sidecar
// contents rather than merging or appending.
func TestCheckpoint_Save_Ugly(t *testing.T) {
	dir := t.TempDir()
	sidecarPath := core.PathJoin(dir, "meta.json")
	if err := Save(sidecarPath, sample{Step: 1}); err != nil {
		t.Fatalf("first Save() error = %v", err)
	}
	if err := Save(sidecarPath, sample{Step: 2}); err != nil {
		t.Fatalf("second Save() error = %v", err)
	}
	got, err := Load[sample](sidecarPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got.Step != 2 {
		t.Fatalf("Step = %d, want 2 (overwritten, not merged)", got.Step)
	}
}

// --- Load ---

// Good: Load round-trips exactly what Save wrote.
func TestCheckpoint_Load_Good(t *testing.T) {
	dir := t.TempDir()
	sidecarPath := core.PathJoin(dir, "meta.json")
	want := sample{Version: 3, Step: 7, Name: "round-trip"}
	if err := Save(sidecarPath, want); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	got, err := Load[sample](sidecarPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if *got != want {
		t.Fatalf("Load() = %+v, want %+v", *got, want)
	}
}

// Bad: a sidecar that was never written is a hard error, not a nil result.
func TestCheckpoint_Load_Bad(t *testing.T) {
	dir := t.TempDir()
	if _, err := Load[sample](core.PathJoin(dir, "never-saved.json")); err == nil {
		t.Fatal("Load() on a missing sidecar: expected error, got nil")
	}
}

// Ugly: a corrupt (non-JSON) sidecar is a parse error, not a zero-value result.
func TestCheckpoint_Load_Ugly(t *testing.T) {
	dir := t.TempDir()
	sidecarPath := core.PathJoin(dir, "corrupt.json")
	if result := core.WriteFile(sidecarPath, []byte("{not json"), 0o600); !result.OK {
		t.Fatalf("write fixture sidecar failed")
	}
	if _, err := Load[sample](sidecarPath); err == nil {
		t.Fatal("Load() on corrupt JSON: expected error, got nil")
	}
}

// --- LoadResume ---

// Good: LoadResume reads back a sidecar Save wrote.
func TestCheckpoint_LoadResume_Good(t *testing.T) {
	dir := t.TempDir()
	sidecarPath := core.PathJoin(dir, "meta.json")
	if err := Save(sidecarPath, sample{Step: 9}); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	got, err := LoadResume[sample](sidecarPath)
	if err != nil {
		t.Fatalf("LoadResume() error = %v", err)
	}
	if got == nil || got.Step != 9 {
		t.Fatalf("LoadResume() = %+v, want Step 9", got)
	}
}

// Bad: a corrupt sidecar is a real parse error, not swallowed into (nil, nil).
func TestCheckpoint_LoadResume_Bad(t *testing.T) {
	dir := t.TempDir()
	sidecarPath := core.PathJoin(dir, "corrupt.json")
	if result := core.WriteFile(sidecarPath, []byte("{not json"), 0o600); !result.OK {
		t.Fatalf("write fixture sidecar failed")
	}
	if _, err := LoadResume[sample](sidecarPath); err == nil {
		t.Fatal("LoadResume() on corrupt JSON: expected error, got nil")
	}
}

// Ugly: a sidecar that was never saved returns (nil, nil) — the soft-
// missing-file semantics --resume relies on to mean "start fresh".
func TestCheckpoint_LoadResume_Ugly(t *testing.T) {
	dir := t.TempDir()
	got, err := LoadResume[sample](core.PathJoin(dir, "never-saved.json"))
	if err != nil {
		t.Fatalf("LoadResume() on absent sidecar error = %v, want nil", err)
	}
	if got != nil {
		t.Fatalf("LoadResume() on absent sidecar = %+v, want nil", got)
	}
}

// --- FormatStepDir ---

// Good: a typical step number is zero-padded to six digits.
func TestCheckpoint_FormatStepDir_Good(t *testing.T) {
	if got := FormatStepDir(42); got != "step-000042" {
		t.Fatalf("FormatStepDir(42) = %q, want step-000042", got)
	}
}

// Bad: a negative step does not panic and produces a deterministic,
// un-padded result.
func TestCheckpoint_FormatStepDir_Bad(t *testing.T) {
	if got := FormatStepDir(-1); got != "step--1" {
		t.Fatalf("FormatStepDir(-1) = %q, want step--1", got)
	}
}

// Ugly: boundary step values — zero, the largest still-padded value, and
// the first value past the padding width.
func TestCheckpoint_FormatStepDir_Ugly(t *testing.T) {
	cases := map[int]string{
		0:      "step-000000",
		99999:  "step-099999",
		100000: "step-100000",
	}
	for step, want := range cases {
		if got := FormatStepDir(step); got != want {
			t.Errorf("FormatStepDir(%d) = %q, want %q", step, got, want)
		}
	}
}
