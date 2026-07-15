// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
)

// --- NewCheckpointMetadata ---

// Good: every input field lands in the expected output field.
func TestNewCheckpointMetadata_Good(t *testing.T) {
	cfg := Config{
		GroupSize:     4,
		KLCoefficient: 0.05,
		LearningRate:  0.001,
		ResumePath:    "/ckpt/prev",
	}
	snapshot := CheckpointSnapshot{
		Samples:  40,
		Rollouts: 160,
		Policy:   eval.Info{Architecture: "policy-arch"},
	}
	update := Update{Step: 10, Epoch: 2, RewardMean: 0.75, RewardStd: 0.2, KLMean: 0.01, Loss: 0.3, KLCoefficient: 0.05}

	meta := NewCheckpointMetadata("/ckpt/step-10", cfg, snapshot, update)

	if meta.Version != CheckpointMetadataVersion {
		t.Errorf("Version = %d, want %d", meta.Version, CheckpointMetadataVersion)
	}
	if meta.Path != "/ckpt/step-10" || meta.ResumePath != "/ckpt/prev" {
		t.Errorf("Path/ResumePath = %q/%q, want /ckpt/step-10 //ckpt/prev", meta.Path, meta.ResumePath)
	}
	if meta.Step != 10 || meta.Epoch != 2 || meta.Samples != 40 || meta.Rollouts != 160 {
		t.Errorf("counters = step %d epoch %d samples %d rollouts %d, want 10 2 40 160", meta.Step, meta.Epoch, meta.Samples, meta.Rollouts)
	}
	if meta.GroupSize != 4 || meta.KLCoefficient != 0.05 || meta.LearningRate != 0.001 {
		t.Errorf("cfg passthrough = groupSize %d klCoef %v lr %v, want 4 0.05 0.001", meta.GroupSize, meta.KLCoefficient, meta.LearningRate)
	}
	if meta.Policy.Architecture != "policy-arch" {
		t.Errorf("Policy = %+v, want architecture policy-arch", meta.Policy)
	}
	if meta.RewardMean != 0.75 || meta.RewardStd != 0.2 || meta.KLMean != 0.01 || meta.Loss != 0.3 {
		t.Errorf("update passthrough = %+v, want RewardMean 0.75 RewardStd 0.2 KLMean 0.01 Loss 0.3", meta)
	}
}

// Bad: every input at its zero value still produces a well-formed
// metadata value (NormalizeConfig backfills GroupSize) rather than
// panicking or leaving zero-poisoned fields.
func TestNewCheckpointMetadata_Bad(t *testing.T) {
	meta := NewCheckpointMetadata("", Config{}, CheckpointSnapshot{}, Update{})
	if meta.Version != CheckpointMetadataVersion {
		t.Errorf("Version = %d, want %d even for zero-value input", meta.Version, CheckpointMetadataVersion)
	}
	if meta.GroupSize != 4 {
		t.Errorf("GroupSize = %d, want the NormalizeConfig default of 4", meta.GroupSize)
	}
}

// Ugly: NormalizeConfig's defaulting is visible in the output even when
// only some Config fields are set, and ResumePath threads through
// unchanged.
func TestNewCheckpointMetadata_Ugly(t *testing.T) {
	meta := NewCheckpointMetadata("/ckpt/x", Config{ResumePath: "/ckpt/prev"}, CheckpointSnapshot{}, Update{Epoch: 1})
	if meta.GroupSize != 4 {
		t.Errorf("GroupSize = %d, want the NormalizeConfig default of 4", meta.GroupSize)
	}
	if meta.ResumePath != "/ckpt/prev" {
		t.Errorf("ResumePath = %q, want /ckpt/prev", meta.ResumePath)
	}
}

// --- SaveCheckpointMetadata ---

// Good: Save writes a readable JSON sidecar under <path>/grpo_checkpoint.json.
func TestSaveCheckpointMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "step-1")
	meta := CheckpointMetadata{Step: 1, Loss: 0.25}

	if err := SaveCheckpointMetadata(path, meta); err != nil {
		t.Fatalf("SaveCheckpointMetadata() error = %v", err)
	}
	read := core.ReadFile(checkpointMetadataPath(path))
	if !read.OK {
		t.Fatalf("sidecar not written at %s", checkpointMetadataPath(path))
	}
}

// Bad: an empty path is rejected before any file-system work happens.
func TestSaveCheckpointMetadata_Bad(t *testing.T) {
	if err := SaveCheckpointMetadata("", CheckpointMetadata{}); err != errCheckpointPath {
		t.Fatalf("SaveCheckpointMetadata(\"\") error = %v, want errCheckpointPath", err)
	}
}

// Ugly: a zero-value Version/Path on the metadata is defaulted before
// writing, and a not-yet-existing nested checkpoint directory is created.
func TestSaveCheckpointMetadata_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "nested", "deeper", "step-2")
	if err := SaveCheckpointMetadata(path, CheckpointMetadata{}); err != nil {
		t.Fatalf("SaveCheckpointMetadata() into a new nested dir error = %v", err)
	}
	loaded, err := LoadCheckpointMetadata(path)
	if err != nil {
		t.Fatalf("LoadCheckpointMetadata() error = %v", err)
	}
	if loaded.Version != CheckpointMetadataVersion || loaded.Path != path {
		t.Fatalf("defaults not applied: %+v, want Version %d Path %q", loaded, CheckpointMetadataVersion, path)
	}
}

// --- LoadCheckpointMetadata ---

// Good: Load round-trips exactly what Save wrote.
func TestLoadCheckpointMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "step-5")
	want := CheckpointMetadata{Step: 5, Epoch: 2, Loss: 1.5, GroupSize: 4}
	if err := SaveCheckpointMetadata(path, want); err != nil {
		t.Fatalf("SaveCheckpointMetadata() error = %v", err)
	}
	got, err := LoadCheckpointMetadata(path)
	if err != nil {
		t.Fatalf("LoadCheckpointMetadata() error = %v", err)
	}
	if got.Step != 5 || got.Epoch != 2 || got.Loss != 1.5 || got.GroupSize != 4 {
		t.Fatalf("LoadCheckpointMetadata() = %+v, want Step 5 Epoch 2 Loss 1.5 GroupSize 4", got)
	}
}

// Bad: an empty path is rejected, and a path with no sidecar file is a
// hard error (unlike LoadResumeMetadata's soft-missing-file semantics).
func TestLoadCheckpointMetadata_Bad(t *testing.T) {
	if _, err := LoadCheckpointMetadata(""); err != errCheckpointPath {
		t.Fatalf("LoadCheckpointMetadata(\"\") error = %v, want errCheckpointPath", err)
	}
	dir := t.TempDir()
	if _, err := LoadCheckpointMetadata(core.PathJoin(dir, "never-saved")); err == nil {
		t.Fatal("LoadCheckpointMetadata() on a missing sidecar: expected error, got nil")
	}
}

// Ugly: a sidecar written with Version 0 (bypassing Save) is backfilled
// to CheckpointMetadataVersion on load.
func TestLoadCheckpointMetadata_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "step-9")
	raw := core.JSONMarshalIndent(CheckpointMetadata{Step: 9}, "", "  ")
	if !raw.OK {
		t.Fatalf("marshal fixture failed")
	}
	if result := core.MkdirAll(path, 0o755); !result.OK {
		t.Fatalf("mkdir fixture dir failed")
	}
	if result := core.WriteFile(checkpointMetadataPath(path), raw.Value.([]byte), 0o600); !result.OK {
		t.Fatalf("write fixture sidecar failed")
	}
	got, err := LoadCheckpointMetadata(path)
	if err != nil {
		t.Fatalf("LoadCheckpointMetadata() error = %v", err)
	}
	if got.Version != CheckpointMetadataVersion {
		t.Fatalf("Version = %d, want backfilled %d", got.Version, CheckpointMetadataVersion)
	}
}

// --- LoadResumeMetadata ---

// Good: Load reads back a checkpoint saved for resume.
func TestLoadResumeMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "resume")
	if err := SaveCheckpointMetadata(path, CheckpointMetadata{Step: 3}); err != nil {
		t.Fatalf("SaveCheckpointMetadata() error = %v", err)
	}
	got, err := LoadResumeMetadata(path)
	if err != nil {
		t.Fatalf("LoadResumeMetadata() error = %v", err)
	}
	if got == nil || got.Step != 3 {
		t.Fatalf("LoadResumeMetadata() = %+v, want Step 3", got)
	}
}

// Bad: a corrupt sidecar is a real parse error, not swallowed.
func TestLoadResumeMetadata_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "corrupt")
	if result := core.MkdirAll(path, 0o755); !result.OK {
		t.Fatalf("mkdir fixture dir failed")
	}
	if result := core.WriteFile(checkpointMetadataPath(path), []byte("{not json"), 0o600); !result.OK {
		t.Fatalf("write fixture sidecar failed")
	}
	if _, err := LoadResumeMetadata(path); err == nil {
		t.Fatal("LoadResumeMetadata() on corrupt JSON: expected error, got nil")
	}
}

// Ugly: a resume path with no sidecar at all (first run, nothing to
// resume from) returns (nil, nil) rather than an error.
func TestLoadResumeMetadata_Ugly(t *testing.T) {
	dir := t.TempDir()
	got, err := LoadResumeMetadata(core.PathJoin(dir, "never-saved"))
	if err != nil {
		t.Fatalf("LoadResumeMetadata() on absent sidecar error = %v, want nil", err)
	}
	if got != nil {
		t.Fatalf("LoadResumeMetadata() on absent sidecar = %+v, want nil", got)
	}
}

// --- FormatStepDir ---

// Good: a typical step number is zero-padded to six digits.
func TestFormatStepDir_Good(t *testing.T) {
	if got := FormatStepDir(42); got != "step-000042" {
		t.Fatalf("FormatStepDir(42) = %q, want step-000042", got)
	}
}

// Bad: a negative step does not panic and produces a deterministic,
// un-padded result.
func TestFormatStepDir_Bad(t *testing.T) {
	if got := FormatStepDir(-1); got != "step--1" {
		t.Fatalf("FormatStepDir(-1) = %q, want step--1", got)
	}
}

// Ugly: boundary step values — zero, the largest still-padded value, and
// the first value past the padding width.
func TestFormatStepDir_Ugly(t *testing.T) {
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
