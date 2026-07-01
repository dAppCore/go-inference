// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
)

// --- NewCheckpointMetadata ---

// Good: every input field lands in the expected output field.
func TestNewCheckpointMetadata_Good(t *testing.T) {
	cfg := Config{
		BatchSize:                 4,
		GradientAccumulationSteps: 2,
		LearningRate:              1e-4,
		MaxSeqLen:                 512,
		SequencePacking:           true,
		EvalPrompts:               []string{"hello"},
		EvalTemperature:           0.25,
		ResumePath:                "/ckpt/prev",
	}
	snapshot := CheckpointSnapshot{
		Step:           12,
		OptimizerStep:  6,
		Samples:        48,
		Loss:           0.22,
		ValLoss:        0.30,
		ScoreComposite: 0.9,
		Model:          eval.Info{Architecture: "gemma4"},
	}

	meta := NewCheckpointMetadata("/ckpt/step-12", cfg, snapshot, 1)

	if meta.Version != CheckpointMetadataVersion {
		t.Errorf("Version = %d, want %d", meta.Version, CheckpointMetadataVersion)
	}
	if meta.Path != "/ckpt/step-12" || meta.ResumePath != "/ckpt/prev" {
		t.Errorf("Path/ResumePath = %q/%q, want /ckpt/step-12 //ckpt/prev", meta.Path, meta.ResumePath)
	}
	if meta.Step != 12 || meta.OptimizerStep != 6 || meta.Epoch != 1 || meta.Samples != 48 {
		t.Errorf("counters = %+v, want step 12 optimizer 6 epoch 1 samples 48", meta)
	}
	if meta.BatchSize != 4 || meta.GradientAccumulationSteps != 2 || meta.EffectiveBatchSize != 8 || meta.MaxSeqLen != 512 || !meta.SequencePacking {
		t.Errorf("cfg passthrough = %+v, want batch 4 accum 2 effective 8 maxseq 512 packing true", meta)
	}
	if meta.Loss != 0.22 || meta.ValLoss != 0.30 || meta.LearningRate != 1e-4 {
		t.Errorf("scalars = loss %v val %v lr %v, want 0.22 0.30 1e-4", meta.Loss, meta.ValLoss, meta.LearningRate)
	}
	if len(meta.EvalPrompts) != 1 || meta.EvalPrompts[0] != "hello" || meta.EvalTemperature != 0.25 {
		t.Errorf("eval passthrough = %+v temp %v, want [hello] 0.25", meta.EvalPrompts, meta.EvalTemperature)
	}
	if meta.ScoreComposite != 0.9 || meta.Model.Architecture != "gemma4" {
		t.Errorf("snapshot passthrough = score %v model %+v, want 0.9 gemma4", meta.ScoreComposite, meta.Model)
	}
}

// Bad: every input at its zero value still produces a well-formed
// metadata value (NormalizeConfig backfills LearningRate/EvalMaxTokens)
// rather than panicking or leaving zero-poisoned fields.
func TestNewCheckpointMetadata_Bad(t *testing.T) {
	meta := NewCheckpointMetadata("", Config{}, CheckpointSnapshot{}, 0)
	if meta.Version != CheckpointMetadataVersion {
		t.Errorf("Version = %d, want %d even for zero-value input", meta.Version, CheckpointMetadataVersion)
	}
	if meta.LearningRate != 1e-5 || meta.BatchSize != 1 || meta.GradientAccumulationSteps != 1 || meta.EffectiveBatchSize != 1 {
		t.Errorf("zero-value Config did not normalise: %+v", meta)
	}
	if meta.Step != 0 || meta.OptimizerStep != 0 || meta.Samples != 0 || meta.Loss != 0 {
		t.Errorf("zero-value snapshot run scalars = %+v, want zeroes", meta)
	}
}

// Ugly: the OptimizerStep fallback — a snapshot with Step but no explicit
// OptimizerStep reports the step count as the optimizer step (the
// degenerate equal-clock case) — plus EvalPrompts is defensively cloned,
// not aliased from the caller's Config.
func TestNewCheckpointMetadata_Ugly(t *testing.T) {
	prompts := []string{"p1", "p2"}
	cfg := Config{EvalPrompts: prompts}
	meta := NewCheckpointMetadata("ckpt", cfg, CheckpointSnapshot{Step: 9}, 2)
	if meta.Step != 9 || meta.OptimizerStep != 9 {
		t.Fatalf("metadata step/optimizer = %d/%d, want 9/9 (fallback to Step)", meta.Step, meta.OptimizerStep)
	}
	if meta.Epoch != 2 {
		t.Fatalf("metadata epoch = %d, want 2", meta.Epoch)
	}
	meta.EvalPrompts[0] = "mutated"
	if prompts[0] != "p1" {
		t.Fatal("NewCheckpointMetadata aliased the caller's EvalPrompts slice, want a clone")
	}
}

// --- SaveCheckpointMetadata ---

// Good: Save writes a readable JSON sidecar under <path>/train_checkpoint.json.
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
// writing, and a .safetensors-suffixed path places the sidecar beside the
// file (in its parent dir) rather than treating the file path itself as a
// directory to write into.
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

	adapterPath := core.PathJoin(dir, "adapter.safetensors")
	if err := SaveCheckpointMetadata(adapterPath, CheckpointMetadata{Step: 3}); err != nil {
		t.Fatalf("SaveCheckpointMetadata(.safetensors) error = %v", err)
	}
	sidecar := core.PathJoin(dir, "train_checkpoint.json")
	if !core.ReadFile(sidecar).OK {
		t.Fatalf(".safetensors sidecar not written beside the file at %s", sidecar)
	}
}

// --- LoadCheckpointMetadata ---

// Good: Load round-trips exactly what Save wrote.
func TestLoadCheckpointMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "step-5")
	want := CheckpointMetadata{Step: 5, Epoch: 2, Loss: 1.5, LearningRate: 1e-4}
	if err := SaveCheckpointMetadata(path, want); err != nil {
		t.Fatalf("SaveCheckpointMetadata() error = %v", err)
	}
	got, err := LoadCheckpointMetadata(path)
	if err != nil {
		t.Fatalf("LoadCheckpointMetadata() error = %v", err)
	}
	if got.Step != 5 || got.Epoch != 2 || got.Loss != 1.5 || got.LearningRate != 1e-4 {
		t.Fatalf("LoadCheckpointMetadata() = %+v, want Step 5 Epoch 2 Loss 1.5 LearningRate 1e-4", got)
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

// --- checkpointMetadataPath ---

// Good: a plain directory path gets the sidecar joined inside it.
func TestCheckpointMetadataPath_Good(t *testing.T) {
	if got := checkpointMetadataPath("/ckpt/step-1"); got != "/ckpt/step-1/train_checkpoint.json" {
		t.Fatalf("checkpointMetadataPath(dir) = %q, want /ckpt/step-1/train_checkpoint.json", got)
	}
}

// Bad: a .safetensors file path places the sidecar in the PARENT dir,
// never inside the file path itself.
func TestCheckpointMetadataPath_Bad(t *testing.T) {
	if got := checkpointMetadataPath("/ckpt/adapter.safetensors"); got != "/ckpt/train_checkpoint.json" {
		t.Fatalf("checkpointMetadataPath(.safetensors) = %q, want /ckpt/train_checkpoint.json", got)
	}
}

// Ugly: a bare filename with no directory component still resolves
// cleanly for both the directory and .safetensors branches.
func TestCheckpointMetadataPath_Ugly(t *testing.T) {
	if got := checkpointMetadataPath("adapter.safetensors"); got != "train_checkpoint.json" {
		t.Fatalf("checkpointMetadataPath(bare .safetensors) = %q, want train_checkpoint.json", got)
	}
	if got := checkpointMetadataPath("ckpt"); got != "ckpt/train_checkpoint.json" {
		t.Fatalf("checkpointMetadataPath(bare dir) = %q, want ckpt/train_checkpoint.json", got)
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
