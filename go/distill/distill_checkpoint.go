// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	core "dappco.re/go"
	"dappco.re/go/inference/checkpoint"
	"dappco.re/go/inference/eval"
)

// CheckpointMetadataVersion is the current schema version stamped into
// every checkpoint sidecar written by SaveCheckpointMetadata.
const CheckpointMetadataVersion = 1

// CheckpointMetadata is the portable JSON sidecar for distillation checkpoints.
type CheckpointMetadata struct {
	Version            int         `json:"version"`
	Path               string      `json:"path"`
	ResumePath         string      `json:"resume_path,omitempty"`
	Step               int         `json:"step"`
	Epoch              int         `json:"epoch"`
	Samples            int         `json:"samples"`
	Tokens             int         `json:"tokens"`
	Loss               float64     `json:"loss"`
	KL                 float64     `json:"kl"`
	SoftCrossEntropy   float64     `json:"soft_cross_entropy"`
	TeacherEntropy     float64     `json:"teacher_entropy"`
	Temperature        float64     `json:"temperature"`
	LossKind           LossKind    `json:"loss_kind"`
	Batch              BatchConfig `json:"batch"`
	Teacher            eval.Info   `json:"teacher"`
	Student            eval.Info   `json:"student"`
	TeacherCacheHits   int         `json:"teacher_cache_hits,omitempty"`
	TeacherCacheMisses int         `json:"teacher_cache_misses,omitempty"`
}

// CheckpointSnapshot carries the running totals a driver's own training
// loop has accumulated by the time it saves a checkpoint — the scalar
// counters NewCheckpointMetadata needs to stamp into the portable
// sidecar. go-mlx's equivalent (NewDistillCheckpointMetadata) read these
// straight off its own *DistillResult run-accumulator; that accumulator
// is orchestration-loop state (each driver runs its own epoch/step loop
// — see the package doc), so the caller passes just the handful of
// totals the sidecar actually records.
type CheckpointSnapshot struct {
	Step               int
	Samples            int
	Tokens             int
	Teacher            eval.Info
	Student            eval.Info
	TeacherCacheHits   int
	TeacherCacheMisses int
}

// NewCheckpointMetadata captures reproducible distillation state.
//
//	meta := distill.NewCheckpointMetadata(path, cfg, snapshot, loss, epoch)
func NewCheckpointMetadata(path string, cfg Config, snapshot CheckpointSnapshot, loss Loss, epoch int) CheckpointMetadata {
	cfg = NormalizeConfig(cfg)
	meta := CheckpointMetadata{
		Version:            CheckpointMetadataVersion,
		Path:               path,
		ResumePath:         cfg.ResumePath,
		Step:               snapshot.Step,
		Epoch:              epoch,
		Samples:            snapshot.Samples,
		Tokens:             snapshot.Tokens,
		Temperature:        cfg.Temperature,
		LossKind:           cfg.Loss,
		Batch:              cfg.Batch,
		Teacher:            snapshot.Teacher,
		Student:            snapshot.Student,
		TeacherCacheHits:   snapshot.TeacherCacheHits,
		TeacherCacheMisses: snapshot.TeacherCacheMisses,
	}
	meta.Loss = loss.Value
	meta.KL = loss.KL
	meta.SoftCrossEntropy = loss.SoftCrossEntropy
	meta.TeacherEntropy = loss.TeacherEntropy
	return meta
}

// SaveCheckpointMetadata writes checkpoint metadata beside student
// artifacts. The marshal-and-write mechanics are the shared checkpoint
// engine (go/checkpoint); only the Version/Path defaulting and the
// sidecar filename below are distill's own.
func SaveCheckpointMetadata(path string, meta CheckpointMetadata) error {
	if path == "" {
		return errCheckpointPath
	}
	if meta.Version == 0 {
		meta.Version = CheckpointMetadataVersion
	}
	if meta.Path == "" {
		meta.Path = path
	}
	return checkpoint.Save(checkpointMetadataPath(path), meta)
}

// LoadCheckpointMetadata reads checkpoint metadata written by SaveCheckpointMetadata.
func LoadCheckpointMetadata(path string) (*CheckpointMetadata, error) {
	if path == "" {
		return nil, errCheckpointPath
	}
	meta, err := checkpoint.Load[CheckpointMetadata](checkpointMetadataPath(path))
	if err != nil {
		return nil, err
	}
	if meta.Version == 0 {
		meta.Version = CheckpointMetadataVersion
	}
	return meta, nil
}

// LoadResumeMetadata reads checkpoint metadata for a resume path,
// returning (nil, nil) when the sidecar does not exist yet — a driver's
// own loop treats an absent resume checkpoint as "start fresh" rather
// than an error.
func LoadResumeMetadata(path string) (*CheckpointMetadata, error) {
	meta, err := checkpoint.LoadResume[CheckpointMetadata](checkpointMetadataPath(path))
	if err != nil || meta == nil {
		return meta, err
	}
	if meta.Version == 0 {
		meta.Version = CheckpointMetadataVersion
	}
	return meta, nil
}

func checkpointMetadataPath(path string) string {
	return core.PathJoin(path, "distill_checkpoint.json")
}

// FormatStepDir builds the "step-NNNNNN" checkpoint dirname — delegates
// to the shared checkpoint engine (go/checkpoint) so distill's copy of
// the zero-pad logic cannot drift from grpo/train's.
//
//	dir := core.PathJoin(cfg.CheckpointDir, distill.FormatStepDir(step))
func FormatStepDir(step int) string {
	return checkpoint.FormatStepDir(step)
}
