// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	core "dappco.re/go"
	"dappco.re/go/inference/checkpoint"
	"dappco.re/go/inference/eval"
)

// CheckpointMetadataVersion is the current schema version stamped into
// every checkpoint sidecar written by SaveCheckpointMetadata.
const CheckpointMetadataVersion = 1

// CheckpointMetadata is the portable JSON sidecar for GRPO checkpoints.
type CheckpointMetadata struct {
	Version       int       `json:"version"`
	Path          string    `json:"path"`
	ResumePath    string    `json:"resume_path,omitempty"`
	Step          int       `json:"step"`
	Epoch         int       `json:"epoch"`
	Samples       int       `json:"samples"`
	Rollouts      int       `json:"rollouts"`
	GroupSize     int       `json:"group_size"`
	RewardMean    float64   `json:"reward_mean"`
	RewardStd     float64   `json:"reward_std"`
	KLMean        float64   `json:"kl_mean,omitempty"`
	Loss          float64   `json:"loss"`
	KLCoefficient float64   `json:"kl_coefficient,omitempty"`
	LearningRate  float64   `json:"learning_rate,omitempty"`
	Policy        eval.Info `json:"policy"`
}

// CheckpointSnapshot carries the running totals a driver's own training
// loop has accumulated by the time it saves a checkpoint — the scalar
// counters NewCheckpointMetadata needs to stamp into the portable
// sidecar. go-mlx's equivalent (NewGRPOCheckpointMetadata) read these
// straight off its own *GRPOResult run-accumulator; that accumulator is
// orchestration-loop state (each driver runs its own epoch/step loop —
// see the package doc), so the caller passes just the handful of totals
// the sidecar actually records.
type CheckpointSnapshot struct {
	Samples  int
	Rollouts int
	Policy   eval.Info
}

// NewCheckpointMetadata captures reproducible GRPO state.
//
//	meta := grpo.NewCheckpointMetadata(path, cfg, snapshot, update)
func NewCheckpointMetadata(path string, cfg Config, snapshot CheckpointSnapshot, update Update) CheckpointMetadata {
	cfg = NormalizeConfig(cfg)
	return CheckpointMetadata{
		Version:       CheckpointMetadataVersion,
		Path:          path,
		ResumePath:    cfg.ResumePath,
		Step:          update.Step,
		Epoch:         update.Epoch,
		Samples:       snapshot.Samples,
		Rollouts:      snapshot.Rollouts,
		GroupSize:     cfg.GroupSize,
		RewardMean:    update.RewardMean,
		RewardStd:     update.RewardStd,
		KLMean:        update.KLMean,
		Loss:          update.Loss,
		KLCoefficient: cfg.KLCoefficient,
		LearningRate:  cfg.LearningRate,
		Policy:        snapshot.Policy,
	}
}

// SaveCheckpointMetadata writes checkpoint metadata beside policy
// artifacts. The marshal-and-write mechanics are the shared checkpoint
// engine (go/checkpoint); only the Version/Path defaulting and the
// sidecar filename below are grpo's own.
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
	return core.PathJoin(path, "grpo_checkpoint.json")
}

// FormatStepDir builds the "step-NNNNNN" checkpoint dirname — delegates
// to the shared checkpoint engine (go/checkpoint) so grpo's copy of the
// zero-pad logic cannot drift from distill/train's.
//
//	dir := core.PathJoin(cfg.CheckpointDir, grpo.FormatStepDir(step))
func FormatStepDir(step int) string {
	return checkpoint.FormatStepDir(step)
}
