// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"strconv"

	core "dappco.re/go"
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

// SaveCheckpointMetadata writes checkpoint metadata beside policy artifacts.
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
	metadataPath := checkpointMetadataPath(path)
	dir := core.PathDir(metadataPath)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.E("CheckpointMetadata.Save", "ensure metadata dir", resultError(result))
		}
	}
	data := core.JSONMarshalIndent(meta, "", "  ")
	if !data.OK {
		return core.E("CheckpointMetadata.Save", "marshal metadata", resultError(data))
	}
	if result := core.WriteFile(metadataPath, data.Value.([]byte), 0o600); !result.OK {
		return core.E("CheckpointMetadata.Save", "write metadata", resultError(result))
	}
	return nil
}

// LoadCheckpointMetadata reads checkpoint metadata written by SaveCheckpointMetadata.
func LoadCheckpointMetadata(path string) (*CheckpointMetadata, error) {
	if path == "" {
		return nil, errCheckpointPath
	}
	read := core.ReadFile(checkpointMetadataPath(path))
	if !read.OK {
		return nil, resultError(read)
	}
	var meta CheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadCheckpointMetadata", "parse metadata", resultError(result))
	}
	if meta.Version == 0 {
		meta.Version = CheckpointMetadataVersion
	}
	return &meta, nil
}

// LoadResumeMetadata reads checkpoint metadata for a resume path,
// returning (nil, nil) when the sidecar does not exist yet — a driver's
// own loop treats an absent resume checkpoint as "start fresh" rather
// than an error.
func LoadResumeMetadata(path string) (*CheckpointMetadata, error) {
	read := core.ReadFile(checkpointMetadataPath(path))
	if !read.OK {
		err := resultError(read)
		if core.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var meta CheckpointMetadata
	if result := core.JSONUnmarshal(read.Value.([]byte), &meta); !result.OK {
		return nil, core.E("LoadResumeMetadata", "parse metadata", resultError(result))
	}
	if meta.Version == 0 {
		meta.Version = CheckpointMetadataVersion
	}
	return &meta, nil
}

func checkpointMetadataPath(path string) string {
	return core.PathJoin(path, "grpo_checkpoint.json")
}

// FormatStepDir builds the "step-NNNNNN" checkpoint dirname using
// strconv.AppendInt with explicit zero padding, avoiding fmt's reflection
// path on the per-checkpoint hot loop. Digit count is computed in place
// instead of via a throwaway strconv.AppendInt(nil, ...) so the function
// allocates exactly once — the returned string itself.
//
//	dir := core.PathJoin(cfg.CheckpointDir, grpo.FormatStepDir(step))
func FormatStepDir(step int) string {
	const prefix = "step-"
	const padTo = 6
	// Allocate room for the prefix plus enough digits — 20 covers the max
	// int64 width.
	buf := make([]byte, 0, len(prefix)+20)
	buf = append(buf, prefix...)
	if step >= 0 && step < 100000 {
		// Hand-rolled zero-pad — strconv.Itoa lacks a Printf-style width
		// modifier, so for the typical sub-1e5 range we count leading
		// zeros ourselves. Above 1e5 strconv emits the full width
		// naturally.
		digits := 1
		for n := step / 10; n > 0; n /= 10 {
			digits++
		}
		for i := digits; i < padTo; i++ {
			buf = append(buf, '0')
		}
	}
	buf = strconv.AppendInt(buf, int64(step), 10)
	return string(buf)
}
