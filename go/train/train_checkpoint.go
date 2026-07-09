// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	core "dappco.re/go"
	"dappco.re/go/inference/eval"
	"dappco.re/go/inference/train/checkpoint"
)

// CheckpointMetadataVersion is the current schema version stamped into
// every checkpoint sidecar written by SaveCheckpointMetadata.
const CheckpointMetadataVersion = 1

// CheckpointMetadata is the portable JSON sidecar for SFT checkpoints.
type CheckpointMetadata struct {
	Version                   int       `json:"version"`
	Path                      string    `json:"path"`
	ResumePath                string    `json:"resume_path,omitempty"`
	Step                      int       `json:"step"`
	OptimizerStep             int       `json:"optimizer_step"`
	Epoch                     int       `json:"epoch"`
	Samples                   int       `json:"samples"`
	Loss                      float64   `json:"loss"`
	ValLoss                   float64   `json:"val_loss,omitempty"`
	LearningRate              float64   `json:"learning_rate"`
	BatchSize                 int       `json:"batch_size"`
	GradientAccumulationSteps int       `json:"gradient_accumulation_steps"`
	EffectiveBatchSize        int       `json:"effective_batch_size"`
	MaxSeqLen                 int       `json:"max_seq_len,omitempty"`
	SequencePacking           bool      `json:"sequence_packing,omitempty"`
	EvalPrompts               []string  `json:"eval_prompts,omitempty"`
	EvalTemperature           float32   `json:"eval_temperature,omitempty"`
	ScoreComposite            float64   `json:"score_composite,omitempty"`
	Model                     eval.Info `json:"model"`
}

// CheckpointSnapshot carries the running totals a driver's own training
// loop has accumulated by the time it saves a checkpoint — the scalar
// counters NewCheckpointMetadata needs to stamp into the portable
// sidecar. go-mlx's equivalent (NewSFTCheckpointMetadata) read these
// straight off its own *SFTResult run-accumulator; that accumulator is
// orchestration-loop state (each driver runs its own epoch/step loop —
// see the package doc), so the caller passes just the handful of totals
// the sidecar actually records. ScoreComposite is whatever quality-cascade
// mechanism the driver has wired (go-mlx uses the lem-scorer's windowed
// composite; this package does not compute one).
type CheckpointSnapshot struct {
	Step           int
	OptimizerStep  int
	Samples        int
	Loss           float64
	ValLoss        float64
	ScoreComposite float64
	Model          eval.Info
}

// NewCheckpointMetadata captures reproducible SFT checkpoint state.
//
//	meta := train.NewCheckpointMetadata(path, cfg, snapshot, epoch)
func NewCheckpointMetadata(path string, cfg Config, snapshot CheckpointSnapshot, epoch int) CheckpointMetadata {
	cfg = NormalizeConfig(cfg)
	optimizerStep := snapshot.OptimizerStep
	if optimizerStep == 0 {
		optimizerStep = snapshot.Step
	}
	return CheckpointMetadata{
		Version:                   CheckpointMetadataVersion,
		Path:                      path,
		ResumePath:                cfg.ResumePath,
		Step:                      snapshot.Step,
		OptimizerStep:             optimizerStep,
		Epoch:                     epoch,
		Samples:                   snapshot.Samples,
		Loss:                      snapshot.Loss,
		ValLoss:                   snapshot.ValLoss,
		LearningRate:              cfg.LearningRate,
		BatchSize:                 cfg.BatchSize,
		GradientAccumulationSteps: cfg.GradientAccumulationSteps,
		EffectiveBatchSize:        EffectiveBatchSize(cfg),
		MaxSeqLen:                 cfg.MaxSeqLen,
		SequencePacking:           cfg.SequencePacking,
		EvalPrompts:               core.SliceClone(cfg.EvalPrompts),
		EvalTemperature:           cfg.EvalTemperature,
		ScoreComposite:            snapshot.ScoreComposite,
		Model:                     snapshot.Model,
	}
}

// SaveCheckpointMetadata writes checkpoint metadata beside an adapter
// package. The marshal-and-write mechanics are the shared checkpoint
// engine (go/checkpoint); only the Version/Path defaulting and the
// sidecar filename below are train's own.
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

// checkpointMetadataPath places the sidecar beside a single adapter file
// — SFT/LoRA adapters are commonly saved as one .safetensors file, unlike
// distill/grpo's directory-based checkpoints — or inside a checkpoint
// directory otherwise. Mirrors go-mlx's sftCheckpointMetadataPath.
func checkpointMetadataPath(path string) string {
	if core.HasSuffix(path, ".safetensors") {
		return core.PathJoin(core.PathDir(path), "train_checkpoint.json")
	}
	return core.PathJoin(path, "train_checkpoint.json")
}

// FormatStepDir builds the "step-NNNNNN" checkpoint dirname — delegates
// to the shared checkpoint engine (go/checkpoint) so train's copy of the
// zero-pad logic cannot drift from distill/grpo's.
//
//	dir := core.PathJoin(cfg.CheckpointDir, train.FormatStepDir(step))
func FormatStepDir(step int) string {
	return checkpoint.FormatStepDir(step)
}
