// SPDX-Licence-Identifier: EUPL-1.2

// Package train provides the engine-neutral LEM training orchestration —
// the SSD (self-distillation) sampling pipeline, the LoRA supervised
// fine-tuning (SFT) loop, checkpoint bookkeeping, the score cascade, and
// the hooks-based SSD code benchmark harness — shared by every inference
// driver (go-mlx, go-rocm, go-cpu, ...).
//
// It is the port of go-mlx's train orchestration onto the engine-neutral
// seams (#270): what was previously threaded through go-mlx's own
// *spine.Tokenizer, metal.LoRAAdapter, and metal.AdamW is re-expressed
// against the neutral primitives:
//
//   - SSD (ssd.go / ssd_model.go / capture.go): sampling rides
//     inference.TextModel.Generate and the neutral inference.DatasetStream;
//     the kernel prefix rides the inference.PromptCacheWarmer capability.
//     SSD never trains — it captures + scores each self-output at birth
//     and STOPS at the trace.
//   - SFT (sft.go / sft_model.go): the LoRA epoch loop drives the neutral
//     [engine.Trainer] seam (Step/StepAccumulated/Loss/Save) instead of a
//     metal adapter + optimiser — the trainable weights + optimiser state
//     never cross the boundary, only the on-disk adapter does.
//   - score_cascade.go: the quality cascade rides an injected [ScoreFunc]
//     hook rather than go-mlx's concrete lem-scorer (mlx/pkg/score, an
//     11-file phonetics/cmudict subsystem with no go-inference home yet) —
//     the cascade machinery (windowed composite, sidecar, best checkpoint)
//     is scorer-neutral; a driver that has a scorer supplies it.
//
// The Model-bound entries (RunSSDModel / RunSFTModel) and the cmd-facing
// runners (RunSSDCommand / RunSFTCommand) live here rather than in package
// inference: the pipeline lives here and package inference cannot import
// train or engine (both import inference — the reverse would cycle), so the
// functions that wire a loaded model into the pipeline are train-package
// functions. cmd/lem {ssd,sft} are thin flag-parsing over the runners.
//
// A driver's own training loop can still call this package's checkpoint
// helpers directly at its own checkpoint cadence:
//
//	meta := train.NewCheckpointMetadata(path, cfg, snapshot, epoch)
//	if err := train.SaveCheckpointMetadata(path, meta); err != nil {
//	    return err
//	}
//
// and, for an SSD recipe's evaluation phase, drives the code benchmark
// with its own generation + test-execution hooks — no model, no Metal,
// and no Python sandbox required by this package itself:
//
//	report, err := train.RunSSDCodeBenchmark(ctx, train.SSDCodeBenchmarkRunner{
//	    Generate: engineGenerate,
//	    RunTests: engineRunTests,
//	}, samples, train.DefaultSSDCodeBenchmarkConfig())
package train

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sentinel errors hoisted to package vars — each previously allocated a
// fresh core.NewError on the (rare but hot under churn) failure path.
var errCheckpointPath = core.NewError("mlx: SFT checkpoint metadata path is required")

// Config is the checkpoint- and schedule-relevant engine-agnostic subset
// of go-mlx's SFTConfig. The LoRA adapter config (spine.LoRAConfig) and
// AdamW optimizer config (metal.AdamWConfig) that SFTConfig also carried
// are engine types with no portable equivalent, so a driver keeps those
// on its own engine-side config and reads this Config for the batch
// shape, checkpoint/eval cadence, and probe emission every driver shares
// — mirroring go-mlx's SFTConfig, which bundled batch shape and loop
// cadence into one struct rather than splitting them.
type Config struct {
	BatchSize                 int                 `json:"batch_size,omitempty"`
	GradientAccumulationSteps int                 `json:"gradient_accumulation_steps,omitempty"`
	Epochs                    int                 `json:"epochs,omitempty"`
	LearningRate              float64             `json:"learning_rate,omitempty"`
	MaxSeqLen                 int                 `json:"max_seq_len,omitempty"`
	SequencePacking           bool                `json:"sequence_packing,omitempty"`
	NoEOS                     bool                `json:"no_eos,omitempty"`
	CheckpointDir             string              `json:"checkpoint_dir,omitempty"`
	CheckpointEvery           int                 `json:"checkpoint_every,omitempty"`
	EvalEvery                 int                 `json:"eval_every,omitempty"`
	EvalPrompts               []string            `json:"eval_prompts,omitempty"`
	EvalMaxTokens             int                 `json:"eval_max_tokens,omitempty"`
	EvalTemperature           float32             `json:"eval_temperature,omitempty"`
	ResumePath                string              `json:"resume_path,omitempty"`
	ProbeSink                 inference.ProbeSink `json:"-"`
}

// NormalizeConfig fills Config defaults: BatchSize and
// GradientAccumulationSteps floor to 1, Epochs floors to 1, LearningRate
// defaults to 1e-5, and EvalMaxTokens defaults to 96. A driver's own
// training loop should call this once per run, exactly as
// NewCheckpointMetadata does internally for every call.
func NormalizeConfig(cfg Config) Config {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1
	}
	if cfg.GradientAccumulationSteps <= 0 {
		cfg.GradientAccumulationSteps = 1
	}
	if cfg.Epochs <= 0 {
		cfg.Epochs = 1
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 1e-5
	}
	if cfg.EvalMaxTokens <= 0 {
		cfg.EvalMaxTokens = 96
	}
	return cfg
}

// EffectiveBatchSize returns the optimizer batch size after gradient
// accumulation.
func EffectiveBatchSize(cfg Config) int {
	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	gradAccum := cfg.GradientAccumulationSteps
	if gradAccum <= 0 {
		gradAccum = 1
	}
	return batchSize * gradAccum
}
