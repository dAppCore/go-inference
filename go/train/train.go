// SPDX-Licence-Identifier: EUPL-1.2

// Package train provides engine-agnostic supervised fine-tuning (SFT)
// checkpoint bookkeeping and a hooks-based SSD (self-distillation) code
// benchmark harness — shared by every inference driver (go-mlx, go-rocm,
// go-cpu, ...).
//
// It is the engine-agnostic half of what was previously go-mlx-only
// (go-mlx/go/train, ~2968 LOC across sft.go, sft_epoch.go, sft_batch.go,
// sft_checkpoint.go, score_cascade.go, dataset_stream.go, capture.go,
// val.go, ssd.go, ssd_eval.go): the checkpoint-sidecar shape, config
// normalisation, and the LiveCodeBench-style pass@k benchmark harness all
// operate on plain Go values and injected function hooks — never an MLX
// array, a live tokenizer, or a model handle — so they belong here where
// every driver can share them.
//
// The bulk of go-mlx's train package is deliberately NOT ported here — it
// is threaded through go-mlx's own *spine.Tokenizer, metal.LoRAAdapter,
// metal.AdamW, and mlx/dataset types, or through the lem-scorer
// (mlx/pkg/score — a large, separate, phonetics/cmudict-backed subsystem
// with no go-inference home of its own) — genuinely engine-bound:
//
//   - sft_epoch.go / sft_batch.go / dataset_stream.go: the LoRA epoch
//     loop, gradient accumulation, and sequence-packed batch building all
//     tokenize against a real *spine.Tokenizer and step a real
//     metal.LoRAAdapter / metal.AdamW.
//   - score_cascade.go: the eval-time quality cascade calls
//     mlx/pkg/score.ScorePair — that package carries no Metal/gemma4
//     dependency of its own, but porting an 11-file phonetics/cmudict
//     scorer is a separate, dedicated task, not a drive-by here.
//   - val.go / capture.go: validation-by-generation and generation
//     capture both operate on SFTEvalResult rows produced by the (not
//     ported) eval loop's own m.Generate calls — a capture/validation
//     helper with no epoch loop to call it would be dead code here.
//   - ssd.go: RunSSD's sampling pipeline embeds the full (engine-bound)
//     SFTConfig, drives model-specific LoRA normalisation
//     (profile.IsGemma4TargetArchitecture / gemma4.NormalizeLoRA), and
//     rides the same unported score cascade — engine-bound end to end.
//
// A driver's own (engine-bound) training loop calls this package's
// checkpoint helpers at its own checkpoint cadence:
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
var (
	errCheckpointPath   = core.NewError("mlx: SFT checkpoint metadata path is required")
	errCoreResultFailed = core.NewError("core result failed")
)

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

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return errCoreResultFailed
}
