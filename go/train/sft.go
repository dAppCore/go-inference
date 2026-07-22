// SPDX-Licence-Identifier: EUPL-1.2

// sft.go: the engine-neutral supervised LoRA fine-tuning loop. Ported from
// go-mlx/go/train (sft.go / sft_epoch.go), re-expressed against the neutral
// [engine.Trainer] seam (go/engine/trainer.go) rather than go-mlx's
// metal.LoRAAdapter + metal.AdamW + *spine.Tokenizer. The optimiser state and
// trainable weights never cross the boundary: the loop drives Trainer.Step /
// StepAccumulated (loss out as float64), checkpoints via the shared checkpoint
// metadata (train_checkpoint.go), evaluates via an injected generation hook,
// and persists the adapter with Trainer.Save.
//
// Scope note (the honest boundary, mirroring engine/metal's LoRATrainer): the
// only Trainer an engine implements today is the HEAD LoRA, which trains
// full-sequence causal LM (each token predicts its own next token). A
// LoRA.TargetKeys naming per-layer projections (q_proj, v_proj, …) is REFUSED
// at trainer open by the metal engine (#31) — never silently trained as
// head-only; leave TargetKeys empty (or ["lm_head"]) for the head. The
// inference.Batch LossMask IS honoured by the engine trainers (response-only
// masking: masked positions contribute zero loss and zero gradient, and the
// loss normaliser divides by the unmasked count).

package train

import (
	"context"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/eval"
	"dappco.re/go/inference/train/dataset"
)

// SFTConfig configures native supervised LoRA fine-tuning. It embeds the shared
// engine-agnostic [Config] (batch shape + checkpoint/eval cadence) and adds the
// adapter identity, save/merge/resume controls, and the score-cascade knobs.
type SFTConfig struct {
	Config
	// LoRA supplies the adapter rank/alpha handed to the engine trainer at
	// open. TargetKeys/BFloat16 are honoured by engines that support them; the
	// head-LoRA trainer reads rank/alpha and REFUSES TargetKeys it will not
	// honour (anything but "lm_head" — see engine/metal #31), so a per-layer
	// projection request fails at open instead of silently training the head.
	LoRA inference.LoRAConfig
	// SavePath is the final adapter destination (an adapter package dir).
	SavePath string
	// Merge asks the engine to fuse the adapter into the base after training,
	// where the engine supports it (the head-LoRA trainer does not — a note is
	// surfaced rather than an error).
	Merge bool
	// ScoreCascade arms the scorer over every eval pass (#50); requires
	// EvalEvery + EvalPrompts + a Score hook.
	ScoreCascade     bool
	ScoreSidecarPath string    // default <CheckpointDir>/score-cascade.jsonl
	ScoreWindow      int       // eval passes per windowed composite (default 3)
	Score            ScoreFunc // the scorer; nil disables the cascade
	// CaptureSidecarPath appends every eval generation as raw JSONL —
	// capture-first, score-independent. Default <CheckpointDir>/captures.jsonl.
	CaptureSidecarPath string
}

// SFTEvalResult records one eval prompt output captured during training.
type SFTEvalResult struct {
	Step   int
	Prompt string
	Text   string
}

// SFTMetrics is the JSON-friendly training summary for dashboards and probes.
type SFTMetrics struct {
	Steps              int     `json:"steps"`
	OptimizerSteps     int     `json:"optimizer_steps"`
	Epochs             int     `json:"epochs"`
	Samples            int     `json:"samples"`
	LastLoss           float64 `json:"last_loss"`
	LearningRate       float64 `json:"learning_rate"`
	BatchSize          int     `json:"batch_size"`
	EffectiveBatchSize int     `json:"effective_batch_size"`
	CheckpointCount    int     `json:"checkpoint_count"`
	EvaluationCount    int     `json:"evaluation_count"`
}

// SFTResult records the outcome of a native SFT LoRA run.
type SFTResult struct {
	Steps              int                  `json:"steps"`
	OptimizerSteps     int                  `json:"optimizer_steps"`
	Epochs             int                  `json:"epochs"`
	Samples            int                  `json:"samples"`
	LastLoss           float64              `json:"last_loss"`
	Losses             []float64            `json:"losses,omitempty"`
	Checkpoints        []string             `json:"checkpoints,omitempty"`
	CheckpointMetadata []CheckpointMetadata `json:"checkpoint_metadata,omitempty"`
	Evaluations        []SFTEvalResult      `json:"-"`
	AdapterPath        string               `json:"adapter_path,omitempty"`
	ResumePath         string               `json:"resume_path,omitempty"`
	// Score cascade (#50) — populated when SFTConfig.ScoreCascade is set with a
	// Score hook: every eval scored at generation time, best checkpoint by
	// windowed composite.
	ScoreRecords       []ScoreRecord `json:"-"`
	BestScoreStep      int           `json:"best_score_step,omitempty"`
	BestScoreComposite float64       `json:"best_score_composite,omitempty"`
	ScoreSidecarPath   string        `json:"score_sidecar_path,omitempty"`
}

// Metrics returns a stable JSON-friendly summary of an SFT run.
func (r *SFTResult) Metrics(cfg SFTConfig) SFTMetrics {
	cfg.Config = NormalizeConfig(cfg.Config)
	if r == nil {
		return SFTMetrics{
			LearningRate:       cfg.LearningRate,
			BatchSize:          cfg.BatchSize,
			EffectiveBatchSize: EffectiveBatchSize(cfg.Config),
		}
	}
	optimizerSteps := r.OptimizerSteps
	if optimizerSteps == 0 {
		optimizerSteps = r.Steps
	}
	return SFTMetrics{
		Steps:              r.Steps,
		OptimizerSteps:     optimizerSteps,
		Epochs:             r.Epochs,
		Samples:            r.Samples,
		LastLoss:           r.LastLoss,
		LearningRate:       cfg.LearningRate,
		BatchSize:          cfg.BatchSize,
		EffectiveBatchSize: EffectiveBatchSize(cfg.Config),
		CheckpointCount:    len(r.Checkpoints),
		EvaluationCount:    len(r.Evaluations),
	}
}

// EncodeFunc tokenises training text into model token ids — the tokeniser seam
// the SFT batch builder rides (the engine trainer consumes token ids, never
// text). RunSFTModel wires it from the model's own tokeniser.
type EncodeFunc func(text string) []int32

// GenerateFunc produces one eval generation for a prompt under a token budget
// and temperature — the eval seam the SFT loop rides (a model's own generation,
// injected so the loop stays engine-neutral).
type GenerateFunc func(ctx context.Context, prompt string, maxTokens int, temperature float32) (string, error)

// RunSFT runs the LoRA SFT epoch loop over ds against an opened engine
// [engine.Trainer]. encode tokenises each sample; gen (optional) drives the eval
// passes. The trainer owns the LoRA weights + optimiser; this loop only steps
// it, checkpoints, evaluates, and saves. RunSFTModel is the model-bound entry
// that opens the trainer and wires encode/gen from a loaded model.
func RunSFT(ctx context.Context, trainer engine.Trainer, encode EncodeFunc, gen GenerateFunc, ds inference.DatasetStream, cfg SFTConfig) (*SFTResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if trainer == nil {
		return nil, core.NewError("mlx: SFT trainer is nil")
	}
	if encode == nil {
		return nil, core.NewError("mlx: SFT encode hook is nil")
	}
	if ds == nil {
		return nil, core.NewError("mlx: SFT dataset is nil")
	}
	cfg.Config = NormalizeConfig(cfg.Config)

	result := &SFTResult{ResumePath: cfg.ResumePath}

	var cascade *scoreCascade
	if cfg.ScoreCascade && cfg.Score != nil && cfg.EvalEvery > 0 && len(cfg.EvalPrompts) > 0 {
		sidecar := cfg.ScoreSidecarPath
		if sidecar == "" && cfg.CheckpointDir != "" {
			sidecar = core.PathJoin(cfg.CheckpointDir, "score-cascade.jsonl")
		}
		cascade = newScoreCascade(sidecar, cfg.ScoreWindow, cfg.Score)
		result.ScoreSidecarPath = sidecar
	}

	capturePath := cfg.CaptureSidecarPath
	if capturePath == "" && cfg.CheckpointDir != "" {
		capturePath = core.PathJoin(cfg.CheckpointDir, "captures.jsonl")
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if epoch > 1 {
			resetter, ok := ds.(dataset.Resetter)
			if !ok {
				return result, core.NewError("mlx: SFT dataset must implement Reset for multiple epochs")
			}
			if err := resetter.Reset(); err != nil {
				return result, err
			}
		}
		if err := runSFTEpoch(ctx, trainer, encode, gen, ds, cfg, result, cascade, capturePath, epoch); err != nil {
			return result, err
		}
		result.Epochs = epoch
	}

	if result.Steps == 0 {
		return result, core.NewError("mlx: SFT dataset produced no trainable batches")
	}
	if cfg.SavePath != "" {
		if err := trainer.Save(cfg.SavePath); err != nil {
			return result, err
		}
		result.AdapterPath = cfg.SavePath
		meta := NewCheckpointMetadata(cfg.SavePath, cfg.Config, sftSnapshot(result), result.Epochs)
		if err := SaveCheckpointMetadata(cfg.SavePath, meta); err != nil {
			return result, err
		}
	}
	if cascade != nil {
		result.ScoreRecords = append([]ScoreRecord(nil), cascade.records...)
		if step, composite, ok := cascade.best(); ok {
			result.BestScoreStep = step
			result.BestScoreComposite = composite
		}
	}
	return result, nil
}

// runSFTEpoch streams the dataset once, gathering BatchSize-sized micro-batches
// and applying one optimiser update per GradientAccumulationSteps micro-batches.
func runSFTEpoch(ctx context.Context, trainer engine.Trainer, encode EncodeFunc, gen GenerateFunc, ds inference.DatasetStream, cfg SFTConfig, result *SFTResult, cascade *scoreCascade, capturePath string, epoch int) error {
	gradAccum := cfg.GradientAccumulationSteps
	if gradAccum <= 0 {
		gradAccum = 1
	}
	pending := make([]inference.Batch, 0, gradAccum)
	micro := make([][]int32, 0, cfg.BatchSize)

	flushMicro := func() {
		if len(micro) == 0 {
			return
		}
		pending = append(pending, inference.Batch{TokenIDs: micro})
		micro = make([][]int32, 0, cfg.BatchSize)
	}
	applyStep := func() error {
		if len(pending) == 0 {
			return nil
		}
		loss, err := trainer.StepAccumulated(pending)
		if err != nil {
			return err
		}
		pending = pending[:0]
		result.Steps++
		result.OptimizerSteps++
		result.LastLoss = loss
		result.Losses = append(result.Losses, loss)
		if cfg.CheckpointEvery > 0 && result.OptimizerSteps%cfg.CheckpointEvery == 0 {
			if err := checkpointSFT(trainer, cfg, result, epoch); err != nil {
				return err
			}
		}
		if cfg.EvalEvery > 0 && result.OptimizerSteps%cfg.EvalEvery == 0 {
			evalSFT(ctx, gen, cfg, result, cascade, capturePath, result.OptimizerSteps)
		}
		return nil
	}

	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		sample, ok, err := ds.Next()
		if err != nil {
			return err
		}
		if !ok {
			break
		}
		ids := sftTokenise(encode, sample, cfg.MaxSeqLen)
		if len(ids) < 2 {
			continue // a training sequence needs at least 2 tokens (causal target)
		}
		result.Samples++
		micro = append(micro, ids)
		if len(micro) >= cfg.BatchSize {
			flushMicro()
			if len(pending) >= gradAccum {
				if err := applyStep(); err != nil {
					return err
				}
			}
		}
	}
	// Drain the tail: any partial micro-batch + any accumulated micro-batches
	// below the gradient-accumulation threshold still make one final update.
	flushMicro()
	return applyStep()
}

// checkpointSFT saves the adapter under a step-NNNNNN directory and writes the
// portable checkpoint sidecar beside it.
func checkpointSFT(trainer engine.Trainer, cfg SFTConfig, result *SFTResult, epoch int) error {
	if cfg.CheckpointDir == "" {
		return nil
	}
	dir := core.PathJoin(cfg.CheckpointDir, FormatStepDir(result.OptimizerSteps))
	if err := trainer.Save(dir); err != nil {
		return err
	}
	meta := NewCheckpointMetadata(dir, cfg.Config, sftSnapshot(result), epoch)
	if err := SaveCheckpointMetadata(dir, meta); err != nil {
		return err
	}
	result.Checkpoints = append(result.Checkpoints, dir)
	result.CheckpointMetadata = append(result.CheckpointMetadata, meta)
	return nil
}

// evalSFT runs the fixed eval probes through the generation hook, captures each
// output, and feeds the score cascade. Best-effort: an eval failure is skipped
// (a probe that will not generate must not abort the training run).
func evalSFT(ctx context.Context, gen GenerateFunc, cfg SFTConfig, result *SFTResult, cascade *scoreCascade, capturePath string, step int) {
	if gen == nil || len(cfg.EvalPrompts) == 0 {
		return
	}
	evals := make([]SFTEvalResult, 0, len(cfg.EvalPrompts))
	for _, prompt := range cfg.EvalPrompts {
		text, err := gen(ctx, prompt, cfg.EvalMaxTokens, cfg.EvalTemperature)
		if err != nil {
			continue
		}
		evals = append(evals, SFTEvalResult{Step: step, Prompt: prompt, Text: text})
	}
	if len(evals) == 0 {
		return
	}
	result.Evaluations = append(result.Evaluations, evals...)
	appendCaptureRows(capturePath, evals)
	if cascade != nil {
		cascade.recordPass(step, evals)
	}
}

// sftSnapshot builds the checkpoint snapshot from the running result totals.
func sftSnapshot(result *SFTResult) CheckpointSnapshot {
	return CheckpointSnapshot{
		Step:          result.Steps,
		OptimizerStep: result.OptimizerSteps,
		Samples:       result.Samples,
		Loss:          result.LastLoss,
		Model:         eval.Info{},
	}
}

// sftTokenise renders a dataset sample to training text and tokenises it,
// truncating to maxSeqLen. The head-LoRA trainer learns full-sequence causal
// LM, so the whole prompt+response text is the sequence (no response mask).
func sftTokenise(encode EncodeFunc, sample dataset.Sample, maxSeqLen int) []int32 {
	ids := encode(sftSampleText(sample))
	if maxSeqLen > 0 && len(ids) > maxSeqLen {
		ids = ids[:maxSeqLen]
	}
	return ids
}

// sftSampleText renders a dataset sample into the training string: joined
// messages when present, else prompt+response, else the raw text.
func sftSampleText(sample dataset.Sample) string {
	if len(sample.Messages) > 0 {
		// Grow once to the exact rendered size, then write the "role: content\n"
		// parts directly — the previous `m.Role + ": " + m.Content + "\n"`
		// concatenation allocated a throwaway string per message before copying
		// it into the builder, on a per-sample-per-epoch path.
		size := 0
		for _, m := range sample.Messages {
			size += len(m.Role) + len(": ") + len(m.Content) + len("\n")
		}
		var text strings.Builder
		text.Grow(size)
		for _, m := range sample.Messages {
			text.WriteString(m.Role)
			text.WriteString(": ")
			text.WriteString(m.Content)
			text.WriteByte('\n')
		}
		return text.String()
	}
	if sample.Prompt != "" && sample.Response != "" {
		return core.Concat(sample.Prompt, "\n", sample.Response)
	}
	if sample.Response != "" {
		return sample.Response
	}
	return sample.Text
}
