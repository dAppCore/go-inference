// SPDX-Licence-Identifier: EUPL-1.2

// Package grpo provides engine-agnostic Group Relative Policy
// Optimisation (GRPO) primitives — reward scoring, group-relative
// advantage normalisation, KL-penalised loss aggregation, SFT-sample
// extraction for reasoning prompts, and checkpoint bookkeeping — shared
// by every inference driver (go-mlx, go-rocm, go-cpu, ...).
//
// It is the engine-agnostic half of what was previously go-mlx-only
// (go-mlx/go/grpo): the reward functions, the group-relative
// advantage/loss maths, and the checkpoint metadata all operate on a
// plain Rollout struct and []float64 arithmetic — never an MLX array or
// a live tokenizer — so they belong here where every driver can share
// them.
//
// The per-step training loop that go-mlx's grpo package also hosts
// (iterating epochs, sampling a group of completions from the policy,
// scoring the reference log-prob, and applying the policy-gradient
// update) is deliberately NOT ported: it is threaded through go-mlx's
// own GRPORunner, which drives a real rollout/generation pass and a
// real backward pass against dappco.re/go/mlx arrays and a live
// tokenizer — genuinely engine-bound. That loop stays engine-side; a
// driver wires its own loop to call ScoreRollout / BuildUpdate per step
// and NewCheckpointMetadata / SaveCheckpointMetadata at its own
// checkpoint cadence:
//
//	rollouts := ... // engine-owned generation, one per group member
//	for i := range rollouts {
//	    // rollouts[i].LogProb set by the engine's own generation pass;
//	    // ReferenceLogProb/KL set by the engine's own reference pass
//	    // when cfg.KLCoefficient != 0.
//	}
//	update, err := grpo.BuildUpdate(step, epoch, sample, rollouts, cfg)
//	if err != nil {
//	    return err
//	}
//	// the driver applies update.Loss to its optimiser step here.
package grpo

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sentinel errors hoisted to package vars — each previously allocated a
// fresh core.NewError on the (rare but hot under churn) failure path.
var (
	errNoRollouts       = core.NewError("grpo: rollout group is empty")
	errRewardNotFinite  = core.NewError("grpo: reward is not finite")
	errLossNotFinite    = core.NewError("grpo: loss is not finite")
	errCheckpointPath   = core.NewError("grpo: checkpoint metadata path is required")
	errCoreResultFailed = core.NewError("core result failed")
)

// Sample is a reasoning prompt extracted from an SFT/JSONL sample.
type Sample struct {
	Prompt          string            `json:"prompt"`
	ReferenceAnswer string            `json:"reference_answer,omitempty"`
	ExpectedAnswer  string            `json:"expected_answer,omitempty"`
	Reasoning       string            `json:"reasoning,omitempty"`
	Meta            map[string]string `json:"meta,omitempty"`
}

// Rollout is one sampled reasoning completion plus training annotations.
//
// Text, Reasoning, Answer, TokenIDs, and LogProb are populated by the
// engine's own generation pass. ReferenceLogProb and KL are populated by
// the engine's own reference-log-prob call whenever a KL penalty applies
// (cfg.KLCoefficient != 0) — this package computes no forward or
// backward pass itself. Reward, RewardParts, Advantage, and
// LossContribution are computed by ScoreRollout / BuildUpdate.
type Rollout struct {
	Text             string   `json:"text,omitempty"`
	Reasoning        string   `json:"reasoning,omitempty"`
	Answer           string   `json:"answer,omitempty"`
	TokenIDs         []int32  `json:"token_ids,omitempty"`
	LogProb          float64  `json:"log_prob,omitempty"`
	ReferenceLogProb float64  `json:"reference_log_prob,omitempty"`
	Reward           float64  `json:"reward,omitempty"`
	RewardParts      []Reward `json:"reward_parts,omitempty"`
	Advantage        float64  `json:"advantage,omitempty"`
	KL               float64  `json:"kl,omitempty"`
	LossContribution float64  `json:"loss_contribution,omitempty"`
}

// Reward is one named reward contribution.
type Reward struct {
	Name   string  `json:"name"`
	Score  float64 `json:"score"`
	Weight float64 `json:"weight,omitempty"`
	Detail string  `json:"detail,omitempty"`
}

// RewardContext is passed to reward functions.
type RewardContext struct {
	Sample  Sample
	Rollout Rollout
	Index   int
}

// RewardFunc scores one rollout.
type RewardFunc func(RewardContext) (Reward, error)

// Update is the grouped policy update computed for one GRPO training
// step. Rollouts is a defensive snapshot detached from any engine-owned
// buffers — see BuildUpdate.
type Update struct {
	Step          int       `json:"step"`
	Epoch         int       `json:"epoch"`
	Sample        Sample    `json:"sample"`
	Rollouts      []Rollout `json:"rollouts"`
	RewardMean    float64   `json:"reward_mean"`
	RewardStd     float64   `json:"reward_std"`
	KLMean        float64   `json:"kl_mean,omitempty"`
	Loss          float64   `json:"loss"`
	KLCoefficient float64   `json:"kl_coefficient,omitempty"`
}

// Config controls Group Relative Policy Optimisation. BuildUpdate itself
// only consumes GroupSize, KLCoefficient, AdvantageEpsilon, and
// RewardFuncs; the remaining fields are the shared vocabulary a driver's
// own training loop reads to drive its checkpoint/eval cadence and probe
// emission — mirroring go-mlx's GRPOConfig, which bundled loss config and
// loop cadence into one struct rather than splitting them.
type Config struct {
	GroupSize        int                 `json:"group_size,omitempty"`
	Epochs           int                 `json:"epochs,omitempty"`
	KLCoefficient    float64             `json:"kl_coefficient,omitempty"`
	AdvantageEpsilon float64             `json:"advantage_epsilon,omitempty"`
	LearningRate     float64             `json:"learning_rate,omitempty"`
	CheckpointDir    string              `json:"checkpoint_dir,omitempty"`
	CheckpointEvery  int                 `json:"checkpoint_every,omitempty"`
	EvalEvery        int                 `json:"eval_every,omitempty"`
	ResumePath       string              `json:"resume_path,omitempty"`
	MaxSamples       int                 `json:"max_samples,omitempty"`
	RewardFuncs      []RewardFunc        `json:"-"`
	ProbeSink        inference.ProbeSink `json:"-"`
}

// NormalizeConfig fills Config defaults: GroupSize floors to 4, Epochs
// floors to 1, AdvantageEpsilon floors to 1e-8. A driver's own training
// loop should call this once per run, exactly as BuildUpdate does
// internally for every call.
func NormalizeConfig(cfg Config) Config {
	if cfg.GroupSize <= 0 {
		cfg.GroupSize = 4
	}
	if cfg.Epochs <= 0 {
		cfg.Epochs = 1
	}
	if cfg.AdvantageEpsilon <= 0 {
		cfg.AdvantageEpsilon = 1e-8
	}
	return cfg
}

// BuildUpdate scores rollouts, computes group-relative advantages, and
// aggregates the KL-penalised loss for one GRPO step.
//
// Rollouts must already carry LogProb and, when a KL penalty applies
// (cfg.KLCoefficient != 0), ReferenceLogProb/KL — populated by the
// engine's own generation and reference-log-prob calls. This function
// performs no generation, no forward/backward pass, and no tokenizer
// access: only reward scoring (see ScoreRollout) and the plain-float
// group-relative aggregation. It mutates each rollout's Reward,
// RewardParts, Advantage, and LossContribution in place; Update.Rollouts
// is a separate, detached snapshot (see snapshotRollouts).
//
//	update, err := grpo.BuildUpdate(step, epoch, sample, rollouts, cfg)
func BuildUpdate(step, epoch int, sample Sample, rollouts []Rollout, cfg Config) (Update, error) {
	cfg = NormalizeConfig(cfg)
	if len(rollouts) == 0 {
		return Update{}, errNoRollouts
	}
	if len(rollouts) != cfg.GroupSize {
		return Update{}, core.NewError(core.Sprintf("grpo: rollout group size mismatch: got %d want %d", len(rollouts), cfg.GroupSize))
	}
	rewardFuncs := cfg.RewardFuncs
	if len(rewardFuncs) == 0 {
		// Default reward funcs slice is shared package-wide — the closure
		// has no per-call state (weight=1 is captured at init), so callers
		// on the default-config path don't pay a fresh closure + slice
		// allocation on every step.
		rewardFuncs = defaultRewardFuncs
	}
	n := len(rollouts)
	rewardCtx := RewardContext{Sample: sample}
	// Pre-allocate one shared []Reward backing for all rollouts' parts in
	// this step; ScoreRollout carves a per-rollout view out of it instead
	// of paying its own make per call.
	partsBacking := make([]Reward, 0, n*len(rewardFuncs))
	for i := range rollouts {
		rewardCtx.Rollout = rollouts[i]
		rewardCtx.Index = i
		start := len(partsBacking)
		filled, total, err := ScoreRollout(&rewardCtx, rewardFuncs, partsBacking)
		if err != nil {
			return Update{}, err
		}
		partsBacking = filled
		end := len(partsBacking)
		rollouts[i].RewardParts = partsBacking[start:end:end]
		rollouts[i].Reward = total
	}

	rewardMean, rewardStd := RewardStats(rollouts)
	advEps := cfg.AdvantageEpsilon
	klCoef := cfg.KLCoefficient
	// Single std-vs-eps branch outside the inner loop — when rewardStd is
	// at or below advEps every rollout's advantage is zero, so the
	// (reward-mean)/std arithmetic can be skipped entirely.
	invStd := 0.0
	useStd := rewardStd > advEps
	if useStd {
		invStd = 1.0 / rewardStd
	}
	var loss, klSum float64
	for i := range rollouts {
		if useStd {
			rollouts[i].Advantage = (rollouts[i].Reward - rewardMean) * invStd
		} else {
			rollouts[i].Advantage = 0
		}
		rollouts[i].LossContribution = -rollouts[i].Advantage*rollouts[i].LogProb + klCoef*rollouts[i].KL
		loss += rollouts[i].LossContribution
		klSum += rollouts[i].KL
	}
	invN := 1.0 / float64(n)
	loss *= invN
	klMean := klSum * invN
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return Update{}, errLossNotFinite
	}
	return Update{
		Step:          step,
		Epoch:         epoch,
		Sample:        sample,
		Rollouts:      snapshotRollouts(rollouts),
		RewardMean:    rewardMean,
		RewardStd:     rewardStd,
		KLMean:        klMean,
		Loss:          loss,
		KLCoefficient: cfg.KLCoefficient,
	}, nil
}

// ScoreRollout walks every reward func against ctx and appends a Reward
// per non-nil func into out, returning the grown slice and the summed
// score. Callers scoring a whole group share one backing slice (out)
// across rollouts and carve a per-rollout view at known offsets, so a
// single n*len(funcs) allocation replaces N per-rollout allocations —
// see BuildUpdate.
//
//	parts, total, err := grpo.ScoreRollout(&ctx, funcs, nil)
func ScoreRollout(ctx *RewardContext, funcs []RewardFunc, out []Reward) ([]Reward, float64, error) {
	var total float64
	for _, fn := range funcs {
		if fn == nil {
			continue
		}
		reward, err := fn(*ctx)
		if err != nil {
			return out, 0, err
		}
		if reward.Name == "" {
			reward.Name = "reward"
		}
		if math.IsNaN(reward.Score) || math.IsInf(reward.Score, 0) {
			return out, 0, errRewardNotFinite
		}
		out = append(out, reward)
		total += reward.Score
	}
	return out, total, nil
}

// RewardStats returns the mean and population standard deviation of
// rollouts' Reward values.
//
//	mean, std := grpo.RewardStats(rollouts)
func RewardStats(rollouts []Rollout) (float64, float64) {
	n := len(rollouts)
	if n == 0 {
		return 0, 0
	}
	// Index iteration — range over []Rollout copies the whole struct
	// (Text/Reasoning/Answer strings, TokenIDs + RewardParts slice
	// headers, all the float fields) on each iteration even though we
	// only ever read the Reward float. Indexing skips the copy.
	var sum float64
	for i := range rollouts {
		sum += rollouts[i].Reward
	}
	invN := 1.0 / float64(n)
	mean := sum * invN
	var variance float64
	for i := range rollouts {
		delta := rollouts[i].Reward - mean
		variance += delta * delta
	}
	variance *= invN
	return mean, math.Sqrt(variance)
}

// snapshotRollouts returns a defensive copy of rollouts for
// Update.Rollouts. TokenIDs is deep-copied into a shared flat backing —
// those slices may be engine-owned (returned by a generation call that
// reuses its own buffers across rollouts), so the update must detach
// from that memory. RewardParts is adopted as-is: BuildUpdate already
// carved each rollout's view out of a step-local backing that is private
// to this call and lives exactly as long as the returned Update needs
// it. Ported from go-mlx's snapshotGRPORollouts.
func snapshotRollouts(rollouts []Rollout) []Rollout {
	out := make([]Rollout, len(rollouts))
	// Bulk struct copy first — copy() lowers to memmove. This already
	// carries each rollout's RewardParts slice header across, so adopting
	// the views is the default; we only override TokenIDs below.
	copy(out, rollouts)
	var totalTokens int
	for i := range rollouts {
		totalTokens += len(rollouts[i].TokenIDs)
	}
	var tokenBacking []int32
	if totalTokens > 0 {
		tokenBacking = make([]int32, totalTokens)
	}
	var tokenCursor int
	for i := range rollouts {
		if src := rollouts[i].TokenIDs; len(src) > 0 {
			next := tokenCursor + len(src)
			dst := tokenBacking[tokenCursor:next:next]
			copy(dst, src)
			out[i].TokenIDs = dst
			tokenCursor = next
		} else {
			out[i].TokenIDs = nil
		}
		// Normalise a zero-length non-nil view to nil for a stable,
		// comparable result.
		if len(out[i].RewardParts) == 0 {
			out[i].RewardParts = nil
		}
	}
	return out
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
