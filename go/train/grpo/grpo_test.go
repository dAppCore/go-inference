// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"math"
	"testing"
)

// --- NormalizeConfig ---

// Good: a zero-value Config gets the documented defaults — group size 4,
// 1 epoch, advantage epsilon 1e-8.
func TestNormalizeConfig_Good(t *testing.T) {
	got := NormalizeConfig(Config{})
	if got.GroupSize != 4 {
		t.Errorf("GroupSize = %d, want 4", got.GroupSize)
	}
	if got.Epochs != 1 {
		t.Errorf("Epochs = %d, want 1", got.Epochs)
	}
	if got.AdvantageEpsilon != 1e-8 {
		t.Errorf("AdvantageEpsilon = %v, want 1e-8", got.AdvantageEpsilon)
	}
}

// Bad: negative inputs are floored to the same defaults as the zero
// value, rather than left negative or panicking.
func TestNormalizeConfig_Bad(t *testing.T) {
	got := NormalizeConfig(Config{GroupSize: -5, AdvantageEpsilon: -1})
	if got.GroupSize != 4 {
		t.Errorf("GroupSize = %d, want floored to 4", got.GroupSize)
	}
	if got.AdvantageEpsilon != 1e-8 {
		t.Errorf("AdvantageEpsilon = %v, want floored to 1e-8", got.AdvantageEpsilon)
	}
}

// Ugly: already-populated fields survive normalisation unchanged —
// NormalizeConfig only fills in defaults, it never overwrites explicit
// caller values.
func TestNormalizeConfig_Ugly(t *testing.T) {
	cfg := Config{GroupSize: 8, Epochs: 3, AdvantageEpsilon: 0.5, KLCoefficient: 0.1}
	got := NormalizeConfig(cfg)
	if got.GroupSize != 8 || got.Epochs != 3 || got.AdvantageEpsilon != 0.5 || got.KLCoefficient != 0.1 {
		t.Fatalf("NormalizeConfig() mutated explicit fields: got %+v", got)
	}
}

// --- ScoreRollout ---

// Good: every non-nil func contributes a part and its score to the total.
func TestScoreRollout_Good(t *testing.T) {
	funcs := []RewardFunc{
		func(RewardContext) (Reward, error) { return Reward{Name: "a", Score: 1}, nil },
		func(RewardContext) (Reward, error) { return Reward{Name: "b", Score: 2}, nil },
	}
	ctx := &RewardContext{}
	parts, total, err := ScoreRollout(ctx, funcs, nil)
	if err != nil {
		t.Fatalf("ScoreRollout() error = %v", err)
	}
	if total != 3 {
		t.Fatalf("total = %v, want 3", total)
	}
	if len(parts) != 2 || parts[0].Name != "a" || parts[1].Name != "b" {
		t.Fatalf("parts = %+v, want [a b]", parts)
	}
}

// Bad: a non-finite score from a reward func is rejected rather than
// silently propagated.
func TestScoreRollout_Bad(t *testing.T) {
	funcs := []RewardFunc{
		func(RewardContext) (Reward, error) { return Reward{Score: math.NaN()}, nil },
	}
	if _, _, err := ScoreRollout(&RewardContext{}, funcs, nil); err != errRewardNotFinite {
		t.Fatalf("ScoreRollout() error = %v, want errRewardNotFinite", err)
	}
}

// Ugly: nil funcs are skipped, an unnamed reward defaults to "reward",
// and passing a shared backing slice appends onto it rather than
// replacing it — the pattern BuildUpdate relies on to carve per-rollout
// views out of one allocation.
func TestScoreRollout_Ugly(t *testing.T) {
	funcs := []RewardFunc{
		nil,
		func(RewardContext) (Reward, error) { return Reward{Score: 5}, nil },
	}
	shared := make([]Reward, 0, 4)
	shared = append(shared, Reward{Name: "prior"})
	parts, total, err := ScoreRollout(&RewardContext{}, funcs, shared)
	if err != nil {
		t.Fatalf("ScoreRollout() error = %v", err)
	}
	if total != 5 {
		t.Fatalf("total = %v, want 5", total)
	}
	if len(parts) != 2 || parts[0].Name != "prior" || parts[1].Name != "reward" {
		t.Fatalf("parts = %+v, want [prior reward] (unnamed reward defaults to \"reward\")", parts)
	}
}

// --- RewardStats ---

// Good: a known population of rewards produces the exact mean and
// population standard deviation.
func TestRewardStats_Good(t *testing.T) {
	rollouts := []Rollout{{Reward: 2}, {Reward: 4}, {Reward: 6}}
	mean, std := RewardStats(rollouts)
	if mean != 4 {
		t.Errorf("mean = %v, want 4", mean)
	}
	wantStd := math.Sqrt(8.0 / 3.0)
	if math.Abs(std-wantStd) > 1e-9 {
		t.Errorf("std = %v, want %v", std, wantStd)
	}
}

// Bad: an empty rollout slice returns (0, 0) rather than dividing by zero.
func TestRewardStats_Bad(t *testing.T) {
	mean, std := RewardStats(nil)
	if mean != 0 || std != 0 {
		t.Fatalf("RewardStats(nil) = %v,%v, want 0,0", mean, std)
	}
}

// Ugly: identical rewards produce a standard deviation within float
// rounding of zero (not a division-by-zero NaN or a materially nonzero
// value) regardless of how many rollouts share the value.
func TestRewardStats_Ugly(t *testing.T) {
	rollouts := []Rollout{{Reward: 3}, {Reward: 3}, {Reward: 3}}
	mean, std := RewardStats(rollouts)
	if mean != 3 {
		t.Fatalf("RewardStats(identical) mean = %v, want 3", mean)
	}
	if math.Abs(std) > 1e-9 {
		t.Fatalf("RewardStats(identical) std = %v, want ~0", std)
	}
}

// --- BuildUpdate ---

// Good: rewards, group-relative advantages, and the KL-penalised loss are
// computed correctly from pre-annotated rollouts.
func TestBuildUpdate_Good(t *testing.T) {
	byIndex := RewardFunc(func(ctx RewardContext) (Reward, error) {
		if ctx.Index == 0 {
			return Reward{Name: "idx", Score: 1}, nil
		}
		return Reward{Name: "idx", Score: 3}, nil
	})
	cfg := Config{GroupSize: 2, KLCoefficient: 0.1, RewardFuncs: []RewardFunc{byIndex}}
	rollouts := []Rollout{
		{LogProb: -1, KL: 0.2},
		{LogProb: -2, KL: 0.4},
	}
	sample := Sample{Prompt: "p"}

	update, err := BuildUpdate(5, 2, sample, rollouts, cfg)
	if err != nil {
		t.Fatalf("BuildUpdate() error = %v", err)
	}
	if update.Step != 5 || update.Epoch != 2 || update.Sample.Prompt != "p" {
		t.Fatalf("Step/Epoch/Sample = %d/%d/%q, want 5/2/p", update.Step, update.Epoch, update.Sample.Prompt)
	}
	if update.RewardMean != 2 || update.RewardStd != 1 {
		t.Fatalf("RewardMean/RewardStd = %v/%v, want 2/1", update.RewardMean, update.RewardStd)
	}
	if math.Abs(update.KLMean-0.3) > 1e-9 {
		t.Fatalf("KLMean = %v, want 0.3", update.KLMean)
	}
	wantLoss := 0.53
	if math.Abs(update.Loss-wantLoss) > 1e-9 {
		t.Fatalf("Loss = %v, want %v", update.Loss, wantLoss)
	}
	if update.KLCoefficient != 0.1 {
		t.Fatalf("KLCoefficient = %v, want 0.1", update.KLCoefficient)
	}
	if len(update.Rollouts) != 2 || update.Rollouts[0].Advantage != -1 || update.Rollouts[1].Advantage != 1 {
		t.Fatalf("Rollouts advantages = %+v, want [-1 1]", update.Rollouts)
	}
	// The caller's own slice is mutated in place too.
	if rollouts[0].Reward != 1 || rollouts[1].Reward != 3 {
		t.Fatalf("input rollouts not annotated in place: %+v", rollouts)
	}
}

// Bad: an empty group, a group-size mismatch, a non-finite reward, and a
// non-finite loss are all rejected with an error rather than a panic or
// a silently poisoned result.
func TestBuildUpdate_Bad(t *testing.T) {
	if _, err := BuildUpdate(1, 1, Sample{}, nil, Config{}); err != errNoRollouts {
		t.Errorf("BuildUpdate(no rollouts) error = %v, want errNoRollouts", err)
	}

	mismatched := []Rollout{{}, {}}
	if _, err := BuildUpdate(1, 1, Sample{}, mismatched, Config{GroupSize: 4}); err == nil {
		t.Error("BuildUpdate(group size mismatch): expected error, got nil")
	}

	badReward := Config{RewardFuncs: []RewardFunc{
		func(RewardContext) (Reward, error) { return Reward{Score: math.Inf(1)}, nil },
	}, GroupSize: 1}
	if _, err := BuildUpdate(1, 1, Sample{}, []Rollout{{}}, badReward); err != errRewardNotFinite {
		t.Errorf("BuildUpdate(non-finite reward) error = %v, want errRewardNotFinite", err)
	}

	infLogProb := Config{GroupSize: 1, RewardFuncs: []RewardFunc{
		func(RewardContext) (Reward, error) { return Reward{Score: 1}, nil },
	}}
	rollouts := []Rollout{{LogProb: math.Inf(1)}}
	if _, err := BuildUpdate(1, 1, Sample{}, rollouts, infLogProb); err != errLossNotFinite {
		t.Errorf("BuildUpdate(non-finite loss) error = %v, want errLossNotFinite", err)
	}
}

// Ugly: when every rollout in the group scores the same reward, the
// population std is 0 and every advantage is exactly 0 (the
// AdvantageEpsilon guard), not a division-by-zero NaN.
func TestBuildUpdate_Ugly(t *testing.T) {
	sameReward := RewardFunc(func(RewardContext) (Reward, error) { return Reward{Score: 7}, nil })
	cfg := Config{RewardFuncs: []RewardFunc{sameReward}} // GroupSize normalises to 4
	rollouts := make([]Rollout, 4)

	update, err := BuildUpdate(1, 1, Sample{}, rollouts, cfg)
	if err != nil {
		t.Fatalf("BuildUpdate() error = %v", err)
	}
	if update.RewardStd != 0 {
		t.Fatalf("RewardStd = %v, want 0 for identical rewards", update.RewardStd)
	}
	for i, r := range update.Rollouts {
		if r.Advantage != 0 {
			t.Errorf("Rollouts[%d].Advantage = %v, want 0", i, r.Advantage)
		}
	}
}
