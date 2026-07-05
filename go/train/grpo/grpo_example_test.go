// SPDX-Licence-Identifier: EUPL-1.2

package grpo_test

import (
	"fmt"

	"dappco.re/go/inference/train/grpo"
)

// ExampleNormalizeConfig shows the defaults NormalizeConfig fills in for
// a zero-value Config.
func ExampleNormalizeConfig() {
	cfg := grpo.NormalizeConfig(grpo.Config{})
	fmt.Println("group size:", cfg.GroupSize)
	fmt.Println("epochs:", cfg.Epochs)
	fmt.Println("advantage epsilon:", cfg.AdvantageEpsilon)
	// Output:
	// group size: 4
	// epochs: 1
	// advantage epsilon: 1e-08
}

// ExampleScoreRollout scores one rollout against a small set of reward
// funcs, summing their weighted contributions.
func ExampleScoreRollout() {
	ctx := &grpo.RewardContext{
		Sample:  grpo.Sample{ExpectedAnswer: "42"},
		Rollout: grpo.Rollout{Answer: "the answer is 42"},
	}
	funcs := []grpo.RewardFunc{grpo.RewardContainsAnswer(1), grpo.RewardExactAnswer(1)}

	parts, total, err := grpo.ScoreRollout(ctx, funcs, nil)
	if err != nil {
		panic(err)
	}
	fmt.Println("parts:", len(parts))
	fmt.Println("total:", total)
	// Output:
	// parts: 2
	// total: 1
}

// ExampleRewardStats computes the group-relative mean and population
// standard deviation of a set of scored rollouts.
func ExampleRewardStats() {
	rollouts := []grpo.Rollout{{Reward: 1}, {Reward: 3}}
	mean, std := grpo.RewardStats(rollouts)
	fmt.Println("mean:", mean)
	fmt.Println("std:", std)
	// Output:
	// mean: 2
	// std: 1
}

// ExampleBuildUpdate shows the per-step call a driver's own training loop
// makes once its engine has populated each rollout's LogProb (and
// ReferenceLogProb/KL, when a KL penalty applies).
func ExampleBuildUpdate() {
	cfg := grpo.Config{GroupSize: 2, RewardFuncs: []grpo.RewardFunc{grpo.RewardExactAnswer(1)}}
	sample := grpo.Sample{ExpectedAnswer: "paris"}
	rollouts := []grpo.Rollout{
		{Answer: "paris", LogProb: -0.1},
		{Answer: "london", LogProb: -0.2},
	}

	update, err := grpo.BuildUpdate(1, 1, sample, rollouts, cfg)
	if err != nil {
		panic(err)
	}
	fmt.Println("reward mean:", update.RewardMean)
	fmt.Println("rollouts:", len(update.Rollouts))
	// Output:
	// reward mean: 0.5
	// rollouts: 2
}
