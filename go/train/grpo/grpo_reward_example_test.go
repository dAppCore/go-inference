// SPDX-Licence-Identifier: EUPL-1.2

package grpo_test

import (
	"fmt"

	"dappco.re/go/inference/train/dataset"
	"dappco.re/go/inference/train/grpo"
)

// ExampleSampleFromSFT extracts a reasoning prompt, reference answer, and
// expected answer from a raw dataset sample.
func ExampleSampleFromSFT() {
	raw := dataset.Sample{Prompt: "What is the capital of France?", Response: "Paris"}
	sample := grpo.SampleFromSFT(raw)
	fmt.Println("prompt:", sample.Prompt)
	fmt.Println("expected:", sample.ExpectedAnswer)
	// Output:
	// prompt: What is the capital of France?
	// expected: Paris
}

// ExampleExtractExpectedAnswer pulls the answer target out of a
// reasoning-style response, stripping a trailing "Answer:" prefix.
func ExampleExtractExpectedAnswer() {
	answer := grpo.ExtractExpectedAnswer(dataset.Sample{Response: "Work it out step by step.\nAnswer: 42"})
	fmt.Println(answer)
	// Output:
	// 42
}

// ExampleRewardContainsAnswer scores a rollout that mentions the expected
// answer anywhere in its answer, text, or reasoning fields.
func ExampleRewardContainsAnswer() {
	fn := grpo.RewardContainsAnswer(1)
	reward, err := fn(grpo.RewardContext{
		Sample:  grpo.Sample{ExpectedAnswer: "42"},
		Rollout: grpo.Rollout{Answer: "The answer is 42."},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println("score:", reward.Score)
	fmt.Println("detail:", reward.Detail)
	// Output:
	// score: 1
	// detail: matched
}

// ExampleRewardExactAnswer scores a rollout whose normalised answer
// matches the expected answer exactly.
func ExampleRewardExactAnswer() {
	fn := grpo.RewardExactAnswer(1)
	reward, err := fn(grpo.RewardContext{
		Sample:  grpo.Sample{ExpectedAnswer: "Paris"},
		Rollout: grpo.Rollout{Answer: "paris"},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println("score:", reward.Score)
	// Output:
	// score: 1
}
