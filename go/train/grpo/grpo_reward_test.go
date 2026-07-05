// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"testing"

	"dappco.re/go/inference/train/dataset"
)

// --- SampleFromSFT ---

// Good: prompt, reference answer, and expected answer are extracted from
// a simple prompt/response sample.
func TestSampleFromSFT_Good(t *testing.T) {
	got := SampleFromSFT(dataset.Sample{Prompt: " what is 2+2? ", Response: " 4 "})
	if got.Prompt != "what is 2+2?" {
		t.Errorf("Prompt = %q, want trimmed prompt", got.Prompt)
	}
	if got.ReferenceAnswer != "4" || got.ExpectedAnswer != "4" {
		t.Errorf("ReferenceAnswer/ExpectedAnswer = %q/%q, want 4/4", got.ReferenceAnswer, got.ExpectedAnswer)
	}
	if got.Reasoning != "" {
		t.Errorf("Reasoning = %q, want empty (answer consumes the whole response)", got.Reasoning)
	}
}

// Bad: a fully empty sample produces a fully empty Sample rather than
// panicking.
func TestSampleFromSFT_Bad(t *testing.T) {
	got := SampleFromSFT(dataset.Sample{})
	if got.Prompt != "" || got.ReferenceAnswer != "" || got.ExpectedAnswer != "" || got.Reasoning != "" || got.Meta != nil {
		t.Fatalf("SampleFromSFT(empty) = %+v, want the zero Sample", got)
	}
}

// Ugly: an empty Prompt falls back to Text, and explicit Labels
// (reasoning) win over the derived suffix-strip even when a
// reasoning-style answer prefix is present in Response.
func TestSampleFromSFT_Ugly(t *testing.T) {
	sample := dataset.Sample{
		Text:     " fallback prompt ",
		Response: "some reasoning here\nFinal Answer: 42",
		Labels:   map[string]string{"reasoning": "explicit reasoning"},
	}
	got := SampleFromSFT(sample)
	if got.Prompt != "fallback prompt" {
		t.Errorf("Prompt = %q, want fallback to Text", got.Prompt)
	}
	if got.ExpectedAnswer != "42" {
		t.Errorf("ExpectedAnswer = %q, want 42 (Final Answer: prefix stripped)", got.ExpectedAnswer)
	}
	if got.Reasoning != "explicit reasoning" {
		t.Errorf("Reasoning = %q, want the Labels override, not the derived suffix-strip", got.Reasoning)
	}
	if got.Meta["reasoning"] != "explicit reasoning" {
		t.Errorf("Meta = %+v, want a clone of Labels", got.Meta)
	}
}

// --- ExtractExpectedAnswer ---

// Good: an explicit "answer" label wins over deriving from Response.
func TestExtractExpectedAnswer_Good(t *testing.T) {
	got := ExtractExpectedAnswer(dataset.Sample{Labels: map[string]string{"answer": " Paris "}, Response: "ignored"})
	if got != "Paris" {
		t.Fatalf("ExtractExpectedAnswer() = %q, want Paris", got)
	}
}

// Bad: a sample with no Response, Text, or Labels returns "" rather than
// panicking on a nil map or empty string.
func TestExtractExpectedAnswer_Bad(t *testing.T) {
	if got := ExtractExpectedAnswer(dataset.Sample{}); got != "" {
		t.Fatalf("ExtractExpectedAnswer(empty) = %q, want empty", got)
	}
}

// Ugly: CRLF line endings are normalised before the backward multi-line
// walk, a mixed-case prefix is folded, and a trailing blank line is
// skipped in favour of the last non-empty line.
func TestExtractExpectedAnswer_Ugly(t *testing.T) {
	got := ExtractExpectedAnswer(dataset.Sample{Response: "line one\r\nANSWER: 7\r\n\r\n"})
	if got != "7" {
		t.Fatalf("ExtractExpectedAnswer() = %q, want 7 (CRLF-normalised, prefix folded, trailing blank skipped)", got)
	}
}

// --- RewardContainsAnswer ---

// Good: the expected answer appears in one of the rollout's fragments.
func TestRewardContainsAnswer_Good(t *testing.T) {
	fn := RewardContainsAnswer(2)
	reward, err := fn(RewardContext{
		Sample:  Sample{ExpectedAnswer: "Paris"},
		Rollout: Rollout{Answer: "The city is Paris."},
	})
	if err != nil {
		t.Fatalf("RewardContainsAnswer()(ctx) error = %v", err)
	}
	if reward.Score != 2 || reward.Detail != "matched" {
		t.Fatalf("reward = %+v, want Score 2 Detail matched", reward)
	}
}

// Bad: an empty expected answer never matches, and a zero weight
// defaults to 1 rather than always scoring 0.
func TestRewardContainsAnswer_Bad(t *testing.T) {
	fn := RewardContainsAnswer(0)
	reward, err := fn(RewardContext{Sample: Sample{ExpectedAnswer: ""}, Rollout: Rollout{Answer: "anything"}})
	if err != nil {
		t.Fatalf("RewardContainsAnswer()(ctx) error = %v", err)
	}
	if reward.Score != 0 || reward.Detail != "no expected answer" || reward.Weight != 1 {
		t.Fatalf("reward = %+v, want Score 0 Detail \"no expected answer\" Weight 1 (zero weight defaults to 1)", reward)
	}
}

// Ugly: a non-ASCII expected answer forces the unicode-aware fallback
// path (rather than the ASCII fast path) and still matches case-
// insensitively.
func TestRewardContainsAnswer_Ugly(t *testing.T) {
	fn := RewardContainsAnswer(1)
	reward, err := fn(RewardContext{
		Sample:  Sample{ExpectedAnswer: "Café"},
		Rollout: Rollout{Text: "I went to a CAFÉ yesterday"},
	})
	if err != nil {
		t.Fatalf("RewardContainsAnswer()(ctx) error = %v", err)
	}
	if reward.Score != 1 || reward.Detail != "matched" {
		t.Fatalf("reward = %+v, want a case-insensitive unicode match", reward)
	}
}

// --- RewardExactAnswer ---

// Good: the normalised (trimmed, lower-cased) answer matches exactly.
func TestRewardExactAnswer_Good(t *testing.T) {
	fn := RewardExactAnswer(1)
	reward, err := fn(RewardContext{Sample: Sample{ExpectedAnswer: " Paris "}, Rollout: Rollout{Answer: "paris"}})
	if err != nil {
		t.Fatalf("RewardExactAnswer()(ctx) error = %v", err)
	}
	if reward.Score != 1 || reward.Detail != "matched" {
		t.Fatalf("reward = %+v, want an exact match", reward)
	}
}

// Bad: an empty expected answer never matches, even against an empty answer.
func TestRewardExactAnswer_Bad(t *testing.T) {
	fn := RewardExactAnswer(1)
	reward, err := fn(RewardContext{Sample: Sample{ExpectedAnswer: ""}, Rollout: Rollout{Answer: ""}})
	if err != nil {
		t.Fatalf("RewardExactAnswer()(ctx) error = %v", err)
	}
	if reward.Score != 0 || reward.Detail != "missing" {
		t.Fatalf("reward = %+v, want Score 0 Detail missing", reward)
	}
}

// Ugly: a superstring answer is a miss — unlike RewardContainsAnswer,
// RewardExactAnswer requires the whole normalised strings to match.
func TestRewardExactAnswer_Ugly(t *testing.T) {
	fn := RewardExactAnswer(1)
	reward, err := fn(RewardContext{Sample: Sample{ExpectedAnswer: "Paris"}, Rollout: Rollout{Answer: "Paris, France"}})
	if err != nil {
		t.Fatalf("RewardExactAnswer()(ctx) error = %v", err)
	}
	if reward.Score != 0 || reward.Detail != "missing" {
		t.Fatalf("reward = %+v, want a miss for a superstring answer", reward)
	}
}
