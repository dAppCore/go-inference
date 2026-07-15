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

// --- containsFoldASCII (leaf case-insensitive ASCII substring search) ---

// Good: the haystack folds A-Z to a-z against an already-lowercased needle,
// and an empty needle is a trivial match.
func TestContainsFoldASCII_Good(t *testing.T) {
	if hit, ok := containsFoldASCII("The ANSWER is 42", "answer"); !hit || !ok {
		t.Fatalf("containsFoldASCII(fold match) = (%v,%v), want (true,true)", hit, ok)
	}
	if hit, ok := containsFoldASCII("anything", ""); !hit || !ok {
		t.Fatalf("containsFoldASCII(empty needle) = (%v,%v), want (true,true)", hit, ok)
	}
}

// Bad: a needle with a non-ASCII byte returns ok=false so the caller falls
// back to the unicode-aware path; a needle longer than the haystack is a clean
// miss with ok=true (the input was valid ASCII).
func TestContainsFoldASCII_Bad(t *testing.T) {
	if hit, ok := containsFoldASCII("a café here", "café"); hit || ok {
		t.Fatalf("containsFoldASCII(non-ascii needle) = (%v,%v), want (false,false)", hit, ok)
	}
	if hit, ok := containsFoldASCII("hi", "hello"); hit || !ok {
		t.Fatalf("containsFoldASCII(needle longer than haystack) = (%v,%v), want (false,true)", hit, ok)
	}
	// A valid-ASCII needle that never occurs — the common "no match" reward
	// outcome — falls through the whole scan to (false, true).
	if hit, ok := containsFoldASCII("hello world", "xyz"); hit || !ok {
		t.Fatalf("containsFoldASCII(absent needle) = (%v,%v), want (false,true)", hit, ok)
	}
}

// Ugly: a folded first-byte false-start must not abort the scan — the real
// match follows later — and a punctuation/digit needle matches raw (folding
// only touches letters).
func TestContainsFoldASCII_Ugly(t *testing.T) {
	if hit, ok := containsFoldASCII("axe ... ANSwer", "answer"); !hit || !ok {
		t.Fatalf("containsFoldASCII(false-start then match) = (%v,%v), want (true,true)", hit, ok)
	}
	if hit, ok := containsFoldASCII("value=+7", "+7"); !hit || !ok {
		t.Fatalf("containsFoldASCII(punct needle) = (%v,%v), want (true,true)", hit, ok)
	}
}

// --- asciiHasPrefixFold ---

// Good: a mixed-case string matches a lowercase prefix under ASCII folding.
func TestAsciiHasPrefixFold_Good(t *testing.T) {
	if !asciiHasPrefixFold("ANSWER: 42", "answer:") {
		t.Fatal("asciiHasPrefixFold(folded prefix) = false, want true")
	}
}

// Bad: a string shorter than the prefix can never match, and a mid-prefix
// divergence fails honestly.
func TestAsciiHasPrefixFold_Bad(t *testing.T) {
	if asciiHasPrefixFold("ans", "answer:") {
		t.Fatal("asciiHasPrefixFold(s shorter than prefix) = true, want false")
	}
	if asciiHasPrefixFold("answXr:", "answer:") {
		t.Fatal("asciiHasPrefixFold(mid divergence) = true, want false")
	}
}

// --- cleanAnswerLine ---

// Good: each recognised answer prefix is folded and stripped, leaving the
// trimmed remainder.
func TestCleanAnswerLine_Good(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"final answer: 7", "7"},
		{"Answer: Paris", "Paris"},
		{"SOLUTION: 42", "42"},
	} {
		if got := cleanAnswerLine(tc.in); got != tc.want {
			t.Fatalf("cleanAnswerLine(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// Bad: a whitespace-only line collapses to empty.
func TestCleanAnswerLine_Bad(t *testing.T) {
	if got := cleanAnswerLine("   "); got != "" {
		t.Fatalf("cleanAnswerLine(blank) = %q, want empty", got)
	}
}

// Ugly: a line whose first byte is in the {a,f,s} gate but which is not an
// answer prefix survives verbatim, and a first byte outside the gate skips the
// prefix scan entirely.
func TestCleanAnswerLine_Ugly(t *testing.T) {
	if got := cleanAnswerLine("september was warm"); got != "september was warm" {
		t.Fatalf("cleanAnswerLine(gate-in non-prefix) = %q, want unchanged", got)
	}
	if got := cleanAnswerLine("42 is the answer"); got != "42 is the answer" {
		t.Fatalf("cleanAnswerLine(gate miss) = %q, want unchanged", got)
	}
}

// --- extractReasoningWithAnswer ---

// Good: the answer suffix is stripped off the response to leave the reasoning.
func TestExtractReasoningWithAnswer_Good(t *testing.T) {
	got := extractReasoningWithAnswer(dataset.Sample{Response: "First A then B. 42"}, "42")
	if got != "First A then B." {
		t.Fatalf("extractReasoningWithAnswer = %q, want the response minus the answer suffix", got)
	}
}

// Bad: no answer, or an empty response, yields no reasoning.
func TestExtractReasoningWithAnswer_Bad(t *testing.T) {
	if got := extractReasoningWithAnswer(dataset.Sample{Response: "some text"}, ""); got != "" {
		t.Fatalf("extractReasoningWithAnswer(no answer) = %q, want empty", got)
	}
	if got := extractReasoningWithAnswer(dataset.Sample{}, "42"); got != "" {
		t.Fatalf("extractReasoningWithAnswer(no response) = %q, want empty", got)
	}
}

// Ugly: an explicit thinking label wins over the derived suffix-strip, and an
// answer that is not actually a suffix leaves the whole response intact.
func TestExtractReasoningWithAnswer_Ugly(t *testing.T) {
	labelled := extractReasoningWithAnswer(dataset.Sample{
		Response: "ignored 42",
		Labels:   map[string]string{"thinking": "the real chain"},
	}, "42")
	if labelled != "the real chain" {
		t.Fatalf("extractReasoningWithAnswer(thinking label) = %q, want the label", labelled)
	}
	if got := extractReasoningWithAnswer(dataset.Sample{Response: "42 came first"}, "different"); got != "42 came first" {
		t.Fatalf("extractReasoningWithAnswer(answer not a suffix) = %q, want the full response", got)
	}
}

// --- expectedIsASCIINoNL ---

// Good: plain ASCII with no newline is the fast-path case.
func TestExpectedIsASCIINoNL_Good(t *testing.T) {
	if !expectedIsASCIINoNL("plain ascii 42") {
		t.Fatal("expectedIsASCIINoNL(ascii) = false, want true")
	}
}

// Bad: a newline (would span the fragment join) or a non-ASCII byte forces the
// slower join + unicode-fold path.
func TestExpectedIsASCIINoNL_Bad(t *testing.T) {
	if expectedIsASCIINoNL("has\nnewline") {
		t.Fatal("expectedIsASCIINoNL(newline) = true, want false")
	}
	if expectedIsASCIINoNL("café") {
		t.Fatal("expectedIsASCIINoNL(non-ascii) = true, want false")
	}
}
