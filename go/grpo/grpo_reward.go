// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

// SampleFromSFT extracts a reasoning prompt and expected answer from a
// dataset sample.
//
//	sample := grpo.SampleFromSFT(raw)
func SampleFromSFT(sample dataset.Sample) Sample {
	prompt := core.Trim(sample.Prompt)
	if prompt == "" {
		prompt = core.Trim(sample.Text)
	}
	// Trim Response once and feed the trimmed string back into the
	// (by-value) sample copy so the inner ExtractExpectedAnswer +
	// extractReasoningWithAnswer both see a pre-trimmed Response.
	// core.Trim is a no-op on already-trimmed input so the inner
	// re-trims become free; we save the two extra whitespace scans the
	// original form paid on every reasoning sample.
	sample.Response = core.Trim(sample.Response)
	// Extract the answer once and forward it to the reasoning step — the
	// without-answer form would otherwise re-run the full label-key
	// sweep + line scan to recover the same value.
	expected := ExtractExpectedAnswer(sample)
	return Sample{
		Prompt:          prompt,
		ReferenceAnswer: sample.Response,
		ExpectedAnswer:  expected,
		Reasoning:       extractReasoningWithAnswer(sample, expected),
		Meta:            cloneStringMap(sample.Labels),
	}
}

// answerMetaKeys are the SFT-label keys ExtractExpectedAnswer consults
// when the dataset carries an explicit answer field. Hoisted to
// package-level so we don't rebuild the four-entry backing array on
// every reasoning sample.
var answerMetaKeys = [...]string{"answer", "expected_answer", "solution", "output"}

// ExtractExpectedAnswer returns the answer target from reasoning-style samples.
func ExtractExpectedAnswer(sample dataset.Sample) string {
	if sample.Labels != nil {
		// Lift the nil check out of the loop — Labels is invariant across
		// the key sweep.
		for _, key := range answerMetaKeys {
			if value := core.Trim(sample.Labels[key]); value != "" {
				return value
			}
		}
	}
	text := core.Trim(sample.Response)
	if text == "" {
		text = core.Trim(sample.Text)
	}
	// Fast path — when the text has no CR we skip the strings.Count scan
	// that ReplaceAll runs to size the result builder. The typical SFT
	// sample is LF-only, so this short-circuits the (small but real)
	// per-call Count walk for the common case.
	normalised := text
	if core.Index(text, "\r") >= 0 {
		normalised = core.Replace(text, "\r\n", "\n")
	}
	// Single-line fast path — when the response is a single line (no
	// "\n"), Split would allocate a one-element []string just to feed it
	// straight to cleanAnswerLine. Skip the slice entirely. Short SFT
	// answers ("42", "Paris", a sentence) hit this branch.
	if core.Index(normalised, "\n") < 0 {
		return cleanAnswerLine(normalised)
	}
	// Multi-line path — walk the input backward by "\n" boundaries instead
	// of pre-splitting into a []string. The original form allocated a
	// fresh []string sized to the line count then indexed backward; for a
	// 2-line response that's an 8-element slice header + 2 string-header
	// backings (~48 B). Now each substring slice is created lazily as we
	// walk.
	end := len(normalised)
	for end > 0 {
		start := core.LastIndex(normalised[:end], "\n")
		line := cleanAnswerLine(normalised[start+1 : end])
		if line != "" {
			return line
		}
		if start < 0 {
			return ""
		}
		end = start
	}
	return ""
}

// extractReasoningWithAnswer is the inner form that takes the
// already-extracted expected answer so callers (the dominant one being
// SampleFromSFT) don't run ExtractExpectedAnswer twice — once for the
// answer field and once again here for the suffix-strip.
func extractReasoningWithAnswer(sample dataset.Sample, answer string) string {
	if sample.Labels != nil {
		if value := core.Trim(sample.Labels["reasoning"]); value != "" {
			return value
		}
		if value := core.Trim(sample.Labels["thinking"]); value != "" {
			return value
		}
	}
	if answer == "" {
		return ""
	}
	response := core.Trim(sample.Response)
	if response == "" {
		return ""
	}
	return core.Trim(core.TrimSuffix(response, answer))
}

// answerPrefixes are the reasoning-style answer prefixes cleanAnswerLine
// looks for. Hoisted to a package-level var so every call doesn't
// re-allocate the three-element backing array (cleanAnswerLine fires for
// every line in every reasoning sample on the SampleFromSFT /
// ExtractExpectedAnswer path).
var answerPrefixes = [...]string{"final answer:", "answer:", "solution:"}

func cleanAnswerLine(line string) string {
	line = core.Trim(line)
	if line == "" {
		return ""
	}
	// First-byte gate — the three answer prefixes all start with one of
	// {a, f, s}. Anything else skips the prefix scan entirely. On
	// free-form text the dominant outcome is "no match".
	switch line[0] {
	case 'a', 'A', 'f', 'F', 's', 'S':
	default:
		return line
	}
	// Case-fold prefix compare directly against the raw line — the
	// prefixes are all ASCII so byte-level case folding suffices. Mixed-
	// case headers like "Answer:" would otherwise pay a core.Lower
	// allocation just so HasPrefix could compare; asciiHasPrefixFold
	// collapses that to zero allocations.
	for _, prefix := range answerPrefixes {
		if asciiHasPrefixFold(line, prefix) {
			return core.Trim(line[len(prefix):])
		}
	}
	return line
}

// asciiHasPrefixFold reports whether prefix is a case-insensitive ASCII
// prefix of s. prefix MUST be lowercase ASCII (a-z + punctuation only) —
// the caller is responsible for that invariant. Used by cleanAnswerLine
// where the prefix set is a fixed package-level array of lowercased
// keywords, so the contract holds by construction.
func asciiHasPrefixFold(s, prefix string) bool {
	if len(s) < len(prefix) {
		return false
	}
	for i := 0; i < len(prefix); i++ {
		c := s[i]
		// Fold ASCII A-Z to a-z by setting bit 5 — bit 5 is the
		// upper/lower case distinguishing bit for ASCII letters and has
		// no effect on the punctuation characters the prefix set contains
		// (':' / ' '). Non-letter bytes outside that range won't match a
		// lowercase letter byte anyway so the compare fails honestly
		// without any further branch.
		if c >= 'A' && c <= 'Z' {
			c |= 0x20
		}
		if c != prefix[i] {
			return false
		}
	}
	return true
}

// containsFoldASCII reports whether s contains substr under ASCII
// case-insensitive comparison. The second return is false when substr
// contains any non-ASCII byte — in that case the caller must fall back
// to the unicode-aware path (core.Lower + Contains) to preserve full
// case-folding semantics. substr is the already-lowered expected answer;
// if it's pure ASCII its bytes are all in 0..0x7f.
func containsFoldASCII(s, substr string) (bool, bool) {
	if len(substr) == 0 {
		return true, true
	}
	// Scan substr once for any byte >= 0x80 — single forward scan is
	// cheaper than checking inside the inner loop on every candidate
	// offset, and the typical expected answer is short (single token /
	// numeral) so the scan touches very few bytes.
	for i := 0; i < len(substr); i++ {
		if substr[i] >= 0x80 {
			return false, false
		}
	}
	if len(s) < len(substr) {
		return false, true
	}
	first := substr[0]
	last := len(s) - len(substr)
	for i := 0; i <= last; i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c |= 0x20
		}
		if c != first {
			continue
		}
		match := true
		for j := 1; j < len(substr); j++ {
			c2 := s[i+j]
			if c2 >= 'A' && c2 <= 'Z' {
				c2 |= 0x20
			}
			if c2 != substr[j] {
				match = false
				break
			}
		}
		if match {
			return true, true
		}
	}
	return false, true
}

// expectedIsASCIINoNL reports whether the expected answer is pure ASCII
// and contains no newline byte. When both conditions hold, the contains-
// answer reward can scan each fragment of the rollout (Answer / Text /
// Reasoning) independently — the expected can't span across the implicit
// "\n" join separator. Lets the caller skip the join allocation entirely
// on the common ASCII path; non-ASCII or newline-bearing expected
// strings fall back to the join + core.Lower path which preserves the
// original cross-fragment + unicode-aware semantics.
func expectedIsASCIINoNL(expected string) bool {
	for i := 0; i < len(expected); i++ {
		c := expected[i]
		if c >= 0x80 || c == '\n' {
			return false
		}
	}
	return true
}

// defaultRewardFuncs is the fallback []RewardFunc used by BuildUpdate
// when Config.RewardFuncs is empty. Package-level so we don't allocate a
// fresh closure + 1-element slice once per training step on the
// default-config path. The captured weight (1) is fixed at init.
var defaultRewardFuncs = []RewardFunc{RewardContainsAnswer(1)}

// RewardContainsAnswer rewards a rollout when it contains the expected answer.
func RewardContainsAnswer(weight float64) RewardFunc {
	if weight == 0 {
		weight = 1
	}
	return func(ctx RewardContext) (Reward, error) {
		expected := core.Lower(core.Trim(ctx.Sample.ExpectedAnswer))
		if expected == "" {
			return Reward{Name: "contains_answer", Weight: weight, Detail: "no expected answer"}, nil
		}
		score := 0.0
		detail := "missing"
		// Fast path: expected is pure ASCII AND contains no separator byte
		// ("\n"). Then the expected can't span across the implicit "\n"
		// join between Answer/Text/Reasoning, so we can scan each fragment
		// independently — no core.Join allocation, no core.Lower(joined)
		// allocation. The common reasoning-dataset shape (short numerals,
		// names, single tokens) hits this path.
		fragments := [3]string{ctx.Rollout.Answer, ctx.Rollout.Text, ctx.Rollout.Reasoning}
		matched := false
		fragmentsOK := true
		// Single ASCII scan: separator-free + pure-ASCII in one walk over
		// expected — the helper's contract is documented above
		// expectedIsASCIINoNL.
		expectedASCII := expectedIsASCIINoNL(expected)
		if expectedASCII {
			for _, f := range fragments {
				if hit, ok := containsFoldASCII(f, expected); !ok {
					// fragment contains substr but substr was rejected —
					// impossible at this point (we already proved ASCII
					// above), so this branch is unreachable but kept for
					// signal-clarity. Use the fallback for completeness.
					fragmentsOK = false
					break
				} else if hit {
					matched = true
					break
				}
			}
		} else {
			fragmentsOK = false
		}
		if !fragmentsOK {
			// Fallback: build the joined text once and case-fold via the
			// unicode-aware core.Lower path. Preserves the original
			// semantics for non-ASCII expected answers and for expected
			// strings that contain newline (cross-fragment spans).
			text := core.Join("\n", ctx.Rollout.Answer, ctx.Rollout.Text, ctx.Rollout.Reasoning)
			matched = core.Contains(core.Lower(text), expected)
		}
		if matched {
			score = weight
			detail = "matched"
		}
		return Reward{Name: "contains_answer", Score: score, Weight: weight, Detail: detail}, nil
	}
}

// RewardExactAnswer rewards exact normalized answer matches.
func RewardExactAnswer(weight float64) RewardFunc {
	if weight == 0 {
		weight = 1
	}
	return func(ctx RewardContext) (Reward, error) {
		expected := core.Lower(core.Trim(ctx.Sample.ExpectedAnswer))
		answer := core.Lower(core.Trim(ctx.Rollout.Answer))
		score := 0.0
		detail := "missing"
		if expected != "" && answer == expected {
			score = weight
			detail = "matched"
		}
		return Reward{Name: "exact_answer", Score: score, Weight: weight, Detail: detail}, nil
	}
}

func cloneStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	return core.MapClone(values)
}
