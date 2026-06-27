// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import core "dappco.re/go"

// defaultReferenceField is the example reference key the heuristic evaluators
// read when no other field is named at construction — the same "answer" slot the
// runner's worked examples use.
//
//	ex.Reference["answer"]
const defaultReferenceField = "answer"

// refString pulls a string reference value out of an example under field,
// returning "" when the field is absent or not a string — so an evaluator never
// panics on a missing or mistyped reference, it simply scores against an empty
// gold value.
//
//	gold := refString(ex, "answer")
func refString(ex Example, field string) string {
	v, _ := ex.Reference[field].(string)
	return v
}

// ExactMatch scores 1 when the output equals the example's reference answer and
// 0 otherwise — the minimal correctness evaluator over the default "answer"
// field (RFC.inference-stack §3.7).
//
//	ev := experiments.ExactMatch()
//	r := e.RunExperiment(ctx, "ethics-probes", target, []experiments.Evaluator{ev})
func ExactMatch() Evaluator {
	return ExactMatchOn(defaultReferenceField)
}

// ExactMatchOn is ExactMatch reading the gold value from a named reference field
// rather than "answer" — for datasets that label their reference differently.
//
//	ev := experiments.ExactMatchOn("gold")
func ExactMatchOn(field string) Evaluator {
	return exactMatchEval{field: field}
}

// exactMatchEval is the equality evaluator. Its field names the reference slot
// holding the expected output.
//
//	exactMatchEval{field: "answer"}.Eval(ex, out)
type exactMatchEval struct {
	field string
}

// Eval scores output against the example's reference answer.
//
//	key, score, comment := experiments.ExactMatch().Eval(ex, "yes")
func (e exactMatchEval) Eval(example Example, output string) (string, float64, string) {
	want := refString(example, e.field)
	if want == output {
		return "exact_match", 1, "hit"
	}
	return "exact_match", 0, "miss"
}

// Contains scores 1 when the output contains the example's reference substring
// and 0 otherwise — a partial-credit evaluator for answers that need only appear
// somewhere in the output. Reads the default "answer" field.
//
//	ev := experiments.Contains()
func Contains() Evaluator {
	return ContainsOn(defaultReferenceField)
}

// ContainsOn is Contains reading the substring from a named reference field
// rather than "answer".
//
//	ev := experiments.ContainsOn("needle")
func ContainsOn(field string) Evaluator {
	return containsEval{field: field}
}

// containsEval is the substring-presence evaluator. Its field names the
// reference slot holding the substring sought in the output.
//
//	containsEval{field: "answer"}.Eval(ex, out)
type containsEval struct {
	field string
}

// Eval scores whether output contains the example's reference substring. An
// empty (or absent) substring is contained by every output, so it scores 1.
//
//	key, score, comment := experiments.Contains().Eval(ex, "always be honest")
func (e containsEval) Eval(example Example, output string) (string, float64, string) {
	sub := refString(example, e.field)
	if core.Contains(output, sub) {
		return "contains", 1, "found"
	}
	return "contains", 0, "absent"
}

// Regexp builds an evaluator that scores 1 when the output matches pattern and 0
// otherwise, returning the evaluator in a Result so an invalid pattern surfaces
// its compile error at construction rather than per-Eval. Uses the core regexp
// primitive (core.Regex), not stdlib.
//
//	r := experiments.Regexp(`\bhonest\b`)
//	if !r.OK { return r }
//	ev := r.Value.(experiments.Evaluator)
func Regexp(pattern string) core.Result {
	rc := core.Regex(pattern)
	if !rc.OK {
		return rc
	}
	return core.Ok(Evaluator(regexpEval{re: rc.Value.(*core.Regexp)}))
}

// regexpEval is the pattern-match evaluator over a pre-compiled core.Regexp, so
// the pattern compiles once at construction and every Eval is a cheap match.
//
//	regexpEval{re: rx}.Eval(ex, out)
type regexpEval struct {
	re *core.Regexp
}

// Eval scores whether output matches the compiled pattern.
//
//	key, score, comment := eval.Regexp(`\d+`).Value.(experiments.Evaluator).Eval(ex, "build 42")
func (e regexpEval) Eval(_ Example, output string) (string, float64, string) {
	if e.re.MatchString(output) {
		return "regexp", 1, "match"
	}
	return "regexp", 0, "no match"
}

// LengthScore builds an evaluator that scores an output's rune length normalised
// against target: a linear ramp from 0 (empty) to 1 (length ≥ target), clamped
// to 1. target must be positive — there is no length to normalise against
// otherwise — so the constructor returns a Result. Length is counted in runes,
// not bytes (core.RuneCount).
//
//	r := experiments.LengthScore(120)
//	if !r.OK { return r }
//	ev := r.Value.(experiments.Evaluator)
func LengthScore(target int) core.Result {
	if target <= 0 {
		return core.Fail(core.E("experiments.LengthScore",
			core.Sprintf("target length must be positive, got %d", target), nil))
	}
	return core.Ok(Evaluator(lengthScoreEval{target: target}))
}

// lengthScoreEval is the normalised-length evaluator. Its target is the rune
// count that scores a full 1.
//
//	lengthScoreEval{target: 120}.Eval(ex, out)
type lengthScoreEval struct {
	target int
}

// Eval scores the output's rune length as a fraction of the target, clamped to
// the 0..1 range.
//
//	key, score, comment := eval.LengthScore(10).Value.(experiments.Evaluator).Eval(ex, "0123456789")
func (e lengthScoreEval) Eval(_ Example, output string) (string, float64, string) {
	n := core.RuneCount(output)
	score := float64(n) / float64(e.target)
	if score > 1 {
		score = 1
	}
	return "length", score, core.Sprintf("%d/%d runes", n, e.target)
}
