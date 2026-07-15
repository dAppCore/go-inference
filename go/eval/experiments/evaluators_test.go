// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import "testing"

// refExample builds an example whose reference carries the given answer string,
// the shape the built-in evaluators read by default.
//
//	ex := refExample("ex-1", "yes")
func refExample(id, answer string) Example {
	return Example{
		ID:        id,
		DatasetID: "ds",
		Inputs:    map[string]any{"id": id},
		Reference: map[string]any{"answer": answer},
	}
}

func TestEvaluators_ExactMatch_Good(t *testing.T) {
	ev := ExactMatch()

	// An output equal to the reference answer scores 1 under the exact_match key.
	key, score, comment := ev.Eval(refExample("ex-1", "yes"), "yes")
	if key != "exact_match" {
		t.Fatalf("key: got %q, want exact_match", key)
	}
	if score != 1 {
		t.Errorf("score: got %v, want 1", score)
	}
	if comment == "" {
		t.Errorf("a hit should carry a comment")
	}
}

func TestEvaluators_ExactMatch_Bad(t *testing.T) {
	ev := ExactMatch()

	// A differing output scores 0 but still reports the metric key (a miss is a
	// score, not a failure).
	key, score, _ := ev.Eval(refExample("ex-1", "yes"), "no")
	if key != "exact_match" {
		t.Fatalf("key: got %q, want exact_match", key)
	}
	if score != 0 {
		t.Errorf("score: got %v, want 0", score)
	}
}

func TestEvaluators_ExactMatch_Ugly(t *testing.T) {
	ev := ExactMatch()

	// A reference with no answer field present: the gold value is the empty
	// string, so an empty output matches it and a non-empty one does not.
	noRef := Example{ID: "ex-1", DatasetID: "ds"}
	if _, s, _ := ev.Eval(noRef, ""); s != 1 {
		t.Errorf("empty output vs absent reference: got %v, want 1", s)
	}
	if _, s, _ := ev.Eval(noRef, "x"); s != 0 {
		t.Errorf("non-empty output vs absent reference: got %v, want 0", s)
	}

	// A reference whose answer is not a string is treated as absent (empty gold),
	// never a panic.
	wrongType := Example{ID: "ex-2", DatasetID: "ds", Reference: map[string]any{"answer": 99}}
	if _, s, _ := ev.Eval(wrongType, "99"); s != 0 {
		t.Errorf("non-string reference: got %v, want 0 (gold is empty)", s)
	}
	if _, s, _ := ev.Eval(wrongType, ""); s != 1 {
		t.Errorf("non-string reference vs empty output: got %v, want 1", s)
	}
}

func TestEvaluators_ExactMatchOn_Good(t *testing.T) {
	// A non-default reference field can be named at construction time.
	ev := ExactMatchOn("gold")
	ex := Example{ID: "ex-2", DatasetID: "ds", Reference: map[string]any{"gold": "42"}}
	if k, s, _ := ev.Eval(ex, "42"); k != "exact_match" || s != 1 {
		t.Errorf("custom reference field: got (%q,%v), want (exact_match,1)", k, s)
	}
}

func TestEvaluators_ExactMatchOn_Bad(t *testing.T) {
	ev := ExactMatchOn("gold")
	ex := Example{ID: "ex-2", DatasetID: "ds", Reference: map[string]any{"gold": "42"}}
	if _, s, _ := ev.Eval(ex, "43"); s != 0 {
		t.Errorf("mismatched output: got %v, want 0", s)
	}
}

func TestEvaluators_ExactMatchOn_Ugly(t *testing.T) {
	// A field name that names no key in the reference reads as the empty gold
	// value, same as the default-field evaluator — an absent custom field is
	// never a panic.
	ev := ExactMatchOn("missing")
	ex := Example{ID: "ex-2", DatasetID: "ds", Reference: map[string]any{"gold": "42"}}
	if _, s, _ := ev.Eval(ex, ""); s != 1 {
		t.Errorf("absent custom field vs empty output: got %v, want 1", s)
	}
}

func TestEvaluators_Contains_Good(t *testing.T) {
	ev := Contains()

	// Output that contains the reference substring scores 1.
	key, score, comment := ev.Eval(refExample("ex-1", "honest"), "always be honest with people")
	if key != "contains" {
		t.Fatalf("key: got %q, want contains", key)
	}
	if score != 1 {
		t.Errorf("score: got %v, want 1", score)
	}
	if comment == "" {
		t.Errorf("a hit should carry a comment")
	}
}

func TestEvaluators_Contains_Bad(t *testing.T) {
	ev := Contains()

	// Output missing the substring scores 0 with the metric key intact.
	key, score, _ := ev.Eval(refExample("ex-1", "honest"), "always lie")
	if key != "contains" {
		t.Fatalf("key: got %q, want contains", key)
	}
	if score != 0 {
		t.Errorf("score: got %v, want 0", score)
	}
}

func TestEvaluators_Contains_Ugly(t *testing.T) {
	ev := Contains()

	// An empty reference substring is vacuously contained by any output — every
	// string contains "" — so it scores 1 (including for an empty output).
	empty := Example{ID: "ex-1", DatasetID: "ds", Reference: map[string]any{"answer": ""}}
	if _, s, _ := ev.Eval(empty, "anything"); s != 1 {
		t.Errorf("empty substring vs output: got %v, want 1", s)
	}
	if _, s, _ := ev.Eval(empty, ""); s != 1 {
		t.Errorf("empty substring vs empty output: got %v, want 1", s)
	}

	// A non-string reference is treated as an empty substring, never a panic.
	wrongType := Example{ID: "ex-2", DatasetID: "ds", Reference: map[string]any{"answer": 7}}
	if _, s, _ := ev.Eval(wrongType, "anything"); s != 1 {
		t.Errorf("non-string substring: got %v, want 1 (empty substring)", s)
	}
}

func TestEvaluators_ContainsOn_Good(t *testing.T) {
	// The substring field is configurable.
	ev := ContainsOn("needle")
	ex := Example{ID: "ex-2", DatasetID: "ds", Reference: map[string]any{"needle": "cat"}}
	if _, s, _ := ev.Eval(ex, "the cat sat"); s != 1 {
		t.Errorf("custom field: got %v, want 1", s)
	}
}

func TestEvaluators_ContainsOn_Bad(t *testing.T) {
	ev := ContainsOn("needle")
	ex := Example{ID: "ex-2", DatasetID: "ds", Reference: map[string]any{"needle": "cat"}}
	if _, s, _ := ev.Eval(ex, "the dog sat"); s != 0 {
		t.Errorf("missing needle: got %v, want 0", s)
	}
}

func TestEvaluators_ContainsOn_Ugly(t *testing.T) {
	// A custom field name absent from the reference reads as an empty
	// substring — vacuously contained by any output, never a panic.
	ev := ContainsOn("missing")
	ex := Example{ID: "ex-2", DatasetID: "ds", Reference: map[string]any{"needle": "cat"}}
	if _, s, _ := ev.Eval(ex, "anything"); s != 1 {
		t.Errorf("absent custom field: got %v, want 1", s)
	}
}

func TestEvaluators_Regexp_Good(t *testing.T) {
	r := Regexp(`\d+`)
	if !r.OK {
		t.Fatalf("compile pattern: %v", r.Error())
	}
	ev := r.Value.(Evaluator)

	// Output matching the pattern scores 1 under the regexp key.
	key, score, comment := ev.Eval(refExample("ex-1", ""), "build 42 passed")
	if key != "regexp" {
		t.Fatalf("key: got %q, want regexp", key)
	}
	if score != 1 {
		t.Errorf("score: got %v, want 1", score)
	}
	if comment == "" {
		t.Errorf("a match should carry a comment")
	}
}

func TestEvaluators_Regexp_Bad(t *testing.T) {
	r := Regexp(`\d+`)
	if !r.OK {
		t.Fatalf("compile pattern: %v", r.Error())
	}
	ev := r.Value.(Evaluator)

	// Output with no match scores 0 but still reports the key.
	key, score, _ := ev.Eval(refExample("ex-1", ""), "no digits here")
	if key != "regexp" {
		t.Fatalf("key: got %q, want regexp", key)
	}
	if score != 0 {
		t.Errorf("score: got %v, want 0", score)
	}
}

func TestEvaluators_Regexp_Ugly(t *testing.T) {
	// An invalid pattern is rejected at construction — the compile error surfaces
	// as a failed Result, not a panic and not a per-Eval surprise.
	bad := Regexp(`(unclosed`)
	if bad.OK {
		t.Fatalf("invalid pattern should fail to compile, got %+v", bad.Value)
	}

	// An anchored, multi-rune pattern matches only a fully-conforming output.
	r := Regexp(`^[a-z]+$`)
	if !r.OK {
		t.Fatalf("compile: %v", r.Error())
	}
	ev := r.Value.(Evaluator)
	if _, s, _ := ev.Eval(refExample("ex-1", ""), "lowercase"); s != 1 {
		t.Errorf("anchored match: got %v, want 1", s)
	}
	if _, s, _ := ev.Eval(refExample("ex-2", ""), "Has Caps"); s != 0 {
		t.Errorf("anchored non-match: got %v, want 0", s)
	}
	// An empty output against a + pattern does not match.
	if _, s, _ := ev.Eval(refExample("ex-3", ""), ""); s != 0 {
		t.Errorf("empty output: got %v, want 0", s)
	}
}

func TestEvaluators_LengthScore_Good(t *testing.T) {
	// Target length 10: an output of exactly 10 runes scores 1.
	r := LengthScore(10)
	if !r.OK {
		t.Fatalf("construct: %v", r.Error())
	}
	ev := r.Value.(Evaluator)

	key, score, comment := ev.Eval(refExample("ex-1", ""), "0123456789") // 10 runes
	if key != "length" {
		t.Fatalf("key: got %q, want length", key)
	}
	if score != 1 {
		t.Errorf("exact-length score: got %v, want 1", score)
	}
	if comment == "" {
		t.Errorf("a score should carry a comment")
	}

	// Half the target length scores 0.5 (linear ramp up to the target).
	if _, s, _ := ev.Eval(refExample("ex-2", ""), "01234"); s != 0.5 { // 5 of 10
		t.Errorf("half-length score: got %v, want 0.5", s)
	}
}

func TestEvaluators_LengthScore_Bad(t *testing.T) {
	// A non-positive target is a caller error — there is no length to normalise
	// against.
	if r := LengthScore(0); r.OK {
		t.Fatalf("zero target should fail, got %+v", r.Value)
	}
	if r := LengthScore(-3); r.OK {
		t.Fatalf("negative target should fail, got %+v", r.Value)
	}
}

func TestEvaluators_LengthScore_Ugly(t *testing.T) {
	r := LengthScore(4)
	if !r.OK {
		t.Fatalf("construct: %v", r.Error())
	}
	ev := r.Value.(Evaluator)

	// An empty output scores 0 (no length).
	if _, s, _ := ev.Eval(refExample("ex-1", ""), ""); s != 0 {
		t.Errorf("empty output: got %v, want 0", s)
	}

	// Output longer than the target is clamped to 1, never above.
	if _, s, _ := ev.Eval(refExample("ex-2", ""), "way too long for four"); s != 1 {
		t.Errorf("over-length clamp: got %v, want 1", s)
	}

	// Length is counted in runes, not bytes: a 2-rune multi-byte string against a
	// target of 4 scores 0.5, not more.
	if _, s, _ := ev.Eval(refExample("ex-3", ""), "é日"); s != 0.5 { // 2 runes of 4
		t.Errorf("rune-counted length: got %v, want 0.5", s)
	}
}
