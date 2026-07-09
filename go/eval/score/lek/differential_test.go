// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "testing"

// --- Imprint ---

func TestDifferential_Imprint_Good(t *testing.T) {
	imp := Imprint("the model considered each constraint in turn before settling on the trade-offs")
	if imp == nil {
		t.Fatal("Imprint returned nil for tokenisable text")
	}
	if imp.VocabRichness < 0 || imp.VocabRichness > 1 {
		t.Errorf("VocabRichness out of [0,1]: %f", imp.VocabRichness)
	}
	if imp.TenseEntropy < 0 || imp.TenseEntropy > 1 {
		t.Errorf("TenseEntropy out of [0,1]: %f", imp.TenseEntropy)
	}
	if imp.VerbDiversity < 0 || imp.VerbDiversity > 1 {
		t.Errorf("VerbDiversity out of [0,1]: %f", imp.VerbDiversity)
	}
	if imp.NounDiversity < 0 || imp.NounDiversity > 1 {
		t.Errorf("NounDiversity out of [0,1]: %f", imp.NounDiversity)
	}
}

func TestDifferential_Imprint_Bad(t *testing.T) {
	imp := Imprint("is this right? what about that? could it be different?")
	if imp == nil {
		t.Fatal("Imprint returned nil for question-heavy text")
	}
	if imp.QuestionRatio <= 0 {
		t.Errorf("question-heavy text QuestionRatio = %f, want > 0", imp.QuestionRatio)
	}
}

func TestDifferential_Imprint_Ugly(t *testing.T) {
	imp := Imprint("")
	if imp != nil {
		t.Errorf("Imprint(\"\") returned non-nil %v, want nil", imp)
	}
}

func TestDifferential_Imprint_PunctuationOnly_Ugly(t *testing.T) {
	imp := Imprint("... !!! ???")
	// Punctuation-only may produce empty tokens; allow either nil or zeroed.
	if imp != nil {
		if imp.VocabRichness != 0 {
			t.Errorf("punctuation-only VocabRichness = %f, want 0", imp.VocabRichness)
		}
	}
}

// --- Differential ---

func TestDifferential_Differential_Good(t *testing.T) {
	d := Differential(
		"is this the right approach?",
		"the constraints suggest weighing trade-offs explicitly first",
	)
	if d == nil {
		t.Fatal("Differential returned nil for divergent texts")
	}
	for name, v := range map[string]float64{
		"Echo": d.Echo, "VerbShift": d.VerbShift, "TenseShift": d.TenseShift,
		"NounEcho": d.NounEcho, "QuestionFlip": d.QuestionFlip, "DomainShift": d.DomainShift,
	} {
		if v < 0 || v > 1 {
			t.Errorf("%s out of [0,1]: %f", name, v)
		}
	}
	// Prompt asked a question, response did not — QuestionFlip should be positive.
	if d.QuestionFlip <= 0 {
		t.Errorf("question→statement QuestionFlip = %f, want > 0", d.QuestionFlip)
	}
}

func TestDifferential_Differential_Bad(t *testing.T) {
	// Response mirrors prompt grammar closely — high echo signal.
	prompt := "the system considered the request and weighed the constraints"
	response := "the system considered the request and weighed the constraints carefully"
	d := Differential(prompt, response)
	if d == nil {
		t.Fatal("Differential returned nil for mirror texts")
	}
	if d.Echo < 0.3 {
		t.Errorf("near-identical Echo = %f, want >= 0.3 (mirroring signal)", d.Echo)
	}
}

func TestDifferential_Differential_Ugly(t *testing.T) {
	d := Differential("", "a perfectly valid response by itself")
	if d != nil {
		t.Errorf("Differential with empty prompt returned %v, want nil", d)
	}
}

func TestDifferential_Differential_EmptyResponse_Ugly(t *testing.T) {
	d := Differential("a prompt with content", "")
	if d != nil {
		t.Errorf("Differential with empty response returned %v, want nil", d)
	}
}

func TestDifferential_Differential_BothEmpty_Ugly(t *testing.T) {
	d := Differential("", "")
	if d != nil {
		t.Errorf("Differential with both empty returned %v, want nil", d)
	}
}

// --- Wired via Score / ScorePair ---

func TestScore_ImprintPopulatedWhenTokensPresent(t *testing.T) {
	r := Score("the response considered the constraints carefully")
	if r.Imprint == nil {
		t.Error("Score did not populate Imprint slot for tokenisable text")
	}
}

func TestScore_ImprintNilOnEmpty(t *testing.T) {
	r := Score("")
	if r.Imprint != nil {
		t.Errorf("Score(\"\") populated Imprint = %v, want nil", r.Imprint)
	}
}

func TestScorePair_DifferentialPopulatedWhenBothTokenised(t *testing.T) {
	d := ScorePair("explain your reasoning", "the trade-offs weighed against each other")
	if d.Differential == nil {
		t.Error("ScorePair did not populate Differential slot for tokenisable pair")
	}
	if d.Prompt.Imprint == nil || d.Response.Imprint == nil {
		t.Error("ScorePair did not populate per-side Imprint slots")
	}
}

func TestScorePair_DifferentialNilWhenSideEmpty(t *testing.T) {
	d := ScorePair("", "a response by itself")
	if d.Differential != nil {
		t.Errorf("ScorePair with empty prompt populated Differential = %v, want nil", d.Differential)
	}
}

// --- Internal helpers (deterministic math) ---
//
// domainCosineSimilarity's partial-overlap branch is not reachable through
// the public Differential path because the reversal tokeniser does not
// populate DomainVocabulary for ordinary prose, so the partial-similarity
// arithmetic is exercised here directly. The package is `package score`,
// so the internal helper is in scope. Values are hand-computable cosine
// similarities — no tautology.

// TestDifferential_DomainCosineSimilarityBranches — the int-map cosine helper.
// Empty/empty → 1.0 (identical), empty/non-empty → 0.0, identical maps →
// 1.0, disjoint keys → 0.0, and the partial-overlap case {x,y}·{x,z} which
// has dot=1, |a|=|b|=√2, so cos = 1/2 = 0.5.
//
// The empty-branch cases short-circuit to exact literals; the cases that
// run the cosine arithmetic go through math.Sqrt, so they're compared
// with a float tolerance rather than exact equality.
func TestDifferential_DomainCosineSimilarityBranches(t *testing.T) {
	const eps = 1e-9
	cases := []struct {
		name string
		a, b map[string]int
		want float64
	}{
		{"both empty", map[string]int{}, map[string]int{}, 1.0},
		{"a empty", map[string]int{}, map[string]int{"x": 1}, 0.0},
		{"b empty", map[string]int{"x": 1}, map[string]int{}, 0.0},
		{"identical", map[string]int{"x": 2, "y": 1}, map[string]int{"x": 2, "y": 1}, 1.0},
		{"disjoint", map[string]int{"a": 1}, map[string]int{"b": 1}, 0.0},
		{"partial overlap", map[string]int{"x": 1, "y": 1}, map[string]int{"x": 1, "z": 1}, 0.5},
	}
	for _, c := range cases {
		got := domainCosineSimilarity(c.a, c.b)
		if diff := got - c.want; diff < -eps || diff > eps {
			t.Errorf("domainCosineSimilarity(%s) = %v, want %v (±%g)", c.name, got, c.want, eps)
		}
	}
}

// TestDifferential_CosineSimilarityZeroVector — when one frequency map has only
// zero-valued entries the magnitude denominator is 0; the helper must
// return 0.0 rather than dividing by zero (NaN).
func TestDifferential_CosineSimilarityZeroVector(t *testing.T) {
	got := cosineSimilarity(map[string]float64{"x": 0}, map[string]float64{"y": 0})
	if got != 0.0 {
		t.Errorf("cosineSimilarity with zero-magnitude vectors = %v, want 0.0 (denom guard)", got)
	}
}

// TestDifferential_ClampUnitBounds — clamp to [0,1]: below clamps to 0, within
// passes through, above clamps to 1.
func TestDifferential_ClampUnitBounds(t *testing.T) {
	cases := []struct {
		in, want float64
	}{
		{-0.5, 0.0}, {-1e9, 0.0}, {0.0, 0.0}, {0.5, 0.5}, {1.0, 1.0}, {1.5, 1.0}, {1e9, 1.0},
	}
	for _, c := range cases {
		if got := clampUnit(c.in); got != c.want {
			t.Errorf("clampUnit(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestDifferential_QuestionFlip_PartialLoss_Good — exercises the partial
// branch of computeQuestionFlip: the prompt is heavily questioning
// (promptQ > 0.1) and the response keeps SOME questioning voice
// (0.02 <= responseQ < promptQ), so the flip is a fractional value
// strictly between 0 and 1, not the saturated 1.0.
func TestDifferential_Differential_QuestionFlipPartialLoss_Good(t *testing.T) {
	d := Differential("is it right? are you sure? do you agree?", "yes it works. but is it tested?")
	if d == nil {
		t.Fatal("Differential returned nil")
	}
	if d.QuestionFlip <= 0 || d.QuestionFlip >= 1 {
		t.Errorf("partial QuestionFlip = %v, want strictly in (0,1)", d.QuestionFlip)
	}
}
