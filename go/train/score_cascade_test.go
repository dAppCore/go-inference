// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the score cascade (#50) machinery, driven with an injected
// ScoreFunc (the concrete lem-scorer has no go-inference home yet). No model,
// no Metal — the fake scorer returns a seeded LEK so the windowed-composite and
// best-checkpoint reads are exercised deterministically.

package train

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// lekScorer returns a ScoreFunc that reads the LEK straight out of a
// prompt→score table — a deterministic stand-in for a real scorer.
func lekScorer(scores map[string]float64) ScoreFunc {
	return func(prompt, text string) ScoreRecord {
		return ScoreRecord{LEK: scores[prompt]}
	}
}

// TestScoreCascade_RecordPassWritesSidecar asserts recordPass scores each eval,
// keeps the vectors in memory, and appends them to the JSONL sidecar with the
// step + prompt + text + birth stamped by the cascade.
func TestScoreCascade_RecordPassWritesSidecar(t *testing.T) {
	sidecar := core.PathJoin(t.TempDir(), "score-cascade.jsonl")
	cascade := newScoreCascade(sidecar, 3, lekScorer(map[string]float64{"q1": 0.4, "q2": 0.6}))
	cascade.recordPass(1, []SFTEvalResult{{Step: 1, Prompt: "q1", Text: "a1"}, {Step: 1, Prompt: "q2", Text: "a2"}})
	if len(cascade.records) != 2 {
		t.Fatalf("records = %d, want 2", len(cascade.records))
	}
	read, err := coreio.Local.Read(sidecar)
	if err != nil {
		t.Fatalf("sidecar read: %v", err)
	}
	var rec ScoreRecord
	first := read[:core.Index(read, "\n")]
	if r := core.JSONUnmarshal([]byte(first), &rec); !r.OK {
		t.Fatalf("record parse: %v", r.Value)
	}
	if rec.Step != 1 || rec.Prompt != "q1" || rec.Text != "a1" || rec.LEK != 0.4 || rec.At == 0 {
		t.Fatalf("record = %+v", rec)
	}
}

// TestScoreCascade_NilScoreFnIsNoOp asserts a cascade with no scorer records
// nothing and answers best() false — scoring simply stays off.
func TestScoreCascade_NilScoreFnIsNoOp(t *testing.T) {
	cascade := newScoreCascade("", 3, nil)
	cascade.recordPass(1, []SFTEvalResult{{Step: 1, Prompt: "q", Text: "a"}})
	if len(cascade.records) != 0 {
		t.Fatalf("records = %d, want 0", len(cascade.records))
	}
	if _, _, ok := cascade.best(); ok {
		t.Fatalf("best() reported a winner with no scores")
	}
}

// TestScoreCascade_BestByWindowedComposite drives three ascending eval passes
// and asserts best() crowns the last step (highest windowed mean) and its
// composite is the trailing-window mean.
func TestScoreCascade_BestByWindowedComposite(t *testing.T) {
	cascade := newScoreCascade("", 2, lekScorer(map[string]float64{"p": 0}))
	// One probe "p" per pass, but a per-step LEK: swap the scorer's read each
	// pass by re-pointing the map value.
	means := []float64{0.2, 0.5, 0.9}
	scores := map[string]float64{"p": 0}
	cascade.scoreFn = func(prompt, text string) ScoreRecord { return ScoreRecord{LEK: scores[prompt]} }
	for i, m := range means {
		scores["p"] = m
		cascade.recordPass(i+1, []SFTEvalResult{{Step: i + 1, Prompt: "p", Text: "a"}})
	}
	step, composite, ok := cascade.best()
	if !ok {
		t.Fatalf("best() found no winner")
	}
	if step != 3 {
		t.Fatalf("best step = %d, want 3", step)
	}
	// window=2 over the last two passes: (0.5 + 0.9)/2 = 0.7.
	if composite < 0.699 || composite > 0.701 {
		t.Fatalf("best composite = %f, want ~0.7", composite)
	}
}

// TestScoreCascade_CompositeAtBeforeAnyEvalIsZero asserts a step before any
// scored pass reads as a zero composite (nothing to average yet).
func TestScoreCascade_CompositeAtBeforeAnyEvalIsZero(t *testing.T) {
	cascade := newScoreCascade("", 3, lekScorer(map[string]float64{"p": 0.5}))
	cascade.recordPass(10, []SFTEvalResult{{Step: 10, Prompt: "p", Text: "a"}})
	if got := cascade.compositeAt(5); got != 0 {
		t.Fatalf("compositeAt(5) = %f, want 0 (before the first eval)", got)
	}
}
