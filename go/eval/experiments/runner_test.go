// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// fakeTarget is the model under test in runner tests: it returns a canned
// output per example id, and reports an error for any id listed in errOn (the
// "one example that fails" case). It records the order it was called in.
//
//	tgt := &fakeTarget{out: map[string]string{"ex-1": "yes"}, errOn: map[string]bool{"ex-3": true}}
type fakeTarget struct {
	out    map[string]string // example id → output
	errOn  map[string]bool   // example id → return an error instead
	called []string          // example ids run, in call order
}

// Run returns the canned output for the example whose prompt was passed, or a
// failure for an id flagged in errOn.
//
//	out, err := tgt.Run(ctx, map[string]any{"id": "ex-1", "prompt": "…"})
func (t *fakeTarget) Run(_ context.Context, inputs map[string]any) (string, error) {
	id, _ := inputs["id"].(string)
	t.called = append(t.called, id)
	if t.errOn[id] {
		return "", core.E("fakeTarget.Run", core.Sprintf("model refused %q", id), nil)
	}
	return t.out[id], nil
}

// exactMatch scores 1.0 when the output equals the example's reference answer,
// 0.0 otherwise — a minimal correctness evaluator.
//
//	key, score, comment := exactMatch{}.Eval(ex, out)
type exactMatch struct{}

func (exactMatch) Eval(ex Example, output string) (string, float64, string) {
	want, _ := ex.Reference["answer"].(string)
	if want == output {
		return "exact_match", 1, "hit"
	}
	return "exact_match", 0, "miss"
}

// lengthScore scores the output's rune length — a second evaluator so the
// runner is exercised with more than one key per example.
//
//	key, score, comment := lengthScore{}.Eval(ex, out)
type lengthScore struct{}

func (lengthScore) Eval(_ Example, output string) (string, float64, string) {
	return "length", float64(len(output)), ""
}

// erroringEvaluator always fails — used to prove an evaluator error is recorded
// as a failure row and does not abort the run.
//
//	key, score, comment := erroringEvaluator{}.Eval(ex, out)
type erroringEvaluator struct{}

func (erroringEvaluator) Eval(_ Example, _ string) (string, float64, string) {
	return "", 0, ""
}

// seedRunnerEval registers a dataset with three examples whose canned outputs
// and references are known, so means are exact.
//
//	e, tgt := seedRunnerEval(t)
func seedRunnerEval(t *testing.T) (*Eval, *fakeTarget) {
	t.Helper()
	e := New()
	if r := e.PutDataset(Dataset{ID: "ethics-probes", Name: "Ethics probes"}); !r.OK {
		t.Fatalf("seed dataset: %v", r.Error())
	}
	want := map[string]string{"ex-1": "yes", "ex-2": "no", "ex-3": "maybe"}
	for _, id := range []string{"ex-1", "ex-2", "ex-3"} {
		ex := Example{
			ID:        id,
			DatasetID: "ethics-probes",
			Inputs:    map[string]any{"id": id, "prompt": "Is honesty always right?"},
			Reference: map[string]any{"answer": want[id]},
		}
		if r := e.AddExample(ex); !r.OK {
			t.Fatalf("add %s: %v", id, r.Error())
		}
	}
	// Target answers ex-1 and ex-2 correctly, ex-3 wrongly (output != reference).
	tgt := &fakeTarget{out: map[string]string{"ex-1": "yes", "ex-2": "no", "ex-3": "nope"}}
	return e, tgt
}

func TestEval_RunExperiment_Good(t *testing.T) {
	e, tgt := seedRunnerEval(t)

	// Run two evaluators across every example.
	r := e.RunExperiment(context.Background(), "ethics-probes", tgt,
		[]Evaluator{exactMatch{}, lengthScore{}})
	if !r.OK {
		t.Fatalf("run experiment: %v", r.Error())
	}
	expID := r.Value.(string)
	if expID == "" {
		t.Fatalf("experiment id is empty")
	}

	// The target was invoked once per example, in id order.
	if len(tgt.called) != 3 {
		t.Fatalf("target calls: got %d, want 3", len(tgt.called))
	}

	// The experiment is recorded and marked complete.
	x := e.GetExperiment(expID)
	if !x.OK {
		t.Fatalf("get experiment: %v", x.Error())
	}
	if got := x.Value.(Experiment).Status; got != StatusComplete {
		t.Errorf("status: got %q, want %q", got, StatusComplete)
	}

	// Feedback rolls up to a mean per key for the experiment.
	agg := e.AggregateFeedback(expID)
	if !agg.OK {
		t.Fatalf("aggregate: %v", agg.Error())
	}
	means := agg.Value.(map[string]float64)
	// exact_match: ex-1 hit, ex-2 hit, ex-3 miss → (1+1+0)/3.
	if got := means["exact_match"]; got != 2.0/3.0 {
		t.Errorf("exact_match mean: got %v, want %v", got, 2.0/3.0)
	}
	// length: len("yes")=3, len("no")=2, len("nope")=4 → (3+2+4)/3 = 3.
	if got := means["length"]; got != 3.0 {
		t.Errorf("length mean: got %v, want 3", got)
	}

	// Every feedback row carries the evaluator source by default.
	rows := e.ListFeedback(expID).Value.([]Feedback)
	if len(rows) != 6 { // 3 examples × 2 evaluators
		t.Fatalf("feedback rows: got %d, want 6", len(rows))
	}
	for _, fb := range rows {
		if fb.Source != SourceEvaluator {
			t.Errorf("row %s source: got %q, want %q", fb.ID, fb.Source, SourceEvaluator)
		}
	}
}

func TestEval_RunExperiment_Bad(t *testing.T) {
	e, tgt := seedRunnerEval(t)

	// An unknown dataset is rejected before any evaluation runs.
	if r := e.RunExperiment(context.Background(), "ghost", tgt, []Evaluator{exactMatch{}}); r.OK {
		t.Fatalf("run over missing dataset should fail, got %+v", r.Value)
	}
	if len(tgt.called) != 0 {
		t.Fatalf("target should not be called for a missing dataset, got %d calls", len(tgt.called))
	}

	// No evaluators is rejected — a run with nothing to score is a caller error.
	if r := e.RunExperiment(context.Background(), "ethics-probes", tgt, nil); r.OK {
		t.Fatalf("run with no evaluators should fail, got %+v", r.Value)
	}

	// A nil target is rejected.
	if r := e.RunExperiment(context.Background(), "ethics-probes", nil, []Evaluator{exactMatch{}}); r.OK {
		t.Fatalf("run with nil target should fail, got %+v", r.Value)
	}
}

func TestEval_RunExperiment_Ugly(t *testing.T) {
	e := New()
	if r := e.PutDataset(Dataset{ID: "ds"}); !r.OK {
		t.Fatalf("seed dataset: %v", r.Error())
	}
	for _, id := range []string{"ex-1", "ex-2", "ex-3"} {
		ex := Example{ID: id, DatasetID: "ds",
			Inputs:    map[string]any{"id": id},
			Reference: map[string]any{"answer": "x"}}
		if r := e.AddExample(ex); !r.OK {
			t.Fatalf("add %s: %v", id, r.Error())
		}
	}

	// The target fails on ex-2; the run must record that failure and carry on,
	// not abort. ex-1 and ex-3 answer correctly.
	tgt := &fakeTarget{
		out:   map[string]string{"ex-1": "x", "ex-3": "x"},
		errOn: map[string]bool{"ex-2": true},
	}
	r := e.RunExperiment(context.Background(), "ds", tgt, []Evaluator{exactMatch{}})
	if !r.OK {
		t.Fatalf("run experiment: %v", r.Error())
	}
	expID := r.Value.(string)

	// All three examples were attempted despite the middle one erroring.
	if len(tgt.called) != 3 {
		t.Fatalf("target calls: got %d, want 3 (failure must not abort)", len(tgt.called))
	}

	// A failure row is recorded for the erroring example under the reserved key.
	rows := e.ListFeedback(expID).Value.([]Feedback)
	var failures, matches int
	for _, fb := range rows {
		switch fb.Key {
		case KeyError:
			failures++
		case "exact_match":
			matches++
		}
	}
	if failures != 1 {
		t.Errorf("error rows: got %d, want 1", failures)
	}
	// exact_match runs only for the two examples the target answered.
	if matches != 2 {
		t.Errorf("exact_match rows: got %d, want 2", matches)
	}

	// The good examples still aggregate (ex-1 and ex-3 both hit → mean 1).
	means := e.AggregateFeedback(expID).Value.(map[string]float64)
	if got := means["exact_match"]; got != 1.0 {
		t.Errorf("exact_match mean: got %v, want 1", got)
	}

	// An evaluator that returns an empty key is recorded as a failure row, not
	// silently dropped, and does not abort the remaining evaluators.
	tgt2 := &fakeTarget{out: map[string]string{"ex-1": "x", "ex-2": "x", "ex-3": "x"}}
	r2 := e.RunExperiment(context.Background(), "ds", tgt2,
		[]Evaluator{erroringEvaluator{}, exactMatch{}})
	if !r2.OK {
		t.Fatalf("run with erroring evaluator: %v", r2.Error())
	}
	exp2 := r2.Value.(string)
	rows2 := e.ListFeedback(exp2).Value.([]Feedback)
	var emptyKeyFailures, good int
	for _, fb := range rows2 {
		if fb.Key == KeyError {
			emptyKeyFailures++
		}
		if fb.Key == "exact_match" {
			good++
		}
	}
	if emptyKeyFailures != 3 { // erroring evaluator fails once per example
		t.Errorf("empty-key failures: got %d, want 3", emptyKeyFailures)
	}
	if good != 3 { // the good evaluator still ran for every example
		t.Errorf("exact_match rows: got %d, want 3", good)
	}
}

func TestEval_Runner_Aggregate_Good(t *testing.T) {
	e, tgt := seedRunnerEval(t)

	// Run a single evaluator; the aggregate is the mean of its per-example
	// scores for the experiment id.
	r := e.RunExperiment(context.Background(), "ethics-probes", tgt, []Evaluator{exactMatch{}})
	if !r.OK {
		t.Fatalf("run: %v", r.Error())
	}
	expID := r.Value.(string)

	means := e.AggregateFeedback(expID).Value.(map[string]float64)
	if got := means["exact_match"]; got != 2.0/3.0 { // two of three hit
		t.Errorf("aggregate: got %v, want %v", got, 2.0/3.0)
	}

	// Two distinct runs produce two distinct experiment ids with independent
	// aggregates — a re-run does not collide with the first.
	r2 := e.RunExperiment(context.Background(), "ethics-probes", tgt, []Evaluator{exactMatch{}})
	if !r2.OK {
		t.Fatalf("re-run: %v", r2.Error())
	}
	if r2.Value.(string) == expID {
		t.Fatalf("re-run reused experiment id %q", expID)
	}
}

func TestEval_Runner_Aggregate_Bad(t *testing.T) {
	e := New()
	if r := e.PutDataset(Dataset{ID: "empty-ds"}); !r.OK {
		t.Fatalf("seed dataset: %v", r.Error())
	}

	// A dataset with no examples runs cleanly but aggregates to nothing — there
	// is no score to take a mean of.
	tgt := &fakeTarget{out: map[string]string{}}
	r := e.RunExperiment(context.Background(), "empty-ds", tgt, []Evaluator{exactMatch{}})
	if !r.OK {
		t.Fatalf("run over empty dataset should still succeed: %v", r.Error())
	}
	if len(tgt.called) != 0 {
		t.Fatalf("no examples means no target calls, got %d", len(tgt.called))
	}
	expID := r.Value.(string)
	means := e.AggregateFeedback(expID).Value.(map[string]float64)
	if len(means) != 0 {
		t.Errorf("empty aggregate: got %d keys, want 0", len(means))
	}
}

func TestEval_Runner_Aggregate_Ugly(t *testing.T) {
	e := New()
	if r := e.PutDataset(Dataset{ID: "ds"}); !r.OK {
		t.Fatalf("seed dataset: %v", r.Error())
	}
	for _, id := range []string{"ex-1", "ex-2"} {
		if r := e.AddExample(Example{ID: id, DatasetID: "ds",
			Inputs:    map[string]any{"id": id},
			Reference: map[string]any{"answer": "x"}}); !r.OK {
			t.Fatalf("add %s: %v", id, r.Error())
		}
	}

	// Both examples error in the target → only failure rows are recorded; the
	// scoring keys never appear, so their aggregate is empty (a failed run is
	// not a zero score).
	tgt := &fakeTarget{errOn: map[string]bool{"ex-1": true, "ex-2": true}}
	r := e.RunExperiment(context.Background(), "ds", tgt, []Evaluator{exactMatch{}})
	if !r.OK {
		t.Fatalf("run: %v", r.Error())
	}
	expID := r.Value.(string)

	rows := e.ListFeedback(expID).Value.([]Feedback)
	if len(rows) != 2 { // one failure row per example, no score rows
		t.Fatalf("rows: got %d, want 2", len(rows))
	}
	means := e.AggregateFeedback(expID).Value.(map[string]float64)
	if _, ok := means["exact_match"]; ok {
		t.Errorf("exact_match should not aggregate when every target call failed: %v", means)
	}
	// The failure key does aggregate (mean of the recorded failure scores).
	if _, ok := means[KeyError]; !ok {
		t.Errorf("error key should aggregate, got %v", means)
	}

	// The experiment is marked failed when every example failed.
	if got := e.GetExperiment(expID).Value.(Experiment).Status; got != StatusFailed {
		t.Errorf("all-failed status: got %q, want %q", got, StatusFailed)
	}
}
