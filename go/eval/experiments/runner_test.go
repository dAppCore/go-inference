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

func TestRunner_Eval_RunExperiment_Good(t *testing.T) {
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

func TestRunner_Eval_RunExperiment_Bad(t *testing.T) {
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

func TestRunner_Eval_RunExperiment_Ugly(t *testing.T) {
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

// failStore wraps a MemStore and can be told to fail specific operations, to
// drive the Eval façade and runner's store-rejection branches the in-memory
// store never exercises on its own (a feedback write that fails mid-run, an
// experiment update that fails on finish, a vanished experiment at finish time).
//
//	s := &failStore{MemStore: NewMemStore(), failFeedback: true}
type failStore struct {
	*MemStore
	failFeedback     bool // PutFeedback returns a failed Result
	failExperiment   bool // PutExperiment returns a failed Result
	failUpdate       bool // UpdateExperiment returns a failed Result
	missExperiment   bool // GetExperiment always reports absent
	failAfterNFeedbk int  // succeed this many PutFeedback calls, then fail (0 = honour failFeedback)
	feedbackCalls    int  // PutFeedback calls seen so far
}

// PutFeedback fails when configured, otherwise delegates to the MemStore.
//
//	s.PutFeedback(experiments.Feedback{ID: "fb-1", Target: "exp-1", Key: "k"})
func (s *failStore) PutFeedback(fb Feedback) core.Result {
	s.feedbackCalls++
	if s.failAfterNFeedbk > 0 {
		if s.feedbackCalls > s.failAfterNFeedbk {
			return core.Fail(core.E("failStore.PutFeedback", "forced failure", nil))
		}
		return s.MemStore.PutFeedback(fb)
	}
	if s.failFeedback {
		return core.Fail(core.E("failStore.PutFeedback", "forced failure", nil))
	}
	return s.MemStore.PutFeedback(fb)
}

// PutExperiment fails when configured, otherwise delegates to the MemStore.
//
//	s.PutExperiment(experiments.Experiment{ID: "exp-1", DatasetID: "ds"})
func (s *failStore) PutExperiment(x Experiment) core.Result {
	if s.failExperiment {
		return core.Fail(core.E("failStore.PutExperiment", "forced failure", nil))
	}
	return s.MemStore.PutExperiment(x)
}

// UpdateExperiment fails when configured, otherwise delegates to the MemStore.
//
//	s.UpdateExperiment(experiments.Experiment{ID: "exp-1", DatasetID: "ds"})
func (s *failStore) UpdateExperiment(x Experiment) core.Result {
	if s.failUpdate {
		return core.Fail(core.E("failStore.UpdateExperiment", "forced failure", nil))
	}
	return s.MemStore.UpdateExperiment(x)
}

// GetExperiment reports absent when missExperiment is set, otherwise delegates.
//
//	s.GetExperiment("exp-1")
func (s *failStore) GetExperiment(id string) core.Result {
	if s.missExperiment {
		return core.Fail(core.E("failStore.GetExperiment", "forced miss", nil))
	}
	return s.MemStore.GetExperiment(id)
}

// seedFailEval builds an Eval over a failStore with one dataset and the given
// example ids, so runner tests can pick exactly which store op blows up.
//
//	e, s := seedFailEval(t, &failStore{...}, "ex-1", "ex-2")
func seedFailEval(t *testing.T, s *failStore, exampleIDs ...string) *Eval {
	t.Helper()
	e := NewWithStore(s)
	if r := e.PutDataset(Dataset{ID: "ds", Name: "ds"}); !r.OK {
		t.Fatalf("seed dataset: %v", r.Error())
	}
	for _, id := range exampleIDs {
		ex := Example{ID: id, DatasetID: "ds",
			Inputs:    map[string]any{"id": id},
			Reference: map[string]any{"answer": "x"}}
		if r := e.AddExample(ex); !r.OK {
			t.Fatalf("add %s: %v", id, r.Error())
		}
	}
	return e
}

func TestEval_RunExperiment_CreateFails(t *testing.T) {
	// A dataset that exists but a store that rejects the experiment insert: the
	// run must surface the CreateExperiment failure before any example is touched.
	s := &failStore{MemStore: NewMemStore(), failExperiment: true}
	e := seedFailEval(t, s, "ex-1")
	tgt := &fakeTarget{out: map[string]string{"ex-1": "x"}}

	r := e.RunExperiment(context.Background(), "ds", tgt, []Evaluator{exactMatch{}})
	if r.OK {
		t.Fatalf("run should fail when the experiment cannot be created, got %+v", r.Value)
	}
	if len(tgt.called) != 0 {
		t.Fatalf("no example should run when create fails, got %d calls", len(tgt.called))
	}
}

func TestEval_RunExperiment_FeedbackWriteFails(t *testing.T) {
	// The experiment is created (first PutFeedback call would be a score row), but
	// the store rejects that feedback write. The run must stop and the experiment
	// is finished as failed via finishExperiment.
	s := &failStore{MemStore: NewMemStore(), failFeedback: true}
	e := seedFailEval(t, s, "ex-1")
	tgt := &fakeTarget{out: map[string]string{"ex-1": "x"}}

	r := e.RunExperiment(context.Background(), "ds", tgt, []Evaluator{exactMatch{}})
	if r.OK {
		t.Fatalf("run should fail when a feedback write is rejected, got %+v", r.Value)
	}
	// The target was still invoked for the example before the write was rejected.
	if len(tgt.called) != 1 {
		t.Fatalf("target calls: got %d, want 1", len(tgt.called))
	}
}

func TestEval_RunExperiment_RecordFailureWriteFails(t *testing.T) {
	// The target errors on the only example, so the runner takes the recordFailure
	// path; the store then rejects that failure-row write. The run surfaces the
	// failure and the experiment is finished as failed.
	s := &failStore{MemStore: NewMemStore(), failFeedback: true}
	e := seedFailEval(t, s, "ex-1")
	tgt := &fakeTarget{errOn: map[string]bool{"ex-1": true}}

	r := e.RunExperiment(context.Background(), "ds", tgt, []Evaluator{exactMatch{}})
	if r.OK {
		t.Fatalf("run should fail when the failure row cannot be written, got %+v", r.Value)
	}
}

func TestEval_RunExperiment_EmptyKeyWriteFails(t *testing.T) {
	// An evaluator returns an empty key, sending the runner down recordFailure;
	// the store rejects that write, so the run fails through finishExperiment.
	s := &failStore{MemStore: NewMemStore(), failFeedback: true}
	e := seedFailEval(t, s, "ex-1")
	tgt := &fakeTarget{out: map[string]string{"ex-1": "x"}}

	r := e.RunExperiment(context.Background(), "ds", tgt, []Evaluator{erroringEvaluator{}})
	if r.OK {
		t.Fatalf("run should fail when the empty-key failure row is rejected, got %+v", r.Value)
	}
}

func TestEval_RunExperiment_FinishUpdateFails(t *testing.T) {
	// Everything scores, but the terminal status write (UpdateExperiment) is
	// rejected: finishExperiment must surface that error over the otherwise-good
	// experiment id.
	s := &failStore{MemStore: NewMemStore(), failUpdate: true}
	e := seedFailEval(t, s, "ex-1")
	tgt := &fakeTarget{out: map[string]string{"ex-1": "x"}}

	r := e.RunExperiment(context.Background(), "ds", tgt, []Evaluator{exactMatch{}})
	if r.OK {
		t.Fatalf("run should fail when the finishing status write is rejected, got %+v", r.Value)
	}
}

func TestEval_RunExperiment_FinishExperimentVanished(t *testing.T) {
	// finishExperiment can't find the experiment (a store that reports it absent):
	// it returns the run's own result unchanged rather than inventing an error.
	// The successful run id therefore still comes back.
	s := &failStore{MemStore: NewMemStore(), missExperiment: true}
	e := seedFailEval(t, s, "ex-1")
	tgt := &fakeTarget{out: map[string]string{"ex-1": "x"}}

	r := e.RunExperiment(context.Background(), "ds", tgt, []Evaluator{exactMatch{}})
	if !r.OK {
		t.Fatalf("run should still return its id when finish can't re-read the experiment: %v", r.Error())
	}
	if r.Value.(string) == "" {
		t.Fatalf("expected a non-empty experiment id")
	}
}

func TestEval_RunExperiment_SecondFeedbackWriteFails(t *testing.T) {
	// First evaluator's feedback row writes fine, the second is rejected: the
	// per-evaluator RecordFeedback failure inside the loop is surfaced and the
	// experiment is finished as failed.
	s := &failStore{MemStore: NewMemStore(), failAfterNFeedbk: 1}
	e := seedFailEval(t, s, "ex-1")
	tgt := &fakeTarget{out: map[string]string{"ex-1": "x"}}

	r := e.RunExperiment(context.Background(), "ds", tgt,
		[]Evaluator{exactMatch{}, lengthScore{}})
	if r.OK {
		t.Fatalf("run should fail when the second feedback write is rejected, got %+v", r.Value)
	}
}
