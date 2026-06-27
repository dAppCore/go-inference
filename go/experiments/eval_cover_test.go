// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import (
	"context"
	"testing"

	core "dappco.re/go"
)

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

func TestEval_Snapshot_Bad(t *testing.T) {
	e := newSeededEval(t)
	// An empty version tag is rejected before the dataset is even consulted.
	if r := e.Snapshot("ethics-probes", ""); r.OK {
		t.Fatalf("empty tag should fail, got %+v", r.Value)
	}
}

func TestEval_Snapshot_Empty(t *testing.T) {
	e := newSeededEval(t)
	// Snapshotting a known dataset with no examples yields an empty (but valid)
	// version under the tag — absence of examples is not an error.
	v := e.Snapshot("ethics-probes", "v0")
	if !v.OK {
		t.Fatalf("snapshot of an empty dataset should succeed: %v", v.Error())
	}
	ver := v.Value.(Version)
	if ver.Tag != "v0" || len(ver.ExampleIDs) != 0 {
		t.Errorf("empty snapshot: got tag %q with %d ids, want v0 with 0", ver.Tag, len(ver.ExampleIDs))
	}
}

func TestEval_Splits_Bad(t *testing.T) {
	e := New()
	// Splits over a dataset that does not exist is an error (distinct from a
	// known dataset that happens to have no examples).
	if r := e.Splits("ghost"); r.OK {
		t.Fatalf("splits of a missing dataset should fail, got %+v", r.Value)
	}
}

func TestEval_Splits_NoSplit(t *testing.T) {
	e := newSeededEval(t)
	// An example with no declared split lands under the empty-string key, kept
	// separate from any named split.
	noSplit := Example{ID: "ns-1", DatasetID: "ethics-probes",
		Inputs: map[string]any{"prompt": "?"}}
	if r := e.AddExample(noSplit); !r.OK {
		t.Fatalf("add no-split example: %v", r.Error())
	}
	named := sampleExample("tr-1", "ethics-probes") // SplitTrain
	if r := e.AddExample(named); !r.OK {
		t.Fatalf("add named-split example: %v", r.Error())
	}

	sp := e.Splits("ethics-probes")
	if !sp.OK {
		t.Fatalf("splits: %v", sp.Error())
	}
	splits := sp.Value.(map[Split][]string)
	if got := splits[Split("")]; len(got) != 1 || got[0] != "ns-1" {
		t.Errorf("empty-split bucket: got %v, want [ns-1]", got)
	}
	if got := splits[SplitTrain]; len(got) != 1 || got[0] != "tr-1" {
		t.Errorf("train bucket: got %v, want [tr-1]", got)
	}
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

func TestMemStore_PutExample_Bad(t *testing.T) {
	// The store's own empty-id guard (the façade catches empty ids first, so this
	// is only reachable by a direct store call).
	s := NewMemStore()
	if r := s.PutExample(Example{ID: "", DatasetID: "ds"}); r.OK {
		t.Fatalf("MemStore.PutExample with empty id should fail, got %+v", r.Value)
	}
}

func TestMemStore_PutExperiment_Bad(t *testing.T) {
	s := NewMemStore()
	if r := s.PutExperiment(Experiment{ID: "", DatasetID: "ds"}); r.OK {
		t.Fatalf("MemStore.PutExperiment with empty id should fail, got %+v", r.Value)
	}
}

func TestMemStore_PutFeedback_Bad(t *testing.T) {
	s := NewMemStore()
	if r := s.PutFeedback(Feedback{ID: "", Target: "exp-1", Key: "k"}); r.OK {
		t.Fatalf("MemStore.PutFeedback with empty id should fail, got %+v", r.Value)
	}
}

func TestMemStore_UpdateExperiment_Bad(t *testing.T) {
	s := NewMemStore()
	// Empty id is rejected.
	if r := s.UpdateExperiment(Experiment{ID: "", DatasetID: "ds"}); r.OK {
		t.Fatalf("MemStore.UpdateExperiment with empty id should fail, got %+v", r.Value)
	}
	// Updating an experiment that was never inserted is rejected (update never
	// creates).
	if r := s.UpdateExperiment(Experiment{ID: "ghost", DatasetID: "ds"}); r.OK {
		t.Fatalf("MemStore.UpdateExperiment of a missing id should fail, got %+v", r.Value)
	}
}

func TestMemStore_UpdateExperiment_Good(t *testing.T) {
	// A genuine in-place update replaces the stored experiment.
	s := NewMemStore()
	if r := s.PutExperiment(Experiment{ID: "exp-1", DatasetID: "ds", Status: StatusPending}); !r.OK {
		t.Fatalf("put: %v", r.Error())
	}
	if r := s.UpdateExperiment(Experiment{ID: "exp-1", DatasetID: "ds", Status: StatusComplete}); !r.OK {
		t.Fatalf("update: %v", r.Error())
	}
	if got := s.GetExperiment("exp-1").Value.(Experiment).Status; got != StatusComplete {
		t.Errorf("updated status: got %q, want %q", got, StatusComplete)
	}
}
