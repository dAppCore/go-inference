// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import "testing"

// sampleDataset builds a populated dataset for tests.
//
//	ds := sampleDataset("ethics-probes")
func sampleDataset(id string) Dataset {
	return Dataset{
		ID:          id,
		Name:        "Ethics probes",
		Description: "Core LEK axiom probes",
	}
}

// sampleExample builds a populated example for tests.
//
//	ex := sampleExample("ex-1", "ethics-probes")
func sampleExample(id, datasetID string) Example {
	return Example{
		ID:        id,
		DatasetID: datasetID,
		Inputs:    map[string]any{"prompt": "Is honesty always right?"},
		Reference: map[string]any{"answer": "context-dependent"},
		Split:     SplitTrain,
		Metadata:  map[string]any{"axiom": "A2"},
	}
}

// newSeededEval returns an Eval with one dataset already registered.
func newSeededEval(t *testing.T) *Eval {
	t.Helper()
	e := New()
	if r := e.PutDataset(sampleDataset("ethics-probes")); !r.OK {
		t.Fatalf("seed dataset: %v", r.Error())
	}
	return e
}

func TestEval_New_Good(t *testing.T) {
	e := New()
	if r := e.PutDataset(Dataset{ID: "ds"}); !r.OK {
		t.Fatalf("put on a fresh Eval: %v", r.Error())
	}
	if r := e.GetDataset("ds"); !r.OK {
		t.Fatalf("get after put: %v", r.Error())
	}
}

func TestEval_New_Bad(t *testing.T) {
	// Two New() calls are independent stores — mutating one must not leak into
	// the other.
	e1, e2 := New(), New()
	if r := e1.PutDataset(Dataset{ID: "ds"}); !r.OK {
		t.Fatalf("put on e1: %v", r.Error())
	}
	if r := e2.GetDataset("ds"); r.OK {
		t.Fatalf("e2 should not see e1's dataset, got %+v", r.Value)
	}
}

func TestEval_New_Ugly(t *testing.T) {
	// A fresh Eval starts with no datasets at all — every read fails absent,
	// not with some stale default.
	e := New()
	if r := e.GetDataset("anything"); r.OK {
		t.Fatalf("fresh Eval should have no datasets, got %+v", r.Value)
	}
	if r := e.ListExamples("anything"); r.OK {
		t.Fatalf("fresh Eval should reject an unknown dataset, got %+v", r.Value)
	}
}

func TestEval_NewWithStore_Good(t *testing.T) {
	s := NewMemStore()
	e := NewWithStore(s)
	if r := e.PutDataset(Dataset{ID: "ds"}); !r.OK {
		t.Fatalf("put via wrapped store: %v", r.Error())
	}
	if r := e.GetDataset("ds"); !r.OK {
		t.Fatalf("get via wrapped store: %v", r.Error())
	}
}

func TestEval_NewWithStore_Bad(t *testing.T) {
	// A store that rejects experiment inserts: NewWithStore does not itself
	// validate the store, so a failing backend's errors surface unchanged
	// through the façade.
	s := &failStore{MemStore: NewMemStore(), failExperiment: true}
	e := NewWithStore(s)
	if r := e.PutDataset(Dataset{ID: "ds"}); !r.OK {
		t.Fatalf("dataset put should still succeed (only experiments are configured to fail): %v", r.Error())
	}
	if r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ds"}); r.OK {
		t.Fatalf("experiment create should surface the store's failure, got %+v", r.Value)
	}
}

func TestEval_NewWithStore_Ugly(t *testing.T) {
	// Two façades over the SAME store instance share state — NewWithStore wraps,
	// it does not clone.
	s := NewMemStore()
	e1 := NewWithStore(s)
	e2 := NewWithStore(s)
	if r := e1.PutDataset(Dataset{ID: "shared"}); !r.OK {
		t.Fatalf("put via e1: %v", r.Error())
	}
	if r := e2.GetDataset("shared"); !r.OK {
		t.Fatalf("dataset put through e1 should be visible via e2 (same store): %v", r.Error())
	}
}

func TestEval_PutDataset_Good(t *testing.T) {
	e := New()
	r := e.PutDataset(sampleDataset("ethics-probes"))
	if !r.OK {
		t.Fatalf("put dataset: %v", r.Error())
	}
	if got := r.Value.(Dataset).Name; got != "Ethics probes" {
		t.Errorf("returned value: got %q, want %q", got, "Ethics probes")
	}
}

func TestEval_PutDataset_Bad(t *testing.T) {
	e := New()
	// A dataset with an empty id is rejected.
	if r := e.PutDataset(Dataset{ID: ""}); r.OK {
		t.Fatalf("empty dataset id should fail, got %+v", r.Value)
	}
}

func TestEval_PutDataset_Ugly(t *testing.T) {
	// PutDataset inserts OR REPLACES — putting the same id twice overwrites
	// rather than failing as a duplicate.
	e := New()
	if r := e.PutDataset(Dataset{ID: "ds", Name: "first"}); !r.OK {
		t.Fatalf("first put: %v", r.Error())
	}
	if r := e.PutDataset(Dataset{ID: "ds", Name: "second"}); !r.OK {
		t.Fatalf("replacing put: %v", r.Error())
	}
	if got := e.GetDataset("ds").Value.(Dataset).Name; got != "second" {
		t.Errorf("after replace: got %q, want %q", got, "second")
	}
}

func TestEval_GetDataset_Good(t *testing.T) {
	e := newSeededEval(t)
	ds := e.GetDataset("ethics-probes")
	if !ds.OK {
		t.Fatalf("get dataset: %v", ds.Error())
	}
	if got := ds.Value.(Dataset).Name; got != "Ethics probes" {
		t.Errorf("dataset name: got %q, want %q", got, "Ethics probes")
	}
}

func TestEval_GetDataset_Bad(t *testing.T) {
	e := New()
	if r := e.GetDataset("ghost"); r.OK {
		t.Fatalf("get of missing dataset should fail, got %+v", r.Value)
	}
}

func TestEval_GetDataset_Ugly(t *testing.T) {
	// A dataset with only its id set round-trips its zero-value optional
	// fields rather than substituting a default.
	e := New()
	if r := e.PutDataset(Dataset{ID: "bare"}); !r.OK {
		t.Fatalf("put: %v", r.Error())
	}
	got := e.GetDataset("bare").Value.(Dataset)
	if got.Name != "" || got.Description != "" {
		t.Errorf("bare dataset: got %+v, want empty Name/Description", got)
	}
}

func TestEval_AddExample_Good(t *testing.T) {
	e := newSeededEval(t)

	// Add two examples to the dataset.
	if r := e.AddExample(sampleExample("ex-1", "ethics-probes")); !r.OK {
		t.Fatalf("add ex-1: %v", r.Error())
	}
	if r := e.AddExample(sampleExample("ex-2", "ethics-probes")); !r.OK {
		t.Fatalf("add ex-2: %v", r.Error())
	}
}

func TestEval_AddExample_Bad(t *testing.T) {
	e := New()
	// Adding an example to a dataset that does not exist fails.
	if r := e.AddExample(sampleExample("ex-1", "ghost")); r.OK {
		t.Fatalf("add to missing dataset should fail, got %+v", r.Value)
	}
}

func TestEval_AddExample_Ugly(t *testing.T) {
	e := newSeededEval(t)

	// An example with an empty id is rejected.
	if r := e.AddExample(Example{ID: "", DatasetID: "ethics-probes"}); r.OK {
		t.Fatalf("empty example id should fail, got %+v", r.Value)
	}

	// First add succeeds.
	if r := e.AddExample(sampleExample("dup", "ethics-probes")); !r.OK {
		t.Fatalf("first add: %v", r.Error())
	}
	// Re-adding the same id within a dataset is a duplicate and is rejected
	// (use UpdateExample to change one in place).
	if r := e.AddExample(sampleExample("dup", "ethics-probes")); r.OK {
		t.Fatalf("duplicate example id should fail, got %+v", r.Value)
	}

	// The same example id may live in a different dataset — ids are scoped per
	// dataset, not global.
	if r := e.PutDataset(sampleDataset("other")); !r.OK {
		t.Fatalf("put other dataset: %v", r.Error())
	}
	if r := e.AddExample(sampleExample("dup", "other")); !r.OK {
		t.Fatalf("same id in a different dataset should succeed: %v", r.Error())
	}
}

func TestEval_ListExamples_Good(t *testing.T) {
	e := newSeededEval(t)
	if r := e.AddExample(sampleExample("ex-1", "ethics-probes")); !r.OK {
		t.Fatalf("add ex-1: %v", r.Error())
	}
	if r := e.AddExample(sampleExample("ex-2", "ethics-probes")); !r.OK {
		t.Fatalf("add ex-2: %v", r.Error())
	}

	// List returns both, sorted by id for determinism.
	got := e.ListExamples("ethics-probes")
	if !got.OK {
		t.Fatalf("list examples: %v", got.Error())
	}
	exs := got.Value.([]Example)
	if len(exs) != 2 {
		t.Fatalf("list: got %d, want 2", len(exs))
	}
	if exs[0].ID != "ex-1" || exs[1].ID != "ex-2" {
		t.Errorf("list order: got %q, %q", exs[0].ID, exs[1].ID)
	}
}

func TestEval_ListExamples_Bad(t *testing.T) {
	e := New()
	if r := e.ListExamples("ghost"); r.OK {
		t.Fatalf("list of missing dataset should fail, got %+v", r.Value)
	}
}

func TestEval_ListExamples_Ugly(t *testing.T) {
	// A known dataset with no examples yet is an empty, successful list — that
	// is distinct from an unknown dataset, which is an error.
	e := newSeededEval(t)
	got := e.ListExamples("ethics-probes")
	if !got.OK {
		t.Fatalf("list of empty dataset should succeed: %v", got.Error())
	}
	if exs := got.Value.([]Example); len(exs) != 0 {
		t.Errorf("empty dataset: got %d examples, want 0", len(exs))
	}
}

func TestEval_Splits_Good(t *testing.T) {
	e := newSeededEval(t)
	holdout := sampleExample("ho-1", "ethics-probes")
	holdout.Split = SplitTest
	if r := e.AddExample(holdout); !r.OK {
		t.Fatalf("add holdout: %v", r.Error())
	}
	if r := e.AddExample(sampleExample("tr-1", "ethics-probes")); !r.OK { // SplitTrain
		t.Fatalf("add train: %v", r.Error())
	}

	sp := e.Splits("ethics-probes")
	if !sp.OK {
		t.Fatalf("splits: %v", sp.Error())
	}
	splits := sp.Value.(map[Split][]string)
	if len(splits[SplitTrain]) != 1 || splits[SplitTrain][0] != "tr-1" {
		t.Errorf("train split: got %v, want [tr-1]", splits[SplitTrain])
	}
	if len(splits[SplitTest]) != 1 || splits[SplitTest][0] != "ho-1" {
		t.Errorf("test split: got %v, want [ho-1]", splits[SplitTest])
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

func TestEval_Splits_Ugly(t *testing.T) {
	// An example with no declared split lands under the empty-string key, kept
	// separate from any named split.
	e := newSeededEval(t)
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

func TestEval_Snapshot_Good(t *testing.T) {
	e := newSeededEval(t)
	if r := e.AddExample(sampleExample("ex-1", "ethics-probes")); !r.OK {
		t.Fatalf("add ex-1: %v", r.Error())
	}
	if r := e.AddExample(sampleExample("ex-2", "ethics-probes")); !r.OK {
		t.Fatalf("add ex-2: %v", r.Error())
	}

	ver := e.Snapshot("ethics-probes", "v1")
	if !ver.OK {
		t.Fatalf("snapshot: %v", ver.Error())
	}
	v := ver.Value.(Version)
	if v.Tag != "v1" || len(v.ExampleIDs) != 2 {
		t.Errorf("version: got tag %q with %d ids, want v1 with 2", v.Tag, len(v.ExampleIDs))
	}
}

func TestEval_Snapshot_Bad(t *testing.T) {
	e := newSeededEval(t)
	// An empty version tag is rejected before the dataset is even consulted.
	if r := e.Snapshot("ethics-probes", ""); r.OK {
		t.Fatalf("empty tag should fail, got %+v", r.Value)
	}
	// Snapshot of an unknown dataset fails.
	if r := e.Snapshot("ghost", "v1"); r.OK {
		t.Fatalf("snapshot of missing dataset should fail, got %+v", r.Value)
	}
}

func TestEval_Snapshot_Ugly(t *testing.T) {
	// Snapshotting a known dataset with no examples yields an empty (but valid)
	// version under the tag — absence of examples is not an error.
	e := newSeededEval(t)
	v := e.Snapshot("ethics-probes", "v0")
	if !v.OK {
		t.Fatalf("snapshot of an empty dataset should succeed: %v", v.Error())
	}
	ver := v.Value.(Version)
	if ver.Tag != "v0" || len(ver.ExampleIDs) != 0 {
		t.Errorf("empty snapshot: got tag %q with %d ids, want v0 with 0", ver.Tag, len(ver.ExampleIDs))
	}
}

func TestEval_CreateExperiment_Good(t *testing.T) {
	e := newSeededEval(t)

	r := e.CreateExperiment(Experiment{
		ID:        "exp-1",
		DatasetID: "ethics-probes",
		Run:       "lemma@step-3000",
	})
	if !r.OK {
		t.Fatalf("create experiment: %v", r.Error())
	}
	// A created experiment defaults to the pending status.
	if got := r.Value.(Experiment).Status; got != StatusPending {
		t.Errorf("default status: got %q, want %q", got, StatusPending)
	}
}

func TestEval_CreateExperiment_Bad(t *testing.T) {
	e := New()
	// An experiment over a dataset that does not exist is rejected.
	if r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ghost"}); r.OK {
		t.Fatalf("experiment over missing dataset should fail, got %+v", r.Value)
	}
}

func TestEval_CreateExperiment_Ugly(t *testing.T) {
	e := newSeededEval(t)

	// An experiment with an empty id is rejected.
	if r := e.CreateExperiment(Experiment{ID: "", DatasetID: "ethics-probes"}); r.OK {
		t.Fatalf("empty experiment id should fail, got %+v", r.Value)
	}

	// First create succeeds.
	if r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ethics-probes"}); !r.OK {
		t.Fatalf("first create: %v", r.Error())
	}
	// Re-creating the same experiment id is a duplicate and is rejected.
	if r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ethics-probes"}); r.OK {
		t.Fatalf("duplicate experiment id should fail, got %+v", r.Value)
	}

	// A caller-supplied status is preserved rather than overwritten with the
	// pending default.
	if r := e.CreateExperiment(Experiment{ID: "exp-running", DatasetID: "ethics-probes", Status: StatusRunning}); !r.OK {
		t.Fatalf("create running: %v", r.Error())
	}
	got := e.GetExperiment("exp-running")
	if !got.OK {
		t.Fatalf("get running: %v", got.Error())
	}
	if s := got.Value.(Experiment).Status; s != StatusRunning {
		t.Errorf("explicit status: got %q, want %q", s, StatusRunning)
	}
}

func TestEval_GetExperiment_Good(t *testing.T) {
	e := newSeededEval(t)
	if r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ethics-probes", Run: "lemma@step-3000"}); !r.OK {
		t.Fatalf("create experiment: %v", r.Error())
	}
	one := e.GetExperiment("exp-1")
	if !one.OK {
		t.Fatalf("get experiment: %v", one.Error())
	}
	if got := one.Value.(Experiment).Run; got != "lemma@step-3000" {
		t.Errorf("run: got %q, want %q", got, "lemma@step-3000")
	}
}

func TestEval_GetExperiment_Bad(t *testing.T) {
	e := New()
	if r := e.GetExperiment("nope"); r.OK {
		t.Fatalf("get of missing experiment should fail, got %+v", r.Value)
	}
}

func TestEval_GetExperiment_Ugly(t *testing.T) {
	// An experiment created with only its required fields round-trips its
	// zero-value optional fields (Run, Created) unchanged.
	e := newSeededEval(t)
	if r := e.CreateExperiment(Experiment{ID: "bare", DatasetID: "ethics-probes"}); !r.OK {
		t.Fatalf("create: %v", r.Error())
	}
	got := e.GetExperiment("bare").Value.(Experiment)
	if got.Run != "" || got.Created != 0 {
		t.Errorf("bare experiment: got %+v, want zero Run/Created", got)
	}
}

func TestEval_ListExperiments_Good(t *testing.T) {
	e := newSeededEval(t)
	if r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ethics-probes"}); !r.OK {
		t.Fatalf("create exp-1: %v", r.Error())
	}
	if r := e.CreateExperiment(Experiment{ID: "exp-2", DatasetID: "ethics-probes", Run: "lemrd@step-1"}); !r.OK {
		t.Fatalf("create exp-2: %v", r.Error())
	}

	got := e.ListExperiments("ethics-probes")
	if !got.OK {
		t.Fatalf("list experiments: %v", got.Error())
	}
	exps := got.Value.([]Experiment)
	if len(exps) != 2 || exps[0].ID != "exp-1" || exps[1].ID != "exp-2" {
		t.Errorf("list experiments: got %d in unexpected order", len(exps))
	}
}

func TestEval_ListExperiments_Bad(t *testing.T) {
	e := New()
	if r := e.ListExperiments("ghost"); r.OK {
		t.Fatalf("list for missing dataset should fail, got %+v", r.Value)
	}
}

func TestEval_ListExperiments_Ugly(t *testing.T) {
	// A known dataset with no experiments yet is an empty, successful list —
	// distinct from an unknown dataset, which is an error.
	e := newSeededEval(t)
	got := e.ListExperiments("ethics-probes")
	if !got.OK {
		t.Fatalf("list of empty dataset should succeed: %v", got.Error())
	}
	if exps := got.Value.([]Experiment); len(exps) != 0 {
		t.Errorf("empty dataset: got %d experiments, want 0", len(exps))
	}
}

func TestEval_RecordFeedback_Good(t *testing.T) {
	e := newSeededEval(t)
	if r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ethics-probes"}); !r.OK {
		t.Fatalf("create experiment: %v", r.Error())
	}
	rows := []Feedback{
		{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8, Source: SourceEvaluator},
		{ID: "fb-2", Target: "exp-1", Key: "ethics", Score: 0.6, Comment: "borderline", Source: SourceHuman},
		{ID: "fb-3", Target: "exp-1", Key: "helpfulness", Score: 0.9, Source: SourceHeuristic},
	}
	for _, fb := range rows {
		if r := e.RecordFeedback(fb); !r.OK {
			t.Fatalf("record %s: %v", fb.ID, r.Error())
		}
	}
}

func TestEval_RecordFeedback_Bad(t *testing.T) {
	e := newSeededEval(t)

	// Feedback with an empty id is rejected.
	if r := e.RecordFeedback(Feedback{ID: "", Target: "exp-1", Key: "ethics"}); r.OK {
		t.Fatalf("empty feedback id should fail, got %+v", r.Value)
	}

	// Feedback with an empty target run/example id is rejected.
	if r := e.RecordFeedback(Feedback{ID: "fb-1", Target: "", Key: "ethics"}); r.OK {
		t.Fatalf("empty target should fail, got %+v", r.Value)
	}

	// Feedback with an empty key is rejected — aggregation is keyed by it.
	if r := e.RecordFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: ""}); r.OK {
		t.Fatalf("empty key should fail, got %+v", r.Value)
	}

	// An unknown feedback source is rejected.
	if r := e.RecordFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Source: Source("robot")}); r.OK {
		t.Fatalf("unknown source should fail, got %+v", r.Value)
	}
}

func TestEval_RecordFeedback_Ugly(t *testing.T) {
	e := newSeededEval(t)

	// An empty source defaults to the evaluator source (the common machine
	// path), so omitting it is allowed.
	if r := e.RecordFeedback(Feedback{ID: "fb-default", Target: "ex-1", Key: "ethics", Score: 0.5}); !r.OK {
		t.Fatalf("empty source should default and succeed: %v", r.Error())
	}
	one := e.ListFeedback("ex-1")
	if !one.OK {
		t.Fatalf("list ex-1: %v", one.Error())
	}
	if src := one.Value.([]Feedback)[0].Source; src != SourceEvaluator {
		t.Errorf("defaulted source: got %q, want %q", src, SourceEvaluator)
	}

	// Re-recording the same feedback id is a duplicate and is rejected.
	if r := e.RecordFeedback(Feedback{ID: "fb-default", Target: "ex-1", Key: "ethics", Score: 0.5}); r.OK {
		t.Fatalf("duplicate feedback id should fail, got %+v", r.Value)
	}
}

func TestEval_ListFeedback_Good(t *testing.T) {
	e := newSeededEval(t)
	if r := e.RecordFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8}); !r.OK {
		t.Fatalf("record fb-1: %v", r.Error())
	}
	if r := e.RecordFeedback(Feedback{ID: "fb-2", Target: "exp-1", Key: "helpfulness", Score: 0.9}); !r.OK {
		t.Fatalf("record fb-2: %v", r.Error())
	}

	got := e.ListFeedback("exp-1")
	if !got.OK {
		t.Fatalf("list feedback: %v", got.Error())
	}
	if fbs := got.Value.([]Feedback); len(fbs) != 2 {
		t.Fatalf("list feedback: got %d, want 2", len(fbs))
	}
}

func TestEval_ListFeedback_Bad(t *testing.T) {
	// Listing feedback for a target with none yet is an empty, successful set —
	// "no feedback" is not an error.
	e := New()
	got := e.ListFeedback("exp-unknown")
	if !got.OK {
		t.Fatalf("list of empty target should succeed: %v", got.Error())
	}
	if fbs := got.Value.([]Feedback); len(fbs) != 0 {
		t.Fatalf("empty target: got %d, want 0", len(fbs))
	}
}

func TestEval_ListFeedback_Ugly(t *testing.T) {
	// ListFeedback never validates that target names a real experiment —
	// feedback rows against an arbitrary string target still list.
	e := New()
	if r := e.RecordFeedback(Feedback{ID: "fb-1", Target: "never-created", Key: "ethics", Score: 0.5}); !r.OK {
		t.Fatalf("record: %v", r.Error())
	}
	got := e.ListFeedback("never-created")
	if !got.OK {
		t.Fatalf("list: %v", got.Error())
	}
	if fbs := got.Value.([]Feedback); len(fbs) != 1 {
		t.Errorf("got %d rows, want 1", len(fbs))
	}
}

func TestEval_AggregateFeedback_Good(t *testing.T) {
	e := newSeededEval(t)
	if r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ethics-probes"}); !r.OK {
		t.Fatalf("create experiment: %v", r.Error())
	}
	rows := []Feedback{
		{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8, Source: SourceEvaluator},
		{ID: "fb-2", Target: "exp-1", Key: "ethics", Score: 0.6, Source: SourceHuman},
		{ID: "fb-3", Target: "exp-1", Key: "helpfulness", Score: 0.9, Source: SourceHeuristic},
	}
	for _, fb := range rows {
		if r := e.RecordFeedback(fb); !r.OK {
			t.Fatalf("record %s: %v", fb.ID, r.Error())
		}
	}

	agg := e.AggregateFeedback("exp-1")
	if !agg.OK {
		t.Fatalf("aggregate: %v", agg.Error())
	}
	means := agg.Value.(map[string]float64)
	if got := means["ethics"]; got != 0.7 { // (0.8 + 0.6) / 2
		t.Errorf("ethics mean: got %v, want 0.7", got)
	}
	if got := means["helpfulness"]; got != 0.9 {
		t.Errorf("helpfulness mean: got %v, want 0.9", got)
	}
}

func TestEval_AggregateFeedback_Bad(t *testing.T) {
	// Aggregating a target with no feedback yields an empty map, not an error.
	e := New()
	agg := e.AggregateFeedback("exp-unknown")
	if !agg.OK {
		t.Fatalf("aggregate of empty target should succeed: %v", agg.Error())
	}
	if means := agg.Value.(map[string]float64); len(means) != 0 {
		t.Fatalf("empty aggregate: got %d keys, want 0", len(means))
	}
}

func TestEval_AggregateFeedback_Ugly(t *testing.T) {
	// A single negative score aggregates to itself (no abs / clamping).
	e := New()
	if r := e.RecordFeedback(Feedback{ID: "fb-neg", Target: "ex-2", Key: "drift", Score: -1.5, Source: SourceHeuristic}); !r.OK {
		t.Fatalf("record negative: %v", r.Error())
	}
	agg := e.AggregateFeedback("ex-2")
	if !agg.OK {
		t.Fatalf("aggregate ex-2: %v", agg.Error())
	}
	if got := agg.Value.(map[string]float64)["drift"]; got != -1.5 {
		t.Errorf("negative mean: got %v, want -1.5", got)
	}
}
