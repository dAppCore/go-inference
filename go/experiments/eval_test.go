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

func TestEval_Examples_Good(t *testing.T) {
	e := newSeededEval(t)

	// Add two examples to the dataset.
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

	// The dataset resolves and reports its own metadata back.
	ds := e.GetDataset("ethics-probes")
	if !ds.OK {
		t.Fatalf("get dataset: %v", ds.Error())
	}
	if got := ds.Value.(Dataset).Name; got != "Ethics probes" {
		t.Errorf("dataset name: got %q, want %q", got, "Ethics probes")
	}

	// A version snapshots the current example ids under a tag.
	ver := e.Snapshot("ethics-probes", "v1")
	if !ver.OK {
		t.Fatalf("snapshot: %v", ver.Error())
	}
	v := ver.Value.(Version)
	if v.Tag != "v1" || len(v.ExampleIDs) != 2 {
		t.Errorf("version: got tag %q with %d ids, want v1 with 2", v.Tag, len(v.ExampleIDs))
	}
}

func TestEval_Examples_Bad(t *testing.T) {
	e := New()

	// Adding an example to a dataset that does not exist fails.
	if r := e.AddExample(sampleExample("ex-1", "ghost")); r.OK {
		t.Fatalf("add to missing dataset should fail, got %+v", r.Value)
	}

	// Listing examples of an unknown dataset fails.
	if r := e.ListExamples("ghost"); r.OK {
		t.Fatalf("list of missing dataset should fail, got %+v", r.Value)
	}

	// Getting an unknown dataset fails.
	if r := e.GetDataset("ghost"); r.OK {
		t.Fatalf("get of missing dataset should fail, got %+v", r.Value)
	}

	// A dataset with an empty id is rejected.
	if r := e.PutDataset(Dataset{ID: ""}); r.OK {
		t.Fatalf("empty dataset id should fail, got %+v", r.Value)
	}

	// Snapshot of an unknown dataset fails.
	if r := e.Snapshot("ghost", "v1"); r.OK {
		t.Fatalf("snapshot of missing dataset should fail, got %+v", r.Value)
	}
}

func TestEval_Examples_Ugly(t *testing.T) {
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

	// Splits group example ids by their declared split name.
	holdout := sampleExample("ho-1", "ethics-probes")
	holdout.Split = SplitTest
	if r := e.AddExample(holdout); !r.OK {
		t.Fatalf("add holdout: %v", r.Error())
	}
	sp := e.Splits("ethics-probes")
	if !sp.OK {
		t.Fatalf("splits: %v", sp.Error())
	}
	splits := sp.Value.(map[Split][]string)
	if len(splits[SplitTrain]) != 1 || splits[SplitTrain][0] != "dup" {
		t.Errorf("train split: got %v, want [dup]", splits[SplitTrain])
	}
	if len(splits[SplitTest]) != 1 || splits[SplitTest][0] != "ho-1" {
		t.Errorf("test split: got %v, want [ho-1]", splits[SplitTest])
	}
}

func TestEval_Experiment_Good(t *testing.T) {
	e := newSeededEval(t)

	// Create an experiment over the dataset.
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

	// A second experiment over the same dataset.
	if r := e.CreateExperiment(Experiment{ID: "exp-2", DatasetID: "ethics-probes", Run: "lemrd@step-1"}); !r.OK {
		t.Fatalf("create exp-2: %v", r.Error())
	}

	// List experiments for the dataset, sorted by id.
	got := e.ListExperiments("ethics-probes")
	if !got.OK {
		t.Fatalf("list experiments: %v", got.Error())
	}
	exps := got.Value.([]Experiment)
	if len(exps) != 2 || exps[0].ID != "exp-1" || exps[1].ID != "exp-2" {
		t.Errorf("list experiments: got %d in unexpected order", len(exps))
	}

	// Get a single experiment by id.
	one := e.GetExperiment("exp-1")
	if !one.OK {
		t.Fatalf("get experiment: %v", one.Error())
	}
	if got := one.Value.(Experiment).Run; got != "lemma@step-3000" {
		t.Errorf("run: got %q, want %q", got, "lemma@step-3000")
	}
}

func TestEval_Experiment_Bad(t *testing.T) {
	e := New()

	// An experiment over a dataset that does not exist is rejected.
	if r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ghost"}); r.OK {
		t.Fatalf("experiment over missing dataset should fail, got %+v", r.Value)
	}

	// Listing experiments for an unknown dataset fails.
	if r := e.ListExperiments("ghost"); r.OK {
		t.Fatalf("list for missing dataset should fail, got %+v", r.Value)
	}

	// Getting an unknown experiment fails.
	if r := e.GetExperiment("nope"); r.OK {
		t.Fatalf("get of missing experiment should fail, got %+v", r.Value)
	}
}

func TestEval_Experiment_Ugly(t *testing.T) {
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

func TestEval_Feedback_Good(t *testing.T) {
	e := newSeededEval(t)
	if r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ethics-probes"}); !r.OK {
		t.Fatalf("create experiment: %v", r.Error())
	}

	// Record three feedback rows against the experiment: two on "ethics", one
	// on "helpfulness".
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

	// List feedback for the experiment, sorted by id.
	got := e.ListFeedback("exp-1")
	if !got.OK {
		t.Fatalf("list feedback: %v", got.Error())
	}
	if fbs := got.Value.([]Feedback); len(fbs) != 3 {
		t.Fatalf("list feedback: got %d, want 3", len(fbs))
	}

	// Aggregate the mean score per key for the experiment.
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

func TestEval_Feedback_Bad(t *testing.T) {
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

func TestEval_Feedback_Ugly(t *testing.T) {
	e := newSeededEval(t)

	// Listing feedback for a target with none yet is an empty, successful set —
	// "no feedback" is not an error.
	got := e.ListFeedback("exp-unknown")
	if !got.OK {
		t.Fatalf("list of empty target should succeed: %v", got.Error())
	}
	if fbs := got.Value.([]Feedback); len(fbs) != 0 {
		t.Fatalf("empty target: got %d, want 0", len(fbs))
	}

	// Aggregating a target with no feedback yields an empty map, not an error.
	agg := e.AggregateFeedback("exp-unknown")
	if !agg.OK {
		t.Fatalf("aggregate of empty target should succeed: %v", agg.Error())
	}
	if means := agg.Value.(map[string]float64); len(means) != 0 {
		t.Fatalf("empty aggregate: got %d keys, want 0", len(means))
	}

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

	// A single negative score aggregates to itself (no abs / clamping).
	if r := e.RecordFeedback(Feedback{ID: "fb-neg", Target: "ex-2", Key: "drift", Score: -1.5, Source: SourceHeuristic}); !r.OK {
		t.Fatalf("record negative: %v", r.Error())
	}
	agg = e.AggregateFeedback("ex-2")
	if !agg.OK {
		t.Fatalf("aggregate ex-2: %v", agg.Error())
	}
	if got := agg.Value.(map[string]float64)["drift"]; got != -1.5 {
		t.Errorf("negative mean: got %v, want -1.5", got)
	}
}
