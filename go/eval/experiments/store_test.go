// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import "testing"

func TestStore_NewMemStore_Good(t *testing.T) {
	s := NewMemStore()
	if r := s.GetDataset("ds"); r.OK {
		t.Fatalf("fresh store should have no datasets, got %+v", r.Value)
	}
}

func TestStore_NewMemStore_Bad(t *testing.T) {
	// Two constructions are independent — writing to one must not leak into the
	// other.
	a, b := NewMemStore(), NewMemStore()
	if r := a.PutDataset(Dataset{ID: "ds"}); !r.OK {
		t.Fatalf("put on a: %v", r.Error())
	}
	if r := b.GetDataset("ds"); r.OK {
		t.Fatalf("b should not see a's dataset, got %+v", r.Value)
	}
}

func TestStore_NewMemStore_Ugly(t *testing.T) {
	// Listing against a dataset that was never seeded is safe (a nil inner map
	// ranges to zero results) rather than panicking.
	s := NewMemStore()
	if exs := s.ListExamples("never-seeded"); len(exs) != 0 {
		t.Errorf("unseeded dataset examples: got %d, want 0", len(exs))
	}
	if exps := s.ListExperiments("never-seeded"); len(exps) != 0 {
		t.Errorf("unseeded dataset experiments: got %d, want 0", len(exps))
	}
	if fbs := s.ListFeedback("never-seeded"); len(fbs) != 0 {
		t.Errorf("unseeded target feedback: got %d, want 0", len(fbs))
	}
}

func TestStore_MemStore_PutDataset_Good(t *testing.T) {
	s := NewMemStore()
	r := s.PutDataset(Dataset{ID: "ds", Name: "Dataset"})
	if !r.OK {
		t.Fatalf("put: %v", r.Error())
	}
	if got := r.Value.(Dataset).Name; got != "Dataset" {
		t.Errorf("returned value: got %q, want %q", got, "Dataset")
	}
}

func TestStore_MemStore_PutDataset_Bad(t *testing.T) {
	s := NewMemStore()
	if r := s.PutDataset(Dataset{ID: ""}); r.OK {
		t.Fatalf("empty id should fail, got %+v", r.Value)
	}
}

func TestStore_MemStore_PutDataset_Ugly(t *testing.T) {
	// PutDataset inserts OR REPLACES — a second put under the same id overwrites
	// the first rather than being rejected as a duplicate.
	s := NewMemStore()
	if r := s.PutDataset(Dataset{ID: "ds", Name: "first"}); !r.OK {
		t.Fatalf("first put: %v", r.Error())
	}
	if r := s.PutDataset(Dataset{ID: "ds", Name: "second"}); !r.OK {
		t.Fatalf("replacing put: %v", r.Error())
	}
	if got := s.GetDataset("ds").Value.(Dataset).Name; got != "second" {
		t.Errorf("after replace: got %q, want %q", got, "second")
	}
}

func TestStore_MemStore_GetDataset_Good(t *testing.T) {
	s := NewMemStore()
	want := Dataset{ID: "ds", Name: "Dataset", Description: "desc"}
	if r := s.PutDataset(want); !r.OK {
		t.Fatalf("put: %v", r.Error())
	}
	got := s.GetDataset("ds")
	if !got.OK {
		t.Fatalf("get: %v", got.Error())
	}
	if got.Value.(Dataset) != want {
		t.Errorf("roundtrip: got %+v, want %+v", got.Value, want)
	}
}

func TestStore_MemStore_GetDataset_Bad(t *testing.T) {
	s := NewMemStore()
	if r := s.GetDataset("ghost"); r.OK {
		t.Fatalf("unknown id should fail, got %+v", r.Value)
	}
}

func TestStore_MemStore_GetDataset_Ugly(t *testing.T) {
	// A dataset with only its id set round-trips its zero-value optional fields
	// rather than substituting a default.
	s := NewMemStore()
	if r := s.PutDataset(Dataset{ID: "bare"}); !r.OK {
		t.Fatalf("put: %v", r.Error())
	}
	got := s.GetDataset("bare").Value.(Dataset)
	if got.Name != "" || got.Description != "" {
		t.Errorf("bare dataset: got %+v, want empty Name/Description", got)
	}
}

func TestStore_MemStore_PutExample_Good(t *testing.T) {
	s := NewMemStore()
	ex := Example{ID: "ex-1", DatasetID: "ds", Inputs: map[string]any{"prompt": "hi"}}
	r := s.PutExample(ex)
	if !r.OK {
		t.Fatalf("put: %v", r.Error())
	}
	if got := r.Value.(Example).ID; got != "ex-1" {
		t.Errorf("returned value: got %q, want ex-1", got)
	}
}

func TestStore_MemStore_PutExample_Bad(t *testing.T) {
	// The store's own empty-id guard (the façade catches empty ids first, so this
	// is only reachable by a direct store call).
	s := NewMemStore()
	if r := s.PutExample(Example{ID: "", DatasetID: "ds"}); r.OK {
		t.Fatalf("MemStore.PutExample with empty id should fail, got %+v", r.Value)
	}
}

func TestStore_MemStore_PutExample_Ugly(t *testing.T) {
	s := NewMemStore()
	if r := s.PutExample(Example{ID: "dup", DatasetID: "ds"}); !r.OK {
		t.Fatalf("first put: %v", r.Error())
	}
	// Re-adding the same id within the SAME dataset is a duplicate.
	if r := s.PutExample(Example{ID: "dup", DatasetID: "ds"}); r.OK {
		t.Fatalf("duplicate id within a dataset should fail, got %+v", r.Value)
	}
	// The same id in a DIFFERENT dataset is not a duplicate — ids are scoped per
	// dataset.
	if r := s.PutExample(Example{ID: "dup", DatasetID: "other"}); !r.OK {
		t.Fatalf("same id in a different dataset should succeed: %v", r.Error())
	}
}

func TestStore_MemStore_ListExamples_Good(t *testing.T) {
	s := NewMemStore()
	s.PutExample(Example{ID: "ex-1", DatasetID: "ds"})
	s.PutExample(Example{ID: "ex-2", DatasetID: "ds"})
	got := s.ListExamples("ds")
	if len(got) != 2 {
		t.Fatalf("got %d examples, want 2", len(got))
	}
}

func TestStore_MemStore_ListExamples_Bad(t *testing.T) {
	s := NewMemStore()
	if got := s.ListExamples("ghost"); len(got) != 0 {
		t.Errorf("unknown dataset: got %d examples, want 0", len(got))
	}
}

func TestStore_MemStore_ListExamples_Ugly(t *testing.T) {
	// Examples come back sorted by id regardless of insertion order.
	s := NewMemStore()
	s.PutExample(Example{ID: "z-last", DatasetID: "ds"})
	s.PutExample(Example{ID: "a-first", DatasetID: "ds"})
	got := s.ListExamples("ds")
	if len(got) != 2 || got[0].ID != "a-first" || got[1].ID != "z-last" {
		t.Errorf("sort order: got %v", got)
	}
}

func TestStore_MemStore_PutExperiment_Good(t *testing.T) {
	s := NewMemStore()
	r := s.PutExperiment(Experiment{ID: "exp-1", DatasetID: "ds"})
	if !r.OK {
		t.Fatalf("put: %v", r.Error())
	}
}

func TestStore_MemStore_PutExperiment_Bad(t *testing.T) {
	s := NewMemStore()
	if r := s.PutExperiment(Experiment{ID: "", DatasetID: "ds"}); r.OK {
		t.Fatalf("MemStore.PutExperiment with empty id should fail, got %+v", r.Value)
	}
}

func TestStore_MemStore_PutExperiment_Ugly(t *testing.T) {
	// Experiment ids are global, unlike example ids — a duplicate is rejected
	// even under a different dataset.
	s := NewMemStore()
	if r := s.PutExperiment(Experiment{ID: "exp-1", DatasetID: "ds-a"}); !r.OK {
		t.Fatalf("first put: %v", r.Error())
	}
	if r := s.PutExperiment(Experiment{ID: "exp-1", DatasetID: "ds-b"}); r.OK {
		t.Fatalf("duplicate experiment id should fail even under a different dataset, got %+v", r.Value)
	}
}

func TestStore_MemStore_GetExperiment_Good(t *testing.T) {
	s := NewMemStore()
	want := Experiment{ID: "exp-1", DatasetID: "ds", Run: "lemma@1", Status: StatusPending, Created: 100}
	s.PutExperiment(want)
	got := s.GetExperiment("exp-1")
	if !got.OK {
		t.Fatalf("get: %v", got.Error())
	}
	if got.Value.(Experiment) != want {
		t.Errorf("roundtrip: got %+v, want %+v", got.Value, want)
	}
}

func TestStore_MemStore_GetExperiment_Bad(t *testing.T) {
	s := NewMemStore()
	if r := s.GetExperiment("ghost"); r.OK {
		t.Fatalf("unknown id should fail, got %+v", r.Value)
	}
}

func TestStore_MemStore_GetExperiment_Ugly(t *testing.T) {
	// An experiment with only its required fields round-trips its zero-value
	// optional fields (Run, Created, Status) unchanged.
	s := NewMemStore()
	s.PutExperiment(Experiment{ID: "bare", DatasetID: "ds"})
	got := s.GetExperiment("bare").Value.(Experiment)
	if got.Run != "" || got.Created != 0 || got.Status != "" {
		t.Errorf("bare experiment: got %+v, want zero optional fields", got)
	}
}

func TestStore_MemStore_UpdateExperiment_Good(t *testing.T) {
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

func TestStore_MemStore_UpdateExperiment_Bad(t *testing.T) {
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

func TestStore_MemStore_UpdateExperiment_Ugly(t *testing.T) {
	// Update REPLACES the full record — it does not merge. A caller who omits a
	// field (e.g. Run) zeroes it out, it does not preserve the old value.
	s := NewMemStore()
	s.PutExperiment(Experiment{ID: "exp-1", DatasetID: "ds", Run: "lemma@1", Status: StatusPending})
	if r := s.UpdateExperiment(Experiment{ID: "exp-1", DatasetID: "ds", Status: StatusComplete}); !r.OK {
		t.Fatalf("update: %v", r.Error())
	}
	got := s.GetExperiment("exp-1").Value.(Experiment)
	if got.Run != "" {
		t.Errorf("update should replace the whole record: Run survived as %q, want zeroed", got.Run)
	}
}

func TestStore_MemStore_ListExperiments_Good(t *testing.T) {
	s := NewMemStore()
	s.PutExperiment(Experiment{ID: "exp-2", DatasetID: "ds"})
	s.PutExperiment(Experiment{ID: "exp-1", DatasetID: "ds"})
	got := s.ListExperiments("ds")
	if len(got) != 2 || got[0].ID != "exp-1" || got[1].ID != "exp-2" {
		t.Errorf("sorted list: got %v", got)
	}
}

func TestStore_MemStore_ListExperiments_Bad(t *testing.T) {
	s := NewMemStore()
	if got := s.ListExperiments("ghost"); len(got) != 0 {
		t.Errorf("unknown dataset: got %d, want 0", len(got))
	}
}

func TestStore_MemStore_ListExperiments_Ugly(t *testing.T) {
	// Experiments over a different dataset are excluded from the list.
	s := NewMemStore()
	s.PutExperiment(Experiment{ID: "exp-a", DatasetID: "ds-a"})
	s.PutExperiment(Experiment{ID: "exp-b", DatasetID: "ds-b"})
	got := s.ListExperiments("ds-a")
	if len(got) != 1 || got[0].ID != "exp-a" {
		t.Errorf("cross-dataset filter: got %v, want just exp-a", got)
	}
}

func TestStore_MemStore_PutFeedback_Good(t *testing.T) {
	s := NewMemStore()
	r := s.PutFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.5})
	if !r.OK {
		t.Fatalf("put: %v", r.Error())
	}
}

func TestStore_MemStore_PutFeedback_Bad(t *testing.T) {
	s := NewMemStore()
	if r := s.PutFeedback(Feedback{ID: "", Target: "exp-1", Key: "k"}); r.OK {
		t.Fatalf("MemStore.PutFeedback with empty id should fail, got %+v", r.Value)
	}
}

func TestStore_MemStore_PutFeedback_Ugly(t *testing.T) {
	// Feedback ids are global — a duplicate is rejected even against a
	// different target.
	s := NewMemStore()
	if r := s.PutFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: "k"}); !r.OK {
		t.Fatalf("first put: %v", r.Error())
	}
	if r := s.PutFeedback(Feedback{ID: "fb-1", Target: "exp-2", Key: "k"}); r.OK {
		t.Fatalf("duplicate feedback id should fail even against a different target, got %+v", r.Value)
	}
}

func TestStore_MemStore_ListFeedback_Good(t *testing.T) {
	s := NewMemStore()
	s.PutFeedback(Feedback{ID: "fb-2", Target: "exp-1", Key: "k"})
	s.PutFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: "k"})
	got := s.ListFeedback("exp-1")
	if len(got) != 2 || got[0].ID != "fb-1" || got[1].ID != "fb-2" {
		t.Errorf("sorted list: got %v", got)
	}
}

func TestStore_MemStore_ListFeedback_Bad(t *testing.T) {
	s := NewMemStore()
	if got := s.ListFeedback("ghost"); len(got) != 0 {
		t.Errorf("unknown target: got %d, want 0", len(got))
	}
}

func TestStore_MemStore_ListFeedback_Ugly(t *testing.T) {
	// Feedback rows for a different target are excluded from the list.
	s := NewMemStore()
	s.PutFeedback(Feedback{ID: "fb-a", Target: "exp-a", Key: "k"})
	s.PutFeedback(Feedback{ID: "fb-b", Target: "exp-b", Key: "k"})
	got := s.ListFeedback("exp-a")
	if len(got) != 1 || got[0].ID != "fb-a" {
		t.Errorf("cross-target filter: got %v, want just fb-a", got)
	}
}
