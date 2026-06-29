// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import (
	"cmp"
	"slices"
	"sync"

	core "dappco.re/go"
)

// Store is the pluggable persistence behind an Eval. The default is MemStore; a
// go-store / DuckDB implementation slots in unchanged (out of scope here).
// Datasets and experiments are keyed by their own id; examples are keyed per
// dataset (the same example id may recur in different datasets); feedback is
// keyed by id and listed by target.
//
//	var s eval.Store = experiments.NewMemStore()
//	s.PutDataset(experiments.Dataset{ID: "ethics-probes"})
type Store interface {
	// PutDataset inserts or replaces a dataset by its id.
	//
	//	s.PutDataset(experiments.Dataset{ID: "ethics-probes"})
	PutDataset(d Dataset) core.Result

	// GetDataset returns the dataset for id, or a failed Result when absent.
	//
	//	r := s.GetDataset("ethics-probes")
	GetDataset(id string) core.Result

	// PutExample inserts an example under its dataset. A failed Result reports a
	// duplicate id within that dataset (callers add, they do not silently
	// overwrite).
	//
	//	s.PutExample(experiments.Example{ID: "ex-1", DatasetID: "ethics-probes"})
	PutExample(ex Example) core.Result

	// ListExamples returns every example in datasetID, sorted by example id.
	//
	//	for _, ex := range s.ListExamples("ethics-probes") { ... }
	ListExamples(datasetID string) []Example

	// PutExperiment inserts an experiment by its id. A failed Result reports a
	// duplicate id.
	//
	//	s.PutExperiment(experiments.Experiment{ID: "exp-1", DatasetID: "ethics-probes"})
	PutExperiment(x Experiment) core.Result

	// GetExperiment returns the experiment for id, or a failed Result when
	// absent.
	//
	//	r := s.GetExperiment("exp-1")
	GetExperiment(id string) core.Result

	// UpdateExperiment replaces an existing experiment in place by its id — the
	// status-transition counterpart to PutExperiment's insert. A failed Result
	// reports an unknown id (update never creates).
	//
	//	s.UpdateExperiment(eval.Experiment{ID: "exp-1", DatasetID: "ethics-probes", Status: experiments.StatusComplete})
	UpdateExperiment(x Experiment) core.Result

	// ListExperiments returns every experiment over datasetID, sorted by id.
	//
	//	for _, x := range s.ListExperiments("ethics-probes") { ... }
	ListExperiments(datasetID string) []Experiment

	// PutFeedback inserts a feedback row by its id. A failed Result reports a
	// duplicate id.
	//
	//	s.PutFeedback(experiments.Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8})
	PutFeedback(fb Feedback) core.Result

	// ListFeedback returns every feedback row for target, sorted by id. An
	// unknown target is an empty slice, not an error.
	//
	//	for _, fb := range s.ListFeedback("exp-1") { ... }
	ListFeedback(target string) []Feedback
}

// MemStore is an in-memory, goroutine-safe Store — the default backing for an
// Eval and the store used in tests.
//
//	s := experiments.NewMemStore()
type MemStore struct {
	mu          sync.RWMutex
	datasets    map[string]Dataset
	examples    map[string]map[string]Example // datasetID → exampleID → Example
	experiments map[string]Experiment
	feedback    map[string]Feedback
}

// NewMemStore returns an empty in-memory Store.
//
//	e := eval.NewWithStore(experiments.NewMemStore())
func NewMemStore() *MemStore {
	return &MemStore{
		datasets:    map[string]Dataset{},
		examples:    map[string]map[string]Example{},
		experiments: map[string]Experiment{},
		feedback:    map[string]Feedback{},
	}
}

// PutDataset inserts or replaces d by its id.
//
//	s.PutDataset(experiments.Dataset{ID: "ethics-probes"})
func (s *MemStore) PutDataset(d Dataset) core.Result {
	if d.ID == "" {
		return core.Fail(core.E("experiments.MemStore.PutDataset", "dataset id is empty", nil))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.datasets[d.ID] = d
	return core.Ok(d)
}

// GetDataset returns the dataset for id.
//
//	r := s.GetDataset("ethics-probes")
func (s *MemStore) GetDataset(id string) core.Result {
	s.mu.RLock()
	defer s.mu.RUnlock()
	d, ok := s.datasets[id]
	if !ok {
		return core.Fail(core.E("experiments.MemStore.GetDataset", core.Sprintf("no dataset with id %q", id), nil))
	}
	return core.Ok(d)
}

// PutExample inserts ex under its dataset, rejecting a duplicate example id
// within that dataset.
//
//	s.PutExample(experiments.Example{ID: "ex-1", DatasetID: "ethics-probes"})
func (s *MemStore) PutExample(ex Example) core.Result {
	if ex.ID == "" {
		return core.Fail(core.E("experiments.MemStore.PutExample", "example id is empty", nil))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	byID, ok := s.examples[ex.DatasetID]
	if !ok {
		byID = map[string]Example{}
		s.examples[ex.DatasetID] = byID
	}
	if _, dup := byID[ex.ID]; dup {
		return core.Fail(core.E("experiments.MemStore.PutExample",
			core.Sprintf("example %q already exists in dataset %q", ex.ID, ex.DatasetID), nil))
	}
	byID[ex.ID] = ex
	return core.Ok(ex)
}

// ListExamples returns every example in datasetID sorted by id.
//
//	all := s.ListExamples("ethics-probes")
func (s *MemStore) ListExamples(datasetID string) []Example {
	s.mu.RLock()
	byID := s.examples[datasetID]
	out := make([]Example, 0, len(byID))
	for _, ex := range byID {
		out = append(out, ex)
	}
	s.mu.RUnlock()
	slices.SortFunc(out, func(a, b Example) int { return cmp.Compare(a.ID, b.ID) })
	return out
}

// PutExperiment inserts x by its id, rejecting a duplicate id.
//
//	s.PutExperiment(experiments.Experiment{ID: "exp-1", DatasetID: "ethics-probes"})
func (s *MemStore) PutExperiment(x Experiment) core.Result {
	if x.ID == "" {
		return core.Fail(core.E("experiments.MemStore.PutExperiment", "experiment id is empty", nil))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, dup := s.experiments[x.ID]; dup {
		return core.Fail(core.E("experiments.MemStore.PutExperiment",
			core.Sprintf("experiment %q already exists", x.ID), nil))
	}
	s.experiments[x.ID] = x
	return core.Ok(x)
}

// GetExperiment returns the experiment for id.
//
//	r := s.GetExperiment("exp-1")
func (s *MemStore) GetExperiment(id string) core.Result {
	s.mu.RLock()
	defer s.mu.RUnlock()
	x, ok := s.experiments[id]
	if !ok {
		return core.Fail(core.E("experiments.MemStore.GetExperiment", core.Sprintf("no experiment with id %q", id), nil))
	}
	return core.Ok(x)
}

// UpdateExperiment replaces x in place by its id, rejecting an unknown id
// (update never inserts — use PutExperiment to create).
//
//	s.UpdateExperiment(eval.Experiment{ID: "exp-1", DatasetID: "ethics-probes", Status: experiments.StatusComplete})
func (s *MemStore) UpdateExperiment(x Experiment) core.Result {
	if x.ID == "" {
		return core.Fail(core.E("experiments.MemStore.UpdateExperiment", "experiment id is empty", nil))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.experiments[x.ID]; !ok {
		return core.Fail(core.E("experiments.MemStore.UpdateExperiment",
			core.Sprintf("no experiment with id %q", x.ID), nil))
	}
	s.experiments[x.ID] = x
	return core.Ok(x)
}

// ListExperiments returns every experiment over datasetID sorted by id.
//
//	all := s.ListExperiments("ethics-probes")
func (s *MemStore) ListExperiments(datasetID string) []Experiment {
	s.mu.RLock()
	out := make([]Experiment, 0, len(s.experiments))
	for _, x := range s.experiments {
		if x.DatasetID == datasetID {
			out = append(out, x)
		}
	}
	s.mu.RUnlock()
	slices.SortFunc(out, func(a, b Experiment) int { return cmp.Compare(a.ID, b.ID) })
	return out
}

// PutFeedback inserts fb by its id, rejecting a duplicate id.
//
//	s.PutFeedback(experiments.Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8})
func (s *MemStore) PutFeedback(fb Feedback) core.Result {
	if fb.ID == "" {
		return core.Fail(core.E("experiments.MemStore.PutFeedback", "feedback id is empty", nil))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, dup := s.feedback[fb.ID]; dup {
		return core.Fail(core.E("experiments.MemStore.PutFeedback",
			core.Sprintf("feedback %q already exists", fb.ID), nil))
	}
	s.feedback[fb.ID] = fb
	return core.Ok(fb)
}

// ListFeedback returns every feedback row for target sorted by id. An unknown
// target yields an empty slice, not an error.
//
//	rows := s.ListFeedback("exp-1")
func (s *MemStore) ListFeedback(target string) []Feedback {
	s.mu.RLock()
	out := make([]Feedback, 0, len(s.feedback))
	for _, fb := range s.feedback {
		if fb.Target == target {
			out = append(out, fb)
		}
	}
	s.mu.RUnlock()
	slices.SortFunc(out, func(a, b Feedback) int { return cmp.Compare(a.ID, b.ID) })
	return out
}
