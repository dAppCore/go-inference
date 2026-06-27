// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import (
	"sort"

	core "dappco.re/go"
)

// Eval is the evaluation-data façade. It wraps a Store with the dataset-aware
// guards and aggregation go-ml's evaluation loop consumes: examples and
// experiments must belong to a known dataset, ids are validated, feedback rows
// roll up to a mean score per key.
//
//	e := eval.New()
//	e.PutDataset(eval.Dataset{ID: "ethics-probes"})
//	e.AddExample(eval.Example{ID: "ex-1", DatasetID: "ethics-probes"})
//	e.CreateExperiment(eval.Experiment{ID: "exp-1", DatasetID: "ethics-probes"})
type Eval struct {
	store Store
}

// New returns an Eval backed by an in-memory Store.
//
//	e := eval.New()
func New() *Eval {
	return NewWithStore(NewMemStore())
}

// NewWithStore returns an Eval over a caller-supplied Store — e.g. a go-store /
// DuckDB implementation in production.
//
//	e := eval.NewWithStore(eval.NewMemStore())
func NewWithStore(s Store) *Eval {
	return &Eval{store: s}
}

// PutDataset inserts or replaces a dataset. An empty id is rejected.
//
//	e.PutDataset(eval.Dataset{ID: "ethics-probes", Name: "Ethics probes"})
func (e *Eval) PutDataset(d Dataset) core.Result {
	return e.store.PutDataset(d)
}

// GetDataset returns the dataset for id, or a failed Result when absent.
//
//	e.GetDataset("ethics-probes")
func (e *Eval) GetDataset(id string) core.Result {
	return e.store.GetDataset(id)
}

// AddExample adds an example to its dataset. The dataset must already exist and
// the example id must be unique within it; an empty example id is rejected.
//
//	e.AddExample(eval.Example{ID: "ex-1", DatasetID: "ethics-probes",
//	    Inputs: map[string]any{"prompt": "Is honesty always right?"}})
func (e *Eval) AddExample(ex Example) core.Result {
	if ex.ID == "" {
		return core.Fail(core.E("eval.AddExample", "example id is empty", nil))
	}
	if d := e.store.GetDataset(ex.DatasetID); !d.OK {
		return core.Fail(core.E("eval.AddExample",
			core.Sprintf("no dataset with id %q", ex.DatasetID), nil))
	}
	return e.store.PutExample(ex)
}

// ListExamples returns the examples in datasetID, sorted by id. An unknown
// dataset is an error (distinct from a known dataset with no examples yet).
//
//	exs := e.ListExamples("ethics-probes").Value.([]eval.Example)
func (e *Eval) ListExamples(datasetID string) core.Result {
	if d := e.store.GetDataset(datasetID); !d.OK {
		return core.Fail(core.E("eval.ListExamples",
			core.Sprintf("no dataset with id %q", datasetID), nil))
	}
	return core.Ok(e.store.ListExamples(datasetID))
}

// Splits groups a dataset's example ids by their declared split. Examples with
// no split fall under the empty-string key. The dataset must exist.
//
//	splits := e.Splits("ethics-probes").Value.(map[eval.Split][]string)
//	holdout := splits[eval.SplitTest]
func (e *Eval) Splits(datasetID string) core.Result {
	if d := e.store.GetDataset(datasetID); !d.OK {
		return core.Fail(core.E("eval.Splits",
			core.Sprintf("no dataset with id %q", datasetID), nil))
	}
	out := map[Split][]string{}
	for _, ex := range e.store.ListExamples(datasetID) {
		out[ex.Split] = append(out[ex.Split], ex.ID)
	}
	// ListExamples is already id-sorted, so each split slice is too.
	return core.Ok(out)
}

// Snapshot captures the current example ids of a dataset under a version tag —
// an immutable record of exactly which examples a run evaluated. The dataset
// must exist and the tag must be non-empty.
//
//	v := e.Snapshot("ethics-probes", "v1").Value.(eval.Version)
func (e *Eval) Snapshot(datasetID, tag string) core.Result {
	if tag == "" {
		return core.Fail(core.E("eval.Snapshot", "version tag is empty", nil))
	}
	if d := e.store.GetDataset(datasetID); !d.OK {
		return core.Fail(core.E("eval.Snapshot",
			core.Sprintf("no dataset with id %q", datasetID), nil))
	}
	exs := e.store.ListExamples(datasetID)
	ids := make([]string, 0, len(exs))
	for _, ex := range exs {
		ids = append(ids, ex.ID)
	}
	sort.Strings(ids)
	return core.Ok(Version{DatasetID: datasetID, Tag: tag, ExampleIDs: ids})
}

// CreateExperiment registers an experiment over a dataset. The dataset must
// exist and the experiment id must be unique; an empty id is rejected. A
// zero-value Status defaults to StatusPending.
//
//	e.CreateExperiment(eval.Experiment{ID: "exp-1", DatasetID: "ethics-probes", Run: "lemma@step-3000"})
func (e *Eval) CreateExperiment(x Experiment) core.Result {
	if x.ID == "" {
		return core.Fail(core.E("eval.CreateExperiment", "experiment id is empty", nil))
	}
	if d := e.store.GetDataset(x.DatasetID); !d.OK {
		return core.Fail(core.E("eval.CreateExperiment",
			core.Sprintf("no dataset with id %q", x.DatasetID), nil))
	}
	if x.Status == "" {
		x.Status = StatusPending
	}
	return e.store.PutExperiment(x)
}

// GetExperiment returns the experiment for id, or a failed Result when absent.
//
//	e.GetExperiment("exp-1")
func (e *Eval) GetExperiment(id string) core.Result {
	return e.store.GetExperiment(id)
}

// ListExperiments returns the experiments over datasetID, sorted by id. An
// unknown dataset is an error.
//
//	exps := e.ListExperiments("ethics-probes").Value.([]eval.Experiment)
func (e *Eval) ListExperiments(datasetID string) core.Result {
	if d := e.store.GetDataset(datasetID); !d.OK {
		return core.Fail(core.E("eval.ListExperiments",
			core.Sprintf("no dataset with id %q", datasetID), nil))
	}
	return core.Ok(e.store.ListExperiments(datasetID))
}

// RecordFeedback stores a score / label against a run or example. The id,
// target, and key must be non-empty; ids are unique. A zero-value Source
// defaults to SourceEvaluator (the common machine path); any other unknown
// source is rejected.
//
//	e.RecordFeedback(eval.Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8, Source: eval.SourceEvaluator})
func (e *Eval) RecordFeedback(fb Feedback) core.Result {
	if fb.ID == "" {
		return core.Fail(core.E("eval.RecordFeedback", "feedback id is empty", nil))
	}
	if fb.Target == "" {
		return core.Fail(core.E("eval.RecordFeedback", "feedback target is empty", nil))
	}
	if fb.Key == "" {
		return core.Fail(core.E("eval.RecordFeedback", "feedback key is empty", nil))
	}
	if fb.Source == "" {
		fb.Source = SourceEvaluator
	}
	if !fb.Source.known() {
		return core.Fail(core.E("eval.RecordFeedback",
			core.Sprintf("unknown feedback source %q", fb.Source), nil))
	}
	return e.store.PutFeedback(fb)
}

// ListFeedback returns the feedback rows for a target, sorted by id. A target
// with no feedback is an empty, successful set — absence is not an error.
//
//	rows := e.ListFeedback("exp-1").Value.([]eval.Feedback)
func (e *Eval) ListFeedback(target string) core.Result {
	return core.Ok(e.store.ListFeedback(target))
}

// AggregateFeedback returns the mean score per key for a target — the roll-up
// the inference-stack run-tree describes (§3.7). A target with no feedback
// yields an empty map (not an error).
//
//	means := e.AggregateFeedback("exp-1").Value.(map[string]float64)
//	ethics := means["ethics"] // mean of every "ethics" score on exp-1
func (e *Eval) AggregateFeedback(target string) core.Result {
	sums := map[string]float64{}
	counts := map[string]int{}
	for _, fb := range e.store.ListFeedback(target) {
		sums[fb.Key] += fb.Score
		counts[fb.Key]++
	}
	means := make(map[string]float64, len(sums))
	for key, sum := range sums {
		means[key] = sum / float64(counts[key])
	}
	return core.Ok(means)
}
