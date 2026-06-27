// SPDX-Licence-Identifier: EUPL-1.2

// Package eval is the evaluation-data surface of go-ml — the datasets,
// examples, experiments, and feedback that drive offline model evaluation. It
// mirrors LangSmith's shape (datasets/experiments/feedback) and is the typed
// record layer the inference-stack spec places in go-ml (RFC.inference-stack
// §5 and §3.7). It holds evaluation metadata only — no model weights, no
// scoring engine, no serving.
//
//	e := eval.New()
//	e.PutDataset(eval.Dataset{ID: "ethics-probes", Name: "Ethics probes"})
//	e.AddExample(eval.Example{ID: "ex-1", DatasetID: "ethics-probes",
//	    Inputs: map[string]any{"prompt": "Is honesty always right?"}})
//	e.CreateExperiment(eval.Experiment{ID: "exp-1", DatasetID: "ethics-probes", Run: "lemma@step-3000"})
//	e.RecordFeedback(eval.Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8, Source: eval.SourceEvaluator})
//	means := e.AggregateFeedback("exp-1").Value.(map[string]float64) // mean score per key
package experiments

// Split is the partition an example belongs to within its dataset — the same
// train/validation/test division an evaluation run respects.
//
//	if ex.Split == eval.SplitTest { holdout = append(holdout, ex) }
type Split string

const (
	// SplitTrain is the training partition.
	SplitTrain Split = "train"
	// SplitValidation is the validation / dev partition.
	SplitValidation Split = "validation"
	// SplitTest is the held-out test partition.
	SplitTest Split = "test"
)

// ExperimentStatus is the lifecycle state of an experiment run.
//
//	if exp.Status == eval.StatusComplete { report(exp) }
type ExperimentStatus string

const (
	// StatusPending is created but not yet started — the default for a new
	// experiment.
	StatusPending ExperimentStatus = "pending"
	// StatusRunning is in progress.
	StatusRunning ExperimentStatus = "running"
	// StatusComplete has finished and its feedback is final.
	StatusComplete ExperimentStatus = "complete"
	// StatusFailed did not finish.
	StatusFailed ExperimentStatus = "failed"
)

// Source is where a piece of feedback came from — read when weighting or
// filtering scores (e.g. human review outranks a heuristic).
//
//	if fb.Source == eval.SourceHuman { trusted = append(trusted, fb) }
type Source string

const (
	// SourceHuman is a human reviewer (annotation queue).
	SourceHuman Source = "human"
	// SourceEvaluator is an automated evaluator / judge model (go-ml suites,
	// pkg/score). The default when a feedback row omits its source.
	SourceEvaluator Source = "evaluator"
	// SourceHeuristic is a rule / metric, not a model.
	SourceHeuristic Source = "heuristic"
)

// known reports whether s is a recognised feedback source. The empty source is
// not "known" — RecordFeedback defaults it to SourceEvaluator before storing.
//
//	if !eval.SourceHuman.known() { ... }
func (s Source) known() bool {
	switch s {
	case SourceHuman, SourceEvaluator, SourceHeuristic:
		return true
	default:
		return false
	}
}

// Dataset is a named collection of examples evaluated together — the unit an
// experiment runs over.
//
//	eval.Dataset{ID: "ethics-probes", Name: "Ethics probes", Description: "Core LEK axiom probes"}
type Dataset struct {
	ID          string `json:"id"`                    // canonical dataset identifier
	Name        string `json:"name,omitempty"`        // human-readable name
	Description string `json:"description,omitempty"` // what the dataset covers
}

// Example is one evaluation case: the inputs fed to a model and the reference
// outputs expected back. Its id is unique within its dataset (not globally).
//
//	eval.Example{
//	    ID:        "ex-1",
//	    DatasetID: "ethics-probes",
//	    Inputs:    map[string]any{"prompt": "Is honesty always right?"},
//	    Reference: map[string]any{"answer": "context-dependent"},
//	    Split:     eval.SplitTrain,
//	}
type Example struct {
	ID        string         `json:"id"`                  // identifier, unique within the dataset
	DatasetID string         `json:"dataset_id"`          // owning dataset id
	Inputs    map[string]any `json:"inputs,omitempty"`    // inputs presented to the model
	Reference map[string]any `json:"reference,omitempty"` // expected / gold outputs
	Split     Split          `json:"split,omitempty"`     // train / validation / test partition
	Metadata  map[string]any `json:"metadata,omitempty"`  // free-form tags (axiom, difficulty, …)
}

// Version is an immutable snapshot of a dataset's example ids under a tag — so
// an experiment can name exactly which examples it ran against.
//
//	v := e.Snapshot("ethics-probes", "v1").Value.(eval.Version)
type Version struct {
	DatasetID  string   `json:"dataset_id"`  // dataset this version snapshots
	Tag        string   `json:"tag"`         // version label (e.g. "v1")
	ExampleIDs []string `json:"example_ids"` // example ids captured, sorted
}

// Experiment is one evaluation run of a model / run reference over a dataset.
// Feedback rows attach to it (or to individual examples) by Target id.
//
//	eval.Experiment{ID: "exp-1", DatasetID: "ethics-probes", Run: "lemma@step-3000"}
type Experiment struct {
	ID        string           `json:"id"`                // canonical experiment identifier
	DatasetID string           `json:"dataset_id"`        // dataset evaluated
	Run       string           `json:"run,omitempty"`     // model / run reference under test
	Status    ExperimentStatus `json:"status,omitempty"`  // pending / running / complete / failed
	Created   int64            `json:"created,omitempty"` // creation time (unix seconds), caller-set
}

// Feedback is a score or label attached to a run or example by id — from a
// human, an evaluator, or a heuristic. Aggregation takes the mean Score per
// Key for a given Target (RFC.inference-stack §3.7).
//
//	eval.Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8, Source: eval.SourceEvaluator}
type Feedback struct {
	ID      string  `json:"id"`                // canonical feedback identifier
	Target  string  `json:"target"`            // run / experiment / example id this scores
	Key     string  `json:"key"`               // metric name (e.g. "ethics", "helpfulness")
	Score   float64 `json:"score"`             // numeric score for the key
	Comment string  `json:"comment,omitempty"` // optional reviewer note
	Source  Source  `json:"source,omitempty"`  // human / evaluator / heuristic
}
