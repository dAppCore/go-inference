// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import (
	"context"

	core "dappco.re/go"
)

// KeyError is the reserved feedback key under which a runner records a
// per-example failure — a target that erred, or an evaluator that returned no
// key. It aggregates separately from scoring keys so a failed example reads as
// a recorded failure, never as a zero score (RFC.inference-stack §3.7).
//
//	if fb.Key == experiments.KeyError { failures++ }
const KeyError = "error"

// Target is the model under test in an experiment: given an example's inputs it
// produces a single output string, or an error the runner records as a failure
// for that example. the serving handoff (a go-mlx-loaded TextModel, or a
// provider endpoint) satisfies this; tests fake it.
//
//	out, err := target.Run(ctx, map[string]any{"prompt": "Is honesty always right?"})
type Target interface {
	Run(ctx context.Context, inputs map[string]any) (output string, err error)
}

// Evaluator scores one (example, output) pair, returning the metric key, its
// numeric score, and an optional comment — the score_cascade unit the LEK
// scorer (go-mlx pkg/score) and the heuristic evaluators both implement. An
// empty key signals the evaluator could not score this pair; the runner records
// that as a failure rather than a silent drop.
//
//	key, score, comment := evaluator.Eval(ex, output)
type Evaluator interface {
	Eval(example Example, output string) (key string, score float64, comment string)
}

// RunExperiment runs target over every example in datasetID, scores each output
// with the evaluators, records the results as Feedback against a fresh
// experiment, and returns that experiment's id. The dataset must exist, target
// must be non-nil, and at least one evaluator is required.
//
// Per example: target.Run is called; on error a single failure row (key
// KeyError) is recorded and the run moves on — one example failing never aborts
// the rest. Otherwise each evaluator runs and records a row; an evaluator that
// returns an empty key is itself recorded under KeyError. The experiment is
// marked StatusComplete, or StatusFailed when every example's target call
// errored. AggregateFeedback then yields the mean score per key.
//
//	r := e.RunExperiment(ctx, "ethics-probes", target, []experiments.Evaluator{exactMatch{}, lekScore{}})
//	expID := r.Value.(string)
//	means := e.AggregateFeedback(expID).Value.(map[string]float64)
func (e *Eval) RunExperiment(ctx context.Context, datasetID string, target Target, evaluators []Evaluator) core.Result {
	if target == nil {
		return core.Fail(core.E("experiments.RunExperiment", "target is nil", nil))
	}
	if len(evaluators) == 0 {
		return core.Fail(core.E("experiments.RunExperiment", "no evaluators given", nil))
	}
	if d := e.store.GetDataset(datasetID); !d.OK {
		return core.Fail(core.E("experiments.RunExperiment",
			core.Sprintf("no dataset with id %q", datasetID), nil))
	}

	expID := core.ID()
	if r := e.CreateExperiment(Experiment{
		ID:        expID,
		DatasetID: datasetID,
		Status:    StatusRunning,
		Created:   core.UnixNow(),
	}); !r.OK {
		return r
	}

	examples := e.store.ListExamples(datasetID) // already id-sorted, deterministic
	attempted, failed := 0, 0
	for _, ex := range examples {
		attempted++
		output, err := target.Run(ctx, ex.Inputs)
		if err != nil {
			failed++
			if r := e.recordFailure(expID, ex.ID, err.Error()); !r.OK {
				return e.finishExperiment(expID, StatusFailed, r)
			}
			continue
		}
		for _, ev := range evaluators {
			key, score, comment := ev.Eval(ex, output)
			if key == "" {
				if r := e.recordFailure(expID, ex.ID, "evaluator returned no key"); !r.OK {
					return e.finishExperiment(expID, StatusFailed, r)
				}
				continue
			}
			fb := Feedback{
				ID:      core.ID(),
				Target:  expID,
				Key:     key,
				Score:   score,
				Comment: comment,
				Source:  SourceEvaluator,
			}
			if r := e.RecordFeedback(fb); !r.OK {
				return e.finishExperiment(expID, StatusFailed, r)
			}
		}
	}

	status := StatusComplete
	if attempted > 0 && failed == attempted {
		// Every example's target call errored — the run produced no scores.
		status = StatusFailed
	}
	return e.finishExperiment(expID, status, core.Ok(expID))
}

// recordFailure stores a single failure row (key KeyError, score 0) for an
// example under an experiment — the audit row for a target or evaluator that
// could not produce a score. It rides the heuristic source: a failure is a
// machine observation, not a human judgement.
//
//	e.recordFailure("exp-1", "ex-3", "model refused")
func (e *Eval) recordFailure(expID, exampleID, comment string) core.Result {
	return e.RecordFeedback(Feedback{
		ID:      core.ID(),
		Target:  expID,
		Key:     KeyError,
		Score:   0,
		Comment: core.Sprintf("%s: %s", exampleID, comment),
		Source:  SourceHeuristic,
	})
}

// finishExperiment stamps the experiment's terminal status and returns out
// unchanged — so the run's outcome (the experiment id, or the error that broke
// the loop) is what RunExperiment hands back. A missing experiment at this
// point is impossible (it was just created), so a status-write failure is
// surfaced over the original result.
//
//	return e.finishExperiment(expID, experiments.StatusComplete, core.Ok(expID))
func (e *Eval) finishExperiment(expID string, status ExperimentStatus, out core.Result) core.Result {
	x := e.store.GetExperiment(expID)
	if !x.OK {
		return out
	}
	exp := x.Value.(Experiment)
	exp.Status = status
	if r := e.store.UpdateExperiment(exp); !r.OK {
		return r
	}
	return out
}
