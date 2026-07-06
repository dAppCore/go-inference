// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import core "dappco.re/go"

// ExampleDataset shows the minimal fields needed to register a dataset.
func ExampleDataset() {
	d := Dataset{ID: "ethics-probes", Name: "Ethics probes", Description: "Core LEK axiom probes"}
	core.Println(d.ID, d.Name)
	// Output: ethics-probes Ethics probes
}

// ExampleExample shows an evaluation case: the inputs a model sees and the
// reference answer an evaluator scores its output against.
func ExampleExample() {
	ex := Example{
		ID:        "ex-1",
		DatasetID: "ethics-probes",
		Inputs:    map[string]any{"prompt": "Is honesty always right?"},
		Reference: map[string]any{"answer": "context-dependent"},
		Split:     SplitTrain,
	}
	core.Println(ex.ID, ex.Split)
	// Output: ex-1 train
}

// ExampleVersion shows the immutable snapshot Eval.Snapshot returns: a tag
// plus the example ids captured under it.
func ExampleVersion() {
	v := Version{DatasetID: "ethics-probes", Tag: "v1", ExampleIDs: []string{"ex-1", "ex-2"}}
	core.Println(v.Tag, len(v.ExampleIDs))
	// Output: v1 2
}

// ExampleExperiment shows one evaluation run of a model / run reference over
// a dataset.
func ExampleExperiment() {
	x := Experiment{ID: "exp-1", DatasetID: "ethics-probes", Run: "lemma@step-3000", Status: StatusPending}
	core.Println(x.ID, x.Status)
	// Output: exp-1 pending
}

// ExampleFeedback shows a score attached to a run or example by target id.
func ExampleFeedback() {
	fb := Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8, Source: SourceEvaluator}
	core.Println(fb.Key, fb.Score, fb.Source)
	// Output: ethics 0.8 evaluator
}
