// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import (
	"context"

	core "dappco.re/go"
)

// exampleTarget is a Target that always answers "yes" — enough to demonstrate
// RunExperiment's shape without pulling in a real model.
type exampleTarget struct{}

func (exampleTarget) Run(_ context.Context, _ map[string]any) (string, error) {
	return "yes", nil
}

func ExampleEval_RunExperiment() {
	e := New()
	e.PutDataset(Dataset{ID: "ds"})
	e.AddExample(Example{ID: "ex-1", DatasetID: "ds", Reference: map[string]any{"answer": "yes"}})

	r := e.RunExperiment(context.Background(), "ds", exampleTarget{}, []Evaluator{ExactMatch()})
	if !r.OK {
		return
	}
	expID := r.Value.(string)
	means := e.AggregateFeedback(expID).Value.(map[string]float64)
	core.Println(means["exact_match"])
	// Output: 1
}
