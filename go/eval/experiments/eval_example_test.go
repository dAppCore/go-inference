// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import core "dappco.re/go"

func ExampleNew() {
	e := New()
	e.PutDataset(Dataset{ID: "ethics-probes", Name: "Ethics probes"})
	core.Println(e.GetDataset("ethics-probes").OK)
	// Output: true
}

func ExampleNewWithStore() {
	e := NewWithStore(NewMemStore())
	e.PutDataset(Dataset{ID: "ethics-probes"})
	core.Println(e.GetDataset("ethics-probes").OK)
	// Output: true
}

func ExampleEval_PutDataset() {
	e := New()
	r := e.PutDataset(Dataset{ID: "ethics-probes", Name: "Ethics probes"})
	core.Println(r.OK)
	// Output: true
}

func ExampleEval_GetDataset() {
	e := New()
	e.PutDataset(Dataset{ID: "ethics-probes", Name: "Ethics probes"})
	core.Println(e.GetDataset("ethics-probes").Value.(Dataset).Name)
	// Output: Ethics probes
}

func ExampleEval_AddExample() {
	e := New()
	e.PutDataset(Dataset{ID: "ethics-probes"})
	r := e.AddExample(Example{ID: "ex-1", DatasetID: "ethics-probes",
		Inputs: map[string]any{"prompt": "Is honesty always right?"}})
	core.Println(r.OK)
	// Output: true
}

func ExampleEval_ListExamples() {
	e := New()
	e.PutDataset(Dataset{ID: "ethics-probes"})
	e.AddExample(Example{ID: "ex-1", DatasetID: "ethics-probes"})
	core.Println(len(e.ListExamples("ethics-probes").Value.([]Example)))
	// Output: 1
}

func ExampleEval_Splits() {
	e := New()
	e.PutDataset(Dataset{ID: "ethics-probes"})
	e.AddExample(Example{ID: "ex-1", DatasetID: "ethics-probes", Split: SplitTest})
	splits := e.Splits("ethics-probes").Value.(map[Split][]string)
	core.Println(splits[SplitTest])
	// Output: [ex-1]
}

func ExampleEval_Snapshot() {
	e := New()
	e.PutDataset(Dataset{ID: "ethics-probes"})
	e.AddExample(Example{ID: "ex-1", DatasetID: "ethics-probes"})
	v := e.Snapshot("ethics-probes", "v1").Value.(Version)
	core.Println(v.Tag, v.ExampleIDs)
	// Output: v1 [ex-1]
}

func ExampleEval_CreateExperiment() {
	e := New()
	e.PutDataset(Dataset{ID: "ethics-probes"})
	r := e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ethics-probes", Run: "lemma@step-3000"})
	core.Println(r.Value.(Experiment).Status)
	// Output: pending
}

func ExampleEval_GetExperiment() {
	e := New()
	e.PutDataset(Dataset{ID: "ethics-probes"})
	e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ethics-probes", Run: "lemma@step-3000"})
	core.Println(e.GetExperiment("exp-1").Value.(Experiment).Run)
	// Output: lemma@step-3000
}

func ExampleEval_ListExperiments() {
	e := New()
	e.PutDataset(Dataset{ID: "ethics-probes"})
	e.CreateExperiment(Experiment{ID: "exp-1", DatasetID: "ethics-probes"})
	core.Println(len(e.ListExperiments("ethics-probes").Value.([]Experiment)))
	// Output: 1
}

func ExampleEval_RecordFeedback() {
	e := New()
	r := e.RecordFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8, Source: SourceEvaluator})
	core.Println(r.OK)
	// Output: true
}

func ExampleEval_ListFeedback() {
	e := New()
	e.RecordFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8})
	core.Println(len(e.ListFeedback("exp-1").Value.([]Feedback)))
	// Output: 1
}

func ExampleEval_AggregateFeedback() {
	e := New()
	e.RecordFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8})
	means := e.AggregateFeedback("exp-1").Value.(map[string]float64)
	core.Println(means["ethics"])
	// Output: 0.8
}
