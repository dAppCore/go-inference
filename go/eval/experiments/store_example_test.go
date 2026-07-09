// SPDX-Licence-Identifier: EUPL-1.2

package experiments

import core "dappco.re/go"

func ExampleNewMemStore() {
	s := NewMemStore()
	core.Println(s.GetDataset("ghost").OK)
	// Output: false
}

func ExampleMemStore_PutDataset() {
	s := NewMemStore()
	r := s.PutDataset(Dataset{ID: "ds", Name: "Dataset"})
	core.Println(r.OK)
	// Output: true
}

func ExampleMemStore_GetDataset() {
	s := NewMemStore()
	s.PutDataset(Dataset{ID: "ds", Name: "Dataset"})
	got := s.GetDataset("ds")
	core.Println(got.Value.(Dataset).Name)
	// Output: Dataset
}

func ExampleMemStore_PutExample() {
	s := NewMemStore()
	r := s.PutExample(Example{ID: "ex-1", DatasetID: "ds"})
	core.Println(r.OK)
	// Output: true
}

func ExampleMemStore_ListExamples() {
	s := NewMemStore()
	s.PutExample(Example{ID: "ex-1", DatasetID: "ds"})
	core.Println(len(s.ListExamples("ds")))
	// Output: 1
}

func ExampleMemStore_PutExperiment() {
	s := NewMemStore()
	r := s.PutExperiment(Experiment{ID: "exp-1", DatasetID: "ds"})
	core.Println(r.OK)
	// Output: true
}

func ExampleMemStore_GetExperiment() {
	s := NewMemStore()
	s.PutExperiment(Experiment{ID: "exp-1", DatasetID: "ds", Status: StatusPending})
	got := s.GetExperiment("exp-1")
	core.Println(got.Value.(Experiment).Status)
	// Output: pending
}

func ExampleMemStore_UpdateExperiment() {
	s := NewMemStore()
	s.PutExperiment(Experiment{ID: "exp-1", DatasetID: "ds", Status: StatusPending})
	s.UpdateExperiment(Experiment{ID: "exp-1", DatasetID: "ds", Status: StatusComplete})
	core.Println(s.GetExperiment("exp-1").Value.(Experiment).Status)
	// Output: complete
}

func ExampleMemStore_ListExperiments() {
	s := NewMemStore()
	s.PutExperiment(Experiment{ID: "exp-1", DatasetID: "ds"})
	core.Println(len(s.ListExperiments("ds")))
	// Output: 1
}

func ExampleMemStore_PutFeedback() {
	s := NewMemStore()
	r := s.PutFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8})
	core.Println(r.OK)
	// Output: true
}

func ExampleMemStore_ListFeedback() {
	s := NewMemStore()
	s.PutFeedback(Feedback{ID: "fb-1", Target: "exp-1", Key: "ethics", Score: 0.8})
	core.Println(len(s.ListFeedback("exp-1")))
	// Output: 1
}
