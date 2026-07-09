// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	"fmt"

	"dappco.re/go/inference/train/dataset"
)

// ExampleNewSliceDataset shows the replayable in-memory dataset: iterate to
// exhaustion, Reset, and iterate again.
func ExampleNewSliceDataset() {
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "2+2", Response: "4"},
		{Text: "raw corpus row"},
	})

	count := 0
	for {
		s, ok, err := ds.Next()
		if err != nil {
			panic(err)
		}
		if !ok {
			break
		}
		count++
		_ = s
	}
	fmt.Println("pass1:", count)

	_ = ds.Reset()
	s, _, _ := ds.Next()
	fmt.Println("after reset:", s.Prompt, s.Response)
	// Output:
	// pass1: 2
	// after reset: 2+2 4
}

// ExampleFunc adapts a generator function into a Dataset.
func ExampleFunc() {
	rows := []dataset.Sample{{Text: "a"}, {Text: "b"}}
	i := 0
	ds := dataset.Func(func() (dataset.Sample, bool, error) {
		if i >= len(rows) {
			return dataset.Sample{}, false, nil
		}
		s := rows[i]
		i++
		return s, true, nil
	})

	for {
		s, ok, _ := ds.Next()
		if !ok {
			break
		}
		fmt.Println(s.Text)
	}
	// Output:
	// a
	// b
}

// ExampleFunc_Next calls the Next method on a Func value directly: the
// adapter forwards straight to the wrapped closure, so one Next call
// yields one row.
func ExampleFunc_Next() {
	fn := dataset.Func(func() (dataset.Sample, bool, error) {
		return dataset.Sample{Prompt: "q", Response: "a"}, true, nil
	})

	s, ok, err := fn.Next()
	if err != nil {
		panic(err)
	}
	fmt.Println("ok:", ok)
	fmt.Println("prompt:", s.Prompt)
	fmt.Println("response:", s.Response)
	// Output:
	// ok: true
	// prompt: q
	// response: a
}

// ExampleSliceDataset_Next iterates a SliceDataset one record at a time via
// the Next method, stopping when ok goes false.
func ExampleSliceDataset_Next() {
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Text: "first"},
		{Text: "second"},
	})

	for {
		s, ok, err := ds.Next()
		if err != nil {
			panic(err)
		}
		if !ok {
			break
		}
		fmt.Println(s.Text)
	}
	// Output:
	// first
	// second
}

// ExampleSliceDataset_Reset rewinds the dataset so a second epoch replays
// the same records from the top.
func ExampleSliceDataset_Reset() {
	ds := dataset.NewSliceDataset([]dataset.Sample{{Text: "row0"}})

	first, _, _ := ds.Next()
	fmt.Println("epoch1:", first.Text)

	if err := ds.Reset(); err != nil {
		panic(err)
	}

	again, _, _ := ds.Next()
	fmt.Println("epoch2:", again.Text)
	// Output:
	// epoch1: row0
	// epoch2: row0
}

// ExampleCloneSample shows the defensive deep copy: mutating the clone's
// Labels does not touch the original.
func ExampleCloneSample() {
	original := dataset.Sample{Text: "doc", Labels: map[string]string{"split": "train"}}
	clone := dataset.CloneSample(original)
	clone.Labels["split"] = "test"

	fmt.Println("original:", original.Labels["split"])
	fmt.Println("clone:", clone.Labels["split"])
	// Output:
	// original: train
	// clone: test
}

// ExampleCloneSamples deep-copies a whole slice: each clone's Labels is
// independent of the source, so mutating one leaves the originals intact.
func ExampleCloneSamples() {
	source := []dataset.Sample{
		{Text: "a", Labels: map[string]string{"split": "train"}},
		{Text: "b", Labels: map[string]string{"split": "train"}},
	}
	clones := dataset.CloneSamples(source)
	clones[0].Labels["split"] = "test"

	fmt.Println("len:", len(clones))
	fmt.Println("source[0]:", source[0].Labels["split"])
	fmt.Println("clone[0]:", clones[0].Labels["split"])
	// Output:
	// len: 2
	// source[0]: train
	// clone[0]: test
}
