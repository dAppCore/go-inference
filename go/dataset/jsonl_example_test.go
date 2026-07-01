// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	"fmt"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/dataset"
)

// ExampleLoadJSONL ingests a small multi-shape JSONL corpus and reports the
// normalised provenance label the loader stamps on each row.
func ExampleLoadJSONL() {
	corpus := strings.Join([]string{
		`{"text":"plain corpus row"}`,
		`{"prompt":"capital of France?","response":"Paris"}`,
		`{"instruction":"summarise","input":"notes","output":"short"}`,
	}, "\n")

	ds, err := dataset.LoadJSONL(strings.NewReader(corpus))
	if err != nil {
		panic(err)
	}
	for {
		s, ok, err := ds.Next()
		if err != nil {
			panic(err)
		}
		if !ok {
			break
		}
		fmt.Println(s.Format)
	}
	// Output:
	// text
	// prompt_response
	// alpaca
}

// ExampleNewJSONL wraps an already-normalised slice of samples into a
// replayable dataset, defensively cloning so later source mutation can't
// reach the dataset.
func ExampleNewJSONL() {
	samples := []dataset.Sample{
		{Text: "row one"},
		{Prompt: "q", Response: "a"},
	}
	ds := dataset.NewJSONL(samples)

	// Mutating the source after construction does not affect the dataset.
	samples[0].Text = "mutated"

	first, _, _ := ds.Next()
	fmt.Println("first:", first.Text)
	// Output:
	// first: row one
}

// ExampleJSONLDataset_Next streams a JSONLDataset one normalised record at
// a time via the Next method.
func ExampleJSONLDataset_Next() {
	ds := dataset.NewJSONL([]dataset.Sample{
		{Text: "alpha"},
		{Text: "beta"},
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
	// alpha
	// beta
}

// ExampleJSONLDataset_Reset rewinds the dataset so a second epoch replays
// the same records.
func ExampleJSONLDataset_Reset() {
	ds := dataset.NewJSONL([]dataset.Sample{{Text: "row0"}})

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

// ExampleJSONLDataset_Samples returns the full record set as a defensive
// copy, independent of the dataset's iteration cursor.
func ExampleJSONLDataset_Samples() {
	ds := dataset.NewJSONL([]dataset.Sample{
		{Text: "a"},
		{Text: "b"},
	})

	all := ds.Samples()
	fmt.Println("count:", len(all))
	fmt.Println("first:", all[0].Text)
	// Output:
	// count: 2
	// first: a
}

// ExampleMessagesToSample converts a normalised message list into a
// supervised sample, using the trailing assistant turn as the response.
func ExampleMessagesToSample() {
	messages := []inference.Message{
		{Role: "user", Content: "ping"},
		{Role: "assistant", Content: "  pong  "},
	}

	sample, ok, err := dataset.MessagesToSample(messages, "openai_messages")
	if err != nil {
		panic(err)
	}
	fmt.Println("ok:", ok)
	fmt.Println("response:", sample.Response)
	fmt.Println("format:", sample.Format)
	// Output:
	// ok: true
	// response: pong
	// format: openai_messages
}

// ExampleNormaliseRole canonicalises common role aliases found in ShareGPT
// and OpenAI-style training exports.
func ExampleNormaliseRole() {
	fmt.Println(dataset.NormaliseRole("gpt"))
	fmt.Println(dataset.NormaliseRole("human"))
	fmt.Println(dataset.NormaliseRole("assistant"))
	// Output:
	// assistant
	// user
	// assistant
}
