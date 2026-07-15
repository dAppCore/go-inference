// SPDX-Licence-Identifier: EUPL-1.2

package train_test

import (
	"context"
	"fmt"
	"os"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/train"
)

// exampleCommandBackendName is a unique registration name so this fake
// backend never collides with a real backend package linked into the same
// binary.
const exampleCommandBackendName = "train-command-example-fake"

// exampleCommandBackend is a minimal inference.Backend whose LoadModel
// always returns the injected model, regardless of path — it stands in for
// a real engine (metal, rocm, ...) so RunSSDCommand/RunSFTCommand can be
// demonstrated without one.
type exampleCommandBackend struct{ model inference.TextModel }

func (b *exampleCommandBackend) Name() string    { return exampleCommandBackendName }
func (b *exampleCommandBackend) Available() bool { return true }
func (b *exampleCommandBackend) LoadModel(string, ...inference.LoadOption) core.Result {
	return core.Ok(b.model)
}

// exampleTokenizerJSON is a tiny char-level BPE tokenizer.json — enough to
// encode the ASCII fixtures these examples train on.
const exampleTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
    "merges": [],
    "byte_fallback": false
  },
  "added_tokens": []
}`

// writeExampleJSONLDataset writes a one-row prompt/response JSONL fixture
// under dir and returns its path.
func writeExampleJSONLDataset(dir, prompt, response string) string {
	path := core.PathJoin(dir, "data.jsonl")
	row := fmt.Sprintf("{\"prompt\":%q,\"response\":%q}\n", prompt, response)
	if result := core.WriteFile(path, []byte(row), 0o600); !result.OK {
		panic(result.Value)
	}
	return path
}

// ExampleRunSSDCommand shows the cmd-facing SSD runner: load the model
// (through whatever backend a driver registers) and the prompt set, sample
// every prompt, and print the summary line a `lem ssd` invocation reports.
func ExampleRunSSDCommand() {
	inference.Register(&exampleCommandBackend{model: &echoModel{}})

	baseResult := core.MkdirTemp("", "train-command-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)
	dataPath := writeExampleJSONLDataset(dir, "hello", "hello")

	err := train.RunSSDCommand(context.Background(), train.SSDCommandConfig{
		ModelPath:  "example-model",
		DataPath:   dataPath,
		Backend:    exampleCommandBackendName, // pin the fake — a linked real engine (metal) must not win selection
		SampleTemp: 0.7,
		Out:        os.Stdout,
	})
	if err != nil {
		panic(err)
	}
	// Output:
	// self-samples 1  sample-temp 0.70  kernel false
	// next: refine the trace in the lab, then  lem sft --data <artifact> --model example-model
}

// ExampleRunSFTCommand shows the cmd-facing SFT runner: load the model +
// tokeniser + training set and run the LoRA SFT loop, printing the summary
// line a `lem sft` invocation reports.
func ExampleRunSFTCommand() {
	inference.Register(&exampleCommandBackend{model: &echoModel{trainer: &echoTrainer{}}})

	baseResult := core.MkdirTemp("", "train-command-example-*")
	if !baseResult.OK {
		panic("tempdir failed")
	}
	dir := baseResult.Value.(string)
	defer core.RemoveAll(dir)
	if result := core.WriteFile(core.PathJoin(dir, "tokenizer.json"), []byte(exampleTokenizerJSON), 0o600); !result.OK {
		panic(result.Value)
	}
	dataPath := writeExampleJSONLDataset(dir, "hello", "hello")

	err := train.RunSFTCommand(context.Background(), train.SFTCommandConfig{
		ModelPath: dir,
		DataPath:  dataPath,
		Backend:   exampleCommandBackendName, // pin the fake — a linked real engine (metal) must not win selection
		Epochs:    1,
		BatchSize: 1,
		GradAccum: 1,
		Rank:      8,
		Alpha:     16,
		Out:       os.Stdout,
	})
	if err != nil {
		panic(err)
	}
	// Output:
	// steps 1  epochs 1  samples 1  last-loss 1.0000
}
