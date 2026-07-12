// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the cmd-facing SSD/SFT runners, driven through a fake
// inference.Backend registered as the sole backend for this test binary —
// RunSSDCommand/RunSFTCommand call the real inference.LoadModel plumbing
// (no test seam bypass), so the fake backend is what proves the command
// layer wires load → run → summary correctly end to end.

package train

import (
	"bytes"
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// commandFakeBackendName is a unique registration name so this fake never
// collides with a real backend package that might also be linked in.
const commandFakeBackendName = "train-command-test-fake"

// commandFakeBackend is a minimal inference.Backend: LoadModel always
// returns the injected model (or the injected error) regardless of path.
// Registering it makes it the sole (hence Default()-selected) backend for
// the remainder of this test binary.
type commandFakeBackend struct {
	model inference.TextModel
	err   error
}

func (b *commandFakeBackend) Name() string    { return commandFakeBackendName }
func (b *commandFakeBackend) Available() bool { return true }
func (b *commandFakeBackend) LoadModel(string, ...inference.LoadOption) core.Result {
	if b.err != nil {
		return core.Fail(b.err)
	}
	return core.Ok(b.model)
}

// registerCommandFakeBackend arms model (or err) as the fake backend's
// LoadModel result for the rest of this test binary.
func registerCommandFakeBackend(t *testing.T, model inference.TextModel, err error) {
	t.Helper()
	inference.Register(&commandFakeBackend{model: model, err: err})
}

// commandMinimalTokenizerJSON is a valid HuggingFace tokenizer.json with a
// tiny char-level BPE vocab — enough to encode ASCII training text ("hello"-
// shaped) into ≥2 tokens without pulling in a real model's tokenizer.
const commandMinimalTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0,
      "e": 1,
      "l": 2,
      "o": 3,
      "he": 5,
      "ll": 6
    },
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": []
}`

// writeCommandJSONLDataset writes a minimal prompt/response JSONL training
// file the SSD/SFT commands can load, using only characters present in
// commandMinimalTokenizerJSON's vocab.
func writeCommandJSONLDataset(t *testing.T, rows ...string) string {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "data.jsonl")
	body := ""
	for _, row := range rows {
		body += row + "\n"
	}
	if result := core.WriteFile(path, []byte(body), 0o600); !result.OK {
		t.Fatalf("write fixture dataset: %v", result.Value)
	}
	return path
}

// writeCommandTokenizerModelDir builds a model directory containing just a
// tokenizer.json — command.go's loadTextModel never touches the disk itself
// (the fake backend answers LoadModel), but RunSFTCommand loads the
// tokenizer straight off ModelPath independently of the backend.
func writeCommandTokenizerModelDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(commandMinimalTokenizerJSON), 0o600); !result.OK {
		t.Fatalf("write fixture tokenizer: %v", result.Value)
	}
	return dir
}

// --- RunSSDCommand ---

// Good: a valid model + dataset samples every prompt and prints the summary
// (self-samples count, sample temperature) to Out.
func TestCommand_RunSSDCommand_Good(t *testing.T) {
	registerCommandFakeBackend(t, &fakeTextModel{}, nil)
	dataPath := writeCommandJSONLDataset(t, `{"prompt":"hello","response":"hello"}`, `{"prompt":"hollo","response":"hello"}`)
	var out bytes.Buffer

	err := RunSSDCommand(context.Background(), SSDCommandConfig{
		ModelPath:      "unused-model-path",
		Backend:        commandFakeBackendName, // a linked real engine (metal) must not win selection
		DataPath:       dataPath,
		SampleTemp:     0.7,
		FilterShortest: 0,
		Out:            &out,
	})
	if err != nil {
		t.Fatalf("RunSSDCommand: %v", err)
	}
	if got := out.String(); core.Index(got, "self-samples 2") < 0 {
		t.Fatalf("summary = %q, want it to report 2 self-samples", got)
	}
}

// Bad: missing --model/--data is rejected up front, before any backend or
// dataset I/O happens.
func TestCommand_RunSSDCommand_Bad(t *testing.T) {
	if err := RunSSDCommand(context.Background(), SSDCommandConfig{}); err == nil {
		t.Fatalf("expected an error for missing --model/--data")
	}
}

// Ugly: --score-samples requested with no Score hook supplied — the honest
// boundary: RunSSDCommand does not error, it notes the gap on Log and
// proceeds capture-only.
func TestCommand_RunSSDCommand_Ugly(t *testing.T) {
	registerCommandFakeBackend(t, &fakeTextModel{}, nil)
	dataPath := writeCommandJSONLDataset(t, `{"prompt":"hello","response":"hello"}`)
	var out, log bytes.Buffer

	err := RunSSDCommand(context.Background(), SSDCommandConfig{
		ModelPath:    "unused-model-path",
		Backend:      commandFakeBackendName, // a linked real engine (metal) must not win selection
		DataPath:     dataPath,
		SampleTemp:   0.7,
		ScoreSamples: true, // Score left nil
		Out:          &out,
		Log:          &log,
	})
	if err != nil {
		t.Fatalf("RunSSDCommand: %v", err)
	}
	if got := log.String(); core.Index(got, "no scorer was supplied") < 0 {
		t.Fatalf("log = %q, want the honest no-scorer notice", got)
	}
}

// --- RunSFTCommand ---

// Good: a valid model + tokenizer + dataset trains to completion and prints
// the steps/epochs/samples summary.
func TestCommand_RunSFTCommand_Good(t *testing.T) {
	registerCommandFakeBackend(t, &fakeTextModel{trainer: &fakeTrainer{}}, nil)
	modelPath := writeCommandTokenizerModelDir(t)
	dataPath := writeCommandJSONLDataset(t, `{"prompt":"hello","response":"hello"}`, `{"prompt":"hollo","response":"hello"}`)
	var out bytes.Buffer

	err := RunSFTCommand(context.Background(), SFTCommandConfig{
		ModelPath: modelPath,
		Backend:   commandFakeBackendName, // a linked real engine (metal) must not win selection
		DataPath:  dataPath,
		Epochs:    1,
		BatchSize: 1,
		GradAccum: 1,
		Rank:      8,
		Alpha:     16,
		Out:       &out,
	})
	if err != nil {
		t.Fatalf("RunSFTCommand: %v", err)
	}
	if got := out.String(); core.Index(got, "steps 2") < 0 {
		t.Fatalf("summary = %q, want it to report 2 steps", got)
	}
}

// Bad: missing --model/--data is rejected up front.
func TestCommand_RunSFTCommand_Bad(t *testing.T) {
	if err := RunSFTCommand(context.Background(), SFTCommandConfig{}); err == nil {
		t.Fatalf("expected an error for missing --model/--data")
	}
}

// Ugly: --packing and --merge are both requested against the head-LoRA
// trainer, which supports neither — RunSFTCommand does not error, it notes
// both gaps on Log and trains anyway (the honest "ignored"/"not supported"
// boundary).
func TestCommand_RunSFTCommand_Ugly(t *testing.T) {
	registerCommandFakeBackend(t, &fakeTextModel{trainer: &fakeTrainer{}}, nil)
	modelPath := writeCommandTokenizerModelDir(t)
	dataPath := writeCommandJSONLDataset(t, `{"prompt":"hello","response":"hello"}`)
	var out, log bytes.Buffer

	err := RunSFTCommand(context.Background(), SFTCommandConfig{
		ModelPath: modelPath,
		Backend:   commandFakeBackendName, // a linked real engine (metal) must not win selection
		DataPath:  dataPath,
		Epochs:    1,
		BatchSize: 1,
		GradAccum: 1,
		Packing:   true,
		Merge:     true,
		Out:       &out,
		Log:       &log,
	})
	if err != nil {
		t.Fatalf("RunSFTCommand: %v", err)
	}
	logged := log.String()
	if core.Index(logged, "--packing has no effect") < 0 {
		t.Fatalf("log = %q, want the packing-ignored notice", logged)
	}
	if core.Index(logged, "--merge is not supported") < 0 {
		t.Fatalf("log = %q, want the merge-not-supported notice", logged)
	}
}
