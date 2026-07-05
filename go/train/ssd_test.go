// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the SSD sampling pipeline (#50/#97), driven with a fake SSDRunner
// (canned generation) and an in-memory dataset. No model, no Metal — SSD never
// trains, so the whole pipeline is generation + capture + score hooks.

package train

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/dataset"
	coreio "dappco.re/go/io"
)

// recordingRunner is a fake SSDRunner: Generate echoes the prompt (so responses
// differ by length), records the exact generation prompts it saw, and counts
// WarmPrefix calls.
type recordingRunner struct {
	prompts    []string
	warmCalls  int
	warmPrefix string
}

func (r *recordingRunner) runner(withWarm, withScore bool) SSDRunner {
	run := SSDRunner{
		Generate: func(_ context.Context, prompt string, _ inference.GenerateConfig) (string, error) {
			r.prompts = append(r.prompts, prompt)
			return "echo:" + prompt, nil
		},
	}
	if withWarm {
		run.WarmPrefix = func(_ context.Context, prefix string) error {
			r.warmCalls++
			r.warmPrefix = prefix
			return nil
		}
	}
	if withScore {
		run.Score = func(prompt, text string) ScoreRecord { return ScoreRecord{LEK: float64(len(text))} }
	}
	return run
}

func ssdDataset(prompts ...string) inference.DatasetStream {
	samples := make([]dataset.Sample, len(prompts))
	for i, p := range prompts {
		samples[i] = dataset.Sample{Prompt: p}
	}
	return dataset.NewSliceDataset(samples)
}

// TestRunSSD_SamplesAndCaptures asserts RunSSD samples every prompt, captures
// each raw return to the sidecar, and returns one SSDSample per prompt.
func TestRunSSD_SamplesAndCaptures(t *testing.T) {
	dir := t.TempDir()
	rec := &recordingRunner{}
	cfg := SSDConfig{SampleTemperature: 0.7, CheckpointDir: dir, FilterShortestPercent: 0}
	result, err := RunSSD(context.Background(), rec.runner(false, false), ssdDataset("alpha", "beta"), cfg)
	if err != nil {
		t.Fatalf("RunSSD: %v", err)
	}
	if len(result.Samples) != 2 {
		t.Fatalf("samples = %d, want 2", len(result.Samples))
	}
	if result.Samples[0].Response != "echo:alpha" {
		t.Fatalf("sample[0].Response = %q", result.Samples[0].Response)
	}
	read, err := coreio.Local.Read(core.PathJoin(dir, "ssd-captures.jsonl"))
	if err != nil {
		t.Fatalf("capture read: %v", err)
	}
	if core.Index(read, "echo:alpha") < 0 {
		t.Fatalf("capture sidecar missing the first return: %q", read)
	}
}

// TestRunSSD_KernelRidesGenerationAndWarmsOnce asserts the kernel prefix is
// prepended to every generation prompt, WarmPrefix is called exactly once with
// the kernel, and the captured/returned sample keeps the BARE prompt (the trace
// records how it speaks under the kernel, never the kernel's words).
func TestRunSSD_KernelRidesGenerationAndWarmsOnce(t *testing.T) {
	rec := &recordingRunner{}
	cfg := SSDConfig{SampleTemperature: 0.7, KernelPrefix: "KERNEL::", DisableCapture: true, FilterShortestPercent: 0}
	result, err := RunSSD(context.Background(), rec.runner(true, false), ssdDataset("q"), cfg)
	if err != nil {
		t.Fatalf("RunSSD: %v", err)
	}
	if rec.warmCalls != 1 || rec.warmPrefix != "KERNEL::" {
		t.Fatalf("warm calls = %d, prefix = %q", rec.warmCalls, rec.warmPrefix)
	}
	if len(rec.prompts) != 1 || rec.prompts[0] != "KERNEL::q" {
		t.Fatalf("generation prompt = %v, want [KERNEL::q]", rec.prompts)
	}
	if result.Samples[0].Prompt != "q" {
		t.Fatalf("sample kept prompt %q, want the bare q", result.Samples[0].Prompt)
	}
	if !result.KernelApplied {
		t.Fatalf("KernelApplied = false, want true")
	}
}

// TestRunSSD_ScoreAtBirth asserts that with ScoreSamples + a Score hook, every
// self-sample is scored at birth and the mean rides into the result.
func TestRunSSD_ScoreAtBirth(t *testing.T) {
	dir := t.TempDir()
	rec := &recordingRunner{}
	cfg := SSDConfig{SampleTemperature: 0.7, CheckpointDir: dir, ScoreSamples: true, FilterShortestPercent: 0}
	result, err := RunSSD(context.Background(), rec.runner(false, true), ssdDataset("a", "bb"), cfg)
	if err != nil {
		t.Fatalf("RunSSD: %v", err)
	}
	if len(result.SampleScores) != 2 {
		t.Fatalf("sample scores = %d, want 2", len(result.SampleScores))
	}
	if result.SampleScoreMean <= 0 {
		t.Fatalf("sample score mean = %f, want > 0", result.SampleScoreMean)
	}
}

// TestRunSSD_FilterShortestDropsShortResponses asserts the shortest-N% filter
// drops the shortest responses before the trace is returned.
func TestRunSSD_FilterShortestDropsShortResponses(t *testing.T) {
	rec := &recordingRunner{}
	cfg := SSDConfig{SampleTemperature: 0.7, DisableCapture: true, FilterShortestPercent: 50}
	// Responses are "echo:"+prompt, so "x" is shortest and "longprompt" longest.
	result, err := RunSSD(context.Background(), rec.runner(false, false), ssdDataset("x", "longprompt"), cfg)
	if err != nil {
		t.Fatalf("RunSSD: %v", err)
	}
	if len(result.Samples) != 1 {
		t.Fatalf("samples after 50%% filter = %d, want 1", len(result.Samples))
	}
	if result.Samples[0].Prompt != "longprompt" {
		t.Fatalf("kept %q, want the longer-response sample", result.Samples[0].Prompt)
	}
}

// TestRunSSD_RejectsUnitTemperature asserts the guard: a unit sampling
// temperature is rejected (diversity is the whole point of SSD sampling).
func TestRunSSD_RejectsUnitTemperature(t *testing.T) {
	rec := &recordingRunner{}
	_, err := RunSSD(context.Background(), rec.runner(false, false), ssdDataset("q"), SSDConfig{SampleTemperature: 1})
	if err == nil {
		t.Fatalf("expected an error for unit sample temperature")
	}
}

// TestRunSSD_EmptyDatasetErrors asserts an empty prompt set is a loud error, not
// a silent empty trace.
func TestRunSSD_EmptyDatasetErrors(t *testing.T) {
	rec := &recordingRunner{}
	_, err := RunSSD(context.Background(), rec.runner(false, false), ssdDataset(), SSDConfig{SampleTemperature: 0.7})
	if err == nil {
		t.Fatalf("expected an error for an empty dataset")
	}
}

// TestRunSSD_NilGenerateErrors asserts a runner with no Generate hook is
// rejected up front.
func TestRunSSD_NilGenerateErrors(t *testing.T) {
	_, err := RunSSD(context.Background(), SSDRunner{}, ssdDataset("q"), SSDConfig{SampleTemperature: 0.7})
	if err == nil {
		t.Fatalf("expected an error for a nil Generate hook")
	}
}
