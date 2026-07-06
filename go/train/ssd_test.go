// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the SSD sampling pipeline (#50/#97), driven with a fake SSDRunner
// (canned generation) and an in-memory dataset. No model, no Metal — SSD never
// trains, so the whole pipeline is generation + capture + score hooks.

package train

import (
	"context"
	"reflect"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/train/dataset"
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

// TestSsd_RunSSD_Good asserts RunSSD samples every prompt, captures
// each raw return to the sidecar, and returns one SSDSample per prompt.
func TestSsd_RunSSD_Good(t *testing.T) {
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

// TestSsd_RunSSD_Ugly asserts the kernel prefix is
// prepended to every generation prompt, WarmPrefix is called exactly once with
// the kernel, and the captured/returned sample keeps the BARE prompt (the trace
// records how it speaks under the kernel, never the kernel's words).
func TestSsd_RunSSD_Ugly(t *testing.T) {
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

// TestSsd_RunSSD_Bad asserts a runner with no Generate hook is
// rejected up front.
func TestSsd_RunSSD_Bad(t *testing.T) {
	_, err := RunSSD(context.Background(), SSDRunner{}, ssdDataset("q"), SSDConfig{SampleTemperature: 0.7})
	if err == nil {
		t.Fatalf("expected an error for a nil Generate hook")
	}
}

// --- DefaultSSDConfig ---

// Good: the ml-ssd data-generation defaults land on every field.
func TestSsd_DefaultSSDConfig_Good(t *testing.T) {
	cfg := DefaultSSDConfig()
	if cfg.SampleMaxTokens != 65536 || cfg.SampleTemperature != 1.5 || cfg.SampleTopK != 20 || cfg.SampleTopP != 0.8 {
		t.Fatalf("sampling defaults = %+v, want max-tokens 65536 temp 1.5 top-k 20 top-p 0.8", cfg)
	}
	if cfg.RepetitionPenalty != 1.0 || cfg.FilterShortestPercent != 10 {
		t.Fatalf("penalty/filter defaults = %+v, want repetition 1.0 filter 10", cfg)
	}
}

// Bad: the shipped defaults are themselves a well-formed SSDConfig — they
// must pass validateSSDConfig cleanly, or every unconfigured `lem ssd` run
// would fail its own guard on the first call.
func TestSsd_DefaultSSDConfig_Bad(t *testing.T) {
	if err := validateSSDConfig(DefaultSSDConfig()); err != nil {
		t.Fatalf("DefaultSSDConfig() failed its own validation: %v", err)
	}
}

// Ugly: DefaultSSDConfig's fields are already non-zero for every knob
// normalizeSSDConfig floors, so running it through the normaliser is a
// no-op — the two independent default systems (recipe defaults here vs the
// zero-value safety net in normalizeSSDConfig) never fight each other.
func TestSsd_DefaultSSDConfig_Ugly(t *testing.T) {
	cfg := DefaultSSDConfig()
	if normalised := normalizeSSDConfig(cfg); normalised != cfg {
		t.Fatalf("normalizeSSDConfig(DefaultSSDConfig()) = %+v, want unchanged %+v", normalised, cfg)
	}
}

// --- DefaultSSDCodeBenchmarkConfig ---

// Good: the LiveCodeBench-v6 evaluation defaults land on every field.
func TestSsd_DefaultSSDCodeBenchmarkConfig_Good(t *testing.T) {
	cfg := DefaultSSDCodeBenchmarkConfig()
	if cfg.Benchmark != "LiveCodeBench-v6" || cfg.NRepeat != 20 {
		t.Fatalf("benchmark/repeat = %+v, want LiveCodeBench-v6 / 20", cfg)
	}
	if len(cfg.Seeds) != 4 || cfg.Seeds[0] != 0 || cfg.Seeds[1] != 1234 {
		t.Fatalf("seeds = %v, want [0 1234 1234 1234]", cfg.Seeds)
	}
	if cfg.Generate.MaxTokens != 32768 || cfg.Generate.Temperature != 0.6 || cfg.Generate.TopP != 0.95 || cfg.Generate.TopK != 20 {
		t.Fatalf("generate defaults = %+v", cfg.Generate)
	}
}

// Bad: the shipped defaults are already normalised — normalizeSSDCodeBenchmarkConfig
// leaves every field unchanged (NRepeat/MaxTokens/TopK/TopP are all non-zero).
func TestSsd_DefaultSSDCodeBenchmarkConfig_Bad(t *testing.T) {
	cfg := DefaultSSDCodeBenchmarkConfig()
	if normalised := normalizeSSDCodeBenchmarkConfig(cfg); normalised.NRepeat != cfg.NRepeat ||
		normalised.Generate.MaxTokens != cfg.Generate.MaxTokens ||
		normalised.Generate.TopK != cfg.Generate.TopK ||
		normalised.Generate.TopP != cfg.Generate.TopP {
		t.Fatalf("normalizeSSDCodeBenchmarkConfig(DefaultSSDCodeBenchmarkConfig()) changed a floored field: %+v", normalised)
	}
}

// Ugly: the returned Seeds slice is a fresh literal every call — mutating one
// call's slice must never bleed into a second call's defaults.
func TestSsd_DefaultSSDCodeBenchmarkConfig_Ugly(t *testing.T) {
	first := DefaultSSDCodeBenchmarkConfig()
	first.Seeds[0] = 9999
	second := DefaultSSDCodeBenchmarkConfig()
	if second.Seeds[0] != 0 {
		t.Fatalf("second call's Seeds = %v, want a fresh [0 ...] unaffected by the first call's mutation", second.Seeds)
	}
}

// --- SSDRecipes ---

// Good: the three released ml-ssd parity recipes are named in order with
// their model identifiers.
func TestSsd_SSDRecipes_Good(t *testing.T) {
	recipes := SSDRecipes()
	if len(recipes) != 3 {
		t.Fatalf("recipes = %d, want 3", len(recipes))
	}
	wantNames := []string{SSDRecipe4BInstruct, SSDRecipe4BThinking, SSDRecipe30BA3BInstruct}
	wantModels := []string{"apple/SimpleSD-4B-instruct", "apple/SimpleSD-4B-thinking", "apple/SimpleSD-30b-a3b-instruct"}
	for i, recipe := range recipes {
		if recipe.Name != wantNames[i] || recipe.Model != wantModels[i] {
			t.Fatalf("recipe[%d] = %+v, want name %q model %q", i, recipe, wantNames[i], wantModels[i])
		}
	}
}

// Bad: every recipe carries the shared rStar-Coder data-generation dataset
// coordinates — a per-recipe typo here would silently point one model card
// at the wrong split.
func TestSsd_SSDRecipes_Bad(t *testing.T) {
	for _, recipe := range SSDRecipes() {
		if recipe.Dataset != "microsoft/rStar-Coder" || recipe.DatasetConfig != "seed_sft" || recipe.DatasetSplit != "train" {
			t.Fatalf("recipe %q dataset coordinates = %+v, want the shared rStar-Coder seed_sft/train triple", recipe.Name, recipe)
		}
	}
}

// Ugly: each call returns an independent slice/Notes — mutating one call's
// result must never bleed into a second call's recipes.
func TestSsd_SSDRecipes_Ugly(t *testing.T) {
	first := SSDRecipes()
	first[0].Notes[0] = "mutated"
	second := SSDRecipes()
	if second[0].Notes[0] == "mutated" {
		t.Fatalf("second call's Notes = %v, want unaffected by the first call's mutation", second[0].Notes)
	}
}

// --- LookupSSDRecipe ---

// Good: a lookup by the recipe's Name resolves it.
func TestSsd_LookupSSDRecipe_Good(t *testing.T) {
	recipe, ok := LookupSSDRecipe(SSDRecipe4BInstruct)
	if !ok || recipe.Name != SSDRecipe4BInstruct {
		t.Fatalf("LookupSSDRecipe(%q) = %+v, %v, want the 4B-instruct recipe", SSDRecipe4BInstruct, recipe, ok)
	}
}

// Bad: an unknown name resolves to the zero-value recipe and false, not a panic.
func TestSsd_LookupSSDRecipe_Bad(t *testing.T) {
	recipe, ok := LookupSSDRecipe("does-not-exist")
	if ok || recipe.Name != "" {
		t.Fatalf("LookupSSDRecipe(unknown) = %+v, %v, want zero-value, false", recipe, ok)
	}
}

// Ugly: a lookup by the recipe's Model identifier ALSO resolves it — the
// dual-key OR-match, not just the Name key.
func TestSsd_LookupSSDRecipe_Ugly(t *testing.T) {
	recipe, ok := LookupSSDRecipe("apple/SimpleSD-30b-a3b-instruct")
	if !ok || recipe.Name != SSDRecipe30BA3BInstruct {
		t.Fatalf("LookupSSDRecipe(by model) = %+v, %v, want the 30B-a3b-instruct recipe", recipe, ok)
	}
}

// --- SSDResult.SampleGenerateConfig ---

// Good: every sampling field on a populated result maps straight through.
func TestSsd_SSDResult_SampleGenerateConfig_Good(t *testing.T) {
	result := &SSDResult{
		SampleMaxTokens:   256,
		SampleTemperature: 0.7,
		SampleTopK:        40,
		SampleTopP:        0.9,
		SampleMinP:        0.05,
		RepetitionPenalty: 1.1,
	}
	got := result.SampleGenerateConfig()
	want := inference.GenerateConfig{MaxTokens: 256, Temperature: 0.7, TopK: 40, TopP: 0.9, MinP: 0.05, RepeatPenalty: 1.1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("SampleGenerateConfig() = %+v, want %+v", got, want)
	}
}

// Bad: a nil *SSDResult returns the zero-value config rather than panicking.
func TestSsd_SSDResult_SampleGenerateConfig_Bad(t *testing.T) {
	var result *SSDResult
	if got := result.SampleGenerateConfig(); !reflect.DeepEqual(got, inference.GenerateConfig{}) {
		t.Fatalf("nil SampleGenerateConfig() = %+v, want the zero value", got)
	}
}

// Ugly: only some fields set on the result — each field of the returned
// config maps from its OWN source field, not a neighbour's (a copy-paste
// swap between e.g. TopK and TopP would slip past a fully-populated test).
func TestSsd_SSDResult_SampleGenerateConfig_Ugly(t *testing.T) {
	result := &SSDResult{SampleTopP: 0.42}
	got := result.SampleGenerateConfig()
	want := inference.GenerateConfig{TopP: 0.42}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("partial SampleGenerateConfig() = %+v, want only TopP set: %+v", got, want)
	}
}
