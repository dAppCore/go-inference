// SPDX-Licence-Identifier: EUPL-1.2

package bench

import (
	"context"
	"errors"
	"testing"
	"time"
)

// fakeRunnerOptions describes the synthetic generation result the test
// runner will return on each Generate call.
type fakeRunnerOptions struct {
	generationMetrics []GenerationMetrics
	generationText    []string
	generationError   error
}

// newFakeRunner returns a Runner whose Generate emits scripted results.
// Callbacks other than Generate are filled with nil-stubs the caller can
// override.
func newFakeRunner(opts fakeRunnerOptions) (Runner, *int) {
	idx := new(int)
	runner := Runner{
		Generate: func(_ context.Context, _ string, _ GenerateOptions) (Generation, error) {
			if opts.generationError != nil {
				return Generation{}, opts.generationError
			}
			i := *idx
			*idx++
			text := ""
			if i < len(opts.generationText) {
				text = opts.generationText[i]
			}
			var metrics GenerationMetrics
			if i < len(opts.generationMetrics) {
				metrics = opts.generationMetrics[i]
			}
			return Generation{Text: text, Metrics: metrics}, nil
		},
	}
	return runner, idx
}

func TestRun_AggregatesGenerationSummary_Good(t *testing.T) {
	runner, _ := newFakeRunner(fakeRunnerOptions{
		generationText: []string{"alpha", "beta"},
		generationMetrics: []GenerationMetrics{
			{
				PromptTokens:        4,
				GeneratedTokens:     6,
				FirstTokenDuration:  12 * time.Millisecond,
				PrefillDuration:     20 * time.Millisecond,
				DecodeDuration:      30 * time.Millisecond,
				TotalDuration:       50 * time.Millisecond,
				PrefillTokensPerSec: 200,
				DecodeTokensPerSec:  60,
				PeakMemoryBytes:     1 << 20,
				ActiveMemoryBytes:   512 << 10,
			},
			{
				PromptTokens:        4,
				GeneratedTokens:     8,
				FirstTokenDuration:  18 * time.Millisecond,
				PrefillDuration:     20 * time.Millisecond,
				DecodeDuration:      40 * time.Millisecond,
				TotalDuration:       60 * time.Millisecond,
				PrefillTokensPerSec: 400,
				DecodeTokensPerSec:  80,
				PeakMemoryBytes:     2 << 20,
				ActiveMemoryBytes:   1 << 20,
			},
		},
	})

	report, err := Run(context.Background(), runner, Config{Prompt: "p", MaxTokens: 16, Runs: 2})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if report.Version != ReportVersion {
		t.Fatalf("Version = %d, want %d", report.Version, ReportVersion)
	}
	summary := report.Generation
	if summary.Runs != 2 {
		t.Fatalf("Runs = %d, want 2", summary.Runs)
	}
	if summary.PromptTokens != 8 || summary.GeneratedTokens != 14 {
		t.Fatalf("tokens = prompt:%d generated:%d", summary.PromptTokens, summary.GeneratedTokens)
	}
	if summary.PrefillTokensPerSec != 300 || summary.DecodeTokensPerSec != 70 {
		t.Fatalf("rates = prefill:%v decode:%v, want averages 300/70",
			summary.PrefillTokensPerSec, summary.DecodeTokensPerSec)
	}
	if summary.PeakMemoryBytes != 2<<20 || summary.ActiveMemoryBytes != 1<<20 {
		t.Fatalf("memory = peak:%d active:%d", summary.PeakMemoryBytes, summary.ActiveMemoryBytes)
	}
	if summary.PrefillDuration != 40*time.Millisecond || summary.DecodeDuration != 70*time.Millisecond {
		t.Fatalf("durations = prefill:%v decode:%v", summary.PrefillDuration, summary.DecodeDuration)
	}
	if summary.TotalDuration != 110*time.Millisecond {
		t.Fatalf("total duration = %v, want 110ms", summary.TotalDuration)
	}
	if summary.FirstTokenDuration != 15*time.Millisecond {
		t.Fatalf("first token duration = %v, want 15ms average", summary.FirstTokenDuration)
	}
	if len(summary.Samples) != 2 || summary.Samples[0].Text != "alpha" || summary.Samples[1].Text != "beta" {
		t.Fatalf("samples = %+v", summary.Samples)
	}
}

func TestRun_FallsBackToElapsedWhenTotalDurationZero_Good(t *testing.T) {
	runner, _ := newFakeRunner(fakeRunnerOptions{
		generationText:    []string{"hi"},
		generationMetrics: []GenerationMetrics{{PromptTokens: 1, GeneratedTokens: 1}},
	})
	report, err := Run(context.Background(), runner, Config{Prompt: "p", MaxTokens: 4, Runs: 1})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if report.Generation.TotalDuration <= 0 {
		t.Fatalf("TotalDuration = %v, want positive fallback from elapsed", report.Generation.TotalDuration)
	}
}

func TestRun_RequiresGenerate_Bad(t *testing.T) {
	if _, err := Run(context.Background(), Runner{}, Config{Prompt: "p", MaxTokens: 4, Runs: 1}); err == nil {
		t.Fatal("Run() without Generate did not error")
	}
}

func TestRun_PropagatesGenerateError_Bad(t *testing.T) {
	want := errors.New("boom")
	runner, _ := newFakeRunner(fakeRunnerOptions{generationError: want})
	if _, err := Run(context.Background(), runner, Config{Prompt: "p", MaxTokens: 4, Runs: 1}); err == nil {
		t.Fatal("Run() did not propagate Generate error")
	}
}

func TestRun_NilContextDefaultsToBackground_Good(t *testing.T) {
	runner, _ := newFakeRunner(fakeRunnerOptions{
		generationText:    []string{"ok"},
		generationMetrics: []GenerationMetrics{{GeneratedTokens: 1}},
	})
	report, err := Run(nil, runner, Config{Prompt: "p", MaxTokens: 4, Runs: 1})
	if err != nil {
		t.Fatalf("Run(nil ctx) error = %v", err)
	}
	if report == nil {
		t.Fatal("Run(nil ctx) report = nil")
	}
}

func TestRun_PopulatesModelInfoFromCallback_Good(t *testing.T) {
	runner, _ := newFakeRunner(fakeRunnerOptions{
		generationText:    []string{"ok"},
		generationMetrics: []GenerationMetrics{{GeneratedTokens: 1}},
	})
	runner.Info = func(context.Context) Info {
		return Info{Architecture: "qwen3", NumLayers: 28, ContextLength: 32768}
	}
	report, err := Run(context.Background(), runner, Config{Prompt: "p", MaxTokens: 4, Runs: 1})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if report.ModelInfo.Architecture != "qwen3" || report.ModelInfo.NumLayers != 28 || report.ModelInfo.ContextLength != 32768 {
		t.Fatalf("ModelInfo = %+v", report.ModelInfo)
	}
}

func TestRun_DispatchesVerbCallbacksWhenIncludeFlagsSet_Good(t *testing.T) {
	runner, _ := newFakeRunner(fakeRunnerOptions{
		generationText:    []string{"ok"},
		generationMetrics: []GenerationMetrics{{GeneratedTokens: 1, TotalDuration: 5 * time.Millisecond}},
	})
	called := struct {
		pc, mvkv, restore, bundle, probe, spec, lookup bool
	}{}
	runner.BenchPromptCache = func(context.Context, Config, GenerationSummary) PromptCacheReport {
		called.pc = true
		return PromptCacheReport{Attempted: true, HitRate: 1}
	}
	runner.BenchMemvidKVBlockWarm = func(context.Context, Config, GenerationSummary) MemvidKVBlockWarmReport {
		called.mvkv = true
		return MemvidKVBlockWarmReport{Attempted: true, BlockSize: 128}
	}
	runner.BenchKVRestore = func(context.Context, Config) LatencyReport {
		called.restore = true
		return LatencyReport{Attempted: true, Duration: time.Millisecond}
	}
	runner.BenchStateBundle = func(context.Context, Config, Info) StateBundleReport {
		called.bundle = true
		return StateBundleReport{Attempted: true, Bytes: 42}
	}
	runner.BenchProbeOverhead = func(context.Context, Config, time.Duration) ProbeReport {
		called.probe = true
		return ProbeReport{Attempted: true, EventCount: 3}
	}
	runner.BenchSpeculativeDecode = func(context.Context, Config) DecodeOptimisationReport {
		called.spec = true
		return DecodeOptimisationReport{Attempted: true, Result: DecodeOptimisationResult{Mode: "speculative"}}
	}
	runner.BenchPromptLookupDecode = func(context.Context, Config) DecodeOptimisationReport {
		called.lookup = true
		return DecodeOptimisationReport{Attempted: true, Result: DecodeOptimisationResult{Mode: "prompt_lookup"}}
	}

	cfg := Config{
		Prompt:                      "p",
		MaxTokens:                   4,
		Runs:                        1,
		IncludePromptCache:          true,
		IncludeMemvidKVBlockWarm:    true,
		IncludeKVRestore:            true,
		IncludeStateBundleRoundTrip: true,
		IncludeProbeOverhead:        true,
		IncludeSpeculativeDecode:    true,
		IncludePromptLookupDecode:   true,
	}
	report, err := Run(context.Background(), runner, cfg)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if !called.pc || !called.mvkv || !called.restore || !called.bundle || !called.probe || !called.spec || !called.lookup {
		t.Fatalf("verb callbacks not all called: %+v", called)
	}
	if !report.PromptCache.Attempted || report.PromptCache.HitRate != 1 {
		t.Fatalf("PromptCache = %+v", report.PromptCache)
	}
	if !report.MemvidKVBlockWarm.Attempted || report.MemvidKVBlockWarm.BlockSize != 128 {
		t.Fatalf("MemvidKVBlockWarm = %+v", report.MemvidKVBlockWarm)
	}
	if !report.KVRestore.Attempted || report.KVRestore.Duration != time.Millisecond {
		t.Fatalf("KVRestore = %+v", report.KVRestore)
	}
	if !report.StateBundle.Attempted || report.StateBundle.Bytes != 42 {
		t.Fatalf("StateBundle = %+v", report.StateBundle)
	}
	if !report.Probes.Attempted || report.Probes.EventCount != 3 {
		t.Fatalf("Probes = %+v", report.Probes)
	}
	if !report.SpeculativeDecode.Attempted || report.SpeculativeDecode.Result.Mode != "speculative" {
		t.Fatalf("SpeculativeDecode = %+v", report.SpeculativeDecode)
	}
	if !report.PromptLookupDecode.Attempted || report.PromptLookupDecode.Result.Mode != "prompt_lookup" {
		t.Fatalf("PromptLookupDecode = %+v", report.PromptLookupDecode)
	}
}

func TestRun_SkipsVerbCallbacksWhenIncludeFlagsFalse_Good(t *testing.T) {
	runner, _ := newFakeRunner(fakeRunnerOptions{
		generationText:    []string{"ok"},
		generationMetrics: []GenerationMetrics{{GeneratedTokens: 1}},
	})
	// Set every callback to a fatal-on-call closure: if Run incorrectly
	// dispatches it, the test fails.
	runner.BenchPromptCache = func(context.Context, Config, GenerationSummary) PromptCacheReport {
		t.Fatal("BenchPromptCache called when IncludePromptCache is false")
		return PromptCacheReport{}
	}
	runner.BenchMemvidKVBlockWarm = func(context.Context, Config, GenerationSummary) MemvidKVBlockWarmReport {
		t.Fatal("BenchMemvidKVBlockWarm called when IncludeMemvidKVBlockWarm is false")
		return MemvidKVBlockWarmReport{}
	}
	runner.BenchKVRestore = func(context.Context, Config) LatencyReport {
		t.Fatal("BenchKVRestore called when IncludeKVRestore is false")
		return LatencyReport{}
	}
	runner.BenchStateBundle = func(context.Context, Config, Info) StateBundleReport {
		t.Fatal("BenchStateBundle called when IncludeStateBundleRoundTrip is false")
		return StateBundleReport{}
	}
	runner.BenchProbeOverhead = func(context.Context, Config, time.Duration) ProbeReport {
		t.Fatal("BenchProbeOverhead called when IncludeProbeOverhead is false")
		return ProbeReport{}
	}
	runner.BenchSpeculativeDecode = func(context.Context, Config) DecodeOptimisationReport {
		t.Fatal("BenchSpeculativeDecode called when IncludeSpeculativeDecode is false")
		return DecodeOptimisationReport{}
	}
	runner.BenchPromptLookupDecode = func(context.Context, Config) DecodeOptimisationReport {
		t.Fatal("BenchPromptLookupDecode called when IncludePromptLookupDecode is false")
		return DecodeOptimisationReport{}
	}

	cfg := Config{Prompt: "p", MaxTokens: 4, Runs: 1}
	if _, err := Run(context.Background(), runner, cfg); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
}

func TestRun_QualityChecks_Good(t *testing.T) {
	runner, _ := newFakeRunner(fakeRunnerOptions{
		generationText: []string{"hello"},
		generationMetrics: []GenerationMetrics{{
			GeneratedTokens: 5,
			TotalDuration:   10 * time.Millisecond,
		}},
	})
	report, err := Run(context.Background(), runner, Config{Prompt: "p", MaxTokens: 8, Runs: 1})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(report.Quality.Checks) != 2 {
		t.Fatalf("Quality.Checks = %d, want 2 default checks", len(report.Quality.Checks))
	}
	for _, check := range report.Quality.Checks {
		switch check.Name {
		case "non_empty_output":
			if !check.Pass {
				t.Fatalf("non_empty_output check failed: %+v", check)
			}
		case "generated_tokens":
			if !check.Pass || check.Detail != "5" {
				t.Fatalf("generated_tokens check = %+v", check)
			}
		default:
			t.Fatalf("unexpected check %q", check.Name)
		}
	}
}

func TestRun_QualityChecksFlagEmptyOutput_Ugly(t *testing.T) {
	runner, _ := newFakeRunner(fakeRunnerOptions{
		generationText:    []string{""},
		generationMetrics: []GenerationMetrics{{}},
	})
	report, err := Run(context.Background(), runner, Config{Prompt: "p", MaxTokens: 4, Runs: 1})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	for _, check := range report.Quality.Checks {
		if check.Pass {
			t.Fatalf("expected quality check %q to fail for empty output, got %+v", check.Name, check)
		}
	}
}

func TestDefaultConfig_Good(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.MaxTokens != 32 || cfg.Runs != 1 {
		t.Fatalf("DefaultConfig() = %+v, want MaxTokens=32 Runs=1", cfg)
	}
	if !cfg.IncludePromptCache || !cfg.IncludeKVRestore || !cfg.IncludeStateBundleRoundTrip || !cfg.IncludeProbeOverhead {
		t.Fatalf("DefaultConfig() includes = %+v, want baseline four-section coverage", cfg)
	}
	if cfg.Prompt == "" {
		t.Fatal("DefaultConfig() Prompt is empty")
	}
}

func TestNormalizeConfig_FillsDefaultsFromZero_Good(t *testing.T) {
	got := normalizeConfig(Config{})
	want := DefaultConfig()
	if got.MaxTokens != want.MaxTokens || got.Runs != want.Runs || got.Prompt != want.Prompt {
		t.Fatalf("normalizeConfig(zero) = %+v, want defaults %+v", got, want)
	}
}

func TestNormalizeConfig_PreservesPartialConfig_Good(t *testing.T) {
	got := normalizeConfig(Config{Prompt: "x", MaxTokens: 7})
	if got.Prompt != "x" || got.MaxTokens != 7 || got.Runs != 1 {
		t.Fatalf("normalizeConfig(partial) = %+v", got)
	}
	if got.CachePrompt != "x" {
		t.Fatalf("CachePrompt = %q, want fallback to Prompt", got.CachePrompt)
	}
}

func TestNormalizeConfig_ClonesSlices_Good(t *testing.T) {
	stops := []int32{1, 2, 3}
	lookup := []int32{4, 5}
	quality := []string{"a"}
	cfg := normalizeConfig(Config{Prompt: "x", MaxTokens: 4, Runs: 1, StopTokens: stops, PromptLookupTokens: lookup, QualityPrompts: quality})
	stops[0] = 99
	lookup[0] = 99
	quality[0] = "z"
	if cfg.StopTokens[0] == 99 || cfg.PromptLookupTokens[0] == 99 || cfg.QualityPrompts[0] == "z" {
		t.Fatalf("normalizeConfig did not clone slices: %+v", cfg)
	}
}

func TestPopulateMemvidKVBlockWarmBench_DerivesSpeedupAndBreakEven_Good(t *testing.T) {
	report := MemvidKVBlockWarmReport{
		Attempted:       true,
		BuildDuration:   100 * time.Millisecond,
		RestoreDuration: 10 * time.Millisecond,
		Metrics:         GenerationMetrics{PeakMemoryBytes: 1 << 20},
	}
	baseline := GenerationSummary{
		PrefillDuration: 50 * time.Millisecond,
		PeakMemoryBytes: 2 << 20,
	}
	PopulateMemvidKVBlockWarmBench(&report, baseline)
	if report.BaselinePrefillDuration != 50*time.Millisecond {
		t.Fatalf("BaselinePrefillDuration = %v", report.BaselinePrefillDuration)
	}
	if report.RestoreSpeedup != 5 {
		t.Fatalf("RestoreSpeedup = %v, want 5", report.RestoreSpeedup)
	}
	if report.PrefillSavedPerQuestion != 40*time.Millisecond {
		t.Fatalf("PrefillSavedPerQuestion = %v, want 40ms", report.PrefillSavedPerQuestion)
	}
	if report.BreakEvenQuestions != 3 {
		t.Fatalf("BreakEvenQuestions = %d, want 3 (ceil(100ms/40ms))", report.BreakEvenQuestions)
	}
	if report.MemoryPeakBytes != 2<<20 {
		t.Fatalf("MemoryPeakBytes = %d, want baseline peak 2MiB", report.MemoryPeakBytes)
	}
}

func TestPopulateMemvidKVBlockWarmBench_SkipsWhenNotAttempted_Ugly(t *testing.T) {
	report := MemvidKVBlockWarmReport{
		BuildDuration:   100 * time.Millisecond,
		RestoreDuration: 10 * time.Millisecond,
	}
	PopulateMemvidKVBlockWarmBench(&report, GenerationSummary{PrefillDuration: 50 * time.Millisecond})
	if report.BaselinePrefillDuration != 0 || report.RestoreSpeedup != 0 || report.BreakEvenQuestions != 0 {
		t.Fatalf("expected no-op when Attempted is false, got %+v", report)
	}
}

func TestPopulateMemvidKVBlockWarmBench_SkipsWhenSavedNonPositive_Ugly(t *testing.T) {
	// Restore took LONGER than baseline prefill — no speedup, no break-even.
	report := MemvidKVBlockWarmReport{
		Attempted:       true,
		BuildDuration:   100 * time.Millisecond,
		RestoreDuration: 80 * time.Millisecond,
	}
	PopulateMemvidKVBlockWarmBench(&report, GenerationSummary{PrefillDuration: 50 * time.Millisecond})
	if report.PrefillSavedPerQuestion != 0 || report.BreakEvenQuestions != 0 {
		t.Fatalf("expected no break-even when restore is slower than baseline, got saved:%v break-even:%d", report.PrefillSavedPerQuestion, report.BreakEvenQuestions)
	}
	if report.RestoreSpeedup == 0 {
		t.Fatalf("RestoreSpeedup should still be derived even when slower, got %v", report.RestoreSpeedup)
	}
}

func TestAdapterInfo_IsEmpty_GoodBad(t *testing.T) {
	if !(AdapterInfo{}).IsEmpty() {
		t.Fatal("zero AdapterInfo IsEmpty = false, want true")
	}
	if (AdapterInfo{Name: "x"}).IsEmpty() {
		t.Fatal("AdapterInfo with Name IsEmpty = true, want false")
	}
	if (AdapterInfo{Rank: 8}).IsEmpty() {
		t.Fatal("AdapterInfo with Rank IsEmpty = true, want false")
	}
	if (AdapterInfo{TargetKeys: []string{"q_proj"}}).IsEmpty() {
		t.Fatal("AdapterInfo with TargetKeys IsEmpty = true, want false")
	}
}

func TestConfigGenerateOptions_PassesProbeSinkThrough_Good(t *testing.T) {
	sentinel := struct{ tag string }{tag: "sink"}
	cfg := Config{MaxTokens: 16, Temperature: 0.7, StopTokens: []int32{1}}
	opts := cfg.GenerateOptions(sentinel)
	if opts.MaxTokens != 16 || opts.Temperature != 0.7 || len(opts.StopTokens) != 1 {
		t.Fatalf("GenerateOptions = %+v", opts)
	}
	got, ok := opts.ProbeSink.(struct{ tag string })
	if !ok || got.tag != "sink" {
		t.Fatalf("ProbeSink = %+v ok=%v, want sentinel passed through", opts.ProbeSink, ok)
	}
}

func TestConfigGenerateOptions_ClonesStopTokens_Good(t *testing.T) {
	stops := []int32{1, 2, 3}
	cfg := Config{MaxTokens: 1, StopTokens: stops}
	opts := cfg.GenerateOptions(nil)
	stops[0] = 99
	if opts.StopTokens[0] == 99 {
		t.Fatal("GenerateOptions did not clone StopTokens — mutating caller-side slice changed snapshot")
	}
}

func TestRun_RunsClampToOneByDefault_Good(t *testing.T) {
	idx := new(int)
	runner := Runner{
		Generate: func(context.Context, string, GenerateOptions) (Generation, error) {
			*idx++
			return Generation{Text: "x", Metrics: GenerationMetrics{GeneratedTokens: 1}}, nil
		},
	}
	// Config with Prompt but Runs=0 — normalize fills default of 1.
	if _, err := Run(context.Background(), runner, Config{Prompt: "p", MaxTokens: 4}); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if *idx != 1 {
		t.Fatalf("Generate called %d times, want 1 after Runs<=0 normalisation", *idx)
	}
}

func TestNonZeroDuration_Good(t *testing.T) {
	if got := NonZeroDuration(0); got != time.Nanosecond {
		t.Fatalf("NonZeroDuration(0) = %v, want 1ns floor", got)
	}
	if got := NonZeroDuration(-5); got != time.Nanosecond {
		t.Fatalf("NonZeroDuration(-5) = %v, want 1ns floor", got)
	}
	if got := NonZeroDuration(123 * time.Millisecond); got != 123*time.Millisecond {
		t.Fatalf("NonZeroDuration(123ms) = %v, want passthrough", got)
	}
}
