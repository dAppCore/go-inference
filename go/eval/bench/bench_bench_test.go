// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the driver-neutral local bench harness — Config
// normalisation, Run orchestration over a synthetic Runner, the
// generation-summary reducer, and the derived-field populator.
//
// Per AX-11 — Run is called once per bench invocation but
// summarizeGenerations + qualityChecks fire over every captured
// sample, and PopulateStateKVBlockWarmBench is called once per
// State-block bench from every driver. The Config copy in
// normalizeConfig touches three slice copies per call.
//
// Run:    go test -bench='BenchmarkBench' -benchmem -run='^$' ./go/bench

package bench

import (
	"context"
	"testing"
	"time"
)

// Sinks defeat compiler DCE.
var (
	benchSinkReport  *Report
	benchSinkErr     error
	benchSinkConfig  Config
	benchSinkSummary GenerationSummary
	benchSinkChecks  []QualityCheck
	benchSinkOpts    GenerateOptions
	benchSinkBool    bool
	benchSinkDur     time.Duration
)

// buildBenchSamples mints n GenerationSample records with representative
// timing + token counts — same shape Run captures from a real driver.
func buildBenchSamples(n int) []GenerationSample {
	samples := make([]GenerationSample, n)
	for i := 0; i < n; i++ {
		samples[i] = GenerationSample{
			Prompt: "Write one precise sentence about local inference.",
			Text:   "Local inference keeps tokens on-device.",
			Tokens: []int32{1, 2, 3, 4, 5, 6, 7, 8},
			Metrics: GenerationMetrics{
				PromptTokens:        12,
				GeneratedTokens:     32,
				FirstTokenDuration:  3 * time.Millisecond,
				PrefillDuration:     5 * time.Millisecond,
				DecodeDuration:      40 * time.Millisecond,
				TotalDuration:       45 * time.Millisecond,
				PrefillTokensPerSec: 2400,
				DecodeTokensPerSec:  800,
				PeakMemoryBytes:     uint64(64 << 20),
				ActiveMemoryBytes:   uint64(48 << 20),
			},
			Elapsed: 45 * time.Millisecond,
		}
	}
	return samples
}

// benchRunner returns a Runner whose Generate emits a fixed scripted
// generation. Used by BenchmarkBench_Run_* below.
func benchRunner(metrics GenerationMetrics) Runner {
	return Runner{
		Generate: func(_ context.Context, prompt string, _ GenerateOptions) (Generation, error) {
			return Generation{
				Text:    "Local inference keeps tokens on-device.",
				Tokens:  []int32{1, 2, 3, 4, 5, 6, 7, 8},
				Metrics: metrics,
			}, nil
		},
	}
}

// --- Run end-to-end with minimal config + scripted generation ---

func BenchmarkBench_Run_Minimal(b *testing.B) {
	cfg := Config{
		Prompt:    "Write one precise sentence about local inference.",
		MaxTokens: 32,
		Runs:      1,
	}
	runner := benchRunner(GenerationMetrics{
		PromptTokens: 12, GeneratedTokens: 32,
		PrefillDuration: 5 * time.Millisecond, DecodeDuration: 40 * time.Millisecond,
		TotalDuration: 45 * time.Millisecond,
	})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkReport, benchSinkErr = Run(ctx, runner, cfg)
	}
}

// 10 runs exercises the summariser inside Run on a bigger sample set.
func BenchmarkBench_Run_TenRuns(b *testing.B) {
	cfg := Config{
		Prompt:    "Write one precise sentence about local inference.",
		MaxTokens: 32,
		Runs:      10,
	}
	runner := benchRunner(GenerationMetrics{
		PromptTokens: 12, GeneratedTokens: 32,
		PrefillDuration: 5 * time.Millisecond, DecodeDuration: 40 * time.Millisecond,
		TotalDuration: 45 * time.Millisecond,
	})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkReport, benchSinkErr = Run(ctx, runner, cfg)
	}
}

// --- DefaultConfig + normalisation hot loop ---

func BenchmarkBench_DefaultConfig(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkConfig = DefaultConfig()
	}
}

func BenchmarkBench_NormalizeConfig_Zero(b *testing.B) {
	cfg := Config{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkConfig = normalizeConfig(cfg)
	}
}

func BenchmarkBench_NormalizeConfig_PopulatedMinimal(b *testing.B) {
	cfg := Config{
		Prompt:    "Write one precise sentence about local inference.",
		MaxTokens: 32,
		Runs:      1,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkConfig = normalizeConfig(cfg)
	}
}

// PopulatedFull exercises every slice-copy + deprecated-field migration
// branch in normalizeConfig.
func BenchmarkBench_NormalizeConfig_PopulatedFull(b *testing.B) {
	cfg := Config{
		Model:                       "qwen3",
		ModelPath:                   "/models/qwen3.gguf",
		Prompt:                      "Write one precise sentence about local inference.",
		CachePrompt:                 "Write one precise sentence about local inference.",
		MaxTokens:                   64,
		Runs:                        4,
		Temperature:                 0.7,
		TopK:                        40,
		TopP:                        0.9,
		MinP:                        0.05,
		StopTokens:                  []int32{0, 1, 2, 3, 4, 5, 6, 7},
		RepeatPenalty:               1.1,
		IncludePromptCache:          true,
		IncludeKVRestore:            true,
		IncludeStateBundleRoundTrip: true,
		IncludeProbeOverhead:        true,
		IncludeMemvidKVBlockWarm:    true,
		MemvidKVBlockSize:           512,
		MemvidKVPrefixTokens:        2048,
		MemvidKVBlockStorePath:      "/cache/state",
		SpeculativeDraftModelPath:   "/models/draft.gguf",
		SpeculativeDraftTokens:      8,
		PromptLookupTokens:          []int32{10, 20, 30, 40, 50},
		QualityPrompts:              []string{"a", "b", "c"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkConfig = normalizeConfig(cfg)
	}
}

// --- GenerateOptions derivation (per-call hot path) ---

func BenchmarkBench_Config_GenerateOptions_Bare(b *testing.B) {
	cfg := DefaultConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkOpts = cfg.GenerateOptions(nil)
	}
}

func BenchmarkBench_Config_GenerateOptions_WithStopTokens(b *testing.B) {
	cfg := DefaultConfig()
	cfg.StopTokens = []int32{0, 1, 2, 3, 4, 5, 6, 7}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkOpts = cfg.GenerateOptions(nil)
	}
}

// --- summarizeGenerations + qualityChecks (called once per Run) ---

func BenchmarkBench_SummarizeGenerations_1Sample(b *testing.B) {
	samples := buildBenchSamples(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkSummary = summarizeGenerations(samples)
	}
}

func BenchmarkBench_SummarizeGenerations_10Samples(b *testing.B) {
	samples := buildBenchSamples(10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkSummary = summarizeGenerations(samples)
	}
}

func BenchmarkBench_SummarizeGenerations_100Samples(b *testing.B) {
	samples := buildBenchSamples(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkSummary = summarizeGenerations(samples)
	}
}

func BenchmarkBench_QualityChecks_10Samples(b *testing.B) {
	samples := buildBenchSamples(10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkChecks = qualityChecks(samples)
	}
}

// --- AdapterInfo.IsEmpty (per-report check, fires from drivers) ---

func BenchmarkBench_AdapterInfo_IsEmpty_Empty(b *testing.B) {
	info := AdapterInfo{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBool = info.IsEmpty()
	}
}

func BenchmarkBench_AdapterInfo_IsEmpty_Populated(b *testing.B) {
	info := AdapterInfo{
		Name:       "qwen3-lora",
		Path:       "/adapters/qwen3.lora",
		Hash:       "sha256:deadbeef",
		Rank:       16,
		Alpha:      32,
		Scale:      0.5,
		TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBool = info.IsEmpty()
	}
}

// --- PopulateStateKVBlockWarmBench (fires once per State-block bench
// from every driver) ---

func BenchmarkBench_PopulateStateKVBlockWarm(b *testing.B) {
	baseline := GenerationSummary{
		PrefillDuration: 200 * time.Millisecond,
		PeakMemoryBytes: uint64(96 << 20),
	}
	report := StateKVBlockWarmReport{
		Attempted:       true,
		BuildDuration:   400 * time.Millisecond,
		RestoreDuration: 8 * time.Millisecond,
		Metrics: GenerationMetrics{
			PeakMemoryBytes:   uint64(120 << 20),
			ActiveMemoryBytes: uint64(64 << 20),
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := report
		PopulateStateKVBlockWarmBench(&r, baseline)
	}
}

// --- NonZeroDuration (exported helper, fires per Run sample) ---

func BenchmarkBench_NonZeroDuration_Positive(b *testing.B) {
	d := 45 * time.Millisecond
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkDur = NonZeroDuration(d)
	}
}

func BenchmarkBench_NonZeroDuration_Zero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkDur = NonZeroDuration(0)
	}
}
