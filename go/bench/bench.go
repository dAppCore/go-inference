// SPDX-Licence-Identifier: EUPL-1.2

// Package bench is the driver-neutral local benchmark/eval harness.
//
// Drivers (go-mlx, go-rocm, go-cuda, …) supply a Runner with
// verb-shaped callbacks for each section of the bench (PromptCache,
// StateKVBlockWarm, KVRestore, StateBundle, SpeculativeDecode,
// PromptLookupDecode, ProbeOverhead). bench.Run orchestrates the
// generation timing + calls each enabled callback + assembles the
// final Report.
package bench

import (
	"context"
	"time"

	core "dappco.re/go"
)

const ReportVersion = 1

// Config controls the local benchmark/eval harness.
type Config struct {
	Model                       string  `json:"model,omitempty"`
	ModelPath                   string  `json:"model_path,omitempty"`
	Prompt                      string  `json:"prompt"`
	CachePrompt                 string  `json:"cache_prompt,omitempty"`
	MaxTokens                   int     `json:"max_tokens"`
	Runs                        int     `json:"runs"`
	Temperature                 float32 `json:"temperature"`
	TopK                        int     `json:"top_k,omitempty"`
	TopP                        float32 `json:"top_p,omitempty"`
	MinP                        float32 `json:"min_p,omitempty"`
	StopTokens                  []int32 `json:"stop_tokens,omitempty"`
	RepeatPenalty               float32 `json:"repeat_penalty,omitempty"`
	IncludePromptCache          bool    `json:"include_prompt_cache"`
	IncludeKVRestore            bool    `json:"include_kv_restore"`
	IncludeStateBundleRoundTrip bool    `json:"include_state_bundle_round_trip"`
	IncludeProbeOverhead        bool    `json:"include_probe_overhead"`
	IncludeStateKVBlockWarm     bool    `json:"include_state_kv_block_warm"`
	// Deprecated: use IncludeStateKVBlockWarm. Kept for old Go callers only.
	IncludeMemvidKVBlockWarm  bool   `json:"-"`
	IncludeSpeculativeDecode  bool   `json:"include_speculative_decode"`
	IncludePromptLookupDecode bool   `json:"include_prompt_lookup_decode"`
	StateKVBlockSize          int    `json:"state_kv_block_size,omitempty"`
	StateKVPrefixTokens       int    `json:"state_kv_prefix_tokens,omitempty"`
	StateKVBlockStorePath     string `json:"state_kv_block_store_path,omitempty"`
	// Deprecated: use StateKVBlockSize. Kept for old Go callers only.
	MemvidKVBlockSize int `json:"-"`
	// Deprecated: use StateKVPrefixTokens. Kept for old Go callers only.
	MemvidKVPrefixTokens int `json:"-"`
	// Deprecated: use StateKVBlockStorePath. Kept for old Go callers only.
	MemvidKVBlockStorePath    string   `json:"-"`
	SpeculativeDraftModelPath string   `json:"speculative_draft_model_path,omitempty"`
	SpeculativeDraftTokens    int      `json:"speculative_draft_tokens,omitempty"`
	PromptLookupTokens        []int32  `json:"prompt_lookup_tokens,omitempty"`
	QualityPrompts            []string `json:"quality_prompts,omitempty"`
}

// DefaultConfig returns a short local benchmark suite suitable for a laptop.
func DefaultConfig() Config {
	return Config{
		Prompt:                      "Write one precise sentence about local inference.",
		MaxTokens:                   32,
		Runs:                        1,
		Temperature:                 0,
		IncludePromptCache:          true,
		IncludeKVRestore:            true,
		IncludeStateBundleRoundTrip: true,
		IncludeProbeOverhead:        true,
	}
}

// Info mirrors a driver's model info — the fields bench consumers care about.
type Info struct {
	Architecture  string      `json:"architecture,omitempty"`
	VocabSize     int         `json:"vocab_size,omitempty"`
	NumLayers     int         `json:"num_layers,omitempty"`
	HiddenSize    int         `json:"hidden_size,omitempty"`
	QuantBits     int         `json:"quant_bits,omitempty"`
	QuantGroup    int         `json:"quant_group,omitempty"`
	ContextLength int         `json:"context_length,omitempty"`
	Adapter       AdapterInfo `json:"adapter,omitempty"`
}

// AdapterInfo identifies a LoRA adapter participating in the bench run.
// Mirrors the shape of go-mlx/lora.AdapterInfo but lives in bench to keep
// the package driver-neutral.
type AdapterInfo struct {
	Name       string   `json:"name,omitempty"`
	Path       string   `json:"path,omitempty"`
	Hash       string   `json:"hash,omitempty"`
	Rank       int      `json:"rank,omitempty"`
	Alpha      float32  `json:"alpha,omitempty"`
	Scale      float32  `json:"scale,omitempty"`
	TargetKeys []string `json:"target_keys,omitempty"`
}

// IsEmpty reports whether the adapter info has no meaningful fields set.
func (info AdapterInfo) IsEmpty() bool {
	return info.Name == "" && info.Path == "" && info.Hash == "" && info.Rank == 0 && info.Alpha == 0 && info.Scale == 0 && len(info.TargetKeys) == 0
}

// GenerateOptions describes one generation request.
type GenerateOptions struct {
	MaxTokens     int     `json:"max_tokens"`
	Temperature   float32 `json:"temperature,omitempty"`
	TopK          int     `json:"top_k,omitempty"`
	TopP          float32 `json:"top_p,omitempty"`
	MinP          float32 `json:"min_p,omitempty"`
	StopTokens    []int32 `json:"stop_tokens,omitempty"`
	RepeatPenalty float32 `json:"repeat_penalty,omitempty"`
	// ProbeSink is opaque to bench. Drivers that support probe-recording
	// attach the recorder here; the value is passed through to the
	// driver's Generate call.
	ProbeSink any `json:"-"`
}

// GenerateOptions returns the per-call generation options derived from
// the Config plus the (optional) probe sink for that call.
func (c Config) GenerateOptions(sink any) GenerateOptions {
	return GenerateOptions{
		MaxTokens:     c.MaxTokens,
		Temperature:   c.Temperature,
		TopK:          c.TopK,
		TopP:          c.TopP,
		MinP:          c.MinP,
		StopTokens:    append([]int32(nil), c.StopTokens...),
		RepeatPenalty: c.RepeatPenalty,
		ProbeSink:     sink,
	}
}

// Generation is one model response plus the driver-reported metrics.
type Generation struct {
	Text    string            `json:"text,omitempty"`
	Tokens  []int32           `json:"tokens,omitempty"`
	Metrics GenerationMetrics `json:"metrics"`
}

// GenerationMetrics is the bench-readable snapshot of generation timing
// + memory + prompt-cache counters. Drivers populate the fields they can
// report; missing fields are zero.
type GenerationMetrics struct {
	PromptTokens               int           `json:"prompt_tokens"`
	GeneratedTokens            int           `json:"generated_tokens"`
	FirstTokenDuration         time.Duration `json:"first_token_duration,omitempty"`
	PrefillDuration            time.Duration `json:"prefill_duration"`
	DecodeDuration             time.Duration `json:"decode_duration"`
	TotalDuration              time.Duration `json:"total_duration"`
	PrefillTokensPerSec        float64       `json:"prefill_tokens_per_sec"`
	DecodeTokensPerSec         float64       `json:"decode_tokens_per_sec"`
	PeakMemoryBytes            uint64        `json:"peak_memory_bytes"`
	ActiveMemoryBytes          uint64        `json:"active_memory_bytes"`
	PromptCacheHits            int           `json:"prompt_cache_hits,omitempty"`
	PromptCacheMisses          int           `json:"prompt_cache_misses,omitempty"`
	PromptCacheHitTokens       int           `json:"prompt_cache_hit_tokens,omitempty"`
	PromptCacheMissTokens      int           `json:"prompt_cache_miss_tokens,omitempty"`
	PromptCacheRestoreDuration time.Duration `json:"prompt_cache_restore_duration,omitempty"`
}

// Runner is the model-side surface bench.Run needs. Generate is required;
// every Bench* callback is optional — if absent, the corresponding
// section of the Report stays Attempted=false.
type Runner struct {
	Info     func(context.Context) Info
	Generate func(context.Context, string, GenerateOptions) (Generation, error)

	BenchPromptCache        func(context.Context, Config, GenerationSummary) PromptCacheReport
	BenchStateKVBlockWarm   func(context.Context, Config, GenerationSummary) StateKVBlockWarmReport
	BenchKVRestore          func(context.Context, Config) LatencyReport
	BenchStateBundle        func(context.Context, Config, Info) StateBundleReport
	BenchProbeOverhead      func(context.Context, Config, time.Duration) ProbeReport
	BenchSpeculativeDecode  func(context.Context, Config) DecodeOptimisationReport
	BenchPromptLookupDecode func(context.Context, Config) DecodeOptimisationReport

	// Deprecated: use BenchStateKVBlockWarm.
	BenchMemvidKVBlockWarm func(context.Context, Config, GenerationSummary) MemvidKVBlockWarmReport
}

// Report is the full benchmark result.
type Report struct {
	Version          int                    `json:"version"`
	Model            string                 `json:"model,omitempty"`
	ModelPath        string                 `json:"model_path,omitempty"`
	ModelInfo        Info                   `json:"model_info"`
	Config           Config                 `json:"config"`
	Generation       GenerationSummary      `json:"generation"`
	PromptCache      PromptCacheReport      `json:"prompt_cache"`
	StateKVBlockWarm StateKVBlockWarmReport `json:"state_kv_block_warm"`
	// Deprecated: use StateKVBlockWarm. Kept for old Go callers only.
	MemvidKVBlockWarm  MemvidKVBlockWarmReport  `json:"-"`
	KVRestore          LatencyReport            `json:"kv_restore"`
	StateBundle        StateBundleReport        `json:"state_bundle"`
	Probes             ProbeReport              `json:"probes"`
	SpeculativeDecode  DecodeOptimisationReport `json:"speculative_decode"`
	PromptLookupDecode DecodeOptimisationReport `json:"prompt_lookup_decode"`
	Quality            QualityReport            `json:"quality"`
}

// GenerationSample stores one measured generation pass.
type GenerationSample struct {
	Prompt  string            `json:"prompt"`
	Text    string            `json:"text,omitempty"`
	Tokens  []int32           `json:"tokens,omitempty"`
	Metrics GenerationMetrics `json:"metrics"`
	Elapsed time.Duration     `json:"elapsed"`
}

// GenerationSummary aggregates baseline generation passes.
type GenerationSummary struct {
	Runs                int                `json:"runs"`
	PromptTokens        int                `json:"prompt_tokens"`
	GeneratedTokens     int                `json:"generated_tokens"`
	FirstTokenDuration  time.Duration      `json:"first_token_duration,omitempty"`
	PrefillTokensPerSec float64            `json:"prefill_tokens_per_sec"`
	DecodeTokensPerSec  float64            `json:"decode_tokens_per_sec"`
	PrefillDuration     time.Duration      `json:"prefill_duration"`
	DecodeDuration      time.Duration      `json:"decode_duration"`
	TotalDuration       time.Duration      `json:"total_duration"`
	PeakMemoryBytes     uint64             `json:"peak_memory_bytes"`
	ActiveMemoryBytes   uint64             `json:"active_memory_bytes"`
	Samples             []GenerationSample `json:"samples,omitempty"`
}

// PromptCacheReport measures warmed prompt-cache reuse.
type PromptCacheReport struct {
	Attempted       bool              `json:"attempted"`
	Hits            int               `json:"hits,omitempty"`
	Misses          int               `json:"misses,omitempty"`
	HitRate         float64           `json:"hit_rate,omitempty"`
	HitTokens       int               `json:"hit_tokens,omitempty"`
	MissTokens      int               `json:"miss_tokens,omitempty"`
	WarmDuration    time.Duration     `json:"warm_duration,omitempty"`
	RestoreDuration time.Duration     `json:"restore_duration,omitempty"`
	Metrics         GenerationMetrics `json:"metrics,omitempty"`
	Error           string            `json:"error,omitempty"`
}

// StateKVBlockWarmReport measures direct prompt-cache warmup from durable
// State KV blocks (driver-specific feature; mlx provides one, others may not).
type StateKVBlockWarmReport struct {
	Attempted                  bool              `json:"attempted"`
	Source                     string            `json:"source,omitempty"`
	BlockSize                  int               `json:"block_size,omitempty"`
	TotalBlocks                int               `json:"total_blocks,omitempty"`
	StorePath                  string            `json:"store_path,omitempty"`
	StoreBytes                 int64             `json:"store_bytes,omitempty"`
	BuildDuration              time.Duration     `json:"build_duration,omitempty"`
	BuildTokens                int               `json:"build_tokens,omitempty"`
	BuildTokensPerSec          float64           `json:"build_tokens_per_sec,omitempty"`
	BlocksRead                 int               `json:"blocks_read,omitempty"`
	ChunksRead                 int               `json:"chunks_read,omitempty"`
	PrefixTokensRestored       int               `json:"prefix_tokens_restored,omitempty"`
	PromptTokensAvoided        int               `json:"prompt_tokens_avoided,omitempty"`
	ReplayTokens               int               `json:"replay_tokens,omitempty"`
	ExactFallbackReplayTokens  int               `json:"exact_fallback_replay_tokens,omitempty"`
	BaselinePrefillDuration    time.Duration     `json:"baseline_prefill_duration,omitempty"`
	RestoreDuration            time.Duration     `json:"restore_duration,omitempty"`
	GenerateDuration           time.Duration     `json:"generate_duration,omitempty"`
	PrefillSavedPerQuestion    time.Duration     `json:"prefill_saved_per_question,omitempty"`
	BuildAmortizationQuestions int               `json:"build_amortization_questions,omitempty"`
	BreakEvenQuestions         int               `json:"break_even_questions,omitempty"`
	RestoreSpeedup             float64           `json:"restore_speedup,omitempty"`
	MemoryPeakBytes            uint64            `json:"memory_peak_bytes,omitempty"`
	Metrics                    GenerationMetrics `json:"metrics,omitempty"`
	Error                      string            `json:"error,omitempty"`
}

// MemvidKVBlockWarmReport measures direct prompt-cache warmup from old
// memvid-named KV blocks.
//
// Deprecated: use StateKVBlockWarmReport.
type MemvidKVBlockWarmReport = StateKVBlockWarmReport

// LatencyReport records a best-effort latency measurement.
type LatencyReport struct {
	Attempted bool          `json:"attempted"`
	Duration  time.Duration `json:"duration,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// StateBundleReport records state-bundle JSON round-trip behavior.
type StateBundleReport struct {
	Attempted bool          `json:"attempted"`
	Duration  time.Duration `json:"duration,omitempty"`
	Bytes     int           `json:"bytes,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// ProbeReport records probe event count and estimated runtime overhead.
//
// Events is opaque (driver-specific probe event vocabulary); KindCounts
// gives bench a portable summary.
type ProbeReport struct {
	Attempted     bool              `json:"attempted"`
	EventCount    int               `json:"event_count,omitempty"`
	KindCounts    map[string]int    `json:"kind_counts,omitempty"`
	Duration      time.Duration     `json:"duration,omitempty"`
	OverheadRatio float64           `json:"overhead_ratio,omitempty"`
	Metrics       GenerationMetrics `json:"metrics,omitempty"`
	Error         string            `json:"error,omitempty"`
	Events        []any             `json:"events,omitempty"`
}

// DecodeOptimisationReport records an optional decode-optimisation
// comparison against the baseline generation path.
type DecodeOptimisationReport struct {
	Attempted bool                      `json:"attempted"`
	Result    DecodeOptimisationResult  `json:"result,omitempty"`
	Metrics   DecodeOptimisationMetrics `json:"metrics,omitempty"`
	Error     string                    `json:"error,omitempty"`
}

// DecodeOptimisationResult mirrors the driver's speculative/prompt-lookup
// decode result. Drivers populate the fields their algorithm produces.
type DecodeOptimisationResult struct {
	Mode    string                    `json:"mode"`
	Prompt  string                    `json:"prompt,omitempty"`
	Text    string                    `json:"text,omitempty"`
	Tokens  []int32                   `json:"tokens,omitempty"`
	Metrics DecodeOptimisationMetrics `json:"metrics"`
}

// DecodeOptimisationMetrics summarises candidate acceptance and timing.
type DecodeOptimisationMetrics struct {
	TargetTokens        int           `json:"target_tokens,omitempty"`
	DraftTokens         int           `json:"draft_tokens,omitempty"`
	LookupTokens        int           `json:"lookup_tokens,omitempty"`
	AcceptedTokens      int           `json:"accepted_tokens,omitempty"`
	RejectedTokens      int           `json:"rejected_tokens,omitempty"`
	EmittedTokens       int           `json:"emitted_tokens,omitempty"`
	AcceptanceRate      float64       `json:"acceptance_rate,omitempty"`
	TargetCalls         int           `json:"target_calls,omitempty"`
	DraftCalls          int           `json:"draft_calls,omitempty"`
	Duration            time.Duration `json:"duration,omitempty"`
	TargetDuration      time.Duration `json:"target_duration,omitempty"`
	DraftDuration       time.Duration `json:"draft_duration,omitempty"`
	VisibleTokensPerSec float64       `json:"visible_tokens_per_sec,omitempty"`
	TargetTokensPerSec  float64       `json:"target_tokens_per_sec,omitempty"`
	DraftTokensPerSec   float64       `json:"draft_tokens_per_sec,omitempty"`
}

// QualityReport contains small deterministic checks over generated text.
type QualityReport struct {
	Checks []QualityCheck `json:"checks,omitempty"`
}

// QualityCheck is one pass/fail bench check.
type QualityCheck struct {
	Name   string  `json:"name"`
	Pass   bool    `json:"pass"`
	Score  float64 `json:"score"`
	Detail string  `json:"detail,omitempty"`
}

// Run executes the local bench/eval suite against the supplied runner.
//
//	report, err := bench.Run(ctx, runner, cfg)
func Run(ctx context.Context, runner Runner, cfg Config) (*Report, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg = normalizeConfig(cfg)
	if runner.Generate == nil {
		return nil, core.NewError("mlx: bench runner requires Generate")
	}
	report := &Report{
		Version:   ReportVersion,
		Model:     cfg.Model,
		ModelPath: cfg.ModelPath,
		Config:    cfg,
	}
	if runner.Info != nil {
		report.ModelInfo = runner.Info(ctx)
	}

	var samples []GenerationSample
	for range cfg.Runs {
		sample, err := runGeneration(ctx, runner, cfg.Prompt, cfg.GenerateOptions(nil))
		if err != nil {
			return nil, err
		}
		samples = append(samples, sample)
	}
	report.Generation = summarizeGenerations(samples)
	report.Quality.Checks = append(report.Quality.Checks, qualityChecks(samples)...)

	if cfg.IncludePromptCache && runner.BenchPromptCache != nil {
		report.PromptCache = runner.BenchPromptCache(ctx, cfg, report.Generation)
	}
	if cfg.IncludeStateKVBlockWarm && runner.BenchStateKVBlockWarm != nil {
		report.StateKVBlockWarm = runner.BenchStateKVBlockWarm(ctx, cfg, report.Generation)
		report.MemvidKVBlockWarm = report.StateKVBlockWarm
	} else if cfg.IncludeStateKVBlockWarm && runner.BenchMemvidKVBlockWarm != nil {
		report.StateKVBlockWarm = runner.BenchMemvidKVBlockWarm(ctx, cfg, report.Generation)
		report.MemvidKVBlockWarm = report.StateKVBlockWarm
	}
	if cfg.IncludeKVRestore && runner.BenchKVRestore != nil {
		report.KVRestore = runner.BenchKVRestore(ctx, cfg)
	}
	if cfg.IncludeStateBundleRoundTrip && runner.BenchStateBundle != nil {
		report.StateBundle = runner.BenchStateBundle(ctx, cfg, report.ModelInfo)
	}
	if cfg.IncludeProbeOverhead && runner.BenchProbeOverhead != nil {
		report.Probes = runner.BenchProbeOverhead(ctx, cfg, report.Generation.TotalDuration)
	}
	if cfg.IncludeSpeculativeDecode && runner.BenchSpeculativeDecode != nil {
		report.SpeculativeDecode = runner.BenchSpeculativeDecode(ctx, cfg)
	}
	if cfg.IncludePromptLookupDecode && runner.BenchPromptLookupDecode != nil {
		report.PromptLookupDecode = runner.BenchPromptLookupDecode(ctx, cfg)
	}
	return report, nil
}

func normalizeConfig(cfg Config) Config {
	def := DefaultConfig()
	if configZero(cfg) {
		return def
	}
	if cfg.Prompt == "" {
		cfg.Prompt = def.Prompt
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = def.MaxTokens
	}
	if cfg.Runs <= 0 {
		cfg.Runs = def.Runs
	}
	if cfg.CachePrompt == "" {
		cfg.CachePrompt = cfg.Prompt
	}
	if cfg.IncludeMemvidKVBlockWarm {
		cfg.IncludeStateKVBlockWarm = true
	}
	if cfg.MemvidKVBlockSize != 0 && cfg.StateKVBlockSize == 0 {
		cfg.StateKVBlockSize = cfg.MemvidKVBlockSize
	}
	if cfg.MemvidKVPrefixTokens != 0 && cfg.StateKVPrefixTokens == 0 {
		cfg.StateKVPrefixTokens = cfg.MemvidKVPrefixTokens
	}
	if cfg.MemvidKVBlockStorePath != "" && cfg.StateKVBlockStorePath == "" {
		cfg.StateKVBlockStorePath = cfg.MemvidKVBlockStorePath
	}
	cfg.StopTokens = append([]int32(nil), cfg.StopTokens...)
	cfg.PromptLookupTokens = append([]int32(nil), cfg.PromptLookupTokens...)
	cfg.QualityPrompts = append([]string(nil), cfg.QualityPrompts...)
	return cfg
}

func configZero(cfg Config) bool {
	return cfg.Model == "" &&
		cfg.ModelPath == "" &&
		cfg.Prompt == "" &&
		cfg.CachePrompt == "" &&
		cfg.MaxTokens == 0 &&
		cfg.Runs == 0 &&
		cfg.Temperature == 0 &&
		cfg.TopK == 0 &&
		cfg.TopP == 0 &&
		cfg.MinP == 0 &&
		len(cfg.StopTokens) == 0 &&
		cfg.RepeatPenalty == 0 &&
		!cfg.IncludePromptCache &&
		!cfg.IncludeKVRestore &&
		!cfg.IncludeStateBundleRoundTrip &&
		!cfg.IncludeProbeOverhead &&
		!cfg.IncludeStateKVBlockWarm &&
		!cfg.IncludeMemvidKVBlockWarm &&
		!cfg.IncludeSpeculativeDecode &&
		!cfg.IncludePromptLookupDecode &&
		cfg.StateKVBlockSize == 0 &&
		cfg.StateKVPrefixTokens == 0 &&
		cfg.StateKVBlockStorePath == "" &&
		cfg.MemvidKVBlockSize == 0 &&
		cfg.MemvidKVPrefixTokens == 0 &&
		cfg.MemvidKVBlockStorePath == "" &&
		cfg.SpeculativeDraftModelPath == "" &&
		cfg.SpeculativeDraftTokens == 0 &&
		len(cfg.PromptLookupTokens) == 0 &&
		len(cfg.QualityPrompts) == 0
}

func runGeneration(ctx context.Context, runner Runner, prompt string, opts GenerateOptions) (GenerationSample, error) {
	start := time.Now()
	generation, err := runner.Generate(ctx, prompt, opts)
	elapsed := NonZeroDuration(time.Since(start))
	if err != nil {
		return GenerationSample{}, err
	}
	return GenerationSample{
		Prompt:  prompt,
		Text:    generation.Text,
		Tokens:  append([]int32(nil), generation.Tokens...),
		Metrics: generation.Metrics,
		Elapsed: elapsed,
	}, nil
}

func summarizeGenerations(samples []GenerationSample) GenerationSummary {
	summary := GenerationSummary{
		Runs:    len(samples),
		Samples: append([]GenerationSample(nil), samples...),
	}
	var prefillRateTotal, decodeRateTotal float64
	firstTokenSamples := 0
	for _, sample := range samples {
		metrics := sample.Metrics
		summary.PromptTokens += metrics.PromptTokens
		summary.GeneratedTokens += metrics.GeneratedTokens
		if metrics.FirstTokenDuration > 0 {
			firstTokenSamples++
			summary.FirstTokenDuration += metrics.FirstTokenDuration
		}
		summary.PrefillDuration += metrics.PrefillDuration
		summary.DecodeDuration += metrics.DecodeDuration
		if metrics.TotalDuration > 0 {
			summary.TotalDuration += metrics.TotalDuration
		} else {
			summary.TotalDuration += sample.Elapsed
		}
		prefillRateTotal += metrics.PrefillTokensPerSec
		decodeRateTotal += metrics.DecodeTokensPerSec
		if metrics.PeakMemoryBytes > summary.PeakMemoryBytes {
			summary.PeakMemoryBytes = metrics.PeakMemoryBytes
		}
		if metrics.ActiveMemoryBytes > summary.ActiveMemoryBytes {
			summary.ActiveMemoryBytes = metrics.ActiveMemoryBytes
		}
	}
	if len(samples) > 0 {
		summary.PrefillTokensPerSec = prefillRateTotal / float64(len(samples))
		summary.DecodeTokensPerSec = decodeRateTotal / float64(len(samples))
	}
	if firstTokenSamples > 0 {
		summary.FirstTokenDuration /= time.Duration(firstTokenSamples)
	}
	return summary
}

func qualityChecks(samples []GenerationSample) []QualityCheck {
	var checks []QualityCheck
	nonEmpty := false
	generatedTokens := 0
	for _, sample := range samples {
		if sample.Text != "" {
			nonEmpty = true
		}
		generatedTokens += sample.Metrics.GeneratedTokens
	}
	checks = append(checks, QualityCheck{
		Name:  "non_empty_output",
		Pass:  nonEmpty,
		Score: boolScore(nonEmpty),
	})
	checks = append(checks, QualityCheck{
		Name:   "generated_tokens",
		Pass:   generatedTokens > 0,
		Score:  boolScore(generatedTokens > 0),
		Detail: core.Sprintf("%d", generatedTokens),
	})
	return checks
}

// PopulateStateKVBlockWarmBench fills in the cross-cutting derived
// fields (Speedup, BreakEvenQuestions, ...) on a StateKVBlockWarmReport
// once the driver-side capture/restore measurements are populated.
//
//	report := runner.BenchStateKVBlockWarm(ctx, cfg, baseline)
//	bench.PopulateStateKVBlockWarmBench(&report, baseline)
func PopulateStateKVBlockWarmBench(report *StateKVBlockWarmReport, baseline GenerationSummary) {
	if report == nil || !report.Attempted {
		return
	}
	report.BaselinePrefillDuration = baseline.PrefillDuration
	report.MemoryPeakBytes = maxUint64(baseline.PeakMemoryBytes, maxUint64(report.Metrics.PeakMemoryBytes, report.Metrics.ActiveMemoryBytes))
	if baseline.PrefillDuration > 0 && report.RestoreDuration > 0 {
		report.RestoreSpeedup = float64(baseline.PrefillDuration) / float64(report.RestoreDuration)
	}
	saved := baseline.PrefillDuration - report.RestoreDuration
	if saved <= 0 || report.BuildDuration <= 0 {
		return
	}
	report.PrefillSavedPerQuestion = saved
	questions := ceilDuration(report.BuildDuration, saved)
	report.BuildAmortizationQuestions = questions
	report.BreakEvenQuestions = questions
}

// PopulateMemvidKVBlockWarmBench fills derived values for the old memvid-named
// State block warm report.
//
// Deprecated: use PopulateStateKVBlockWarmBench.
func PopulateMemvidKVBlockWarmBench(report *MemvidKVBlockWarmReport, baseline GenerationSummary) {
	PopulateStateKVBlockWarmBench(report, baseline)
}

func ceilDuration(value, divisor time.Duration) int {
	if value <= 0 || divisor <= 0 {
		return 0
	}
	return int((value + divisor - 1) / divisor)
}

func maxUint64(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}

func boolScore(pass bool) float64 {
	if pass {
		return 1
	}
	return 0
}

// NonZeroDuration returns d if positive, else 1 nanosecond. Exported for
// drivers that want consistent non-zero durations in their bench reports.
func NonZeroDuration(d time.Duration) time.Duration {
	if d <= 0 {
		return time.Nanosecond
	}
	return d
}
