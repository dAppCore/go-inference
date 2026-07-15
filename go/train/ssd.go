// SPDX-Licence-Identifier: EUPL-1.2

// ssd.go: the native SSD sampling pipeline (sample raw outputs from a FROZEN
// model, capture + score each self-output at birth, STOP at the scored trace)
// — hooks-based and model-free. Ported from go-mlx/go/train/ssd.go and made
// engine-neutral: the go-mlx original threaded generation through
// *spine.GenerateConfig / spine.ModelInfo and its own mlx/dataset package;
// this port rides inference.GenerateConfig, inference.DatasetStream, and the
// scorer-neutral score cascade (score_cascade.go).
//
// SSD NEVER trains (this is the whole point of the no-correct-answer lane): it
// samples the frozen base, scores each self-output at birth (score_cascade),
// captures the trace (capture.go), and STOPS. The captures + sample scores ARE
// the deliverable — a stronger lab model picks steps from them and re-performs
// the sequence into an SFT artefact, which a SEPARATE `sft` run trains on. So
// this pipeline needs generation + scoring + capture only; there is no Trainer
// in the path.

package train

import (
	"context"
	"math"
	"sort"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/train/dataset"
)

// defaultSSDTemperature is the frozen-base sampling temperature default;
// defaultSSDMaxTokens / defaultSSDTopK / defaultSSDTopP are shared with the SSD
// code-benchmark harness (ssd_eval.go) and reused here.
const defaultSSDTemperature = 0.7

// Named SSD parity recipes (mirroring the ml-ssd released model cards).
const (
	SSDRecipe4BInstruct     = "SimpleSD-4B-instruct"
	SSDRecipe4BThinking     = "SimpleSD-4B-thinking"
	SSDRecipe30BA3BInstruct = "SimpleSD-30b-a3b-instruct"
)

// SSDConfig configures native self-distillation sampling. Unlike go-mlx's
// SSDConfig it embeds no SFTConfig (SSD never trains): CheckpointDir is the
// output directory for the scored trace and the sidecar defaults derive from
// it.
type SSDConfig struct {
	SampleMaxTokens       int     `json:"sample_max_tokens,omitempty"`
	SampleTemperature     float32 `json:"sample_temperature,omitempty"`
	SampleTopK            int     `json:"sample_top_k,omitempty"`
	SampleTopP            float32 `json:"sample_top_p,omitempty"`
	SampleMinP            float32 `json:"sample_min_p,omitempty"`
	RepetitionPenalty     float32 `json:"repetition_penalty,omitempty"`
	FilterShortestPercent float32 `json:"filter_shortest_percent,omitempty"`
	// CheckpointDir is the trace output directory; the capture + score
	// sidecars default beside it.
	CheckpointDir string `json:"checkpoint_dir,omitempty"`
	// ScoreSamples arms the scorer over the SAMPLING phase (#50): every
	// self-generated (prompt, response) is scored at the moment it's born —
	// the no-correct-answer lane's quality read, before any filter. Requires
	// the runner to supply a Score hook.
	ScoreSamples     bool   `json:"score_samples,omitempty"`
	ScoreSidecarPath string `json:"score_sidecar_path,omitempty"` // default <CheckpointDir>/ssd-samples-score.jsonl
	// KernelPrefix (#97): a standing text prefix — the LEK-2 kernel —
	// prefilled ONCE as KV state (runner.WarmPrefix) and reused for every
	// sample via the engine's exact token-prefix cache. Generation happens
	// UNDER the kernel, but the captured rows keep the BARE prompt: the trace
	// records how it speaks under the kernel, never the kernel's words. The
	// text is used verbatim — no normalisation.
	KernelPrefix string `json:"kernel_prefix,omitempty"`
	// Capture-first (#97): every raw return is appended to the capture sidecar
	// at the moment it exists — BEFORE any filter, independent of scoring.
	// Defaults beside the checkpoints; DisableCapture opts out.
	CaptureSidecarPath string `json:"capture_sidecar_path,omitempty"` // default <CheckpointDir>/ssd-captures.jsonl
	DisableCapture     bool   `json:"disable_capture,omitempty"`
}

// SSDRecipe describes a native SSD parity recipe.
type SSDRecipe struct {
	Name          string                 `json:"name"`
	Model         string                 `json:"model"`
	Dataset       string                 `json:"dataset,omitempty"`
	DatasetConfig string                 `json:"dataset_config,omitempty"`
	DatasetSplit  string                 `json:"dataset_split,omitempty"`
	Train         SSDConfig              `json:"train"`
	Eval          SSDCodeBenchmarkConfig `json:"eval"`
	Notes         []string               `json:"notes,omitempty"`
}

// SSDRunner supplies the native generation, prefix warming, and scoring hooks
// the SSD sampling loop drives. Generate is required; WarmPrefix and Score are
// optional.
type SSDRunner struct {
	// Generate samples one response for a prompt under the given config.
	Generate func(context.Context, string, inference.GenerateConfig) (string, error)
	// FormatPrompt frames a bare prompt for GENERATION — an instruct base needs
	// its own turn template around the prompt (raw completion of a
	// complete-sounding utterance samples end-of-turn immediately and the trace
	// comes back empty). The captured rows keep the BARE prompt regardless; the
	// kernel prefix rides BEFORE the framed prompt (kernel, then turn
	// structure). Optional — nil feeds the prompt raw.
	FormatPrompt func(string) string
	// WarmPrefix prefills the engine's exact token-prefix cache with the
	// kernel ONCE, so every sample's generation reuses the kernel's KV state
	// instead of recomputing it. Optional — without it the kernel lane is
	// still correct (the prefix rides every generation prompt), just not
	// cached.
	WarmPrefix func(context.Context, string) error
	// Score scores one self-generated (prompt, response) at birth. Optional —
	// nil disables sampling-phase scoring even when SSDConfig.ScoreSamples is
	// set.
	Score ScoreFunc
}

// SSDSample records one raw sampled response.
type SSDSample struct {
	Prompt   string            `json:"prompt"`
	Response string            `json:"response"`
	Meta     map[string]string `json:"meta,omitempty"`
}

// SSDResult records a native SSD run.
type SSDResult struct {
	Samples               []SSDSample `json:"samples"`
	SampleTemperature     float32     `json:"sample_temperature"`
	SampleMaxTokens       int         `json:"sample_max_tokens"`
	SampleTopK            int         `json:"sample_top_k,omitempty"`
	SampleTopP            float32     `json:"sample_top_p,omitempty"`
	SampleMinP            float32     `json:"sample_min_p,omitempty"`
	RepetitionPenalty     float32     `json:"repetition_penalty,omitempty"`
	FilterShortestPercent float32     `json:"filter_shortest_percent,omitempty"`
	// Sampling-phase cascade (#50) — populated when SSDConfig.ScoreSamples is
	// set and the runner supplies a Score hook: every self-generated sample
	// scored at birth.
	SampleScores       []ScoreRecord `json:"sample_scores,omitempty"`
	SampleScoreMean    float64       `json:"sample_score_mean,omitempty"`
	SampleScoreSidecar string        `json:"sample_score_sidecar,omitempty"`
	// Kernel + capture lanes (#97).
	KernelApplied  bool   `json:"kernel_applied,omitempty"`
	CaptureSidecar string `json:"capture_sidecar,omitempty"`
}

// RunSSD samples raw outputs from a frozen model, captures + scores each at
// birth, and STOPS at the scored trace. It intentionally has no verifier,
// teacher, or training step.
func RunSSD(ctx context.Context, runner SSDRunner, ds inference.DatasetStream, cfg SSDConfig) (*SSDResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if ds == nil {
		return nil, core.NewError("mlx: SSD dataset is nil")
	}
	if runner.Generate == nil {
		return nil, core.NewError("mlx: SSD generate function is nil")
	}
	cfg = normalizeSSDConfig(cfg)
	if err := validateSSDConfig(cfg); err != nil {
		return nil, err
	}

	// The sampling-phase cascade (#50): every self-generated sample scored at
	// the moment it's born — the no-correct-answer lane's quality read.
	var sampleCascade *scoreCascade
	if cfg.ScoreSamples && runner.Score != nil {
		sidecar := cfg.ScoreSidecarPath
		if sidecar == "" && cfg.CheckpointDir != "" {
			sidecar = core.PathJoin(cfg.CheckpointDir, "ssd-samples-score.jsonl")
		}
		sampleCascade = newScoreCascade(sidecar, 1, runner.Score)
	}

	// Capture-first (#97): on by default beside the checkpoints — the raw
	// returns ARE the candidate corpus in the no-correct-answer lane.
	if cfg.DisableCapture {
		cfg.CaptureSidecarPath = ""
	} else if cfg.CaptureSidecarPath == "" && cfg.CheckpointDir != "" {
		cfg.CaptureSidecarPath = core.PathJoin(cfg.CheckpointDir, "ssd-captures.jsonl")
	}

	// The kernel as standing KV state (#97): prefilled once, reused by every
	// sample. A failed warm is loud — the operator armed the kernel lane, and
	// silently sampling without it would forge the run.
	if cfg.KernelPrefix != "" && runner.WarmPrefix != nil {
		if err := runner.WarmPrefix(ctx, cfg.KernelPrefix); err != nil {
			return nil, err
		}
	}

	samples, err := buildSSDSamples(ctx, runner, ds, cfg, sampleCascade)
	if err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, core.NewError("mlx: SSD dataset produced no prompts")
	}
	return newSSDResult(samples, cfg, sampleCascade), nil
}

// DefaultSSDConfig returns the ml-ssd data-generation defaults.
func DefaultSSDConfig() SSDConfig {
	return SSDConfig{
		SampleMaxTokens:       65536,
		SampleTemperature:     1.5,
		SampleTopK:            20,
		SampleTopP:            0.8,
		RepetitionPenalty:     1.0,
		FilterShortestPercent: 10,
	}
}

// DefaultSSDCodeBenchmarkConfig returns the ml-ssd LiveCodeBench-v6 evaluation
// defaults (the eval side ships in ssd_eval.go; the sampling defaults live
// here).
func DefaultSSDCodeBenchmarkConfig() SSDCodeBenchmarkConfig {
	return SSDCodeBenchmarkConfig{
		Benchmark: "LiveCodeBench-v6",
		NRepeat:   20,
		Seeds:     []uint64{0, 1234, 1234, 1234},
		Generate: inference.GenerateConfig{
			MaxTokens:   32768,
			Temperature: 0.6,
			TopP:        0.95,
			TopK:        20,
			MinP:        0,
		},
	}
}

// SSDRecipes returns the released ml-ssd model recipe descriptors with native
// data-generation and evaluation defaults.
func SSDRecipes() []SSDRecipe {
	train := DefaultSSDConfig()
	eval := DefaultSSDCodeBenchmarkConfig()
	return []SSDRecipe{
		newSSDRecipe(SSDRecipe4BInstruct, "apple/SimpleSD-4B-instruct", train, eval),
		newSSDRecipe(SSDRecipe4BThinking, "apple/SimpleSD-4B-thinking", train, eval),
		newSSDRecipe(SSDRecipe30BA3BInstruct, "apple/SimpleSD-30b-a3b-instruct", train, eval),
	}
}

// LookupSSDRecipe returns a named SSD parity recipe.
func LookupSSDRecipe(name string) (SSDRecipe, bool) {
	for _, recipe := range SSDRecipes() {
		if recipe.Name == name || recipe.Model == name {
			return recipe, true
		}
	}
	return SSDRecipe{}, false
}

// SampleGenerateConfig returns the frozen-model sampling configuration used to
// create the raw SSD training rows.
func (r *SSDResult) SampleGenerateConfig() inference.GenerateConfig {
	if r == nil {
		return inference.GenerateConfig{}
	}
	return inference.GenerateConfig{
		MaxTokens:     r.SampleMaxTokens,
		Temperature:   r.SampleTemperature,
		TopK:          r.SampleTopK,
		TopP:          r.SampleTopP,
		MinP:          r.SampleMinP,
		RepeatPenalty: r.RepetitionPenalty,
	}
}

func newSSDResult(samples []SSDSample, cfg SSDConfig, sampleCascade *scoreCascade) *SSDResult {
	result := &SSDResult{
		Samples:               samples,
		SampleTemperature:     cfg.SampleTemperature,
		SampleMaxTokens:       cfg.SampleMaxTokens,
		SampleTopK:            cfg.SampleTopK,
		SampleTopP:            cfg.SampleTopP,
		SampleMinP:            cfg.SampleMinP,
		RepetitionPenalty:     cfg.RepetitionPenalty,
		FilterShortestPercent: cfg.FilterShortestPercent,
		KernelApplied:         cfg.KernelPrefix != "",
		CaptureSidecar:        cfg.CaptureSidecarPath,
	}
	if sampleCascade != nil {
		result.SampleScores = append([]ScoreRecord(nil), sampleCascade.records...)
		result.SampleScoreSidecar = sampleCascade.sidecarPath
		if len(result.SampleScores) > 0 {
			sum := 0.0
			for _, rec := range result.SampleScores {
				sum += rec.LEK
			}
			result.SampleScoreMean = sum / float64(len(result.SampleScores))
		}
	}
	return result
}

func newSSDRecipe(name, model string, train SSDConfig, eval SSDCodeBenchmarkConfig) SSDRecipe {
	return SSDRecipe{
		Name:          name,
		Model:         model,
		Dataset:       "microsoft/rStar-Coder",
		DatasetConfig: "seed_sft",
		DatasetSplit:  "train",
		Train:         train,
		Eval:          eval,
		Notes: []string{
			"Use the released model card for model-specific decode sampling when it differs from the upstream eval example.",
			"Store runtime artefacts under docs/runtime/ when reproducing this recipe locally.",
		},
	}
}

// buildSSDSamples samples every prompt in the dataset, captures + scores each
// return at birth, and applies the shortest-response filter to the result.
func buildSSDSamples(ctx context.Context, runner SSDRunner, ds inference.DatasetStream, cfg SSDConfig, sampleCascade *scoreCascade) ([]SSDSample, error) {
	samples := make([]SSDSample, 0, 16)
	genCfg := ssdGenerateConfig(cfg)
	for index := 0; ; index++ {
		if err := ctx.Err(); err != nil {
			return samples, err
		}
		sample, ok, err := ds.Next()
		if err != nil {
			return samples, err
		}
		if !ok {
			break
		}
		prompt := ssdPrompt(sample)
		if prompt == "" {
			continue
		}
		// The kernel rides the GENERATION prompt only — verbatim prefix,
		// reused from the warmed KV state. The bare prompt is what the capture
		// + scorer read; the turn framing (FormatPrompt) sits between kernel
		// and prompt so the kernel stays standing KV state ahead of the turns.
		generationPrompt := prompt
		if runner.FormatPrompt != nil {
			generationPrompt = runner.FormatPrompt(prompt)
		}
		if cfg.KernelPrefix != "" {
			generationPrompt = cfg.KernelPrefix + generationPrompt
		}
		response, err := runner.Generate(ctx, generationPrompt, genCfg)
		if err != nil {
			return samples, err
		}
		appendCaptureRows(cfg.CaptureSidecarPath, []SFTEvalResult{{Step: index, Prompt: prompt, Text: response}})
		meta := cloneSampleLabels(sample)
		meta["ssd"] = "simple_self_distillation"
		meta["ssd_source_index"] = strconv.Itoa(index)
		meta["ssd_sample_temperature"] = formatSSDFloat32(cfg.SampleTemperature)
		if cfg.KernelPrefix != "" {
			meta["ssd_kernel"] = "1"
		}
		if sampleCascade != nil {
			// Score at birth — the sample's own quality read rides its meta so
			// the filter (and any later curation) is explainable downstream.
			sampleCascade.recordPass(index, []SFTEvalResult{{Step: index, Prompt: prompt, Text: response}})
			if recs := sampleCascade.records; len(recs) > 0 {
				meta["ssd_lek"] = strconv.FormatFloat(recs[len(recs)-1].LEK, 'f', 2, 64)
			}
		}
		samples = append(samples, SSDSample{Prompt: prompt, Response: response, Meta: meta})
	}
	return filterSSDShortest(samples, cfg.FilterShortestPercent), nil
}

// filterSSDShortest drops the shortest-response percent% of samples (by
// response length), keeping at least one.
func filterSSDShortest(rows []SSDSample, percent float32) []SSDSample {
	if percent <= 0 || len(rows) <= 1 {
		return rows
	}
	drop := int(math.Ceil(float64(len(rows)) * float64(percent) / 100))
	if drop <= 0 {
		return rows
	}
	if drop >= len(rows) {
		drop = len(rows) - 1
	}
	order := make([]int, len(rows))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		return len(rows[order[i]].Response) < len(rows[order[j]].Response)
	})
	dropped := make(map[int]struct{}, drop)
	for _, index := range order[:drop] {
		dropped[index] = struct{}{}
	}
	filtered := make([]SSDSample, 0, len(rows)-drop)
	for index, row := range rows {
		if _, ok := dropped[index]; ok {
			continue
		}
		filtered = append(filtered, row)
	}
	return filtered
}

// cloneSampleLabels copies a sample's Labels (go-mlx's dataset.Sample.Meta) into
// a fresh, always-non-nil map the SSD loop can annotate.
func cloneSampleLabels(sample dataset.Sample) map[string]string {
	meta := core.MapClone(sample.Labels)
	if meta == nil {
		meta = make(map[string]string, 4)
	}
	return meta
}

func ssdPrompt(sample dataset.Sample) string {
	if sample.Prompt != "" {
		return sample.Prompt
	}
	return sample.Text
}

func ssdGenerateConfig(cfg SSDConfig) inference.GenerateConfig {
	return inference.GenerateConfig{
		MaxTokens:     cfg.SampleMaxTokens,
		Temperature:   cfg.SampleTemperature,
		TopK:          cfg.SampleTopK,
		TopP:          cfg.SampleTopP,
		MinP:          cfg.SampleMinP,
		RepeatPenalty: cfg.RepetitionPenalty,
	}
}

func normalizeSSDConfig(cfg SSDConfig) SSDConfig {
	if cfg.SampleMaxTokens <= 0 {
		cfg.SampleMaxTokens = defaultSSDMaxTokens
	}
	if cfg.SampleTemperature == 0 {
		cfg.SampleTemperature = defaultSSDTemperature
	}
	if cfg.SampleTopK == 0 {
		cfg.SampleTopK = defaultSSDTopK
	}
	if cfg.SampleTopP == 0 {
		cfg.SampleTopP = defaultSSDTopP
	}
	return cfg
}

func validateSSDConfig(cfg SSDConfig) error {
	if cfg.SampleTemperature <= 0 || math.IsNaN(float64(cfg.SampleTemperature)) || math.IsInf(float64(cfg.SampleTemperature), 0) {
		return core.NewError("mlx: SSD sample temperature must be positive and finite")
	}
	if cfg.SampleTemperature == 1 {
		return core.NewError("mlx: SSD sample temperature must be non-unit")
	}
	if cfg.SampleMaxTokens <= 0 {
		return core.NewError("mlx: SSD sample max tokens must be positive")
	}
	if cfg.RepetitionPenalty < 0 || math.IsNaN(float64(cfg.RepetitionPenalty)) || math.IsInf(float64(cfg.RepetitionPenalty), 0) {
		return core.NewError("mlx: SSD repetition penalty must be finite and non-negative")
	}
	if cfg.FilterShortestPercent < 0 || cfg.FilterShortestPercent > 100 || math.IsNaN(float64(cfg.FilterShortestPercent)) || math.IsInf(float64(cfg.FilterShortestPercent), 0) {
		return core.NewError("mlx: SSD filter shortest percent must be finite between 0 and 100")
	}
	return nil
}

func formatSSDFloat32(value float32) string {
	return strconv.FormatFloat(float64(value), 'g', -1, 32)
}
