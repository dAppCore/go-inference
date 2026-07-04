// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"
	"sort"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	defaultSimpleSelfDistillationMaxTokens       = 65536
	defaultSimpleSelfDistillationTemperature     = 1.5
	defaultSimpleSelfDistillationTopK            = 20
	defaultSimpleSelfDistillationTopP            = 0.8
	defaultSimpleSelfDistillationRepetition      = 1.0
	defaultSimpleSelfDistillationFilterShortest  = 10
	defaultSimpleSelfDistillationEvalMaxTokens   = 32768
	defaultSimpleSelfDistillationEvalTemperature = 0.6
	defaultSimpleSelfDistillationEvalTopP        = 0.95

	simpleSelfDistillationDecodeTemperatureLabel = "ssd_decode_temperature"
	simpleSelfDistillationEvalTemperatureLabel   = "ssd_eval_temperature"
)

// SimpleSelfDistillationConfig configures native self-distillation.
type SimpleSelfDistillationConfig struct {
	SampleMaxTokens   int                      `json:"sample_max_tokens,omitempty"`
	SampleTemperature float32                  `json:"sample_temperature,omitempty"`
	SampleTopK        int                      `json:"sample_top_k,omitempty"`
	SampleTopP        float32                  `json:"sample_top_p,omitempty"`
	SampleMinP        float32                  `json:"sample_min_p,omitempty"`
	RepetitionPenalty float32                  `json:"repetition_penalty,omitempty"`
	FilterShortestPct float32                  `json:"filter_shortest_percent,omitempty"`
	DecodeTemperature float32                  `json:"decode_temperature,omitempty"`
	SFT               inference.TrainingConfig `json:"sft,omitempty"`
}

// SimpleSelfDistillationRunner supplies the native generation step.
type SimpleSelfDistillationRunner struct {
	Generate func(context.Context, string, inference.GenerateConfig) (string, error)
}

// NativeSimpleSelfDistillationAdamWConfig configures ROCm-local SSD generation
// followed by an SFT loss plus AdamW update pass.
type NativeSimpleSelfDistillationAdamWConfig struct {
	SSD       SimpleSelfDistillationConfig
	State     *NativeAdamWState
	Gradients [][]float32
	TrackPath string
}

// SimpleSelfDistillationSample records one raw sampled response.
type SimpleSelfDistillationSample struct {
	Prompt   string            `json:"prompt"`
	Response string            `json:"response"`
	Labels   map[string]string `json:"labels,omitempty"`
}

// SimpleSelfDistillationResult records a native SSD run.
type SimpleSelfDistillationResult struct {
	Samples           []SimpleSelfDistillationSample `json:"samples"`
	SFT               *inference.TrainingResult      `json:"-"`
	SampleTemperature float32                        `json:"sample_temperature"`
	DecodeTemperature float32                        `json:"decode_temperature"`
	SampleMaxTokens   int                            `json:"sample_max_tokens"`
	SampleTopK        int                            `json:"sample_top_k,omitempty"`
	SampleTopP        float32                        `json:"sample_top_p,omitempty"`
	SampleMinP        float32                        `json:"sample_min_p,omitempty"`
	RepetitionPenalty float32                        `json:"repetition_penalty,omitempty"`
	FilterShortestPct float32                        `json:"filter_shortest_percent,omitempty"`
}

// RunSimpleSelfDistillation samples raw outputs from a frozen model and stops
// at the generated trace. Training is a separate SFT step over a curated trace.
func RunSimpleSelfDistillation(ctx context.Context, runner SimpleSelfDistillationRunner, dataset inference.DatasetStream, cfg SimpleSelfDistillationConfig) (*SimpleSelfDistillationResult, error) {
	result, _, _, err := runSimpleSelfDistillationTrace(ctx, runner, dataset, cfg, false)
	return result, err
}

func runSimpleSelfDistillationTrace(ctx context.Context, runner SimpleSelfDistillationRunner, dataset inference.DatasetStream, cfg SimpleSelfDistillationConfig, preserveUnsetSampleMaxTokens bool) (*SimpleSelfDistillationResult, []inference.DatasetSample, SimpleSelfDistillationConfig, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if dataset == nil {
		return nil, nil, cfg, core.NewError("rocm: SSD dataset is nil")
	}
	if runner.Generate == nil {
		return nil, nil, cfg, core.NewError("rocm: SSD generate function is nil")
	}
	cfg = normalizeSimpleSelfDistillationConfig(cfg, preserveUnsetSampleMaxTokens)
	if err := validateSimpleSelfDistillationConfig(cfg, preserveUnsetSampleMaxTokens); err != nil {
		return nil, nil, cfg, err
	}

	generated, samples, err := buildSimpleSelfDistillationDataset(ctx, runner, dataset, cfg)
	if err != nil {
		return nil, nil, cfg, err
	}
	result := &SimpleSelfDistillationResult{
		Samples:           samples,
		SampleTemperature: cfg.SampleTemperature,
		DecodeTemperature: cfg.DecodeTemperature,
		SampleMaxTokens:   cfg.SampleMaxTokens,
		SampleTopK:        cfg.SampleTopK,
		SampleTopP:        cfg.SampleTopP,
		SampleMinP:        cfg.SampleMinP,
		RepetitionPenalty: cfg.RepetitionPenalty,
		FilterShortestPct: cfg.FilterShortestPct,
	}
	if len(samples) == 0 {
		return result, generated, cfg, core.NewError("rocm: SSD dataset produced no prompts")
	}
	return result, generated, cfg, nil
}

// RunModelSimpleSelfDistillation wires a TextModel into the SSD trace runner.
func RunModelSimpleSelfDistillation(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, cfg SimpleSelfDistillationConfig) (*SimpleSelfDistillationResult, error) {
	if model == nil {
		return nil, core.NewError("rocm: SSD model is nil")
	}
	result, _, _, err := runSimpleSelfDistillationTrace(ctx, SimpleSelfDistillationRunner{
		Generate: func(ctx context.Context, prompt string, cfg inference.GenerateConfig) (string, error) {
			return generateTextForSimpleSelfDistillation(ctx, model, prompt, cfg)
		},
	}, dataset, cfg, simpleSelfDistillationPreserveUnsetMaxTokensForModel(model))
	return result, err
}

// RunModelNativeSimpleSelfDistillationAdamWUpdatePass wires TextModel
// generation into the ROCm SFT loss plus AdamW update pass without making the
// model claim SFTTrainer support.
func RunModelNativeSimpleSelfDistillationAdamWUpdatePass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, cfg NativeSimpleSelfDistillationAdamWConfig) (*SimpleSelfDistillationResult, bool, error) {
	if model == nil {
		return nil, false, core.NewError("rocm: SSD model is nil")
	}
	if cfg.State == nil {
		return nil, false, core.NewError("rocm: SSD AdamW state is nil")
	}
	var nativeLoss bool
	result, generated, normalized, err := runSimpleSelfDistillationTrace(ctx, SimpleSelfDistillationRunner{
		Generate: func(ctx context.Context, prompt string, cfg inference.GenerateConfig) (string, error) {
			return generateTextForSimpleSelfDistillation(ctx, model, prompt, cfg)
		},
	}, dataset, cfg.SSD, simpleSelfDistillationPreserveUnsetMaxTokensForModel(model))
	if err != nil {
		return result, nativeLoss, err
	}
	trainDataset := newSimpleSelfDistillationDataset(filterSimpleSelfDistillationShortest(generated, normalized.FilterShortestPct))
	if cfg.TrackPath != "" {
		sft, _, ok, err := RunNativeSFTAdamWUpdateTrackPass(ctx, model, trainDataset, cfg.State, cfg.Gradients, cfg.TrackPath, normalized.SFT)
		nativeLoss = ok
		if result != nil {
			result.SFT = sft
		}
		return result, nativeLoss, err
	}
	sft, ok, err := RunNativeSFTAdamWUpdatePass(ctx, model, trainDataset, cfg.State, cfg.Gradients, normalized.SFT)
	nativeLoss = ok
	if result != nil {
		result.SFT = sft
	}
	return result, nativeLoss, err
}

// SampleGenerateConfig returns the frozen-model sampling configuration used to
// create the raw SSD trace rows.
func (result *SimpleSelfDistillationResult) SampleGenerateConfig() inference.GenerateConfig {
	if result == nil {
		return inference.GenerateConfig{}
	}
	return inference.GenerateConfig{
		MaxTokens:     result.SampleMaxTokens,
		Temperature:   result.SampleTemperature,
		TopK:          result.SampleTopK,
		TopP:          result.SampleTopP,
		MinP:          result.SampleMinP,
		RepeatPenalty: result.RepetitionPenalty,
	}
}

// DecodeGenerateConfig returns the post-SSD decode configuration with the
// separately tuned decode temperature. The token budget remains caller-owned.
func (result *SimpleSelfDistillationResult) DecodeGenerateConfig(maxTokens int) inference.GenerateConfig {
	if result == nil {
		return inference.GenerateConfig{MaxTokens: maxTokens}
	}
	return inference.GenerateConfig{
		MaxTokens:   maxTokens,
		Temperature: result.DecodeTemperature,
	}
}

// SimpleSelfDistillationEvalGenerateConfig reconstructs the post-SSD eval
// generation config carried through TrainingConfig labels. The bool reports
// whether SSD eval/decode temperature evidence was present.
func SimpleSelfDistillationEvalGenerateConfig(labels map[string]string, maxTokens int) (inference.GenerateConfig, bool, error) {
	cfg := inference.GenerateConfig{MaxTokens: maxTokens}
	value := labels[simpleSelfDistillationEvalTemperatureLabel]
	if value == "" {
		value = labels[simpleSelfDistillationDecodeTemperatureLabel]
	}
	if value == "" {
		return cfg, false, nil
	}
	temperature, err := strconv.ParseFloat(value, 32)
	if err != nil || temperature < 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return inference.GenerateConfig{}, false, core.NewError("rocm: SSD eval temperature label must be non-negative and finite")
	}
	cfg.Temperature = float32(temperature)
	return cfg, true, nil
}

func buildSimpleSelfDistillationDataset(ctx context.Context, runner SimpleSelfDistillationRunner, dataset inference.DatasetStream, cfg SimpleSelfDistillationConfig) ([]inference.DatasetSample, []SimpleSelfDistillationSample, error) {
	generated := make([]inference.DatasetSample, 0, 16)
	samples := make([]SimpleSelfDistillationSample, 0, 16)
	generateCfg := simpleSelfDistillationGenerateConfig(cfg)
	for index := 0; ; index++ {
		if err := ctx.Err(); err != nil {
			return generated, samples, err
		}
		sample, ok, err := dataset.Next()
		if err != nil {
			return generated, samples, err
		}
		if !ok {
			break
		}
		prompt := simpleSelfDistillationPrompt(sample)
		if prompt == "" {
			continue
		}
		response, err := runner.Generate(ctx, prompt, generateCfg)
		if err != nil {
			return generated, samples, err
		}
		labels := rocmCloneLabels(sample.Labels)
		if labels == nil {
			labels = make(map[string]string, 4)
		}
		labels["ssd"] = "simple_self_distillation"
		labels["ssd_source_index"] = strconv.Itoa(index)
		labels["ssd_sample_temperature"] = formatSimpleSelfDistillationFloat32(cfg.SampleTemperature)
		row := inference.DatasetSample{Prompt: prompt, Response: response, Labels: labels}
		generated = append(generated, row)
		samples = append(samples, SimpleSelfDistillationSample{
			Prompt:   prompt,
			Response: response,
			Labels:   rocmCloneLabels(labels),
		})
	}
	return generated, samples, nil
}

func simpleSelfDistillationPrompt(sample inference.DatasetSample) string {
	if sample.Prompt != "" {
		return sample.Prompt
	}
	return sample.Text
}

func simpleSelfDistillationGenerateConfig(cfg SimpleSelfDistillationConfig) inference.GenerateConfig {
	return inference.GenerateConfig{
		MaxTokens:     cfg.SampleMaxTokens,
		Temperature:   cfg.SampleTemperature,
		TopK:          cfg.SampleTopK,
		TopP:          cfg.SampleTopP,
		MinP:          cfg.SampleMinP,
		RepeatPenalty: cfg.RepetitionPenalty,
	}
}

func normalizeSimpleSelfDistillationConfig(cfg SimpleSelfDistillationConfig, preserveUnsetSampleMaxTokens bool) SimpleSelfDistillationConfig {
	if cfg.SampleMaxTokens <= 0 && !preserveUnsetSampleMaxTokens {
		cfg.SampleMaxTokens = defaultSimpleSelfDistillationMaxTokens
	}
	if cfg.SampleTemperature == 0 {
		cfg.SampleTemperature = defaultSimpleSelfDistillationTemperature
	}
	if cfg.SampleTopK == 0 {
		cfg.SampleTopK = defaultSimpleSelfDistillationTopK
	}
	if cfg.SampleTopP == 0 {
		cfg.SampleTopP = defaultSimpleSelfDistillationTopP
	}
	if cfg.DecodeTemperature != 0 && cfg.SFT.Labels == nil {
		cfg.SFT.Labels = map[string]string{}
	}
	if cfg.DecodeTemperature != 0 {
		formatted := formatSimpleSelfDistillationFloat32(cfg.DecodeTemperature)
		cfg.SFT.Labels[simpleSelfDistillationDecodeTemperatureLabel] = formatted
		cfg.SFT.Labels[simpleSelfDistillationEvalTemperatureLabel] = formatted
	}
	return cfg
}

func validateSimpleSelfDistillationConfig(cfg SimpleSelfDistillationConfig, preserveUnsetSampleMaxTokens bool) error {
	if cfg.SampleTemperature <= 0 || math.IsNaN(float64(cfg.SampleTemperature)) || math.IsInf(float64(cfg.SampleTemperature), 0) {
		return core.NewError("rocm: SSD sample temperature must be positive and finite")
	}
	if cfg.SampleTemperature == 1 {
		return core.NewError("rocm: SSD sample temperature must be non-unit")
	}
	if cfg.DecodeTemperature < 0 || math.IsNaN(float64(cfg.DecodeTemperature)) || math.IsInf(float64(cfg.DecodeTemperature), 0) {
		return core.NewError("rocm: SSD decode temperature must be finite")
	}
	if cfg.SampleMaxTokens < 0 {
		return core.NewError("rocm: SSD sample max tokens must be non-negative")
	}
	if cfg.SampleMaxTokens == 0 && !preserveUnsetSampleMaxTokens {
		return core.NewError("rocm: SSD sample max tokens must be positive")
	}
	if cfg.RepetitionPenalty < 0 || math.IsNaN(float64(cfg.RepetitionPenalty)) || math.IsInf(float64(cfg.RepetitionPenalty), 0) {
		return core.NewError("rocm: SSD repetition penalty must be finite and non-negative")
	}
	if cfg.FilterShortestPct < 0 || cfg.FilterShortestPct > 100 || math.IsNaN(float64(cfg.FilterShortestPct)) || math.IsInf(float64(cfg.FilterShortestPct), 0) {
		return core.NewError("rocm: SSD filter shortest percent must be finite between 0 and 100")
	}
	return nil
}

func simpleSelfDistillationPreserveUnsetMaxTokensForModel(model inference.TextModel) bool {
	return isROCmGemma4Architecture(rocmDecodeModelInfo(model).Architecture)
}

func generateTextForSimpleSelfDistillation(ctx context.Context, model inference.TextModel, prompt string, cfg inference.GenerateConfig) (string, error) {
	builder := core.NewBuilder()
	builder.Grow(cfg.MaxTokens * 4)
	for token := range model.Generate(ctx, prompt, simpleSelfDistillationOptions(cfg)...) {
		builder.WriteString(token.Text)
	}
	if r := model.Err(); !r.OK {
		return "", r.Value.(error)
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return "", err
		}
	}
	return builder.String(), nil
}

func simpleSelfDistillationOptions(cfg inference.GenerateConfig) []inference.GenerateOption {
	opts := []inference.GenerateOption{
		inference.WithMaxTokens(cfg.MaxTokens),
		inference.WithTemperature(cfg.Temperature),
		inference.WithTopK(cfg.TopK),
		inference.WithTopP(cfg.TopP),
	}
	if cfg.MinP != 0 {
		opts = append(opts, inference.WithMinP(cfg.MinP))
	}
	if cfg.RepeatPenalty != 0 {
		opts = append(opts, inference.WithRepeatPenalty(cfg.RepeatPenalty))
	}
	return opts
}

func filterSimpleSelfDistillationShortest(rows []inference.DatasetSample, percent float32) []inference.DatasetSample {
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
	for index := range order {
		order[index] = index
	}
	sort.SliceStable(order, func(i, j int) bool {
		return len(rows[order[i]].Response) < len(rows[order[j]].Response)
	})
	dropped := make(map[int]struct{}, drop)
	for _, index := range order[:drop] {
		dropped[index] = struct{}{}
	}
	filtered := make([]inference.DatasetSample, 0, len(rows)-drop)
	for index, row := range rows {
		if _, ok := dropped[index]; ok {
			continue
		}
		filtered = append(filtered, row)
	}
	return filtered
}

func formatSimpleSelfDistillationFloat32(value float32) string {
	return strconv.FormatFloat(float64(value), 'f', -1, 32)
}

func rocmCloneLabels(labels map[string]string) map[string]string {
	if labels == nil {
		return nil
	}
	clone := make(map[string]string, len(labels))
	for key, value := range labels {
		clone[key] = value
	}
	return clone
}

type simpleSelfDistillationDataset struct {
	samples []inference.DatasetSample
	index   int
}

func newSimpleSelfDistillationDataset(samples []inference.DatasetSample) *simpleSelfDistillationDataset {
	return &simpleSelfDistillationDataset{samples: append([]inference.DatasetSample(nil), samples...)}
}

func (dataset *simpleSelfDistillationDataset) Next() (inference.DatasetSample, bool, error) {
	if dataset == nil || dataset.index >= len(dataset.samples) {
		return inference.DatasetSample{}, false, nil
	}
	sample := dataset.samples[dataset.index]
	dataset.index++
	sample.Labels = rocmCloneLabels(sample.Labels)
	return sample, true, nil
}
