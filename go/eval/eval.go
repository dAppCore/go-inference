// SPDX-Licence-Identifier: EUPL-1.2

// Package eval provides dataset-native perplexity + small quality probes
// for any inference driver (go-mlx, go-rocm, go-cuda, etc.).
//
// It is decoupled from driver concrete types: Sample, Batch, and
// BatchConfig are opaque (any), Dataset is an interface, and the
// runner adapter provides callbacks for the few fields eval needs to
// inspect (BatchTokens, SampleText). Driver wrappers convert their
// native types into an eval.Runner.
package eval

import (
	"context"
	"math"
	"strconv"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/train/lora"
)

const ReportVersion = 1

// Sample is one dataset row. Opaque to eval; the runner provides
// SampleText for quality probes that need to read the text body.
type Sample = any

// Batch is one tokenised batch. Opaque to eval; the runner evaluates
// it and may provide BatchTokens for token-count fallback.
type Batch = any

// BatchConfig is the dataset batching configuration. Opaque to eval —
// passed through to the runner's BuildBatches.
type BatchConfig = any

// Dataset is an iterator over Samples.
//
//	for {
//	    sample, ok, err := ds.Next()
//	    if !ok || err != nil { break }
//	}
type Dataset interface {
	Next() (Sample, bool, error)
}

// AdapterInfo identifies a LoRA adapter participating in the eval run.
// lora is the shared domain home for this identity (see lora.AdapterInfo)
// — eval aliases it rather than keeping its own copy so the field set and
// IsEmpty behaviour cannot drift between packages.
type AdapterInfo = lora.AdapterInfo

// Info mirrors a driver's model info — flat fields that travel through
// reports for downstream consumers.
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

// Config controls dataset-native perplexity and small quality probes.
type Config struct {
	Batch         BatchConfig    `json:"batch"`
	AdapterPath   string         `json:"adapter_path,omitempty"`
	MaxSamples    int            `json:"max_samples,omitempty"`
	QualityProbes []QualityProbe `json:"-"`
}

// Runner supplies the model operations needed for dataset evaluation.
// BuildBatches and EvaluateBatch are required; the rest are optional.
type Runner struct {
	Info          func(context.Context) Info
	LoadAdapter   func(context.Context, string) (AdapterInfo, error)
	BuildBatches  func(context.Context, Dataset, BatchConfig) ([]Batch, error)
	EvaluateBatch func(context.Context, Batch) (BatchMetrics, error)
	// BatchTokens is a fallback for BatchMetrics.Tokens when the runner
	// reports zero. Returns the loss-eligible token count.
	BatchTokens func(Batch) int
	// SampleText extracts the human-readable text body from a Sample for
	// quality probes that need to inspect it.
	SampleText func(Sample) (text, response string)
}

// BatchMetrics is the loss result for one tokenized batch.
type BatchMetrics struct {
	Samples int     `json:"samples,omitempty"`
	Tokens  int     `json:"tokens,omitempty"`
	Loss    float64 `json:"loss,omitempty"`
}

// Metrics aggregates loss and perplexity over a dataset stream.
type Metrics struct {
	Samples    int     `json:"samples,omitempty"`
	Batches    int     `json:"batches,omitempty"`
	Tokens     int     `json:"tokens,omitempty"`
	Loss       float64 `json:"loss,omitempty"`
	Perplexity float64 `json:"perplexity,omitempty"`
}

// Report is a JSON-friendly native eval result.
type Report struct {
	Version   int           `json:"version"`
	ModelInfo Info          `json:"model_info"`
	Adapter   AdapterInfo   `json:"adapter,omitempty"`
	Config    Config        `json:"config"`
	Metrics   Metrics       `json:"metrics"`
	Quality   QualityReport `json:"quality"`
	Duration  time.Duration `json:"duration,omitempty"`
}

// QualityProbe adds a custom deterministic quality check.
type QualityProbe struct {
	Name  string                            `json:"name"`
	Check func(QualityContext) QualityCheck `json:"-"`
}

// QualityContext is passed to custom eval probes.
type QualityContext struct {
	Config    Config
	Samples   []Sample
	Metrics   Metrics
	ModelInfo Info
	Adapter   AdapterInfo
	// SampleText is the runner's accessor for reading text/response from
	// an opaque Sample. Probes that introspect sample content go through
	// this rather than type-asserting.
	SampleText func(Sample) (text, response string)
}

// QualityReport contains small deterministic checks over eval data + metrics.
type QualityReport struct {
	Checks []QualityCheck `json:"checks,omitempty"`
}

// QualityCheck is one quality probe result.
type QualityCheck struct {
	Name   string  `json:"name"`
	Pass   bool    `json:"pass"`
	Score  float64 `json:"score"`
	Detail string  `json:"detail,omitempty"`
}

// RunDataset evaluates perplexity and quality probes over a dataset stream.
//
//	report, err := eval.RunDataset(ctx, runner, dataset, cfg)
func RunDataset(ctx context.Context, runner Runner, dataset Dataset, cfg Config) (*Report, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if runner.EvaluateBatch == nil {
		return nil, core.NewError("mlx: eval runner requires EvaluateBatch")
	}
	if runner.BuildBatches == nil {
		return nil, core.NewError("mlx: eval runner requires BuildBatches")
	}
	if dataset == nil {
		return nil, core.NewError("mlx: eval dataset is nil")
	}

	start := time.Now()
	samples, err := collectSamples(ctx, dataset, cfg.MaxSamples)
	if err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, core.NewError("mlx: eval dataset produced no samples")
	}

	report := &Report{
		Version: ReportVersion,
		Config:  cfg,
	}
	if runner.Info != nil {
		report.ModelInfo = runner.Info(ctx)
		report.Adapter = report.ModelInfo.Adapter
	}
	if cfg.AdapterPath != "" {
		if runner.LoadAdapter == nil {
			return nil, core.NewError("mlx: eval runner does not support LoRA adapter loading")
		}
		adapter, err := runner.LoadAdapter(ctx, cfg.AdapterPath)
		if err != nil {
			return nil, err
		}
		report.Adapter = adapter
		if runner.Info != nil {
			report.ModelInfo = runner.Info(ctx)
		}
		if report.ModelInfo.Adapter.IsEmpty() {
			report.ModelInfo.Adapter = adapter
		}
	}
	if report.Adapter.IsEmpty() {
		report.Adapter = report.ModelInfo.Adapter
	}

	batches, err := runner.BuildBatches(ctx, newSliceDataset(samples), cfg.Batch)
	if err != nil {
		return nil, err
	}
	if len(batches) == 0 {
		return nil, core.NewError("mlx: eval dataset produced no tokenized batches")
	}

	metrics, err := evaluateBatches(ctx, runner, batches, len(samples))
	if err != nil {
		return nil, err
	}
	report.Metrics = metrics
	report.Duration = nonZeroDuration(time.Since(start))
	report.Quality = runQualityProbes(QualityContext{
		Config:     cfg,
		Samples:    samples,
		Metrics:    metrics,
		ModelInfo:  report.ModelInfo,
		Adapter:    report.Adapter,
		SampleText: runner.SampleText,
	})
	return report, nil
}

func collectSamples(ctx context.Context, dataset Dataset, maxSamples int) ([]Sample, error) {
	// Pre-allocate when maxSamples is known — saves the
	// log2(maxSamples) doubling grows that append would otherwise pay.
	// For the 0-hint case (unknown dataset size), let append handle
	// growth as before.
	var samples []Sample
	if maxSamples > 0 {
		samples = make([]Sample, 0, maxSamples)
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if maxSamples > 0 && len(samples) >= maxSamples {
			break
		}
		sample, ok, err := dataset.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		samples = append(samples, sample)
	}
	return samples, nil
}

type sliceDataset struct {
	samples []Sample
	idx     int
}

func newSliceDataset(samples []Sample) Dataset {
	return &sliceDataset{samples: samples}
}

func (d *sliceDataset) Next() (Sample, bool, error) {
	if d.idx >= len(d.samples) {
		return nil, false, nil
	}
	sample := d.samples[d.idx]
	d.idx++
	return sample, true, nil
}

func evaluateBatches(ctx context.Context, runner Runner, batches []Batch, samples int) (Metrics, error) {
	metrics := Metrics{Samples: samples, Batches: len(batches)}
	var weightedLoss float64
	for _, batch := range batches {
		if err := ctx.Err(); err != nil {
			return Metrics{}, err
		}
		batchMetrics, err := runner.EvaluateBatch(ctx, batch)
		if err != nil {
			return Metrics{}, err
		}
		if batchMetrics.Tokens <= 0 && runner.BatchTokens != nil {
			batchMetrics.Tokens = runner.BatchTokens(batch)
		}
		if batchMetrics.Tokens <= 0 {
			continue
		}
		if math.IsNaN(batchMetrics.Loss) || math.IsInf(batchMetrics.Loss, 0) {
			return Metrics{}, core.NewError("mlx: eval batch loss is not finite")
		}
		metrics.Tokens += batchMetrics.Tokens
		weightedLoss += batchMetrics.Loss * float64(batchMetrics.Tokens)
	}
	if metrics.Tokens == 0 {
		return Metrics{}, core.NewError("mlx: eval produced no loss tokens")
	}
	metrics.Loss = weightedLoss / float64(metrics.Tokens)
	metrics.Perplexity = math.Exp(metrics.Loss)
	return metrics, nil
}

func runQualityProbes(ctx QualityContext) QualityReport {
	checks := defaultQualityChecks(ctx)
	for _, probe := range ctx.Config.QualityProbes {
		check := QualityCheck{Name: probe.Name}
		if probe.Check == nil {
			check.Pass = false
			check.Detail = "probe has no check function"
		} else {
			check = probe.Check(ctx)
			if check.Name == "" {
				check.Name = probe.Name
			}
		}
		checks = append(checks, check)
	}
	return QualityReport{Checks: checks}
}

func defaultQualityChecks(ctx QualityContext) []QualityCheck {
	samples := len(ctx.Samples)
	lossFinite := !math.IsNaN(ctx.Metrics.Loss) && !math.IsInf(ctx.Metrics.Loss, 0) && ctx.Metrics.Loss >= 0
	pplFinite := !math.IsNaN(ctx.Metrics.Perplexity) && !math.IsInf(ctx.Metrics.Perplexity, 0) && ctx.Metrics.Perplexity >= 1
	// strconv.Itoa / FormatFloat skip the fmt formatter pipeline that
	// core.Sprintf would walk for every Detail string. Each Sprintf
	// was 1-2 allocs; FormatX returns a single fresh string.
	return []QualityCheck{
		{Name: "samples_present", Pass: samples > 0, Score: boolScore(samples > 0), Detail: strconv.Itoa(samples)},
		{Name: "token_coverage", Pass: ctx.Metrics.Tokens > 0, Score: boolScore(ctx.Metrics.Tokens > 0), Detail: strconv.Itoa(ctx.Metrics.Tokens)},
		{Name: "loss_finite", Pass: lossFinite, Score: boolScore(lossFinite), Detail: strconv.FormatFloat(ctx.Metrics.Loss, 'f', 6, 64)},
		{Name: "perplexity_finite", Pass: pplFinite, Score: boolScore(pplFinite), Detail: strconv.FormatFloat(ctx.Metrics.Perplexity, 'f', 6, 64)},
	}
}

// ResponseCoverageProbe is a quality probe that counts samples with
// non-empty Text or Response. Driver wrappers attach this probe so
// eval doesn't need to know about the driver's sample field shape.
//
//	cfg.QualityProbes = append(cfg.QualityProbes, eval.ResponseCoverageProbe())
func ResponseCoverageProbe() QualityProbe {
	return QualityProbe{
		Name: "response_coverage",
		Check: func(ctx QualityContext) QualityCheck {
			if ctx.SampleText == nil {
				return QualityCheck{Name: "response_coverage", Pass: false, Detail: "no SampleText accessor"}
			}
			samples := len(ctx.Samples)
			responseLike := 0
			for _, sample := range ctx.Samples {
				text, response := ctx.SampleText(sample)
				if core.Trim(text) != "" || core.Trim(response) != "" {
					responseLike++
				}
			}
			// Hand-build the "%d/%d" Detail without Sprintf — 1 alloc
			// vs Sprintf's 2-3 (formatter scratch + result).
			detail := make([]byte, 0, 16)
			detail = strconv.AppendInt(detail, int64(responseLike), 10)
			detail = append(detail, '/')
			detail = strconv.AppendInt(detail, int64(samples), 10)
			return QualityCheck{
				Name:   "response_coverage",
				Pass:   responseLike == samples,
				Score:  fractionScore(responseLike, samples),
				Detail: core.AsString(detail),
			}
		},
	}
}

func boolScore(ok bool) float64 {
	if ok {
		return 1
	}
	return 0
}

func fractionScore(numerator, denominator int) float64 {
	if denominator <= 0 {
		return 0
	}
	return float64(numerator) / float64(denominator)
}

func nonZeroDuration(d time.Duration) time.Duration {
	if d <= 0 {
		return time.Nanosecond
	}
	return d
}
