package classify

import (
	"context"
	"time"

	"dappco.re/go"
	"dappco.re/go/inference"
	golog "dappco.re/go/log"
)

// CalibrationSample is a single text entry for model comparison.
type CalibrationSample struct {
	Text       string
	TrueDomain string // optional ground truth label (empty if unknown)
}

// CalibrationResult holds per-sample classification from two models.
type CalibrationResult struct {
	Text       string `json:"text"`
	TrueDomain string `json:"true_domain,omitempty"`
	DomainA    string `json:"domain_a"`
	DomainB    string `json:"domain_b"`
	Agree      bool   `json:"agree"`
}

// CalibrationStats holds aggregate metrics from CalibrateDomains.
type CalibrationStats struct {
	Total          int                 `json:"total"`
	Agreed         int                 `json:"agreed"`
	AgreementRate  float64             `json:"agreement_rate"`
	ByDomainA      map[string]int      `json:"by_domain_a"`
	ByDomainB      map[string]int      `json:"by_domain_b"`
	ConfusionPairs map[string]int      `json:"confusion_pairs"` // "technical->creative": count
	AccuracyA      float64             `json:"accuracy_a"`      // vs ground truth (0 if none)
	AccuracyB      float64             `json:"accuracy_b"`      // vs ground truth (0 if none)
	CorrectA       int                 `json:"correct_a"`
	CorrectB       int                 `json:"correct_b"`
	WithTruth      int                 `json:"with_truth"` // samples that had ground truth
	DurationA      time.Duration       `json:"duration_a"`
	DurationB      time.Duration       `json:"duration_b"`
	Results        []CalibrationResult `json:"results"`
}

type classificationBatch struct {
	Domains  []string
	Duration time.Duration
}

// CalibrateDomains classifies all samples with both models and computes agreement.
// Model A is typically the smaller/faster model (1B), model B the larger reference (27B).
// Samples with non-empty TrueDomain also contribute to accuracy metrics.
func CalibrateDomains(ctx context.Context, modelA, modelB inference.TextModel,
	samples []CalibrationSample, opts ...ClassifyOption) core.Result {

	if len(samples) == 0 {
		return failResult(golog.E("CalibrateDomains", "empty sample set", nil))
	}

	cfg := defaultClassifyConfig()
	for _, o := range opts {
		o(&cfg)
	}

	stats := &CalibrationStats{
		ByDomainA:      make(map[string]int),
		ByDomainB:      make(map[string]int),
		ConfusionPairs: make(map[string]int),
	}

	// Build classification prompts from sample texts.
	prompts := make([]string, len(samples))
	for i, s := range samples {
		prompts[i] = core.Sprintf(cfg.promptTemplate, s.Text)
	}

	// Classify with model A.
	classifiedA := classifyAll(ctx, modelA, prompts, cfg.batchSize)
	if !classifiedA.OK {
		return failResult(golog.E("CalibrateDomains", "classify with model A", core.NewError(classifiedA.Error())))
	}
	batchA := classifiedA.Value.(classificationBatch)
	domainsA := batchA.Domains
	stats.DurationA = batchA.Duration

	// Classify with model B.
	classifiedB := classifyAll(ctx, modelB, prompts, cfg.batchSize)
	if !classifiedB.OK {
		return failResult(golog.E("CalibrateDomains", "classify with model B", core.NewError(classifiedB.Error())))
	}
	batchB := classifiedB.Value.(classificationBatch)
	domainsB := batchB.Domains
	stats.DurationB = batchB.Duration

	// Compare results.
	stats.Total = len(samples)
	stats.Results = make([]CalibrationResult, len(samples))

	for i, s := range samples {
		a, b := domainsA[i], domainsB[i]
		agree := a == b
		if agree {
			stats.Agreed++
		} else {
			key := core.Sprintf("%s->%s", a, b)
			stats.ConfusionPairs[key]++
		}
		stats.ByDomainA[a]++
		stats.ByDomainB[b]++

		if s.TrueDomain != "" {
			stats.WithTruth++
			if a == s.TrueDomain {
				stats.CorrectA++
			}
			if b == s.TrueDomain {
				stats.CorrectB++
			}
		}

		stats.Results[i] = CalibrationResult{
			Text:       s.Text,
			TrueDomain: s.TrueDomain,
			DomainA:    a,
			DomainB:    b,
			Agree:      agree,
		}
	}

	if stats.Total > 0 {
		stats.AgreementRate = float64(stats.Agreed) / float64(stats.Total)
	}
	if stats.WithTruth > 0 {
		stats.AccuracyA = float64(stats.CorrectA) / float64(stats.WithTruth)
		stats.AccuracyB = float64(stats.CorrectB) / float64(stats.WithTruth)
	}

	return core.Ok(stats)
}

// classifyAll runs batch classification over all prompts, returning domain labels.
func classifyAll(ctx context.Context, model inference.TextModel, prompts []string, batchSize int) core.Result {
	start := time.Now()
	domains := make([]string, len(prompts))

	for i := 0; i < len(prompts); i += batchSize {
		end := min(i+batchSize, len(prompts))
		batch := prompts[i:end]

		results, err := model.Classify(ctx, batch, inference.WithMaxTokens(1))
		if err != nil {
			return failResult(golog.E("classifyAll", core.Sprintf("classify batch [%d:%d]", i, end), err))
		}

		for j, r := range results {
			domains[i+j] = mapTokenToDomain(r.Token.Text)
		}
	}

	return core.Ok(classificationBatch{Domains: domains, Duration: time.Since(start)})
}
