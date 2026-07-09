// SPDX-Licence-Identifier: EUPL-1.2

package eval

import (
	"context"
	"math"
	"time"

	core "dappco.re/go"
)

// evalErrDataset always fails on Next — used to exercise the
// dataset-read-error propagation path in collectSamples/RunDataset
// without depending on a real dataset backend.
type evalErrDataset struct{ err error }

func (d evalErrDataset) Next() (Sample, bool, error) {
	return nil, false, d.err
}

// --- RunDataset -------------------------------------------------------------

func TestEval_RunDataset_Good(t *core.T) {
	samples := buildEvalSamples(5)
	report, err := RunDataset(context.Background(), evalRunner(samples), &evalSampleIter{samples: samples}, Config{})
	core.RequireNoError(t, err)
	core.AssertNotNil(t, report)
	core.AssertEqual(t, ReportVersion, report.Version)
	core.AssertEqual(t, 5, report.Metrics.Samples)
	core.AssertTrue(t, report.Metrics.Batches > 0)
	core.AssertEqual(t, "qwen3", report.ModelInfo.Architecture)
	core.AssertTrue(t, report.Duration > 0)
}

func TestEval_RunDataset_Bad(t *core.T) {
	samples := buildEvalSamples(2)

	// A runner without EvaluateBatch is rejected.
	_, err := RunDataset(context.Background(),
		Runner{BuildBatches: evalRunner(samples).BuildBatches},
		&evalSampleIter{samples: samples}, Config{})
	core.AssertError(t, err)

	// A nil dataset is rejected.
	_, err = RunDataset(context.Background(), evalRunner(samples), nil, Config{})
	core.AssertError(t, err)

	// A runner without BuildBatches is rejected.
	_, err = RunDataset(context.Background(),
		Runner{EvaluateBatch: evalRunner(samples).EvaluateBatch},
		&evalSampleIter{samples: samples}, Config{})
	core.AssertError(t, err, "BuildBatches")

	// AdapterPath set but the runner has no LoadAdapter callback.
	_, err = RunDataset(context.Background(), evalRunner(samples), &evalSampleIter{samples: samples}, Config{AdapterPath: "/adapters/x.lora"})
	core.AssertError(t, err, "LoRA")

	// LoadAdapter itself failing propagates verbatim.
	loadErr := core.NewError("adapter load failed")
	badAdapterRunner := evalRunner(samples)
	badAdapterRunner.LoadAdapter = func(context.Context, string) (AdapterInfo, error) {
		return AdapterInfo{}, loadErr
	}
	_, err = RunDataset(context.Background(), badAdapterRunner, &evalSampleIter{samples: samples}, Config{AdapterPath: "/adapters/x.lora"})
	core.AssertErrorIs(t, err, loadErr)

	// BuildBatches itself failing propagates verbatim.
	buildErr := core.NewError("build batches failed")
	badBuildRunner := evalRunner(samples)
	badBuildRunner.BuildBatches = func(context.Context, Dataset, BatchConfig) ([]Batch, error) {
		return nil, buildErr
	}
	_, err = RunDataset(context.Background(), badBuildRunner, &evalSampleIter{samples: samples}, Config{})
	core.AssertErrorIs(t, err, buildErr)

	// BuildBatches returning zero batches with no error is rejected as
	// "no tokenized batches".
	emptyBuildRunner := evalRunner(samples)
	emptyBuildRunner.BuildBatches = func(context.Context, Dataset, BatchConfig) ([]Batch, error) {
		return nil, nil
	}
	_, err = RunDataset(context.Background(), emptyBuildRunner, &evalSampleIter{samples: samples}, Config{})
	core.AssertError(t, err, "no tokenized batches")

	// EvaluateBatch failing inside evaluateBatches propagates through
	// RunDataset's own error-return branch.
	evalErr := core.NewError("evaluate batch failed")
	badEvalRunner := evalRunner(samples)
	badEvalRunner.EvaluateBatch = func(context.Context, Batch) (BatchMetrics, error) {
		return BatchMetrics{}, evalErr
	}
	_, err = RunDataset(context.Background(), badEvalRunner, &evalSampleIter{samples: samples}, Config{})
	core.AssertErrorIs(t, err, evalErr)
}

func TestEval_RunDataset_Ugly(t *core.T) {
	// MaxSamples caps an over-long stream.
	samples := buildEvalSamples(50)
	report, err := RunDataset(context.Background(), evalRunner(samples), &evalSampleIter{samples: samples}, Config{MaxSamples: 10})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 10, report.Metrics.Samples)

	// An empty dataset produces no samples, which is an error.
	_, err = RunDataset(context.Background(), evalRunner(nil), &evalSampleIter{samples: nil}, Config{})
	core.AssertError(t, err)

	// A nil context is normalised to context.Background() rather than
	// panicking downstream.
	small := buildEvalSamples(3)
	report, err = RunDataset(nil, evalRunner(small), &evalSampleIter{samples: small}, Config{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 3, report.Metrics.Samples)

	// dataset.Next() failing mid-stream surfaces through collectSamples.
	nextErr := core.NewError("dataset read failed")
	_, err = RunDataset(context.Background(), evalRunner(small), evalErrDataset{err: nextErr}, Config{})
	core.AssertErrorIs(t, err, nextErr)

	// A runner with no Info callback still produces a report — ModelInfo
	// stays zero-valued rather than panicking.
	noInfo := evalRunner(small)
	noInfo.Info = nil
	report, err = RunDataset(context.Background(), noInfo, &evalSampleIter{samples: small}, Config{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "", report.ModelInfo.Architecture)

	// AdapterPath succeeds: the loaded adapter backfills both
	// report.Adapter and report.ModelInfo.Adapter because the runner's
	// Info callback reports no adapter of its own.
	loaded := AdapterInfo{Name: "lora-1", Rank: 8}
	adapterRunner := evalRunner(small)
	adapterRunner.LoadAdapter = func(context.Context, string) (AdapterInfo, error) {
		return loaded, nil
	}
	report, err = RunDataset(context.Background(), adapterRunner, &evalSampleIter{samples: small}, Config{AdapterPath: "/adapters/lora-1"})
	core.RequireNoError(t, err)
	core.AssertEqual(t, loaded, report.Adapter)
	core.AssertEqual(t, loaded, report.ModelInfo.Adapter)
}

// --- AdapterInfo.IsEmpty ----------------------------------------------------

func TestEval_IsEmpty_Good(t *core.T) {
	core.AssertTrue(t, AdapterInfo{}.IsEmpty())
}

func TestEval_IsEmpty_Bad(t *core.T) {
	core.AssertFalse(t, AdapterInfo{Name: "lora-1"}.IsEmpty())
	core.AssertFalse(t, AdapterInfo{Rank: 8}.IsEmpty())
	core.AssertFalse(t, AdapterInfo{Scale: 2.0}.IsEmpty())
}

func TestEval_IsEmpty_Ugly(t *core.T) {
	// Only the slice field set — still not empty.
	core.AssertFalse(t, AdapterInfo{TargetKeys: []string{"q_proj"}}.IsEmpty())
}

// --- ResponseCoverageProbe --------------------------------------------------

func sampleText(s Sample) (string, string) {
	es := s.(evalSampleShape)
	return es.Text, es.Response
}

func TestEval_ResponseCoverageProbe_Good(t *core.T) {
	probe := ResponseCoverageProbe()
	core.AssertEqual(t, "response_coverage", probe.Name)

	check := probe.Check(QualityContext{
		Samples: []Sample{
			evalSampleShape{Text: "q1", Response: "a1"},
			evalSampleShape{Text: "q2", Response: "a2"},
		},
		SampleText: sampleText,
	})
	core.AssertTrue(t, check.Pass)
	core.AssertInDelta(t, 1.0, check.Score, 0.001)
	core.AssertEqual(t, "2/2", check.Detail)
}

func TestEval_ResponseCoverageProbe_Bad(t *core.T) {
	// No SampleText accessor — the probe cannot inspect samples.
	check := ResponseCoverageProbe().Check(QualityContext{
		Samples: []Sample{evalSampleShape{Text: "q", Response: "a"}},
	})
	core.AssertFalse(t, check.Pass)
	core.AssertContains(t, check.Detail, "no SampleText accessor")
}

func TestEval_ResponseCoverageProbe_Ugly(t *core.T) {
	// Half the samples carry content — a partial-coverage fraction.
	check := ResponseCoverageProbe().Check(QualityContext{
		Samples: []Sample{
			evalSampleShape{Text: "q", Response: "a"},
			evalSampleShape{},
		},
		SampleText: sampleText,
	})
	core.AssertFalse(t, check.Pass)
	core.AssertInDelta(t, 0.5, check.Score, 0.001)
	core.AssertEqual(t, "1/2", check.Detail)
}

// --- collectSamples ----------------------------------------------------------

func TestCollectSamples_Good(t *core.T) {
	source := buildEvalSamples(5)
	samples, err := collectSamples(context.Background(), &evalSampleIter{samples: source}, 0)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 5, len(samples))

	capped, err := collectSamples(context.Background(), &evalSampleIter{samples: source}, 3)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 3, len(capped))
}

func TestCollectSamples_Bad(t *core.T) {
	nextErr := core.NewError("dataset read failed")
	_, err := collectSamples(context.Background(), evalErrDataset{err: nextErr}, 0)
	core.AssertErrorIs(t, err, nextErr)
}

func TestCollectSamples_Ugly(t *core.T) {
	// A cancelled context short-circuits before dataset.Next() is ever
	// called.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	source := buildEvalSamples(2)
	_, err := collectSamples(ctx, &evalSampleIter{samples: source}, 0)
	core.AssertErrorIs(t, err, context.Canceled)
}

// --- evaluateBatches -----------------------------------------------------------

func TestEvaluateBatches_Good(t *core.T) {
	runner := Runner{
		EvaluateBatch: func(_ context.Context, batch Batch) (BatchMetrics, error) {
			eb := batch.(evalBatchShape)
			return BatchMetrics{Tokens: eb.Tokens, Loss: eb.Loss}, nil
		},
	}
	batches := []Batch{evalBatchShape{Tokens: 4, Loss: 1.0}, evalBatchShape{Tokens: 4, Loss: 2.0}}
	metrics, err := evaluateBatches(context.Background(), runner, batches, 2)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 8, metrics.Tokens)
	core.AssertInDelta(t, 1.5, metrics.Loss, 0.0001)
	core.AssertInDelta(t, math.Exp(1.5), metrics.Perplexity, 0.0001)

	// BatchTokens supplies the token count when EvaluateBatch reports
	// zero — the fallback-recovery branch.
	fallbackRunner := Runner{
		EvaluateBatch: func(context.Context, Batch) (BatchMetrics, error) {
			return BatchMetrics{Tokens: 0, Loss: 0.5}, nil
		},
		BatchTokens: func(Batch) int { return 6 },
	}
	metrics, err = evaluateBatches(context.Background(), fallbackRunner, []Batch{evalBatchShape{}}, 1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 6, metrics.Tokens)
}

func TestEvaluateBatches_Bad(t *core.T) {
	evalErr := core.NewError("evaluate batch failed")
	erroringRunner := Runner{
		EvaluateBatch: func(context.Context, Batch) (BatchMetrics, error) {
			return BatchMetrics{}, evalErr
		},
	}
	_, err := evaluateBatches(context.Background(), erroringRunner, []Batch{evalBatchShape{}}, 1)
	core.AssertErrorIs(t, err, evalErr)

	nanRunner := Runner{
		EvaluateBatch: func(context.Context, Batch) (BatchMetrics, error) {
			return BatchMetrics{Tokens: 4, Loss: math.NaN()}, nil
		},
	}
	_, err = evaluateBatches(context.Background(), nanRunner, []Batch{evalBatchShape{}}, 1)
	core.AssertError(t, err, "not finite")

	infRunner := Runner{
		EvaluateBatch: func(context.Context, Batch) (BatchMetrics, error) {
			return BatchMetrics{Tokens: 4, Loss: math.Inf(1)}, nil
		},
	}
	_, err = evaluateBatches(context.Background(), infRunner, []Batch{evalBatchShape{}}, 1)
	core.AssertError(t, err, "not finite")
}

func TestEvaluateBatches_Ugly(t *core.T) {
	// Every batch reports zero tokens with no BatchTokens fallback — all
	// are skipped (the continue branch) and the reducer rejects the run
	// as loss-token-free.
	zeroRunner := Runner{
		EvaluateBatch: func(context.Context, Batch) (BatchMetrics, error) {
			return BatchMetrics{Tokens: 0, Loss: 1.0}, nil
		},
	}
	_, err := evaluateBatches(context.Background(), zeroRunner, []Batch{evalBatchShape{}}, 1)
	core.AssertError(t, err, "no loss tokens")

	// A cancelled context short-circuits mid-loop before any batch is
	// evaluated.
	calls := 0
	countingRunner := Runner{
		EvaluateBatch: func(context.Context, Batch) (BatchMetrics, error) {
			calls++
			return BatchMetrics{Tokens: 4, Loss: 1.0}, nil
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = evaluateBatches(ctx, countingRunner, []Batch{evalBatchShape{}, evalBatchShape{}}, 2)
	core.AssertErrorIs(t, err, context.Canceled)
	core.AssertEqual(t, 0, calls)
}

// --- runQualityProbes ----------------------------------------------------------

func TestRunQualityProbes_Good(t *core.T) {
	source := buildEvalSamples(2)
	samples := make([]Sample, len(source))
	for i, s := range source {
		samples[i] = s
	}
	qc := QualityContext{
		Samples: samples,
		Metrics: Metrics{Samples: 2, Tokens: 16, Loss: 1.0, Perplexity: 2.7},
		Config: Config{
			QualityProbes: []QualityProbe{
				ResponseCoverageProbe(), // Check sets its own Name.
				{
					Name: "always_pass",
					Check: func(QualityContext) QualityCheck {
						return QualityCheck{Pass: true, Score: 1} // empty Name falls back to probe.Name.
					},
				},
			},
		},
		SampleText: sampleText,
	}

	report := runQualityProbes(qc)
	core.AssertEqual(t, 6, len(report.Checks)) // 4 defaults + 2 custom.

	var sawCoverage, sawAlwaysPass bool
	for _, check := range report.Checks {
		switch check.Name {
		case "response_coverage":
			sawCoverage = true
		case "always_pass":
			sawAlwaysPass = true
			core.AssertTrue(t, check.Pass)
		}
	}
	core.AssertTrue(t, sawCoverage)
	core.AssertTrue(t, sawAlwaysPass)
}

func TestRunQualityProbes_Bad(t *core.T) {
	qc := QualityContext{
		Config: Config{
			QualityProbes: []QualityProbe{{Name: "broken"}}, // Check left nil.
		},
	}

	report := runQualityProbes(qc)
	core.AssertEqual(t, 5, len(report.Checks)) // 4 defaults + 1 broken probe.

	last := report.Checks[len(report.Checks)-1]
	core.AssertEqual(t, "broken", last.Name)
	core.AssertFalse(t, last.Pass)
	core.AssertEqual(t, "probe has no check function", last.Detail)
}

// --- boolScore / fractionScore / nonZeroDuration --------------------------------

func TestBoolScore_Good(t *core.T) {
	core.AssertEqual(t, 1.0, boolScore(true))
	core.AssertEqual(t, 0.0, boolScore(false))
}

func TestFractionScore_Good(t *core.T) {
	core.AssertInDelta(t, 0.5, fractionScore(1, 2), 0.0001)
	core.AssertInDelta(t, 0, fractionScore(0, 5), 0.0001)
}

func TestFractionScore_Bad(t *core.T) {
	core.AssertEqual(t, 0.0, fractionScore(3, 0))
	core.AssertEqual(t, 0.0, fractionScore(3, -1))
}

func TestNonZeroDuration_Good(t *core.T) {
	core.AssertEqual(t, 45*time.Millisecond, nonZeroDuration(45*time.Millisecond))
}

func TestNonZeroDuration_Bad(t *core.T) {
	core.AssertEqual(t, time.Nanosecond, nonZeroDuration(0))
	core.AssertEqual(t, time.Nanosecond, nonZeroDuration(-5*time.Second))
}
