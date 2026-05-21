// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the driver-neutral dataset-eval harness — RunDataset
// over a synthetic Runner, the sample-collector hot loop, the batch
// reducer, quality-probe runners, and the AdapterInfo emptiness check.
//
// Per AX-11 — RunDataset fires once per eval invocation, but
// collectSamples + evaluateBatches walk every sample/batch the dataset
// emits, and runQualityProbes runs every check after every eval. The
// `quick_eval` lane in lthn/LEM-Eval uses ~200 samples per probe.
//
// Run:    go test -bench='BenchmarkEval' -benchmem -run='^$' ./go/eval

package eval

import (
	"context"
	"testing"
	"time"
)

// Sinks defeat compiler DCE.
var (
	evalSinkReport     *Report
	evalSinkErr        error
	evalSinkSamples    []Sample
	evalSinkMetrics    Metrics
	evalSinkQuality    QualityReport
	evalSinkBool       bool
	evalSinkDur        time.Duration
	evalSinkBatchTok   int
	evalSinkQualScore  float64
	evalSinkBoolScore  float64
	evalSinkFracScore  float64
	evalSinkSampleText string
)

// evalSampleShape is the synthetic Sample type the benches feed through
// eval — eval treats Sample as opaque (any), so the shape only needs
// to be readable by the runner's SampleText callback.
type evalSampleShape struct {
	Text     string
	Response string
}

// evalBatchShape is the synthetic Batch type. eval treats Batch as
// opaque (any); the runner's EvaluateBatch + BatchTokens callbacks
// extract loss + token count.
type evalBatchShape struct {
	Tokens int
	Loss   float64
}

// buildEvalSamples mints n samples shaped like the LEM-Eval rows
// (text body + response). Each carries a non-empty text/response so
// response_coverage doesn't short-circuit.
func buildEvalSamples(n int) []evalSampleShape {
	samples := make([]evalSampleShape, n)
	for i := 0; i < n; i++ {
		samples[i] = evalSampleShape{
			Text:     "What is the capital of Lethean?",
			Response: "The capital is in the network.",
		}
	}
	return samples
}

// evalSampleIter wraps a slice in the Dataset interface.
type evalSampleIter struct {
	samples []evalSampleShape
	idx     int
}

func (it *evalSampleIter) Next() (Sample, bool, error) {
	if it.idx >= len(it.samples) {
		return nil, false, nil
	}
	s := it.samples[it.idx]
	it.idx++
	return s, true, nil
}

// evalRunner returns a Runner whose callbacks emit deterministic
// per-sample metrics. Used by every RunDataset bench below.
func evalRunner(samples []evalSampleShape) Runner {
	return Runner{
		Info: func(context.Context) Info {
			return Info{Architecture: "qwen3", ContextLength: 4096}
		},
		BuildBatches: func(_ context.Context, ds Dataset, _ BatchConfig) ([]Batch, error) {
			var batches []Batch
			for {
				s, ok, err := ds.Next()
				if err != nil {
					return nil, err
				}
				if !ok {
					break
				}
				_ = s
				batches = append(batches, evalBatchShape{Tokens: 8, Loss: 1.5})
			}
			return batches, nil
		},
		EvaluateBatch: func(_ context.Context, batch Batch) (BatchMetrics, error) {
			eb := batch.(evalBatchShape)
			return BatchMetrics{Samples: 1, Tokens: eb.Tokens, Loss: eb.Loss}, nil
		},
		BatchTokens: func(batch Batch) int {
			return batch.(evalBatchShape).Tokens
		},
		SampleText: func(sample Sample) (string, string) {
			s := sample.(evalSampleShape)
			return s.Text, s.Response
		},
	}
}

// --- RunDataset end-to-end at 10 / 100 question scales ---

func BenchmarkEval_RunDataset_10Samples(b *testing.B) {
	cfg := Config{}
	ctx := context.Background()
	source := buildEvalSamples(10)
	runner := evalRunner(source)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkReport, evalSinkErr = RunDataset(ctx, runner, &evalSampleIter{samples: source}, cfg)
	}
}

func BenchmarkEval_RunDataset_100Samples(b *testing.B) {
	cfg := Config{}
	ctx := context.Background()
	source := buildEvalSamples(100)
	runner := evalRunner(source)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkReport, evalSinkErr = RunDataset(ctx, runner, &evalSampleIter{samples: source}, cfg)
	}
}

// MaxSamples short-circuits collectSamples — exercises the limited
// path that quick_eval lanes use.
func BenchmarkEval_RunDataset_100Samples_MaxSamples50(b *testing.B) {
	cfg := Config{MaxSamples: 50}
	ctx := context.Background()
	source := buildEvalSamples(100)
	runner := evalRunner(source)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkReport, evalSinkErr = RunDataset(ctx, runner, &evalSampleIter{samples: source}, cfg)
	}
}

// RunDataset with a custom QualityProbe attached — measures the cost
// of running per-sample text inspection (the ResponseCoverageProbe
// path drivers wire up by default).
func BenchmarkEval_RunDataset_100Samples_WithProbe(b *testing.B) {
	cfg := Config{QualityProbes: []QualityProbe{ResponseCoverageProbe()}}
	ctx := context.Background()
	source := buildEvalSamples(100)
	runner := evalRunner(source)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkReport, evalSinkErr = RunDataset(ctx, runner, &evalSampleIter{samples: source}, cfg)
	}
}

// --- collectSamples in isolation ---

func BenchmarkEval_CollectSamples_10(b *testing.B) {
	ctx := context.Background()
	source := buildEvalSamples(10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkSamples, evalSinkErr = collectSamples(ctx, &evalSampleIter{samples: source}, 0)
	}
}

func BenchmarkEval_CollectSamples_100(b *testing.B) {
	ctx := context.Background()
	source := buildEvalSamples(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkSamples, evalSinkErr = collectSamples(ctx, &evalSampleIter{samples: source}, 0)
	}
}

func BenchmarkEval_CollectSamples_100_Cap50(b *testing.B) {
	ctx := context.Background()
	source := buildEvalSamples(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkSamples, evalSinkErr = collectSamples(ctx, &evalSampleIter{samples: source}, 50)
	}
}

// --- evaluateBatches in isolation ---

func BenchmarkEval_EvaluateBatches_10(b *testing.B) {
	source := buildEvalSamples(10)
	runner := evalRunner(source)
	batches, err := runner.BuildBatches(context.Background(), &evalSampleIter{samples: source}, nil)
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkMetrics, evalSinkErr = evaluateBatches(ctx, runner, batches, len(source))
	}
}

func BenchmarkEval_EvaluateBatches_100(b *testing.B) {
	source := buildEvalSamples(100)
	runner := evalRunner(source)
	batches, err := runner.BuildBatches(context.Background(), &evalSampleIter{samples: source}, nil)
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkMetrics, evalSinkErr = evaluateBatches(ctx, runner, batches, len(source))
	}
}

// --- defaultQualityChecks + runQualityProbes (per-eval probe surface) ---

func BenchmarkEval_DefaultQualityChecks(b *testing.B) {
	source := buildEvalSamples(10)
	samples := make([]Sample, len(source))
	for i, s := range source {
		samples[i] = s
	}
	qc := QualityContext{
		Samples: samples,
		Metrics: Metrics{Samples: 10, Tokens: 80, Loss: 1.5, Perplexity: 4.48},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = defaultQualityChecks(qc)
	}
}

func BenchmarkEval_RunQualityProbes_NoCustom(b *testing.B) {
	source := buildEvalSamples(10)
	samples := make([]Sample, len(source))
	for i, s := range source {
		samples[i] = s
	}
	qc := QualityContext{
		Samples: samples,
		Metrics: Metrics{Samples: 10, Tokens: 80, Loss: 1.5, Perplexity: 4.48},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkQuality = runQualityProbes(qc)
	}
}

// 100 samples × ResponseCoverageProbe — the body the probe walks per call.
func BenchmarkEval_ResponseCoverageProbe_100Samples(b *testing.B) {
	source := buildEvalSamples(100)
	samples := make([]Sample, len(source))
	for i, s := range source {
		samples[i] = s
	}
	probe := ResponseCoverageProbe()
	qc := QualityContext{
		Samples: samples,
		Metrics: Metrics{Samples: 100, Tokens: 800, Loss: 1.5, Perplexity: 4.48},
		SampleText: func(sample Sample) (string, string) {
			s := sample.(evalSampleShape)
			return s.Text, s.Response
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = probe.Check(qc)
	}
}

// --- AdapterInfo.IsEmpty ---

func BenchmarkEval_AdapterInfo_IsEmpty_Empty(b *testing.B) {
	info := AdapterInfo{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkBool = info.IsEmpty()
	}
}

func BenchmarkEval_AdapterInfo_IsEmpty_Populated(b *testing.B) {
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
		evalSinkBool = info.IsEmpty()
	}
}

// --- Score helpers (called per quality check) ---

func BenchmarkEval_BoolScore_True(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkBoolScore = boolScore(true)
	}
}

func BenchmarkEval_FractionScore_HalfPopulated(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkFracScore = fractionScore(50, 100)
	}
}

// --- nonZeroDuration ---

func BenchmarkEval_NonZeroDuration_Positive(b *testing.B) {
	d := 45 * time.Millisecond
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkDur = nonZeroDuration(d)
	}
}

func BenchmarkEval_NonZeroDuration_Zero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evalSinkDur = nonZeroDuration(0)
	}
}

// --- sliceDataset.Next (the iterator created by RunDataset to feed
// BuildBatches; fires once per sample) ---

func BenchmarkEval_SliceDataset_Next_100Samples(b *testing.B) {
	source := buildEvalSamples(100)
	samples := make([]Sample, len(source))
	for i, s := range source {
		samples[i] = s
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := newSliceDataset(samples)
		for {
			_, ok, err := ds.Next()
			if err != nil || !ok {
				break
			}
		}
	}
}
