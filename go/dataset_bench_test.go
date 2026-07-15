// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for dataset / batch / report shapes — JSON marshal for
// EvalReport + BenchReport (the wire format trainers + UIs reach for)
// plus the DatasetStream Next-loop floor (per-sample iteration cost).
// Per AX-11 — these shapes carry per-sample/per-result data so any
// allocation-per-call cost compounds across a full training run.
//
// Run:    go test -bench='BenchmarkDataset' -benchmem -run='^$' .

package inference

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	datasetBenchSinkString string
	datasetBenchSinkSample DatasetSample
	datasetBenchSinkBatch  Batch
	datasetBenchSinkOK     bool
	datasetBenchSinkErr    error
	datasetBenchSinkCount  int
)

// benchDatasetStream is a deterministic in-memory stream — same shape as
// the test-suite stub but exposed at file scope so the per-Next floor
// can be measured without t.Helper bookkeeping.
type benchDatasetStream struct {
	samples []DatasetSample
	index   int
}

func (s *benchDatasetStream) Next() (DatasetSample, bool, error) {
	if s.index >= len(s.samples) {
		return DatasetSample{}, false, nil
	}
	sample := s.samples[s.index]
	s.index++
	return sample, true, nil
}

func (s *benchDatasetStream) Reset() error {
	s.index = 0
	return nil
}

func buildBenchDatasetSamples(n int) []DatasetSample {
	samples := make([]DatasetSample, n)
	for i := range samples {
		samples[i] = DatasetSample{
			Prompt:   core.Sprintf("prompt-%d", i),
			Response: core.Sprintf("response-%d", i),
			Messages: []Message{
				{Role: "user", Content: core.Sprintf("turn-%d", i)},
				{Role: "assistant", Content: core.Sprintf("reply-%d", i)},
			},
			Labels: map[string]string{"source": "bench", "split": "train"},
		}
	}
	return samples
}

// --- DatasetStream.Next — per-sample iteration floor ---

func BenchmarkDataset_StreamNext_Hit(b *testing.B) {
	stream := &benchDatasetStream{samples: buildBenchDatasetSamples(1)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stream.index = 0
		datasetBenchSinkSample, datasetBenchSinkOK, datasetBenchSinkErr = stream.Next()
	}
}

func BenchmarkDataset_StreamNext_Exhausted(b *testing.B) {
	stream := &benchDatasetStream{samples: nil}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		datasetBenchSinkSample, datasetBenchSinkOK, datasetBenchSinkErr = stream.Next()
	}
}

func BenchmarkDataset_StreamLoop_100Samples(b *testing.B) {
	samples := buildBenchDatasetSamples(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stream := &benchDatasetStream{samples: samples}
		count := 0
		for {
			_, ok, err := stream.Next()
			if !ok || err != nil {
				break
			}
			count++
		}
		datasetBenchSinkCount = count
	}
}

// --- Batch struct copies (per-batch carry cost) ---

func BenchmarkDataset_BatchAssemble_Small(b *testing.B) {
	samples := buildBenchDatasetSamples(8)
	tokenIDs := [][]int32{{1, 2, 3, 4}, {5, 6, 7, 8}}
	attention := [][]float32{{1, 1, 1, 1}, {1, 1, 1, 0}}
	lossMask := LossMask{Values: [][]float32{{0, 0, 1, 1}, {0, 1, 1, 0}}}
	labels := map[string]string{"split": "train"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		datasetBenchSinkBatch = Batch{
			TokenIDs:      tokenIDs,
			AttentionMask: attention,
			LossMask:      lossMask,
			Samples:       samples,
			Labels:        labels,
		}
	}
}

// --- JSON serialisation of the portable report types ---

func BenchmarkDataset_EvalReport_Marshal(b *testing.B) {
	report := EvalReport{
		Model: ModelIdentity{Architecture: "qwen3", QuantBits: 4},
		Metrics: EvalMetrics{
			Samples:    2048,
			Tokens:     262144,
			Loss:       1.234,
			Perplexity: 3.4321,
		},
		Probes: []QualityProbeResult{
			{Name: "integrity", Passed: true, Score: 0.91},
			{Name: "calibration", Passed: true, Score: 0.82},
			{Name: "stability", Passed: false, Score: 0.43},
		},
		Labels: map[string]string{"run": "nightly-2026-05-21"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		datasetBenchSinkString = core.JSONMarshalString(report)
	}
}

func BenchmarkDataset_BenchReport_Marshal(b *testing.B) {
	report := BenchReport{
		Model:                 ModelIdentity{Architecture: "gemma4", QuantBits: 4},
		Adapter:               AdapterIdentity{Path: "/adapters/v3", Rank: 16, Alpha: 32},
		PromptTokens:          2048,
		GeneratedTokens:       512,
		PrefillTokensPerSec:   1240.5,
		DecodeTokensPerSec:    45.2,
		PeakMemoryBytes:       12 << 30,
		PromptCacheHitRate:    0.81,
		KVRestoreMilliseconds: 12.4,
		Labels:                map[string]string{"workload": "long_context"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		datasetBenchSinkString = core.JSONMarshalString(report)
	}
}

func BenchmarkDataset_MemoryPlan_Marshal(b *testing.B) {
	plan := MemoryPlan{
		MachineClass:      "m3-ultra-96gb",
		DeviceMemoryBytes: 96 << 30,
		ContextLength:     131072,
		BatchSize:         4,
		CacheMode:         "paged-q8",
		Quantization:      "q4_k_m",
		KVCacheBytes:      18 << 30,
		TrainingFeasible:  true,
		Notes:             []string{"reserve 4GB for OS", "leave 8GB headroom"},
		Labels:            map[string]string{"profile": "long_context"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		datasetBenchSinkString = core.JSONMarshalString(plan)
	}
}

func BenchmarkDataset_ModelFitReport_Marshal(b *testing.B) {
	report := ModelFitReport{
		Model:          ModelIdentity{Architecture: "qwen3", QuantBits: 4, ContextLength: 32768},
		Fits:           true,
		ArchitectureOK: true,
		QuantizationOK: true,
		MemoryPlan: MemoryPlan{
			MachineClass:     "m3-ultra-96gb",
			ContextLength:    32768,
			CacheMode:        "paged-q4",
			TrainingFeasible: false,
		},
		Notes: []string{"context fits", "training not feasible at this quant"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		datasetBenchSinkString = core.JSONMarshalString(report)
	}
}
