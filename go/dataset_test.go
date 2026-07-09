// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"testing"
)

type datasetStreamStub struct {
	samples []DatasetSample
	index   int
}

func (s *datasetStreamStub) Next() (DatasetSample, bool, error) {
	if s.index >= len(s.samples) {
		return DatasetSample{}, false, nil
	}
	sample := s.samples[s.index]
	s.index++
	return sample, true, nil
}

func (s *datasetStreamStub) Reset() error {
	s.index = 0
	return nil
}

type evaluatorStub struct {
	report *EvalReport
}

func (e evaluatorStub) Evaluate(context.Context, DatasetStream, EvalConfig) (*EvalReport, error) {
	return e.report, nil
}

func TestDataset_DatasetSample_Good(t *testing.T) {
	sample := DatasetSample{
		Prompt:    "question",
		Response:  "answer",
		Reasoning: "work",
		Messages:  []Message{{Role: "user", Content: "question"}},
		Labels:    map[string]string{"source": "unit"},
	}

	checkEqual(t, "question", sample.Prompt)
	checkLen(t, sample.Messages, 1)
	checkEqual(t, "unit", sample.Labels["source"])
}

func TestDatasetBatchLossMask(t *testing.T) {
	batch := Batch{
		TokenIDs: [][]int32{{1, 2, 3}},
		LossMask: LossMask{Values: [][]float32{{
			0,
			1,
			1,
		}}},
	}

	checkEqual(t, float32(1), batch.LossMask.Values[0][1])
}

func TestDatasetStreamReset(t *testing.T) {
	stream := &datasetStreamStub{
		samples: []DatasetSample{{Text: "one"}},
	}

	sample, ok, err := stream.Next()
	checkNoError(t, err)
	checkTrue(t, ok)
	checkEqual(t, "one", sample.Text)

	sample, ok, err = stream.Next()
	checkNoError(t, err)
	checkFalse(t, ok)
	checkEqual(t, DatasetSample{}, sample)

	checkNoError(t, stream.Reset())
	sample, ok, err = stream.Next()
	checkNoError(t, err)
	checkTrue(t, ok)
	checkEqual(t, "one", sample.Text)
}

func TestDataset_EvalReport_Good(t *testing.T) {
	report := EvalReport{
		Model: ModelIdentity{Architecture: "qwen3"},
		Metrics: EvalMetrics{
			Samples:    2,
			Tokens:     64,
			Loss:       1.25,
			Perplexity: 3.49,
		},
		Probes: []QualityProbeResult{{
			Name:   "integrity",
			Passed: true,
			Score:  0.9,
		}},
	}
	evaluator := evaluatorStub{report: &report}

	got, err := evaluator.Evaluate(context.Background(), &datasetStreamStub{}, EvalConfig{MaxSamples: 2})

	checkNoError(t, err)
	checkEqual(t, "qwen3", got.Model.Architecture)
	checkEqual(t, 64, got.Metrics.Tokens)
	checkLen(t, got.Probes, 1)
}

func TestDatasetBenchAndMemoryPlan(t *testing.T) {
	report := BenchReport{
		Model:                 ModelIdentity{Architecture: "gemma4"},
		PromptTokens:          2048,
		GeneratedTokens:       128,
		PrefillTokensPerSec:   1200,
		DecodeTokensPerSec:    32,
		PeakMemoryBytes:       8 << 30,
		PromptCacheHitRate:    0.8,
		KVRestoreMilliseconds: 12.5,
	}
	plan := MemoryPlan{
		MachineClass:      "m3-ultra-96gb",
		DeviceMemoryBytes: 96 << 30,
		ContextLength:     131072,
		CacheMode:         "paged-q8",
		TrainingFeasible:  true,
	}

	checkEqual(t, "gemma4", report.Model.Architecture)
	checkEqual(t, float64(0.8), report.PromptCacheHitRate)
	checkEqual(t, "paged-q8", plan.CacheMode)
	checkTrue(t, plan.TrainingFeasible)
}

func TestDataset_TrainingResult_Ugly_CheckpointsOnly(t *testing.T) {
	result := TrainingResult{
		Checkpoints: []StateRef{{
			Kind: "checkpoint",
			URI:  "file:///tmp/step-10",
		}},
	}

	checkLen(t, result.Checkpoints, 1)
	checkEqual(t, "", result.Model.Architecture)
}
