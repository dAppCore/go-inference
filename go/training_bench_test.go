// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the training contract shapes — DefaultLoRAConfig
// constructor + TrainingConfig / TrainingResult / DistillConfig / GRPOConfig
// JSON marshal. Per AX-11 — TrainingResult is the canonical wire format
// every trainer emits on every checkpoint; the per-step Metrics record is
// the tightest serialise loop. DefaultLoRAConfig fires once per training
// run but is exercised heavily in tests + tooling.
//
// Run:    go test -bench='BenchmarkTraining' -benchmem -run='^$' .

package inference

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE. Distinct names from the other bench files.
var (
	trainingBenchSinkConfig LoRAConfig
	trainingBenchSinkString string
)

// --- DefaultLoRAConfig (constructor allocation cost) ---

func BenchmarkTraining_DefaultLoRAConfig(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainingBenchSinkConfig = DefaultLoRAConfig()
	}
}

// --- TrainingConfig marshal (per-run checkpoint envelope) ---

func BenchmarkTraining_TrainingConfig_Marshal(b *testing.B) {
	cfg := TrainingConfig{
		Epochs:               3,
		BatchSize:            4,
		GradientAccumulation: 8,
		LearningRate:         1e-4,
		LoRA: LoRAConfig{
			Rank:       16,
			Alpha:      32,
			TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
			BFloat16:   true,
		},
		Labels: map[string]string{"run": "nightly", "dataset": "lthn-corpus"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainingBenchSinkString = core.JSONMarshalString(cfg)
	}
}

// --- TrainingMetrics marshal (per-step record — tightest loop) ---

func BenchmarkTraining_TrainingMetrics_Marshal(b *testing.B) {
	metrics := TrainingMetrics{
		Epoch:        2,
		Step:         512,
		Samples:      16384,
		Tokens:       2097152,
		Loss:         1.234,
		LearningRate: 5e-5,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainingBenchSinkString = core.JSONMarshalString(metrics)
	}
}

// --- TrainingResult marshal (per-checkpoint envelope) ---

func BenchmarkTraining_TrainingResult_Marshal(b *testing.B) {
	result := TrainingResult{
		Model: ModelIdentity{
			Path:         "/models/qwen3-4b",
			Architecture: "qwen3",
			QuantBits:    4,
		},
		Adapter: AdapterIdentity{
			Path:   "/adapters/run-2026-05-21/epoch-2",
			Format: "safetensors",
			Rank:   16,
			Alpha:  32,
		},
		Metrics: TrainingMetrics{
			Epoch:        2,
			Step:         512,
			Samples:      16384,
			Tokens:       2097152,
			Loss:         1.234,
			LearningRate: 5e-5,
		},
		Checkpoints: []StateRef{
			{Kind: "checkpoint", URI: "file:///tmp/step-256", SizeBytes: 1 << 20},
			{Kind: "checkpoint", URI: "file:///tmp/step-512", SizeBytes: 1 << 20},
		},
		Labels: map[string]string{"run": "nightly"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainingBenchSinkString = core.JSONMarshalString(result)
	}
}

// --- DistillConfig marshal (teacher/student wire envelope) ---

func BenchmarkTraining_DistillConfig_Marshal(b *testing.B) {
	cfg := DistillConfig{
		TrainingConfig: TrainingConfig{
			Epochs:               2,
			BatchSize:            8,
			GradientAccumulation: 4,
			LearningRate:         2e-4,
			LoRA: LoRAConfig{
				Rank:       8,
				Alpha:      16,
				TargetKeys: []string{"q_proj", "v_proj"},
			},
		},
		Temperature: 2.0,
		Alpha:       0.7,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainingBenchSinkString = core.JSONMarshalString(cfg)
	}
}

// --- GRPOConfig marshal (reasoning policy optimisation envelope) ---

func BenchmarkTraining_GRPOConfig_Marshal(b *testing.B) {
	cfg := GRPOConfig{
		TrainingConfig: TrainingConfig{
			Epochs:       1,
			BatchSize:    2,
			LearningRate: 5e-6,
			LoRA: LoRAConfig{
				Rank:       16,
				Alpha:      32,
				TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
				BFloat16:   true,
			},
		},
		GroupSize: 8,
		KLWeight:  0.04,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainingBenchSinkString = core.JSONMarshalString(cfg)
	}
}

// --- LoRAConfig marshal (per-adapter sidecar) ---

func BenchmarkTraining_LoRAConfig_Marshal(b *testing.B) {
	cfg := LoRAConfig{
		Rank:       64,
		Alpha:      128,
		TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
		BFloat16:   true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trainingBenchSinkString = core.JSONMarshalString(cfg)
	}
}
