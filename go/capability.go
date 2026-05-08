// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "context"

// TokenizerModel exposes native tokenisation and chat-template handling.
type TokenizerModel interface {
	Encode(text string) []int32
	Decode(ids []int32) string
	ApplyChatTemplate(messages []Message) (string, error)
}

// AdapterModel exposes LoRA adapter lifecycle operations for inference.
type AdapterModel interface {
	LoadAdapter(path string) (AdapterIdentity, error)
	UnloadAdapter() error
	ActiveAdapter() AdapterIdentity
}

// StatefulModel exposes portable model-state capture and restore.
type StatefulModel interface {
	CaptureState(ctx context.Context, prompt string, opts ...GenerateOption) (*StateBundle, error)
	RestoreState(ctx context.Context, bundle *StateBundle) error
}

// ProbeableModel accepts a typed probe sink for inference or training events.
type ProbeableModel interface {
	SetProbeSink(sink ProbeSink)
}

// BenchableModel runs local benchmark workloads.
type BenchableModel interface {
	Benchmark(ctx context.Context, cfg BenchConfig) (*BenchReport, error)
}

// ModelFitPlanner estimates whether a model fits a memory budget.
type ModelFitPlanner interface {
	PlanModelFit(ctx context.Context, model ModelIdentity, memoryBytes uint64) (*ModelFitReport, error)
}

// SFTTrainer trains a model or adapter with supervised fine tuning.
type SFTTrainer interface {
	TrainSFT(ctx context.Context, dataset DatasetStream, cfg TrainingConfig) (*TrainingResult, error)
}

// DistillTrainer trains a student model from teacher outputs.
type DistillTrainer interface {
	Distill(ctx context.Context, dataset DatasetStream, cfg DistillConfig) (*TrainingResult, error)
}

// GRPOTrainer trains grouped reasoning rollouts.
type GRPOTrainer interface {
	TrainGRPO(ctx context.Context, dataset DatasetStream, cfg GRPOConfig) (*TrainingResult, error)
}
