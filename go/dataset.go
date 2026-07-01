// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "context"

// DatasetSample is a backend-neutral training or evaluation item.
type DatasetSample struct {
	Text      string    `json:"text,omitempty"`
	Prompt    string    `json:"prompt,omitempty"`
	Response  string    `json:"response,omitempty"`
	Reasoning string    `json:"reasoning,omitempty"`
	Messages  []Message `json:"messages,omitempty"`
	// Format is the source-corpus row shape this sample was normalised from
	// (e.g. "text", "openai_messages", "sharegpt", "prompt_response",
	// "alpaca", "reasoning") — stamped by dataset.LoadJSONL (go/dataset).
	// Empty for samples built directly rather than parsed from a corpus.
	Format string            `json:"format,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`
}

// DatasetStream is the smallest pull-based dataset contract shared by
// training, evaluation, distillation, and reasoning rollouts.
type DatasetStream interface {
	Next() (DatasetSample, bool, error)
}

// DatasetResetter marks streams that can replay from the start.
type DatasetResetter interface {
	Reset() error
}

// LossMask marks which token positions contribute to training loss.
type LossMask struct {
	Values [][]float32 `json:"values,omitempty"`
}

// Batch is a tokenizer-ready batch with optional response-loss masking.
type Batch struct {
	TokenIDs      [][]int32         `json:"token_ids,omitempty"`
	AttentionMask [][]float32       `json:"attention_mask,omitempty"`
	LossMask      LossMask          `json:"loss_mask,omitempty"`
	Samples       []DatasetSample   `json:"samples,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// EvalConfig controls model evaluation over a dataset stream.
type EvalConfig struct {
	MaxSamples int            `json:"max_samples,omitempty"`
	BatchSize  int            `json:"batch_size,omitempty"`
	MaxSeqLen  int            `json:"max_seq_len,omitempty"`
	Probes     []QualityProbe `json:"probes,omitempty"`
}

// EvalMetrics records aggregate loss and perplexity counters.
type EvalMetrics struct {
	Samples    int     `json:"samples,omitempty"`
	Tokens     int     `json:"tokens,omitempty"`
	Loss       float64 `json:"loss,omitempty"`
	Perplexity float64 `json:"perplexity,omitempty"`
}

// QualityProbe is a small named prompt used for qualitative checks.
type QualityProbe struct {
	Name   string            `json:"name,omitempty"`
	Prompt string            `json:"prompt,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`
}

// QualityProbeResult records one qualitative probe result.
type QualityProbeResult struct {
	Name   string  `json:"name,omitempty"`
	Passed bool    `json:"passed,omitempty"`
	Score  float64 `json:"score,omitempty"`
	Text   string  `json:"text,omitempty"`
}

// EvalReport is the portable output of dataset evaluation.
type EvalReport struct {
	Model   ModelIdentity        `json:"model,omitempty"`
	Adapter AdapterIdentity      `json:"adapter,omitempty"`
	Metrics EvalMetrics          `json:"metrics,omitempty"`
	Probes  []QualityProbeResult `json:"probes,omitempty"`
	Labels  map[string]string    `json:"labels,omitempty"`
}

// BenchConfig controls reusable local inference benchmarks.
type BenchConfig struct {
	Prompts      []string `json:"prompts,omitempty"`
	MaxTokens    int      `json:"max_tokens,omitempty"`
	WarmupRuns   int      `json:"warmup_runs,omitempty"`
	MeasuredRuns int      `json:"measured_runs,omitempty"`
}

// BenchReport records fast local benchmark counters.
type BenchReport struct {
	Model                 ModelIdentity     `json:"model,omitempty"`
	Adapter               AdapterIdentity   `json:"adapter,omitempty"`
	PromptTokens          int               `json:"prompt_tokens,omitempty"`
	GeneratedTokens       int               `json:"generated_tokens,omitempty"`
	PrefillTokensPerSec   float64           `json:"prefill_tokens_per_sec,omitempty"`
	DecodeTokensPerSec    float64           `json:"decode_tokens_per_sec,omitempty"`
	PeakMemoryBytes       uint64            `json:"peak_memory_bytes,omitempty"`
	PromptCacheHitRate    float64           `json:"prompt_cache_hit_rate,omitempty"`
	KVRestoreMilliseconds float64           `json:"kv_restore_milliseconds,omitempty"`
	Labels                map[string]string `json:"labels,omitempty"`
}

// MemoryPlan records device-informed runtime settings.
type MemoryPlan struct {
	MachineClass      string            `json:"machine_class,omitempty"`
	DeviceMemoryBytes uint64            `json:"device_memory_bytes,omitempty"`
	ContextLength     int               `json:"context_length,omitempty"`
	BatchSize         int               `json:"batch_size,omitempty"`
	CacheMode         string            `json:"cache_mode,omitempty"`
	Quantization      string            `json:"quantization,omitempty"`
	KVCacheBytes      uint64            `json:"kv_cache_bytes,omitempty"`
	TrainingFeasible  bool              `json:"training_feasible,omitempty"`
	Notes             []string          `json:"notes,omitempty"`
	Labels            map[string]string `json:"labels,omitempty"`
}

// ModelFitReport records whether a model is expected to fit a machine.
type ModelFitReport struct {
	Model          ModelIdentity `json:"model,omitempty"`
	Fits           bool          `json:"fits,omitempty"`
	MemoryPlan     MemoryPlan    `json:"memory_plan,omitempty"`
	ArchitectureOK bool          `json:"architecture_ok,omitempty"`
	QuantizationOK bool          `json:"quantization_ok,omitempty"`
	Notes          []string      `json:"notes,omitempty"`
}

// TrainingConfig is the shared SFT LoRA training configuration envelope.
type TrainingConfig struct {
	Epochs               int               `json:"epochs,omitempty"`
	BatchSize            int               `json:"batch_size,omitempty"`
	GradientAccumulation int               `json:"gradient_accumulation,omitempty"`
	LearningRate         float64           `json:"learning_rate,omitempty"`
	LoRA                 LoRAConfig        `json:"lora,omitempty"`
	Labels               map[string]string `json:"labels,omitempty"`
}

// TrainingMetrics records live or final training counters.
type TrainingMetrics struct {
	Epoch        int     `json:"epoch,omitempty"`
	Step         int     `json:"step,omitempty"`
	Samples      int     `json:"samples,omitempty"`
	Tokens       int     `json:"tokens,omitempty"`
	Loss         float64 `json:"loss,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
}

// TrainingResult is the portable output of a training run.
type TrainingResult struct {
	Model       ModelIdentity     `json:"model,omitempty"`
	Adapter     AdapterIdentity   `json:"adapter,omitempty"`
	Metrics     TrainingMetrics   `json:"metrics,omitempty"`
	Checkpoints []StateRef        `json:"checkpoints,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// DistillConfig controls teacher/student distillation.
type DistillConfig struct {
	TrainingConfig
	Temperature float64 `json:"temperature,omitempty"`
	Alpha       float64 `json:"alpha,omitempty"`
}

// GRPOConfig controls grouped reasoning policy optimisation.
type GRPOConfig struct {
	TrainingConfig
	GroupSize int     `json:"group_size,omitempty"`
	KLWeight  float64 `json:"kl_weight,omitempty"`
}

// Evaluator marks backends or adapters that can evaluate dataset streams.
type Evaluator interface {
	Evaluate(ctx context.Context, dataset DatasetStream, cfg EvalConfig) (*EvalReport, error)
}
