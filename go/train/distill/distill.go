// SPDX-Licence-Identifier: EUPL-1.2

// Package distill provides engine-agnostic knowledge-distillation
// primitives — KL / soft-cross-entropy loss over plain teacher/student
// logit tensors, teacher-logit caching, and checkpoint bookkeeping —
// shared by every inference driver (go-mlx, go-rocm, go-cpu, ...).
//
// It is the engine-agnostic half of what was previously go-mlx-only
// (go-mlx/go/distill): the loss math, the teacher-logit cache, and the
// checkpoint metadata carry no engine dependency — Logits is a plain
// [][][]float32 tensor, never an MLX array — so they belong here where
// every driver can share them.
//
// The per-step training loop that go-mlx's distill package also hosts
// (iterating epochs, driving the teacher/student forward pass, applying
// the optimiser step) is deliberately NOT ported: it is threaded through
// go-mlx's own SFTBatch / Tokenizer / ModelInfo types (see
// go-mlx/go/distill_compat.go), which load real tokenizer vocabularies
// and drive an actual Metal forward pass — genuinely engine-bound. That
// loop stays engine-side; a driver wires its own loop to call BatchLoss
// per step and NewCheckpointMetadata / SaveCheckpointMetadata at its own
// checkpoint cadence:
//
//	loss, err := distill.BatchLoss(teacherLogits, studentLogits, mask, cfg)
//	if err != nil {
//	    return err
//	}
//	// the driver applies loss.Value to its optimiser step here.
package distill

import (
	"context"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/train/dataset"
)

// Sentinel errors hoisted to package vars — each previously allocated a
// fresh core.NewError on the (rare but hot under churn) failure path.
// Centralised here (rather than beside each call site) so the whole
// package's error vocabulary reads in one place, matching the source
// this package was ported from (go-mlx/go/distill.go).
var (
	errDatasetNil         = core.NewError("distill: dataset is nil")
	errLogitNotFinite     = core.NewError("mlx: distillation logit is not finite")
	errCheckpointPath     = core.NewError("mlx: distillation checkpoint metadata path is required")
	errTeacherLogitsEmpty = core.NewError("mlx: teacher logits are empty")
	errTempInvalid        = core.NewError("mlx: distillation temperature must be finite and positive")
	errNoMaskedTokens     = core.NewError("mlx: distillation loss has no masked tokens")
	errLogitVocab         = core.NewError("mlx: distillation logit shape mismatch: vocabulary")
	errLogitSeq           = core.NewError("mlx: distillation logit shape mismatch: sequence")
	errLogitEmptyVocab    = core.NewError("mlx: distillation logit shape mismatch: empty vocabulary")
	errLogitBatch         = core.NewError("mlx: distillation logit shape mismatch: batch")
	errKLNotFinite        = core.NewError("mlx: distillation KL loss is not finite")
	errCoreResultFailed   = core.NewError("core result failed")
)

// LossKind selects the scalar used to train the student.
type LossKind string

const (
	LossKL               LossKind = "kl"
	LossSoftCrossEntropy LossKind = "soft_cross_entropy"
)

// Logits is a batch x sequence x vocabulary tensor in Go-native form —
// never an engine tensor handle, always a plain, engine-agnostic value.
type Logits [][][]float32

// BatchConfig controls how a driver's tokenizer batches dataset samples
// for distillation. An alias for dataset.BatchConfig: none of its four
// fields (batch size, max sequence length, packing, EOS handling) are
// distillation-specific — they are the same generic batch shape every
// training/eval/distillation driver needs — so the canonical definition
// lives in dataset and this name stays as a compatibility spelling within
// this package. Every existing distill.BatchConfig{...} literal and the
// Config.Batch field below keep working unchanged.
//
//	cfg := distill.BatchConfig{BatchSize: 8}
type BatchConfig = dataset.BatchConfig

// Compile-time proof BatchConfig is a genuine alias for dataset.BatchConfig
// rather than a look-alike type that merely shares its fields — a value of
// one only assigns to a variable of the other without conversion when both
// names denote the identical type.
var _ dataset.BatchConfig = BatchConfig{}

// Config controls native knowledge distillation over dataset streams.
// BatchLoss itself only consumes Temperature and Loss; the remaining
// fields are the shared vocabulary a driver's own training loop reads to
// drive its checkpoint/eval cadence and probe emission — mirroring
// go-mlx's DistillConfig, which bundled loss config and loop cadence into
// one struct rather than splitting them.
type Config struct {
	Batch           BatchConfig         `json:"batch"`
	Epochs          int                 `json:"epochs,omitempty"`
	Temperature     float64             `json:"temperature,omitempty"`
	Loss            LossKind            `json:"loss,omitempty"`
	LearningRate    float64             `json:"learning_rate,omitempty"`
	CheckpointDir   string              `json:"checkpoint_dir,omitempty"`
	CheckpointEvery int                 `json:"checkpoint_every,omitempty"`
	EvalEvery       int                 `json:"eval_every,omitempty"`
	ResumePath      string              `json:"resume_path,omitempty"`
	MaxSamples      int                 `json:"max_samples,omitempty"`
	ProbeSink       inference.ProbeSink `json:"-"`
}

// Loss records per-batch distillation loss components.
type Loss struct {
	Value            float64  `json:"value"`
	KL               float64  `json:"kl"`
	SoftCrossEntropy float64  `json:"soft_cross_entropy"`
	TeacherEntropy   float64  `json:"teacher_entropy"`
	Tokens           int      `json:"tokens"`
	Temperature      float64  `json:"temperature"`
	Kind             LossKind `json:"kind"`
}

// TeacherLogitCache provides cache hooks for offline teacher logits.
type TeacherLogitCache interface {
	GetTeacherLogits(context.Context, string) (Logits, bool, error)
	PutTeacherLogits(context.Context, string, Logits) error
}

// MemoryLogitCache is a small in-process teacher-logit cache for tests and local runs.
type MemoryLogitCache struct {
	mu     sync.RWMutex
	logits map[string]Logits
}

// NewMemoryLogitCache creates an in-memory teacher-logit cache.
func NewMemoryLogitCache() *MemoryLogitCache {
	return &MemoryLogitCache{logits: map[string]Logits{}}
}

// GetTeacherLogits returns cached teacher logits for key.
func (c *MemoryLogitCache) GetTeacherLogits(_ context.Context, key string) (Logits, bool, error) {
	if c == nil {
		return nil, false, nil
	}
	c.mu.RLock()
	logits, ok := c.logits[key]
	c.mu.RUnlock()
	// Skip the clone on miss — defer + clone overhead is wasted when
	// there's nothing to copy. Releasing the read lock manually also
	// shrinks the critical section: the clone now runs lock-free, which
	// matters when teacher logits are large (B*S*V float32).
	if !ok {
		return nil, false, nil
	}
	return cloneLogits(logits), true, nil
}

// PutTeacherLogits stores teacher logits for key.
func (c *MemoryLogitCache) PutTeacherLogits(_ context.Context, key string, logits Logits) error {
	if c == nil {
		return nil
	}
	// Clone outside the write lock — the clone is a pure copy of caller
	// data with no shared state, so it can race freely with other
	// goroutines. Acquiring the lock only for the map assignment shrinks
	// the critical section from O(B*S*V) to O(1).
	cloned := cloneLogits(logits)
	c.mu.Lock()
	if c.logits == nil {
		c.logits = map[string]Logits{}
	}
	c.logits[key] = cloned
	c.mu.Unlock()
	return nil
}

// CollectSamples pulls up to maxSamples samples from ds (or all remaining
// samples when maxSamples <= 0), returning a defensively cloned slice
// ready to replay via dataset.NewSliceDataset. Ported from go-mlx's
// distillation runner, which used exactly this pull-until-exhausted-or-
// capped loop to build the MaxSamples-truncated source dataset for a
// distillation epoch.
//
//	samples, err := distill.CollectSamples(ctx, ds, cfg.MaxSamples)
//	source := dataset.NewSliceDataset(samples)
func CollectSamples(ctx context.Context, ds dataset.Dataset, maxSamples int) ([]dataset.Sample, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if ds == nil {
		return nil, errDatasetNil
	}
	var samples []dataset.Sample
	if maxSamples > 0 {
		samples = make([]dataset.Sample, 0, maxSamples)
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if maxSamples > 0 && len(samples) >= maxSamples {
			break
		}
		sample, ok, err := ds.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		samples = append(samples, dataset.CloneSample(sample))
	}
	return samples, nil
}
