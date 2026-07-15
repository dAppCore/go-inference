// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"dappco.re/go/inference"
)

// trainer.go adds the engine-neutral TRAINING seam that sits beside [Session]:
// where Session is the retained decode surface, [Trainer] is the retained LoRA
// SFT surface. A concrete engine (the Apple-GPU "metal" engine, the AMD "hip"
// engine) supplies a Trainer that owns the LoRA weights + optimiser state and
// runs the gradient steps; this package only names the primitives the SFT loop
// drives, so a caller trains through one contract regardless of engine.
//
// The contract is deliberately weight-opaque: the trained tensors and optimiser
// state NEVER leave the engine package (no *Array / *AdamW / device handle in
// the signatures). Losses cross the boundary as float64, batches as the neutral
// [inference.Batch] (token ids + optional response loss-mask), and the trained
// adapter is persisted to disk by [Trainer.Save] — so the only cross-engine
// artefact is the on-disk adapter, exactly as it is for serving. The metal
// engine implements Trainer with its own no-cgo gradient kernels; the hip engine
// implements it later over its model.LoRAAdapter — same seam, different maths.

// Trainer is the retained LoRA SFT training session over a loaded model. It wraps
// the frozen base, the trainable LoRA weights, and the optimiser state; each
// [Trainer.Step] runs one gradient step and returns the training loss, and
// [Trainer.Save] writes the trained adapter to disk. Single-goroutine, mirroring
// the [Session] contract: one Trainer drives one training loop.
//
//	tr, err := model.OpenTrainer(inference.TrainingConfig{LoRA: inference.LoRAConfig{Rank: 8, Alpha: 16}})
//	for step := 0; step < steps; step++ {
//	    loss, err := tr.Step(batch) // one AdamW step over batch; loss falls as it learns
//	}
//	err = tr.Save("/models/lora/domain-v1") // adapter.safetensors + adapter_config.json
type Trainer interface {
	// Step runs one SFT gradient step over batch (one optimiser update) and returns
	// the mean training loss. Targets are the causal next token of each sequence in
	// batch.TokenIDs; batch.LossMask, when set, restricts the loss to response
	// positions. The optimiser state is held inside the Trainer (set at open), so
	// the caller drives the loop by repeated Step calls, not by threading an
	// optimiser through — the trainable weights never cross this boundary.
	Step(batch inference.Batch) (loss float64, err error)

	// StepAccumulated accumulates the gradients of several micro-batches and applies
	// ONE optimiser update from their mean — the large-effective-batch path when a
	// single batch does not fit. Returns the mean loss across the micro-batches.
	StepAccumulated(batches []inference.Batch) (loss float64, err error)

	// Loss is the forward-only mean loss over batch under the current adapter
	// weights: no gradients, no optimiser update. The validation lane of the
	// training instrument — the same objective Step minimises, none of the movement.
	Loss(batch inference.Batch) (loss float64, err error)

	// Save persists the trained LoRA adapter to path as a reloadable package
	// (adapter.safetensors + adapter_config.json, the go-mlx on-disk format), so
	// `serve --adapter <path>` reapplies it. Only the adapter weights are written;
	// the frozen base is not.
	Save(path string) error

	// Close releases the retained training session (the base session and any device
	// state). The saved adapter on disk is unaffected.
	Close() error
}

// TrainerModel is the optional capability of a loaded model that can open an
// engine [Trainer] — the "open a train session from a loaded model" entry point.
// An engine whose token model supports LoRA SFT implements it; callers probe for
// it exactly as they probe inference.TrainableModel:
//
//	tr, ok := model.(engine.TrainerModel)
//	if !ok { return core.NewError("engine does not support training") }
//	trainer, err := tr.OpenTrainer(cfg)
type TrainerModel interface {
	// OpenTrainer opens a retained LoRA SFT [Trainer] over this model with the given
	// training configuration (LoRA rank/alpha, learning rate). The returned Trainer
	// owns a fresh base training session and zero-initialised adapter.
	OpenTrainer(cfg inference.TrainingConfig) (Trainer, error)
}
