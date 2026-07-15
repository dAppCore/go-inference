// SPDX-Licence-Identifier: EUPL-1.2

// sft_model.go: the Model-bound SFT entry point, re-expressed against the
// neutral [engine.Trainer] seam. go-mlx's Model.TrainSFT bound to the concrete
// mlx.Model and drove metal.LoRAAdapter + metal.AdamW directly; this probes the
// loaded inference.TextModel for the [engine.TrainerModel] capability (the metal
// engine forwards it — see engine.TextModel.OpenTrainer), opens a trainer, and
// runs the neutral SFT loop (sft.go) over it. The trainable weights + optimiser
// state stay inside the engine; only the on-disk adapter crosses out.
//
// It lives in dappco.re/go/inference/train for the same reason as RunSSDModel:
// this is where the loop lives and where engine is importable (package
// inference cannot import engine — engine imports inference).

package train

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
)

// RunSFTModel opens an engine trainer over the loaded model and runs the LoRA
// SFT loop over ds. encode tokenises training text with the model's own
// tokeniser (the engine trainer consumes token ids); gen (optional) drives the
// eval passes. Returns a clear error when the loaded engine does not implement
// the [engine.TrainerModel] seam.
func RunSFTModel(ctx context.Context, model inference.TextModel, encode EncodeFunc, ds inference.DatasetStream, cfg SFTConfig) (*SFTResult, error) {
	if model == nil {
		return nil, core.NewError("mlx: SFT model is nil")
	}
	tm, ok := model.(engine.TrainerModel)
	if !ok {
		return nil, core.NewError("mlx: loaded engine does not support training (no engine.TrainerModel)")
	}
	cfg.Config = NormalizeConfig(cfg.Config)
	trainer, err := tm.OpenTrainer(inference.TrainingConfig{
		Epochs:               cfg.Epochs,
		BatchSize:            cfg.BatchSize,
		GradientAccumulation: cfg.GradientAccumulationSteps,
		LearningRate:         cfg.LearningRate,
		LoRA:                 cfg.LoRA,
	})
	if err != nil {
		return nil, err
	}
	defer func() { _ = trainer.Close() }()

	gen := func(ctx context.Context, prompt string, maxTokens int, temperature float32) (string, error) {
		opts := []inference.GenerateOption{inference.WithMaxTokens(maxTokens)}
		if temperature > 0 {
			opts = append(opts, inference.WithTemperature(temperature))
		}
		return generateText(ctx, model, prompt, opts...)
	}
	return RunSFT(ctx, trainer, encode, gen, ds, cfg)
}
