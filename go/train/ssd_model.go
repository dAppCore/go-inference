// SPDX-Licence-Identifier: EUPL-1.2

// ssd_model.go: the Model-bound SSD entry point, re-expressed engine-neutrally.
// go-mlx's Model.RunSSD bound to the concrete mlx.Model (metal); this binds to
// the neutral inference.TextModel, building the SSDRunner from the model's own
// GenerateStream (via TextModel.Generate) and the capability-probed
// PromptCacheWarmer (the neutral WarmPromptCache seam). No engine type appears
// in the signature — a driver on any registered backend can self-distil.
//
// It lives here in dappco.re/go/inference/train, NOT in package inference: the
// pipeline (RunSSD) lives in train, and package inference cannot import train
// (train imports inference — the reverse would cycle), so the Model-bound entry
// that wires a TextModel into the pipeline is a train-package function.

package train

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// RunSSDModel samples the frozen model over ds, capturing + scoring each
// self-output at birth, and STOPS at the scored trace. scoreFn is the optional
// sampling-phase scorer (nil disables scoring even when cfg.ScoreSamples is
// set). Generation rides model.Generate; the kernel prefix rides the neutral
// PromptCacheWarmer when the engine exposes it.
func RunSSDModel(ctx context.Context, model inference.TextModel, ds inference.DatasetStream, cfg SSDConfig, scoreFn ScoreFunc) (*SSDResult, error) {
	if model == nil {
		return nil, core.NewError("mlx: SSD model is nil")
	}
	runner := SSDRunner{
		Generate: func(ctx context.Context, prompt string, gcfg inference.GenerateConfig) (string, error) {
			return generateText(ctx, model, prompt, ssdGenerateOptions(gcfg)...)
		},
		Score: scoreFn,
	}
	// The kernel-prefix lane (#97): prefill the kernel ONCE as the exact
	// token-prefix cache; every sample's generation reuses that KV state. An
	// engine without the warm capability simply omits WarmPrefix — the prefix
	// still rides every generation prompt (recomputed, same output).
	if warmer, ok := model.(inference.PromptCacheWarmer); ok {
		runner.WarmPrefix = warmer.WarmPromptCache
	}
	return RunSSD(ctx, runner, ds, cfg)
}

// generateText drives model.Generate to completion and returns the concatenated
// decoded text, surfacing any generation error and context cancellation.
func generateText(ctx context.Context, model inference.TextModel, prompt string, opts ...inference.GenerateOption) (string, error) {
	builder := core.NewBuilder()
	for token := range model.Generate(ctx, prompt, opts...) {
		builder.WriteString(token.Text)
	}
	if r := model.Err(); !r.OK {
		if err, ok := r.Value.(error); ok {
			return "", err
		}
		return "", core.NewError("mlx: generation failed")
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return "", err
		}
	}
	return builder.String(), nil
}

// ssdGenerateOptions turns a sampling GenerateConfig into the option list
// TextModel.Generate takes. Mirrors go-mlx's ssdOptions.
func ssdGenerateOptions(cfg inference.GenerateConfig) []inference.GenerateOption {
	opts := []inference.GenerateOption{
		inference.WithMaxTokens(cfg.MaxTokens),
		inference.WithTemperature(cfg.Temperature),
	}
	if cfg.TopK != 0 {
		opts = append(opts, inference.WithTopK(cfg.TopK))
	}
	if cfg.TopP != 0 {
		opts = append(opts, inference.WithTopP(cfg.TopP))
	}
	if cfg.MinP != 0 {
		opts = append(opts, inference.WithMinP(cfg.MinP))
	}
	if cfg.RepeatPenalty != 0 {
		opts = append(opts, inference.WithRepeatPenalty(cfg.RepeatPenalty))
	}
	return opts
}
