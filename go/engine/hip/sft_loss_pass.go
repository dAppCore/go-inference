// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// RunNativeSFTLossPass runs the supervised loss half of SFT over a dataset. It
// intentionally does not apply gradients or update adapters; ok is true only
// when the linked HIP cross-entropy kernel produced the loss.
func RunNativeSFTLossPass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, cfg inference.TrainingConfig) (*inference.TrainingResult, bool, error) {
	if model == nil {
		return nil, false, core.NewError("rocm: native SFT loss pass model is nil")
	}
	rocm, ok := model.(*rocmModel)
	if !ok {
		return nil, false, core.NewError("rocm: native SFT loss pass requires a ROCm model")
	}
	if dataset == nil {
		return nil, false, core.NewError("rocm: native SFT loss pass dataset is nil")
	}
	labels := rocmCloneLabels(cfg.Labels)
	if labels == nil {
		labels = make(map[string]string, 12)
	}
	if evalGenerate, ok, err := SimpleSelfDistillationEvalGenerateConfig(labels, 0); err != nil {
		return nil, false, err
	} else if ok {
		formatted := formatSimpleSelfDistillationFloat32(evalGenerate.Temperature)
		labels["eval.temperature"] = formatted
		labels["training_eval_temperature"] = formatted
	}
	eval, err := rocm.Evaluate(ctx, dataset, inference.EvalConfig{
		BatchSize: cfg.BatchSize,
	})
	if err != nil {
		return nil, false, err
	}
	for key, value := range eval.Labels {
		labels["eval."+key] = value
	}
	labels["training_stage"] = "sft_loss_pass"
	labels["training_interface"] = "loss_only"
	labels["training_update_status"] = "not_applied"
	labels["trainer_interface"] = "not_implemented"
	labels["loss_backend"] = eval.Labels["loss_backend"]
	labels["loss_status"] = eval.Labels["loss_status"]
	labels["loss_kernel"] = eval.Labels["loss_kernel"]
	labels["loss_kernel_name"] = eval.Labels["loss_kernel_name"]
	result := &inference.TrainingResult{
		Model:   eval.Model,
		Adapter: eval.Adapter,
		Metrics: inference.TrainingMetrics{
			Samples: eval.Metrics.Samples,
			Tokens:  eval.Metrics.Tokens,
			Loss:    eval.Metrics.Loss,
		},
		Labels: labels,
	}
	return result, eval.Labels["loss_backend"] == "hip" && eval.Labels["loss_status"] == "experimental", nil
}
