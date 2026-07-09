// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// RunNativeDistillationAdamWUpdatePass composes the ROCm distillation KL loss
// pass with the packed AdamW update primitive. It is a package-local training
// step toward the production distillation lane; caller-supplied gradients are
// applied, but the shared DistillTrainer interface remains unimplemented.
func RunNativeDistillationAdamWUpdatePass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, state *NativeAdamWState, gradients [][]float32, cfg inference.DistillConfig) (*inference.TrainingResult, bool, error) {
	if state == nil {
		return nil, false, core.NewError("rocm: native distillation AdamW update pass state is nil")
	}
	loss, nativeLoss, err := RunNativeDistillationLossPass(ctx, model, dataset, cfg)
	if err != nil {
		return nil, false, err
	}
	update, err := RunNativeAdamWUpdatePass(ctx, model, state, gradients, cfg.TrainingConfig)
	if err != nil {
		return loss, nativeLoss, err
	}

	labels := rocmCloneLabels(loss.Labels)
	if labels == nil {
		labels = make(map[string]string, 24)
	}
	mergeNativeAdamWUpdateLabels(labels, update)
	labels["training_stage"] = "distillation_loss_adamw_update_pass"
	labels["training_interface"] = "loss_plus_optimizer_update"
	labels["training_update_status"] = "applied"
	labels["trainer_interface"] = "not_implemented"
	labels["loss_native_ready"] = boolLabel(nativeLoss)

	result := *loss
	result.Metrics.Step = update.Metrics.Step
	result.Metrics.LearningRate = update.Metrics.LearningRate
	result.Labels = labels
	return &result, nativeLoss, nil
}

// RunNativeDistillationAdamWUpdateTrackPass applies one distillation loss +
// AdamW update step, then appends the updated optimizer state to an append-only
// track.
func RunNativeDistillationAdamWUpdateTrackPass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, state *NativeAdamWState, gradients [][]float32, trackPath string, cfg inference.DistillConfig) (*inference.TrainingResult, NativeAdamWTrackRecord, bool, error) {
	if trackPath == "" {
		return nil, NativeAdamWTrackRecord{}, false, core.NewError("rocm: native distillation AdamW update track path is required")
	}
	result, nativeLoss, err := RunNativeDistillationAdamWUpdatePass(ctx, model, dataset, state, gradients, cfg)
	if err != nil {
		return result, NativeAdamWTrackRecord{}, nativeLoss, err
	}
	record, err := AppendNativeAdamWStateTrack(trackPath, state)
	if err != nil {
		return result, NativeAdamWTrackRecord{}, nativeLoss, err
	}
	labels := rocmCloneLabels(result.Labels)
	if labels == nil {
		labels = make(map[string]string, 32)
	}
	if err := addNativeAdamWTrackLabels(labels, trackPath, record); err != nil {
		return result, NativeAdamWTrackRecord{}, nativeLoss, err
	}
	labels["training_stage"] = "distillation_loss_adamw_update_track_pass"

	out := *result
	out.Labels = labels
	return &out, record, nativeLoss, nil
}
