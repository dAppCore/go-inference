// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// RunNativeSFTAdamWUpdatePass composes the ROCm SFT loss pass with the packed
// AdamW update primitive. It still is not a full SFTTrainer: gradients are
// caller-supplied, no backward graph is built, and the shared trainer interface
// remains deliberately unimplemented.
func RunNativeSFTAdamWUpdatePass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, state *NativeAdamWState, gradients [][]float32, cfg inference.TrainingConfig) (*inference.TrainingResult, bool, error) {
	if state == nil {
		return nil, false, core.NewError("rocm: native SFT AdamW update pass state is nil")
	}
	loss, nativeLoss, err := RunNativeSFTLossPass(ctx, model, dataset, cfg)
	if err != nil {
		return nil, false, err
	}
	update, err := RunNativeAdamWUpdatePass(ctx, model, state, gradients, cfg)
	if err != nil {
		return loss, nativeLoss, err
	}

	labels := rocmCloneLabels(loss.Labels)
	if labels == nil {
		labels = make(map[string]string, 24)
	}
	mergeNativeAdamWUpdateLabels(labels, update)
	labels["training_stage"] = "sft_loss_adamw_update_pass"
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

// RunNativeSFTAdamWUpdateTrackPass applies one SFT loss + AdamW update step,
// then appends the updated optimizer state to an append-only track.
func RunNativeSFTAdamWUpdateTrackPass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, state *NativeAdamWState, gradients [][]float32, trackPath string, cfg inference.TrainingConfig) (*inference.TrainingResult, NativeAdamWTrackRecord, bool, error) {
	if trackPath == "" {
		return nil, NativeAdamWTrackRecord{}, false, core.NewError("rocm: native SFT AdamW update track path is required")
	}
	result, nativeLoss, err := RunNativeSFTAdamWUpdatePass(ctx, model, dataset, state, gradients, cfg)
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
	labels["training_stage"] = "sft_loss_adamw_update_track_pass"

	out := *result
	out.Labels = labels
	return &out, record, nativeLoss, nil
}

func mergeNativeAdamWUpdateLabels(labels map[string]string, update *inference.TrainingResult) {
	if labels == nil || update == nil {
		return
	}
	for _, key := range []string{
		"optimizer",
		"optimizer_backend",
		"optimizer_kernel",
		"optimizer_kernel_name",
		"hip_optimizer_update",
		"optimizer_state_layout",
		"optimizer_tensors",
		"optimizer_parameters",
		"optimizer_step",
		"optimizer_packed",
	} {
		if value := update.Labels[key]; value != "" {
			labels[key] = value
		}
	}
}
