// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// RunNativeGRPOAdamWUpdatePass composes the ROCm grouped-reward advantage pass
// with the packed AdamW update primitive. It is not a full GRPOTrainer:
// rollouts, policy loss, KL control, and backward graph construction are still
// outside this helper.
func RunNativeGRPOAdamWUpdatePass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, state *NativeAdamWState, gradients [][]float32, cfg inference.GRPOConfig) (*inference.TrainingResult, bool, error) {
	if state == nil {
		return nil, false, core.NewError("rocm: native GRPO AdamW update pass state is nil")
	}
	advantage, nativeAdvantage, err := RunNativeGRPOAdvantagePass(ctx, model, dataset, cfg)
	if err != nil {
		return nil, false, err
	}
	update, err := RunNativeAdamWUpdatePass(ctx, model, state, gradients, cfg.TrainingConfig)
	if err != nil {
		return advantage, nativeAdvantage, err
	}

	labels := rocmCloneLabels(advantage.Labels)
	if labels == nil {
		labels = make(map[string]string, 24)
	}
	mergeNativeAdamWUpdateLabels(labels, update)
	labels["training_stage"] = "grpo_advantage_adamw_update_pass"
	labels["training_interface"] = "advantage_plus_optimizer_update"
	labels["training_update_status"] = "applied"
	labels["trainer_interface"] = "not_implemented"
	labels["advantage_native_ready"] = boolLabel(nativeAdvantage)

	result := *advantage
	result.Metrics.Step = update.Metrics.Step
	result.Metrics.LearningRate = update.Metrics.LearningRate
	result.Labels = labels
	return &result, nativeAdvantage, nil
}

// RunNativeGRPOAdamWUpdateTrackPass applies one grouped-reward advantage +
// AdamW update step, then appends the updated optimizer state to an append-only
// track.
func RunNativeGRPOAdamWUpdateTrackPass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, state *NativeAdamWState, gradients [][]float32, trackPath string, cfg inference.GRPOConfig) (*inference.TrainingResult, NativeAdamWTrackRecord, bool, error) {
	if trackPath == "" {
		return nil, NativeAdamWTrackRecord{}, false, core.NewError("rocm: native GRPO AdamW update track path is required")
	}
	result, nativeAdvantage, err := RunNativeGRPOAdamWUpdatePass(ctx, model, dataset, state, gradients, cfg)
	if err != nil {
		return result, NativeAdamWTrackRecord{}, nativeAdvantage, err
	}
	record, err := AppendNativeAdamWStateTrack(trackPath, state)
	if err != nil {
		return result, NativeAdamWTrackRecord{}, nativeAdvantage, err
	}
	labels := rocmCloneLabels(result.Labels)
	if labels == nil {
		labels = make(map[string]string, 32)
	}
	if err := addNativeAdamWTrackLabels(labels, trackPath, record); err != nil {
		return result, NativeAdamWTrackRecord{}, nativeAdvantage, err
	}
	labels["training_stage"] = "grpo_advantage_adamw_update_track_pass"

	out := *result
	out.Labels = labels
	return &out, record, nativeAdvantage, nil
}
