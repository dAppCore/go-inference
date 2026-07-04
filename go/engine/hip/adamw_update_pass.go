// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const nativeAdamWUpdateKernelName = hipKernelNameAdamWUpdate

type nativeAdamWUpdateKernelModel interface {
	RunAdamWUpdate(ctx context.Context, state *NativeAdamWState, gradients [][]float32) (bool, error)
}

// RunNativeAdamWUpdatePass applies one packed AdamW update to caller-owned
// optimizer state. It is an optimizer stepping stone, not a full trainer: no
// backward pass is computed, no dataset is consumed, and HIP AdamW kernels are
// used only when the loaded ROCm runtime reports a linked optimizer kernel.
func RunNativeAdamWUpdatePass(ctx context.Context, model inference.TextModel, state *NativeAdamWState, gradients [][]float32, cfg inference.TrainingConfig) (*inference.TrainingResult, error) {
	if model == nil {
		return nil, core.NewError("rocm: native AdamW update pass model is nil")
	}
	rocm, ok := model.(*rocmModel)
	if !ok {
		return nil, core.NewError("rocm: native AdamW update pass requires a ROCm model")
	}
	if state == nil {
		return nil, core.NewError("rocm: native AdamW update pass state is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.LearningRate != 0 {
		state.Config.LearningRate = cfg.LearningRate
		state.Config.LearningRateSet = true
	}

	kernelStatus := rocm.kernelStatus()
	optimizerBackend := "reference"
	if native, ok := rocm.native.(nativeAdamWUpdateKernelModel); ok {
		handled, err := native.RunAdamWUpdate(ctx, state, gradients)
		if err != nil {
			return nil, err
		}
		if handled {
			optimizerBackend = "hip"
		}
	}
	if optimizerBackend == "reference" {
		if err := state.StepInPlace(gradients); err != nil {
			return nil, err
		}
	}
	labels := rocmCloneLabels(cfg.Labels)
	if labels == nil {
		labels = make(map[string]string, 16)
	}
	total := stateTotalLen(state)
	labels["training_stage"] = "adamw_update_pass"
	labels["training_interface"] = "optimizer_update_only"
	labels["training_update_status"] = "applied"
	labels["trainer_interface"] = "not_implemented"
	labels["optimizer"] = "adamw"
	labels["optimizer_backend"] = optimizerBackend
	labels["optimizer_kernel"] = kernelStatus.Optimizer
	labels["optimizer_kernel_name"] = nativeAdamWUpdateKernelName
	labels["optimizer_launch_args"] = "hipAdamWUpdateLaunchArgs"
	labels["optimizer_launch_args_bytes"] = strconv.Itoa(hipAdamWUpdateLaunchArgsBytes)
	labels["hip_optimizer_update"] = kernelStatus.Optimizer
	labels["optimizer_state_layout"] = "packed_contiguous_parameters_m_v"
	labels["optimizer_tensors"] = strconv.Itoa(len(state.Layout))
	labels["optimizer_parameters"] = strconv.Itoa(total)
	labels["optimizer_step"] = strconv.Itoa(state.Step)
	labels["optimizer_packed"] = strconv.FormatBool(state.Config.Packed)

	return &inference.TrainingResult{
		Model:   rocm.modelIdentity(),
		Adapter: rocm.ActiveAdapter(),
		Metrics: inference.TrainingMetrics{
			Step:         state.Step,
			LearningRate: state.Config.LearningRate,
		},
		Labels: labels,
	}, nil
}
