// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// NativeCrossEntropyLossResult records a native cross-entropy loss kernel result.
type NativeCrossEntropyLossResult struct {
	Loss       float64 `json:"loss"`
	Perplexity float64 `json:"perplexity"`
}

// NativeDistillationKLLossResult records a native distillation KL kernel result.
type NativeDistillationKLLossResult struct {
	KL float64 `json:"kl"`
}

type nativeDistillationLossKernelModel interface {
	RunDistillationKLLoss(ctx context.Context, studentLogits, teacherLogits [][]float32, temperature float64) (hipDistillationKLLossResult, bool, error)
}

type nativeGRPOAdvantageKernelModel interface {
	RunGRPOAdvantage(ctx context.Context, rewards []float64) ([]float64, bool, error)
}

// RunNativeAdamWUpdate runs the linked HIP AdamW optimizer update kernel for a
// ROCm model. ok=false means the model is ROCm but the native optimizer path is
// not available for the loaded runtime.
func RunNativeAdamWUpdate(ctx context.Context, model inference.TextModel, state *NativeAdamWState, gradients [][]float32) (bool, error) {
	native, err := rocmNativeTrainingKernelModel[nativeAdamWUpdateKernelModel](model, "AdamW update")
	if err != nil {
		return false, err
	}
	return native.RunAdamWUpdate(ctx, state, gradients)
}

// RunNativeCrossEntropyLoss runs the linked HIP cross-entropy loss kernel for a
// ROCm model. ok=false means the model is ROCm but the native kernel path is not
// available for the loaded runtime.
func RunNativeCrossEntropyLoss(ctx context.Context, model inference.TextModel, logits [][]float32, targets []int) (NativeCrossEntropyLossResult, bool, error) {
	native, err := rocmNativeTrainingKernelModel[nativeEvalLossKernelModel](model, "cross entropy")
	if err != nil {
		return NativeCrossEntropyLossResult{}, false, err
	}
	result, ok, err := native.RunEvalCrossEntropyLoss(ctx, logits, targets)
	return NativeCrossEntropyLossResult{
		Loss:       result.Loss,
		Perplexity: result.Perplexity,
	}, ok, err
}

// RunNativeDistillationKLLoss runs the linked HIP teacher/student KL loss kernel
// for a ROCm model. ok=false means the model is ROCm but the native kernel path
// is not available for the loaded runtime.
func RunNativeDistillationKLLoss(ctx context.Context, model inference.TextModel, studentLogits, teacherLogits [][]float32, temperature float64) (NativeDistillationKLLossResult, bool, error) {
	native, err := rocmNativeTrainingKernelModel[nativeDistillationLossKernelModel](model, "distillation KL")
	if err != nil {
		return NativeDistillationKLLossResult{}, false, err
	}
	result, ok, err := native.RunDistillationKLLoss(ctx, studentLogits, teacherLogits, temperature)
	return NativeDistillationKLLossResult{KL: result.KL}, ok, err
}

// RunNativeGRPOAdvantage runs the linked HIP grouped-reward advantage kernel for
// a ROCm model. ok=false means the model is ROCm but the native kernel path is
// not available for the loaded runtime.
func RunNativeGRPOAdvantage(ctx context.Context, model inference.TextModel, rewards []float64) ([]float64, bool, error) {
	native, err := rocmNativeTrainingKernelModel[nativeGRPOAdvantageKernelModel](model, "GRPO advantage")
	if err != nil {
		return nil, false, err
	}
	return native.RunGRPOAdvantage(ctx, rewards)
}

func rocmNativeTrainingKernelModel[T any](model inference.TextModel, operation string) (T, error) {
	var zero T
	if model == nil {
		return zero, core.NewError("rocm: native " + operation + " model is nil")
	}
	rocm, ok := model.(*rocmModel)
	if !ok {
		return zero, core.NewError("rocm: native " + operation + " requires a ROCm model")
	}
	if rocm.native == nil {
		return zero, core.NewError("rocm: native " + operation + " model runtime is nil")
	}
	native, ok := rocm.native.(T)
	if !ok {
		return zero, core.NewError("rocm: native " + operation + " kernel interface is not available")
	}
	return native, nil
}
