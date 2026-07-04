// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// RunNativeLoRABackwardPass computes reference LoRA A/B gradients for one
// projection from an input activation and upstream output gradients. It is a
// backward primitive, not a public trainer implementation.
func RunNativeLoRABackwardPass(input, loraA, loraB, upstream []float32, rows, cols, rank int, alpha float32) ([][]float32, error) {
	if rank <= 0 || rows <= 0 || cols <= 0 {
		return nil, core.NewError("rocm: LoRA backward rows, cols, and rank must be positive")
	}
	if !hipQ8ScaleIsPositiveFinite(alpha) {
		return nil, core.NewError("rocm: LoRA backward alpha must be positive and finite")
	}
	if len(input) != cols {
		return nil, core.Errorf("rocm: LoRA backward input length %d does not match cols %d", len(input), cols)
	}
	if len(upstream) != rows {
		return nil, core.Errorf("rocm: LoRA backward upstream length %d does not match rows %d", len(upstream), rows)
	}
	if len(loraA) != rank*cols {
		return nil, core.Errorf("rocm: LoRA backward A length %d does not match rank*cols %d", len(loraA), rank*cols)
	}
	if len(loraB) != rows*rank {
		return nil, core.Errorf("rocm: LoRA backward B length %d does not match rows*rank %d", len(loraB), rows*rank)
	}
	if !rocmFloat32SliceFinite(input) || !rocmFloat32SliceFinite(upstream) || !rocmFloat32SliceFinite(loraA) || !rocmFloat32SliceFinite(loraB) {
		return nil, core.NewError("rocm: LoRA backward inputs must be finite")
	}

	down := make([]float32, rank)
	for r := 0; r < rank; r++ {
		for c := 0; c < cols; c++ {
			down[r] += loraA[r*cols+c] * input[c]
		}
	}
	scale := alpha / float32(rank)
	gradA := make([]float32, len(loraA))
	gradB := make([]float32, len(loraB))
	for row := 0; row < rows; row++ {
		grad := upstream[row] * scale
		for r := 0; r < rank; r++ {
			gradB[row*rank+r] += grad * down[r]
		}
	}
	for r := 0; r < rank; r++ {
		back := float32(0)
		for row := 0; row < rows; row++ {
			back += upstream[row] * loraB[row*rank+r]
		}
		back *= scale
		for c := 0; c < cols; c++ {
			gradA[r*cols+c] += back * input[c]
		}
	}
	return [][]float32{gradA, gradB}, nil
}

// RunNativeLoRAAdamWUpdatePass computes one reference LoRA backward pass from
// the packed LoRA AdamW state and applies the resulting A/B gradients.
func RunNativeLoRAAdamWUpdatePass(ctx context.Context, model inference.TextModel, state *NativeAdamWState, input, upstream []float32, rows, cols, rank int, alpha float32, cfg inference.TrainingConfig) (*inference.TrainingResult, error) {
	if err := ctxErr(ctx); err != nil {
		return nil, err
	}
	if state == nil {
		return nil, core.NewError("rocm: native LoRA AdamW update pass state is nil")
	}
	loraA, loraB, err := nativeLoRAAdamWStateViews(state, rows, cols, rank)
	if err != nil {
		return nil, err
	}
	gradients, err := RunNativeLoRABackwardPass(input, loraA, loraB, upstream, rows, cols, rank, alpha)
	if err != nil {
		return nil, err
	}
	result, err := RunNativeAdamWUpdatePass(ctx, model, state, gradients, cfg)
	if err != nil {
		return nil, err
	}
	labels := rocmCloneLabels(result.Labels)
	if labels == nil {
		labels = make(map[string]string, 20)
	}
	labels["lora_backward_backend"] = "reference"
	labels["lora_backward_kernel"] = hipKernelStatusNotLinked
	labels["lora_backward_parameters"] = "lora_a,lora_b"
	labels["lora_backward_rank"] = core.Sprintf("%d", rank)
	labels["training_interface"] = "lora_backward_plus_optimizer_update"
	labels["training_stage"] = "lora_backward_adamw_update_pass"
	labels["trainer_interface"] = "not_implemented"

	out := *result
	out.Labels = labels
	return &out, nil
}

// RunNativeLoRAAdamWUpdateTrackPass applies one LoRA backward + AdamW update
// step, then appends the updated optimizer state to an append-only track.
func RunNativeLoRAAdamWUpdateTrackPass(ctx context.Context, model inference.TextModel, state *NativeAdamWState, input, upstream []float32, rows, cols, rank int, alpha float32, trackPath string, cfg inference.TrainingConfig) (*inference.TrainingResult, NativeAdamWTrackRecord, error) {
	if trackPath == "" {
		return nil, NativeAdamWTrackRecord{}, core.NewError("rocm: native LoRA AdamW update track path is required")
	}
	result, err := RunNativeLoRAAdamWUpdatePass(ctx, model, state, input, upstream, rows, cols, rank, alpha, cfg)
	if err != nil {
		return result, NativeAdamWTrackRecord{}, err
	}
	record, err := AppendNativeAdamWStateTrack(trackPath, state)
	if err != nil {
		return result, NativeAdamWTrackRecord{}, err
	}
	labels := rocmCloneLabels(result.Labels)
	if labels == nil {
		labels = make(map[string]string, 24)
	}
	if err := addNativeAdamWTrackLabels(labels, trackPath, record); err != nil {
		return result, NativeAdamWTrackRecord{}, err
	}
	labels["training_stage"] = "lora_backward_adamw_update_track_pass"

	out := *result
	out.Labels = labels
	return &out, record, nil
}

func nativeLoRAAdamWStateViews(state *NativeAdamWState, rows, cols, rank int) ([]float32, []float32, error) {
	if state == nil {
		return nil, nil, core.NewError("rocm: LoRA AdamW state is nil")
	}
	if len(state.Layout) != 2 {
		return nil, nil, core.Errorf("rocm: LoRA AdamW state layout length %d does not match A/B tensors", len(state.Layout))
	}
	if state.Layout[0].Name != "lora_a" || state.Layout[1].Name != "lora_b" {
		return nil, nil, core.NewError("rocm: LoRA AdamW state layout must contain lora_a then lora_b")
	}
	if state.Layout[0].Length != rank*cols || state.Layout[1].Length != rows*rank {
		return nil, nil, core.NewError("rocm: LoRA AdamW state layout does not match projection shape")
	}
	loraA, ok := state.ParamView(0)
	if !ok {
		return nil, nil, core.NewError("rocm: LoRA AdamW A view is unavailable")
	}
	loraB, ok := state.ParamView(1)
	if !ok {
		return nil, nil, core.NewError("rocm: LoRA AdamW B view is unavailable")
	}
	return loraA, loraB, nil
}

func ctxErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}
