// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// RunNativeGRPOAdvantagePass runs the grouped-reward advantage normalisation
// half of GRPO over labelled samples. It intentionally does not perform
// rollouts, policy loss, KL control, or adapter updates. ok is true only when
// the linked HIP GRPO advantage kernel produced the advantages. Samples provide
// reward or comma-separated rewards labels.
func RunNativeGRPOAdvantagePass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, cfg inference.GRPOConfig) (*inference.TrainingResult, bool, error) {
	if model == nil {
		return nil, false, core.NewError("rocm: native GRPO advantage pass model is nil")
	}
	rocm, ok := model.(*rocmModel)
	if !ok {
		return nil, false, core.NewError("rocm: native GRPO advantage pass requires a ROCm model")
	}
	if dataset == nil {
		return nil, false, core.NewError("rocm: native GRPO advantage pass dataset is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	rewards, samples, err := collectGRPORewardRows(ctx, dataset)
	if err != nil {
		return nil, false, err
	}
	if len(rewards) == 0 {
		return nil, false, core.NewError("rocm: native GRPO advantage pass dataset produced no rewards")
	}
	labels := rocmCloneLabels(cfg.Labels)
	if labels == nil {
		labels = make(map[string]string, 16)
	}
	labels["training_stage"] = "grpo_advantage_pass"
	labels["training_interface"] = "advantage_only"
	labels["training_update_status"] = "not_applied"
	labels["trainer_interface"] = "not_implemented"
	labels["grpo_samples"] = strconv.Itoa(samples)
	labels["grpo_rewards"] = strconv.Itoa(len(rewards))
	if cfg.GroupSize > 0 {
		labels["grpo_group_size"] = strconv.Itoa(cfg.GroupSize)
	}
	if cfg.KLWeight != 0 {
		labels["grpo_kl_weight"] = formatFloat64Label(cfg.KLWeight)
	}
	result := &inference.TrainingResult{
		Model:   rocm.modelIdentity(),
		Adapter: rocm.ActiveAdapter(),
		Metrics: inference.TrainingMetrics{
			Samples: len(rewards),
			Step:    1,
		},
		Labels: labels,
	}
	if advantages, ok, err := RunNativeGRPOAdvantage(ctx, model, rewards); ok {
		labels["advantage_backend"] = "hip"
		labels["advantage_kernel"] = hipKernelStatusLinked
		labels["advantage_kernel_name"] = hipKernelNameGRPOAdvantage
		if err != nil {
			labels["advantage_status"] = "error"
			labels["advantage_error"] = err.Error()
			return result, true, nil
		}
		labels["advantages"] = formatFloat64CSVLabel(advantages)
		labels["advantage_status"] = "experimental"
		return result, true, nil
	}
	advantages, err := rocmReferenceNormalizeAdvantages(rewards)
	if err != nil {
		labels["advantage_status"] = "error"
		labels["advantage_error"] = err.Error()
		return result, false, nil
	}
	labels["advantages"] = formatFloat64CSVLabel(advantages)
	labels["advantage_backend"] = "reference"
	labels["advantage_kernel"] = rocm.kernelStatus().GRPO
	labels["advantage_kernel_name"] = hipKernelNameGRPOAdvantage
	labels["advantage_status"] = "experimental"
	return result, false, nil
}

func collectGRPORewardRows(ctx context.Context, dataset inference.DatasetStream) ([]float64, int, error) {
	var rewards []float64
	samples := 0
	if hint := grpoDatasetRemainingHint(dataset); hint > 0 {
		rewards = reserveFloat64Capacity(rewards, hint)
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		sample, ok, err := dataset.Next()
		if err != nil {
			return nil, 0, err
		}
		if !ok {
			break
		}
		start := len(rewards)
		rewards, err = grpoAppendRewardsFromLabels(rewards, sample.Labels)
		if err != nil {
			return nil, 0, err
		}
		if len(rewards) == start {
			continue
		}
		samples++
	}
	return rewards, samples, nil
}

func parseFloat64CSVLabel(raw string) ([]float64, error) {
	return parseFloat64CSVLabelAppend(make([]float64, 0, strings.Count(raw, ",")+1), raw)
}

func parseFloat64CSVLabelAppend(out []float64, raw string) ([]float64, error) {
	raw = core.Trim(raw)
	if raw == "" {
		return nil, core.NewError("empty float")
	}
	for {
		part := raw
		index := strings.IndexByte(raw, ',')
		if index >= 0 {
			part = raw[:index]
			raw = raw[index+1:]
		} else {
			raw = ""
		}
		text := core.Trim(part)
		if text == "" {
			return nil, core.NewError("empty float")
		}
		value, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return nil, err
		}
		out = append(out, value)
		if index < 0 {
			break
		}
	}
	return out, nil
}

func formatFloat64CSVLabel(values []float64) string {
	builder := core.NewBuilder()
	for i, value := range values {
		if i > 0 {
			builder.WriteString(",")
		}
		builder.WriteString(formatFloat64Label(value))
	}
	return builder.String()
}
