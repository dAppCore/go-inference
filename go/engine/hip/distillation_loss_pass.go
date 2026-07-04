// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// RunNativeDistillationLossPass runs the teacher/student KL-loss half of
// distillation over labelled samples. It intentionally does not update a
// student; ok is true only when the linked HIP distillation kernel produced the
// loss. Samples provide comma-separated float labels named student_logits and
// teacher_logits.
func RunNativeDistillationLossPass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, cfg inference.DistillConfig) (*inference.TrainingResult, bool, error) {
	if model == nil {
		return nil, false, core.NewError("rocm: native distillation loss pass model is nil")
	}
	rocm, ok := model.(*rocmModel)
	if !ok {
		return nil, false, core.NewError("rocm: native distillation loss pass requires a ROCm model")
	}
	if dataset == nil {
		return nil, false, core.NewError("rocm: native distillation loss pass dataset is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	student, teacher, samples, err := collectDistillationLossRows(ctx, dataset)
	if err != nil {
		return nil, false, err
	}
	if samples == 0 {
		return nil, false, core.NewError("rocm: native distillation loss pass dataset produced no labelled samples")
	}
	temperature := cfg.Temperature
	if temperature == 0 {
		temperature = 1
	}
	labels := rocmCloneLabels(cfg.Labels)
	if labels == nil {
		labels = make(map[string]string, 12)
	}
	labels["training_stage"] = "distillation_loss_pass"
	labels["training_interface"] = "loss_only"
	labels["training_update_status"] = "not_applied"
	labels["trainer_interface"] = "not_implemented"
	labels["distillation_temperature"] = formatFloat64Label(temperature)
	labels["distillation_samples"] = strconv.Itoa(samples)
	if len(student) > 0 {
		labels["distillation_vocab"] = strconv.Itoa(len(student[0]))
	}
	result := &inference.TrainingResult{
		Model:   rocm.modelIdentity(),
		Adapter: rocm.ActiveAdapter(),
		Metrics: inference.TrainingMetrics{
			Samples: samples,
		},
		Labels: labels,
	}
	if native, ok, err := RunNativeDistillationKLLoss(ctx, model, student, teacher, temperature); ok {
		labels["loss_backend"] = "hip"
		labels["loss_kernel"] = hipKernelStatusLinked
		labels["loss_kernel_name"] = hipKernelNameDistillKL
		if err != nil {
			labels["loss_status"] = "error"
			labels["loss_error"] = err.Error()
			return result, true, nil
		}
		result.Metrics.Loss = native.KL
		labels["loss"] = core.Sprintf("%.6f", native.KL)
		labels["loss_status"] = "experimental"
		return result, true, nil
	}
	value, err := rocmReferenceDistillationKL(student, teacher, temperature)
	if err != nil {
		labels["loss_status"] = "error"
		labels["loss_error"] = err.Error()
		return result, false, nil
	}
	result.Metrics.Loss = value
	labels["loss"] = core.Sprintf("%.6f", value)
	labels["loss_backend"] = "reference"
	labels["loss_kernel"] = rocm.kernelStatus().Distillation
	labels["loss_kernel_name"] = hipKernelNameDistillKL
	labels["loss_status"] = "experimental"
	return result, false, nil
}

func collectDistillationLossRows(ctx context.Context, dataset inference.DatasetStream) ([][]float32, [][]float32, int, error) {
	var students [][]float32
	var teachers [][]float32
	for {
		if err := ctx.Err(); err != nil {
			return nil, nil, 0, err
		}
		sample, ok, err := dataset.Next()
		if err != nil {
			return nil, nil, 0, err
		}
		if !ok {
			break
		}
		student, teacher, ok, err := distillationLossRowsFromLabels(sample.Labels)
		if err != nil {
			return nil, nil, 0, err
		}
		if !ok {
			continue
		}
		students = append(students, student)
		teachers = append(teachers, teacher)
	}
	return students, teachers, len(students), nil
}

func distillationLossRowsFromLabels(labels map[string]string) ([]float32, []float32, bool, error) {
	studentRaw := core.Trim(labels["student_logits"])
	teacherRaw := core.Trim(labels["teacher_logits"])
	if studentRaw == "" && teacherRaw == "" {
		return nil, nil, false, nil
	}
	if studentRaw == "" || teacherRaw == "" {
		return nil, nil, false, core.NewError("rocm: distillation sample requires student_logits and teacher_logits labels")
	}
	student, err := parseFloat32CSVLabel(studentRaw)
	if err != nil {
		return nil, nil, false, core.E("rocm.DistillationLossPass", "parse student_logits", err)
	}
	teacher, err := parseFloat32CSVLabel(teacherRaw)
	if err != nil {
		return nil, nil, false, core.E("rocm.DistillationLossPass", "parse teacher_logits", err)
	}
	if len(student) == 0 || len(student) != len(teacher) {
		return nil, nil, false, core.NewError("rocm: distillation logits must be non-empty and equal length")
	}
	return student, teacher, true, nil
}

func parseFloat32CSVLabel(raw string) ([]float32, error) {
	parts := core.Split(raw, ",")
	out := make([]float32, 0, len(parts))
	for _, part := range parts {
		text := core.Trim(part)
		if text == "" {
			return nil, core.NewError("empty float")
		}
		value, err := strconv.ParseFloat(text, 32)
		if err != nil {
			return nil, err
		}
		out = append(out, float32(value))
	}
	return out, nil
}

func formatFloat64Label(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
