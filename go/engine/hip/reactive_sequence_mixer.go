// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"context"
	"maps"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const ReactiveInferenceContract = "reactive-inference-v1"

// ReactiveSequenceMixerReport is the native ROCm view of go-mlx's config-
// composed sequence-mixer loader contract.
type ReactiveSequenceMixerReport struct {
	Version            int                            `json:"version"`
	Kind               string                         `json:"kind"`
	Backend            string                         `json:"backend"`
	CLIContract        string                         `json:"cli_contract"`
	ModelPath          string                         `json:"model_path"`
	Model              inference.ModelIdentity        `json:"model"`
	Inspection         *inference.ModelPackInspection `json:"inspection,omitempty"`
	Registry           []SequenceMixerFamily          `json:"registry"`
	Plan               *SequenceMixerLoadPlan         `json:"plan,omitempty"`
	Status             string                         `json:"status"`
	ExecutionStatus    string                         `json:"execution_status"`
	PlanningReady      bool                           `json:"planning_ready"`
	TensorBindingReady bool                           `json:"tensor_binding_ready"`
	ComposedStackReady bool                           `json:"composed_stack_ready"`
	RunnerReady        bool                           `json:"runner_ready"`
	MissingTensors     []string                       `json:"missing_tensors,omitempty"`
	Labels             map[string]string              `json:"labels,omitempty"`
	Notes              []string                       `json:"notes,omitempty"`
}

// PlanReactiveSequenceMixer inspects a local model pack and reports whether it
// can enter the reactive sequence-mixer fast lane.
func PlanReactiveSequenceMixer(ctx context.Context, modelPath string) (*ReactiveSequenceMixerReport, error) {
	return (&rocmBackend{}).PlanReactiveSequenceMixer(ctx, modelPath)
}

func (b *rocmBackend) PlanReactiveSequenceMixer(ctx context.Context, modelPath string) (*ReactiveSequenceMixerReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" {
		return nil, core.NewError("model path is required")
	}
	inspection, err := b.InspectModelPack(ctx, modelPath)
	if err != nil {
		return nil, err
	}
	report := baseReactiveSequenceMixerReport(modelPath, inspection)
	if inspection.Format != "safetensors" {
		report.Status = "unsupported_format"
		report.ExecutionStatus = "not_safetensors"
		report.Notes = append(report.Notes, "Reactive sequence-mixer planning currently uses safetensors tensor names for subpath discovery.")
		return report, nil
	}
	switch inspection.Labels["sequence_mixer_load_plan_status"] {
	case "valid":
		plan, tensorNames, err := reactiveSequenceMixerLoadPlanAndTensorNames(modelPath, inspection)
		if err != nil {
			report.Status = "invalid"
			report.ExecutionStatus = "plan_rebuild_failed"
			report.Labels["sequence_mixer_report_error"] = err.Error()
			report.Notes = append(report.Notes, err.Error())
			return report, nil
		}
		if plan == nil {
			report.Status = "not_declared"
			report.ExecutionStatus = "not_required"
			report.Labels["sequence_mixer_report_status"] = report.Status
			report.Notes = append(report.Notes, "The model pack does not declare a config-composed sequence-mixer plan.")
			return report, nil
		}
		report.Plan = plan
		report.PlanningReady = true
		if missing := reactiveComposedStackMissingTensors(plan, tensorNames); len(missing) > 0 {
			report.Status = "incomplete"
			report.ExecutionStatus = "composed_stack_missing_runner_retired"
			report.MissingTensors = missing
			report.Labels["sequence_mixer_report_status"] = report.Status
			report.Labels["sequence_mixer_tensor_binding"] = "mixer_ready"
			report.Labels["sequence_mixer_composed_stack"] = "missing"
			report.Labels["sequence_mixer_composed_stack_missing"] = core.Join(",", missing...)
			report.Notes = append(report.Notes,
				"ROCm validated the sequence-mixer plan, but the full go-mlx composed block stack is incomplete; the composed session runner is retired (#50) regardless, so completing the stack would not by itself unlock execution.",
				"Missing composed tensors: "+core.Join(",", missing...),
			)
			return report, nil
		}
		report.Status = "runner_retired"
		report.ExecutionStatus = "composed_route_retired"
		report.TensorBindingReady = true
		report.ComposedStackReady = true
		report.Labels["sequence_mixer_report_status"] = report.Status
		report.Labels["sequence_mixer_tensor_binding"] = "ready"
		report.Labels["sequence_mixer_composed_stack"] = "ready"
		report.Notes = append(report.Notes,
			"ROCm validated the composed loader contract, but the portable incremental session runner is retired (#50) — loadHIPComposedTextModel declines this architecture outright; no native ROCm execution path serves this layer-mixer plan, factory-native port pending.",
		)
	case "invalid":
		report.Status = "invalid"
		report.ExecutionStatus = "load_plan_invalid"
		report.Labels["sequence_mixer_report_status"] = report.Status
		if detail := strings.TrimSpace(inspection.Labels["sequence_mixer_load_plan_error"]); detail != "" {
			report.Notes = append(report.Notes, detail)
		}
	default:
		report.Status = "not_declared"
		report.ExecutionStatus = "not_required"
		report.Labels["sequence_mixer_report_status"] = report.Status
		report.Notes = append(report.Notes, "The model pack does not declare a config-composed sequence-mixer plan.")
	}
	return report, nil
}

// baseReactiveSequenceMixerReport builds the report shell every PlanReactiveSequenceMixer outcome starts
// from. RunnerReady starts — and, as of #50, STAYS — false: every architecture this report can ever
// describe is, by construction, one whose config declares config-composed sequence-mixer layer types, and
// composed_runtime.go's loadHIPComposedTextModel now declines every such architecture outright (the
// "portable composed incremental runner" model/composed used to back this report is retired; nothing in
// this package replaced it). No branch below re-sets RunnerReady true; a future factory-native HIP port for
// one of these mixer kinds would need to flip it deliberately, not by an unexamined default.
func baseReactiveSequenceMixerReport(modelPath string, inspection *inference.ModelPackInspection) *ReactiveSequenceMixerReport {
	labels := map[string]string{
		"backend":                      "rocm",
		"cli_contract":                 ReactiveInferenceContract,
		"production_requires_env_gate": "false",
		"production_requires_cli_flag": "false",
		"sequence_mixer_report":        "true",
		"sequence_mixer_runner_status": "composed_route_retired",
	}
	var model inference.ModelIdentity
	if inspection != nil {
		model = inspection.Model
		maps.Copy(labels, inspection.Labels)
		if inspection.Supported {
			labels["model_pack_supported"] = "true"
		} else {
			labels["model_pack_supported"] = "false"
		}
	}
	return &ReactiveSequenceMixerReport{
		Version:         1,
		Kind:            "reactive-sequence-mixer-report",
		Backend:         "rocm",
		CLIContract:     ReactiveInferenceContract,
		ModelPath:       modelPath,
		Model:           model,
		Inspection:      inspection,
		Registry:        cloneSequenceMixerFamilies(DefaultSequenceMixerFamilies()),
		Status:          "unknown",
		ExecutionStatus: "unknown",
		RunnerReady:     false,
		Labels:          labels,
	}
}

func reactiveSequenceMixerLoadPlanAndTensorNames(modelPath string, inspection *inference.ModelPackInspection) (*SequenceMixerLoadPlan, []string, error) {
	weightPaths, err := rocmSafetensorsWeightFiles(modelPath)
	if err != nil {
		return nil, nil, err
	}
	var tensors []nativeTensorInfo
	for _, weightPath := range weightPaths {
		weightTensors, err := readROCmSafetensorsNativeTensors(weightPath)
		if err != nil {
			return nil, nil, err
		}
		tensors = append(tensors, weightTensors...)
	}
	names := make([]string, 0, len(tensors))
	for _, tensor := range tensors {
		names = append(names, tensor.Name)
	}
	plan, err := sequenceMixerLoadPlanFromInspection(inspection, tensors)
	return plan, names, err
}

func reactiveComposedStackMissingTensors(plan *SequenceMixerLoadPlan, tensorNames []string) []string {
	if plan == nil {
		return nil
	}
	nameSet := make(map[string]bool, len(tensorNames))
	for _, name := range tensorNames {
		nameSet[name] = true
	}
	required := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
	}
	for _, layer := range plan.Layers {
		prefix := core.Sprintf("model.layers.%d", layer.Layer)
		required = append(required,
			prefix+".input_layernorm.weight",
			prefix+".post_attention_layernorm.weight",
			prefix+".mlp.gate_proj.weight",
			prefix+".mlp.up_proj.weight",
			prefix+".mlp.down_proj.weight",
		)
	}
	missing := make([]string, 0)
	for _, name := range required {
		if HasResolvedDenseWeightName(nameSet, name) {
			continue
		}
		missing = append(missing, name)
	}
	return missing
}
