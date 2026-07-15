// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"context"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const runtimeLaneBackendPendingDetail = "runtime lane is compiled and packaged, but backend-specific runtime dispatch is pending"

type runtimeLaneBackend struct {
	backend string
}

func init() {
	registerRuntimeLaneBackends()
}

// RuntimeLaneBackends exposes the non-ROCm release lanes as registered,
// fail-closed backends so API consumers can negotiate the full artifact family.
func RuntimeLaneBackends() []inference.Backend {
	backends := []string{"cuda", "cpu"}
	out := make([]inference.Backend, 0, len(backends))
	for _, backend := range backends {
		if len(RuntimeLanesForBackend(backend)) == 0 {
			continue
		}
		out = append(out, &runtimeLaneBackend{backend: backend})
	}
	return out
}

func registerRuntimeLaneBackends() {
	for _, backend := range RuntimeLaneBackends() {
		inference.Register(backend)
	}
}

func (backend *runtimeLaneBackend) Name() string {
	if backend == nil {
		return ""
	}
	return backend.backend
}

func (*runtimeLaneBackend) Available() bool {
	return false
}

func (backend *runtimeLaneBackend) LoadModel(string, ...inference.LoadOption) core.Result {
	name := ""
	if backend != nil {
		name = backend.backend
	}
	return core.Fail(runtimeLaneBackendPendingError("LoadModel", name, RuntimeLanesForBackend(name)))
}

func (backend *runtimeLaneBackend) InspectModelPack(ctx context.Context, path string) (*inference.ModelPackInspection, error) {
	name := ""
	if backend != nil {
		name = backend.backend
	}
	lanes := RuntimeLanesForBackend(name)
	inspection, err := InspectModelPack(ctx, path)
	if err != nil {
		return nil, err
	}
	runtimeLaneBackendAnnotateInspection(inspection, runtimeLaneBackendLabels(name, lanes))
	return inspection, nil
}

func (backend *runtimeLaneBackend) PlanModelFit(ctx context.Context, model inference.ModelIdentity, memoryBytes uint64) (*inference.ModelFitReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	name := ""
	if backend != nil {
		name = backend.backend
	}
	labels := runtimeLaneBackendLabels(name, RuntimeLanesForBackend(name))
	model.Labels = runtimeLaneBackendMergeLabels(model.Labels, labels)

	planner, ok := any(&rocmBackend{}).(interface {
		PlanModelFit(context.Context, inference.ModelIdentity, uint64) (*inference.ModelFitReport, error)
	})
	if ok {
		report, err := planner.PlanModelFit(ctx, model, memoryBytes)
		if err != nil {
			return nil, err
		}
		runtimeLaneBackendAnnotateModelFit(report, labels)
		report.Notes = append(report.Notes, runtimeLaneBackendPendingDetail)
		return report, nil
	}

	return &inference.ModelFitReport{
		Model: model,
		MemoryPlan: inference.MemoryPlan{
			DeviceMemoryBytes: memoryBytes,
			Quantization:      model.QuantType,
			ContextLength:     model.ContextLength,
			Labels:            cloneStringMap(labels),
			Notes: []string{
				"backend-specific model-fit planner is pending for this runtime lane",
				runtimeLaneBackendPendingDetail,
			},
		},
		ArchitectureOK: model.Architecture != "",
		QuantizationOK: model.QuantBits > 0 || strings.TrimSpace(model.QuantType) != "",
		Notes: []string{
			"backend-specific model-fit planner is pending for this runtime lane",
			runtimeLaneBackendPendingDetail,
		},
	}, nil
}

func (backend *runtimeLaneBackend) Capabilities() inference.CapabilityReport {
	name := ""
	if backend != nil {
		name = backend.backend
	}
	lanes := RuntimeLanesForBackend(name)
	labels := runtimeLaneBackendLabels(name, lanes)
	capabilities := []inference.Capability{
		runtimeLaneBackendCapability(inference.SupportedCapability(
			inference.CapabilityRuntimeDiscovery,
			inference.CapabilityGroupRuntime,
		), labels),
		runtimeLaneBackendCapability(inference.SupportedCapability(
			inference.CapabilityModelFit,
			inference.CapabilityGroupRuntime,
		), labels),
		runtimeLaneBackendCapability(inference.SupportedCapability(
			inference.CapabilityMemoryPlanning,
			inference.CapabilityGroupRuntime,
		), labels),
		runtimeLaneBackendCapability(inference.SupportedCapability(
			inference.CapabilityKVCachePlanning,
			inference.CapabilityGroupRuntime,
		), labels),
		runtimeLaneBackendCapability(inference.PlannedCapability(
			inference.CapabilityModelLoad,
			inference.CapabilityGroupRuntime,
			runtimeLaneBackendPendingDetail,
		), labels),
		runtimeLaneBackendCapability(inference.PlannedCapability(
			inference.CapabilityGenerate,
			inference.CapabilityGroupModel,
			runtimeLaneBackendPendingDetail,
		), labels),
		runtimeLaneBackendCapability(inference.PlannedCapability(
			inference.CapabilityChat,
			inference.CapabilityGroupModel,
			runtimeLaneBackendPendingDetail,
		), labels),
		runtimeLaneBackendCapability(inference.PlannedCapability(
			inference.CapabilityBatchGenerate,
			inference.CapabilityGroupModel,
			runtimeLaneBackendPendingDetail,
		), labels),
		runtimeLaneBackendCapability(inference.PlannedCapability(
			inference.CapabilityResponsesAPI,
			inference.CapabilityGroupRuntime,
			"OpenAI-compatible serving is packaged through the CLI lane, but backend-specific runtime dispatch is pending",
		), labels),
		runtimeLaneBackendCapability(inference.PlannedCapability(
			inference.CapabilityStateBundle,
			inference.CapabilityGroupRuntime,
			"retained-state contracts are shared with ROCm, but backend-specific runtime state ownership is pending",
		), labels),
		runtimeLaneBackendCapability(inference.PlannedCapability(
			inference.CapabilityStateWake,
			inference.CapabilityGroupRuntime,
			"retained-state wake is shared with ROCm, but backend-specific runtime state ownership is pending",
		), labels),
		runtimeLaneBackendCapability(inference.PlannedCapability(
			inference.CapabilityStateSleep,
			inference.CapabilityGroupRuntime,
			"retained-state sleep is shared with ROCm, but backend-specific runtime state ownership is pending",
		), labels),
		runtimeLaneBackendCapability(inference.PlannedCapability(
			inference.CapabilityStateFork,
			inference.CapabilityGroupRuntime,
			"retained-state fork is shared with ROCm, but backend-specific runtime state ownership is pending",
		), labels),
	}
	return inference.CapabilityReport{
		Runtime: inference.RuntimeIdentity{
			Backend:       name,
			NativeRuntime: false,
			Labels:        cloneStringMap(labels),
		},
		Available:    false,
		Capabilities: capabilities,
		Labels:       cloneStringMap(labels),
	}
}

func runtimeLaneBackendCapability(capability inference.Capability, labels map[string]string) inference.Capability {
	capability.Labels = cloneStringMap(labels)
	return capability
}

func runtimeLaneBackendAnnotateInspection(inspection *inference.ModelPackInspection, labels map[string]string) {
	if inspection == nil {
		return
	}
	inspection.Labels = runtimeLaneBackendMergeLabels(inspection.Labels, labels)
	inspection.Model.Labels = runtimeLaneBackendMergeLabels(inspection.Model.Labels, labels)
	inspection.Tokenizer.Labels = runtimeLaneBackendMergeLabels(inspection.Tokenizer.Labels, labels)
	for index := range inspection.Capabilities {
		inspection.Capabilities[index].Labels = runtimeLaneBackendMergeLabels(inspection.Capabilities[index].Labels, labels)
	}
	inspection.Notes = append(inspection.Notes, runtimeLaneBackendPendingDetail)
}

func runtimeLaneBackendAnnotateModelFit(report *inference.ModelFitReport, labels map[string]string) {
	if report == nil {
		return
	}
	report.Model.Labels = runtimeLaneBackendMergeLabels(report.Model.Labels, labels)
	report.MemoryPlan.Labels = runtimeLaneBackendMergeLabels(report.MemoryPlan.Labels, labels)
	report.MemoryPlan.Notes = append(report.MemoryPlan.Notes, runtimeLaneBackendPendingDetail)
}

func runtimeLaneBackendMergeLabels(current, lane map[string]string) map[string]string {
	out := cloneStringMap(current)
	if out == nil {
		out = map[string]string{}
	}
	for key, value := range lane {
		if strings.TrimSpace(value) == "" {
			continue
		}
		out[key] = value
	}
	return out
}

func runtimeLaneBackendLabels(backend string, lanes []RuntimeLaneStatus) map[string]string {
	labels := map[string]string{
		"active_backend":                  "rocm",
		"backend":                         backend,
		"library":                         "go-rocm",
		"production_requires_cli_flag":    "false",
		"production_requires_env_gate":    "false",
		"runtime_dispatch_status":         RuntimeDispatchStatusCompileReadyPending,
		"runtime_status":                  "dispatch_pending",
		"runtime_lane_backend_registered": "true",
	}
	if len(lanes) == 0 {
		return labels
	}
	artifacts := make([]string, 0, len(lanes))
	kernelOutputs := make([]string, 0, len(lanes))
	platforms := make([]string, 0, len(lanes))
	runtimeLanes := make([]string, 0, len(lanes))
	sidecars := make([]string, 0, len(lanes))
	statuses := make([]string, 0, len(lanes))
	productionArtifact := false
	stubRuntime := false
	for _, lane := range lanes {
		lane = lane.Clone()
		artifacts = append(artifacts, lane.Artifact)
		kernelOutputs = append(kernelOutputs, lane.KernelOutput)
		platforms = append(platforms, lane.HIPPlatform)
		runtimeLanes = append(runtimeLanes, lane.RuntimeLane)
		sidecars = append(sidecars, lane.Sidecars...)
		statuses = append(statuses, lane.RuntimeDispatchStatus)
		productionArtifact = productionArtifact || lane.ProductionArtifact
		stubRuntime = stubRuntime || lane.StubRuntime
	}
	labels["artifacts"] = runtimeLaneBackendJoinUnique(artifacts)
	labels["hip_platform"] = runtimeLaneBackendJoinUnique(platforms)
	labels["kernel_output"] = runtimeLaneBackendJoinUnique(kernelOutputs)
	labels["production_artifact"] = boolRuntimeLaneLabel(productionArtifact)
	labels["runtime_lane"] = runtimeLaneBackendJoinUnique(runtimeLanes)
	labels["runtime_lanes"] = labels["runtime_lane"]
	labels["runtime_dispatch_status"] = runtimeLaneBackendJoinUnique(statuses)
	labels["runtime_dispatch_statuses"] = labels["runtime_dispatch_status"]
	labels["sidecars"] = runtimeLaneBackendJoinUnique(sidecars)
	labels["stub_runtime"] = boolRuntimeLaneLabel(stubRuntime)
	return labels
}

func runtimeLaneBackendJoinUnique(values []string) string {
	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return strings.Join(out, ",")
}

func runtimeLaneBackendPendingError(operation, backend string, lanes []RuntimeLaneStatus) error {
	labels := runtimeLaneBackendLabels(backend, lanes)
	runtimeLanes := labels["runtime_lanes"]
	if runtimeLanes == "" {
		runtimeLanes = backend
	}
	sidecars := labels["sidecars"]
	if sidecars == "" {
		sidecars = "no packaged sidecar"
	}
	status := labels["runtime_dispatch_status"]
	if status == "" {
		status = RuntimeDispatchStatusCompileReadyPending
	}
	return core.E(
		core.Sprintf("rocm %s %s", backend, operation),
		core.Sprintf("%s runtime lane is compiled and packaged for %s (%s), but runtime dispatch is pending (runtime_dispatch_status=%s); use the rocm backend or lthn-amd until %s runtime dispatch is registered", backend, runtimeLanes, sidecars, status, backend),
		nil,
	)
}
