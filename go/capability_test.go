// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"testing"

	core "dappco.re/go"
)

type capabilityModel struct {
	*stubTextModel
	sink    ProbeSink
	adapter AdapterIdentity
}

func (m *capabilityModel) Encode(text string) []int32 {
	return []int32{int32(len(text))}
}

func (m *capabilityModel) Decode(ids []int32) string {
	return core.Sprintf("%d", len(ids))
}

func (m *capabilityModel) ApplyChatTemplate(messages []Message) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}
	return messages[0].Content, nil
}

func (m *capabilityModel) LoadAdapter(path string) (AdapterIdentity, error) {
	m.adapter = AdapterIdentity{Path: path, Format: "lora"}
	return m.adapter, nil
}

func (m *capabilityModel) UnloadAdapter() error {
	m.adapter = AdapterIdentity{}
	return nil
}

func (m *capabilityModel) ActiveAdapter() AdapterIdentity {
	return m.adapter
}

func (m *capabilityModel) CaptureState(context.Context, string, ...GenerateOption) (*StateBundle, error) {
	return &StateBundle{Model: ModelIdentity{Architecture: "stub"}}, nil
}

func (m *capabilityModel) RestoreState(context.Context, *StateBundle) error {
	return nil
}

func (m *capabilityModel) SetProbeSink(sink ProbeSink) {
	m.sink = sink
}

func (m *capabilityModel) Benchmark(context.Context, BenchConfig) (*BenchReport, error) {
	return &BenchReport{Model: ModelIdentity{Architecture: "stub"}}, nil
}

func (m *capabilityModel) PlanModelFit(context.Context, ModelIdentity, uint64) (*ModelFitReport, error) {
	return &ModelFitReport{Fits: true}, nil
}

func (m *capabilityModel) TrainSFT(context.Context, DatasetStream, TrainingConfig) (*TrainingResult, error) {
	return &TrainingResult{Adapter: AdapterIdentity{Format: "lora"}}, nil
}

func (m *capabilityModel) Distill(context.Context, DatasetStream, DistillConfig) (*TrainingResult, error) {
	return &TrainingResult{Model: ModelIdentity{Architecture: "student"}}, nil
}

func (m *capabilityModel) TrainGRPO(context.Context, DatasetStream, GRPOConfig) (*TrainingResult, error) {
	return &TrainingResult{Metrics: TrainingMetrics{Step: 1}}, nil
}

func (m *capabilityModel) Capabilities() CapabilityReport {
	return CapabilityReport{
		Runtime:   RuntimeIdentity{Backend: "stub", NativeRuntime: true},
		Available: true,
		Capabilities: []Capability{
			SupportedCapability(CapabilityGenerate, CapabilityGroupModel),
			ExperimentalCapability(CapabilityProbeEvents, CapabilityGroupProbe, "test sink"),
			PlannedCapability(CapabilityQuantization, CapabilityGroupRuntime, "not in stub"),
		},
	}
}

func TestCapabilityInterfaces(t *testing.T) {
	model := &capabilityModel{stubTextModel: &stubTextModel{}}

	_, ok := any(model).(TokenizerModel)
	checkTrue(t, ok)
	_, ok = any(model).(AdapterModel)
	checkTrue(t, ok)
	_, ok = any(model).(StatefulModel)
	checkTrue(t, ok)
	_, ok = any(model).(ProbeableModel)
	checkTrue(t, ok)
	_, ok = any(model).(BenchableModel)
	checkTrue(t, ok)
	_, ok = any(model).(ModelFitPlanner)
	checkTrue(t, ok)
	_, ok = any(model).(SFTTrainer)
	checkTrue(t, ok)
	_, ok = any(model).(DistillTrainer)
	checkTrue(t, ok)
	_, ok = any(model).(GRPOTrainer)
	checkTrue(t, ok)
	_, ok = any(model).(CapabilityReporter)
	checkTrue(t, ok)
}

func TestCapability_TokenizerModel_Good(t *testing.T) {
	model := &capabilityModel{}
	tokenizer := any(model).(TokenizerModel)

	ids := tokenizer.Encode("hello")
	text := tokenizer.Decode([]int32{1, 2, 3})
	prompt, err := tokenizer.ApplyChatTemplate([]Message{{Role: "user", Content: "hi"}})

	checkNoError(t, err)
	checkEqual(t, []int32{5}, ids)
	checkEqual(t, "3", text)
	checkEqual(t, "hi", prompt)
}

func TestCapability_AdapterModel_Good(t *testing.T) {
	model := &capabilityModel{}
	adapter := any(model).(AdapterModel)

	identity, err := adapter.LoadAdapter("/tmp/adapter.safetensors")
	checkNoError(t, err)
	checkEqual(t, "/tmp/adapter.safetensors", identity.Path)
	checkEqual(t, "lora", adapter.ActiveAdapter().Format)

	checkNoError(t, adapter.UnloadAdapter())
	checkEqual(t, AdapterIdentity{}, adapter.ActiveAdapter())
}

func TestCapability_StateAndProbe_Ugly_MinimalModel(t *testing.T) {
	model := &capabilityModel{}
	stateful := any(model).(StatefulModel)
	probeable := any(model).(ProbeableModel)

	bundle, err := stateful.CaptureState(context.Background(), "prompt")
	checkNoError(t, err)
	checkEqual(t, "stub", bundle.Model.Architecture)

	probeable.SetProbeSink(ProbeSinkFunc(func(ProbeEvent) {}))
	checkNotNil(t, model.sink)
}

func TestCapability_ReportHelpers_Good(t *testing.T) {
	report := CapabilityReport{
		Capabilities: []Capability{
			SupportedCapability(CapabilityGenerate, CapabilityGroupModel),
			ExperimentalCapability(CapabilityProbeEvents, CapabilityGroupProbe, "research telemetry"),
			PlannedCapability(CapabilityQuantization, CapabilityGroupRuntime, "future"),
			UnsupportedCapability(CapabilityGRPO, CapabilityGroupTraining, "stub"),
		},
	}

	checkTrue(t, report.Supports(CapabilityGenerate))
	checkTrue(t, report.Supports(CapabilityProbeEvents))
	checkFalse(t, report.Supports(CapabilityQuantization))
	checkFalse(t, report.Supports(CapabilityGRPO))
	checkEqual(t, []CapabilityID{CapabilityGenerate, CapabilityProbeEvents}, report.SupportedCapabilityIDs())
	checkEqual(t, []CapabilityID{CapabilityGenerate, CapabilityGRPO, CapabilityProbeEvents, CapabilityQuantization}, report.CapabilityIDs())
}

func TestCapability_CapabilityClone_Ugly(t *testing.T) {
	report := CapabilityReport{Capabilities: []Capability{{
		ID:     CapabilityGenerate,
		Group:  CapabilityGroupModel,
		Status: CapabilityStatusSupported,
		Labels: map[string]string{"backend": "stub"},
	}}}

	capability, ok := report.Capability(CapabilityGenerate)
	checkTrue(t, ok)
	capability.Labels["backend"] = "mutated"

	again, ok := report.Capability(CapabilityGenerate)
	checkTrue(t, ok)
	checkEqual(t, "stub", again.Labels["backend"])
}

func TestCapability_CapabilitiesOfReporter_Good(t *testing.T) {
	model := &capabilityModel{stubTextModel: &stubTextModel{}}

	report, ok := CapabilitiesOf(model)

	checkTrue(t, ok)
	checkTrue(t, report.Available)
	checkEqual(t, "stub", report.Runtime.Backend)
	checkTrue(t, report.Supports(CapabilityGenerate))
	checkTrue(t, report.Supports(CapabilityProbeEvents))
}

func TestCapability_TextModelCapabilities_Good(t *testing.T) {
	model := &capabilityModel{stubTextModel: &stubTextModel{}}

	report := TextModelCapabilities(RuntimeIdentity{Backend: "test"}, model)

	checkEqual(t, "test", report.Runtime.Backend)
	checkTrue(t, report.Supports(CapabilityGenerate))
	checkTrue(t, report.Supports(CapabilityTokenizer))
	checkTrue(t, report.Supports(CapabilityLoRAInference))
	checkTrue(t, report.Supports(CapabilityStateBundle))
	checkTrue(t, report.Supports(CapabilityBenchmark))
	checkTrue(t, report.Supports(CapabilityLoRATraining))
	checkTrue(t, report.Supports(CapabilityDistillation))
	checkTrue(t, report.Supports(CapabilityGRPO))
}

func TestCapability_BackendCapabilities_BadUnavailable(t *testing.T) {
	backend := &stubBackend{name: "gpu", available: false}

	report, ok := CapabilitiesOf(backend)

	checkTrue(t, ok)
	checkFalse(t, report.Available)
	checkEqual(t, "gpu", report.Runtime.Backend)
	checkTrue(t, report.Supports(CapabilityModelLoad))
}

func TestCapability_CapabilitiesOfUnknown_Ugly(t *testing.T) {
	report, ok := CapabilitiesOf(struct{}{})

	checkFalse(t, ok)
	checkEqual(t, CapabilityReport{}, report)
}

type memoryLimitBackend struct {
	stubBackend
	seen RuntimeMemoryLimits
}

func (backend *memoryLimitBackend) SetRuntimeMemoryLimits(limits RuntimeMemoryLimits) RuntimeMemoryLimits {
	backend.seen = limits
	limits.PreviousCacheLimitBytes = 128
	limits.PreviousMemoryLimitBytes = 256
	return limits
}

func TestCapability_SetRuntimeMemoryLimits_Good(t *testing.T) {
	resetBackends(t)
	backend := &memoryLimitBackend{stubBackend: stubBackend{name: "metal", available: true}}
	Register(backend)

	applied, ok := SetRuntimeMemoryLimits("metal", RuntimeMemoryLimits{CacheLimitBytes: 1024, MemoryLimitBytes: 2048})

	checkTrue(t, ok)
	checkEqual(t, uint64(1024), backend.seen.CacheLimitBytes)
	checkEqual(t, uint64(2048), backend.seen.MemoryLimitBytes)
	checkEqual(t, uint64(128), applied.PreviousCacheLimitBytes)
	checkEqual(t, uint64(256), applied.PreviousMemoryLimitBytes)
}

func TestCapability_SetRuntimeMemoryLimits_BadMissing(t *testing.T) {
	resetBackends(t)

	applied, ok := SetRuntimeMemoryLimits("metal", RuntimeMemoryLimits{CacheLimitBytes: 1024})

	checkFalse(t, ok)
	checkEqual(t, RuntimeMemoryLimits{}, applied)
}

func TestCapability_SetRuntimeMemoryLimits_UglyUnsupported(t *testing.T) {
	resetBackends(t)
	Register(&stubBackend{name: "plain", available: true})

	applied, ok := SetRuntimeMemoryLimits("plain", RuntimeMemoryLimits{CacheLimitBytes: 1024})

	checkFalse(t, ok)
	checkEqual(t, RuntimeMemoryLimits{}, applied)
}
