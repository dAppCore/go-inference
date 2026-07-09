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

func TestCapability_CapabilitiesOf_Good(t *testing.T) {
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

func TestCapability_CapabilitiesOf_Ugly(t *testing.T) {
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

// AX-11: alloc + behavioural lock for TextModelCapabilities on a
// model implementing every optional capability interface. Mirrors
// BenchmarkCapability_TextModelCapabilities_FullSurface — every
// backend pays this once per Load() when reporting its surface to
// the dispatcher, so a regression here ripples through every
// consumer (go-mlx, go-rocm, go-cuda).
//
// Baselines (Apple M3 Ultra, -benchmem):
//
//	pre-presize  (literal-4 + append × N grows): 3 allocs / 3479ns / 2208B
//	post-presize (make([], 0, 28) once):         1 alloc  /  403ns / 2048B
//
// Trade-off: pre-sized slice is ~1.7KB larger per call on the
// "no-optional-interfaces" path (Plain) because we always allocate
// for the upper bound. Acceptable because (a) model load is one-shot
// per backend per app session, and (b) the alloc-count drop +
// 8x speedup matters far more than the bytes delta at this scale.
//
// Twin assertions:
//  1. ALLOCS — stays at 1 (the single pre-sized backing slice)
//  2. BEHAVIOUR — the reported capability set matches expectations
//     for the full-surface model fixture
func TestCapability_AllocBudget_TextModelCapabilities_FullSurface(t *testing.T) {
	model := &capabilityModel{stubTextModel: &stubTextModel{}}
	runtime := RuntimeIdentity{Backend: "test"}

	// Behavioural lock — output must contain the expected capabilities.
	// Spot-check that optional interfaces were detected; full coverage
	// lives in TestCapability_CapabilitiesOf_TextModel.
	report := TextModelCapabilities(runtime, model)
	if !report.Available {
		t.Fatalf("expected report.Available=true for FullSurface model")
	}
	// The capabilityModel fixture implements the optional interfaces
	// the test suite covers — exact count is the contract. If the
	// fixture grows to cover new interface branches, bump both this
	// number AND maxTextModelCapabilities together so the alloc gate
	// stays at 1 (single backing slice).
	const expectedCapabilities = 14
	if got := len(report.Capabilities); got != expectedCapabilities {
		t.Fatalf("FullSurface capability count drifted: expected %d, got %d", expectedCapabilities, got)
	}

	// Alloc-budget lock. Bump maxTextModelCapabilities in capability.go
	// AND this comment if new optional-interface branches land.
	avg := testing.AllocsPerRun(5, func() {
		_ = TextModelCapabilities(runtime, model)
	})
	const budget = 2.0 // current measured: 1
	if avg > budget {
		t.Fatalf("TextModelCapabilities alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Every backend pays this per Load() when reporting capabilities.\n"+
			"If this jumped because a new optional-interface branch was added, "+
			"bump maxTextModelCapabilities in capability.go to match.",
			avg, budget)
	}
}
