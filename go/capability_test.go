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

// --- AlgorithmProfile -> Capability (arch-neutral) ---------------------------

// TestCapability_AlgorithmProfile_Capability_Good pins the profile->portable
// conversion AND its arch-neutrality: a backend declares which architectures an
// algorithm serves through AlgorithmProfile.Architectures, and the portable
// Capability must carry that declaration verbatim in its labels — no gemma
// assumption, whatever the backend lists (here a non-gemma qwen3-moe/llama pair).
func TestCapability_AlgorithmProfile_Capability_Good(t *testing.T) {
	profile := AlgorithmProfile{
		ID:               CapabilityMoERouting,
		Group:            CapabilityGroupModel,
		CapabilityStatus: CapabilityStatusExperimental,
		RuntimeStatus:    FeatureRuntimeMetadataOnly,
		Algorithm:        "top-k-router",
		Detail:           "expert routing",
		Architectures:    []string{"qwen3-moe", "llama"},
		Requires:         []CapabilityID{CapabilityTokenizer, CapabilityChatTemplate},
		Provides:         []string{"router-logits"},
	}

	capability := profile.Capability()
	if capability.ID != CapabilityMoERouting || capability.Status != CapabilityStatusExperimental {
		t.Fatalf("capability head = %+v, want id=moe.routing status=experimental", capability)
	}
	if got := capability.Labels["architectures"]; got != "qwen3-moe,llama" {
		t.Fatalf("architectures label = %q, want the declared non-gemma pair verbatim", got)
	}
	if got := capability.Labels["runtime_status"]; got != string(FeatureRuntimeMetadataOnly) {
		t.Fatalf("runtime_status label = %q, want metadata_only", got)
	}
	if got := capability.Labels["algorithm"]; got != "top-k-router" {
		t.Fatalf("algorithm label = %q, want top-k-router", got)
	}
	if got := capability.Labels["requires"]; got != "tokenizer,chat.template" {
		t.Fatalf("requires label = %q, want the joined capability ids", got)
	}
	if got := capability.Labels["provides"]; got != "router-logits" {
		t.Fatalf("provides label = %q, want router-logits", got)
	}
}

// TestCapability_AlgorithmProfile_Capability_Bad pins the sparse-profile arm: a
// profile with no algorithm / architectures / requires / provides emits only the
// mandatory runtime_status label, so a bare declaration stays minimal.
func TestCapability_AlgorithmProfile_Capability_Bad(t *testing.T) {
	capability := AlgorithmProfile{
		ID:            CapabilityGenerate,
		Group:         CapabilityGroupModel,
		RuntimeStatus: FeatureRuntimeNative,
	}.Capability()
	if len(capability.Labels) != 1 {
		t.Fatalf("sparse profile labels = %v, want only runtime_status", capability.Labels)
	}
	if capability.Labels["runtime_status"] != string(FeatureRuntimeNative) {
		t.Fatalf("runtime_status = %q, want native", capability.Labels["runtime_status"])
	}
}

// TestCapability_CloneAlgorithmProfile_Ugly pins the deep copy: mutating any
// slice of the clone must not reach back into the original's backing arrays.
func TestCapability_CloneAlgorithmProfile_Ugly(t *testing.T) {
	original := AlgorithmProfile{
		ID:            CapabilitySplitInference,
		Architectures: []string{"qwen3"},
		Requires:      []CapabilityID{CapabilityModelLoad},
		Provides:      []string{"shard-plan"},
		Notes:         []string{"n"},
	}
	clone := CloneAlgorithmProfile(original)
	clone.Architectures[0] = "MUTATED"
	clone.Requires[0] = "MUTATED"
	clone.Provides[0] = "MUTATED"
	clone.Notes[0] = "MUTATED"
	if original.Architectures[0] != "qwen3" || original.Requires[0] != CapabilityModelLoad ||
		original.Provides[0] != "shard-plan" || original.Notes[0] != "n" {
		t.Fatalf("CloneAlgorithmProfile shares backing arrays: original = %+v", original)
	}
}

// TestCapability_CapabilityIDLabel_Good pins the id-join helper directly,
// including the empty-slice edge (no ids -> empty label).
func TestCapability_CapabilityIDLabel_Good(t *testing.T) {
	if got := capabilityIDLabel([]CapabilityID{CapabilityChat, CapabilityGenerate}); got != "chat,generate" {
		t.Fatalf("capabilityIDLabel = %q, want chat,generate", got)
	}
	if got := capabilityIDLabel(nil); got != "" {
		t.Fatalf("capabilityIDLabel(nil) = %q, want empty", got)
	}
}

// --- report / CapabilitiesOf / BackendCapabilities edges ---------------------

// TestCapability_Report_Capability_NotFound_Bad pins the miss arm: a report
// that lacks an id returns the zero capability and false.
func TestCapability_Report_Capability_NotFound_Bad(t *testing.T) {
	report := CapabilityReport{Capabilities: []Capability{SupportedCapability(CapabilityGenerate, CapabilityGroupModel)}}
	got, ok := report.Capability(CapabilityGRPO)
	if ok || got.ID != "" {
		t.Fatalf("Capability(absent) = (%+v, %v), want (zero, false)", got, ok)
	}
}

// TestCapability_CapabilitiesOf_Nil_Bad pins the nil-value arm: no report, false.
func TestCapability_CapabilitiesOf_Nil_Bad(t *testing.T) {
	if report, ok := CapabilitiesOf(nil); ok || report.Available {
		t.Fatalf("CapabilitiesOf(nil) = (%+v, %v), want (empty, false)", report, ok)
	}
}

// TestCapability_CapabilitiesOf_PlainTextModel_Good pins the TextModel inference
// arm: a plain model that does NOT implement CapabilityReporter still yields an
// inferred report through the TextModel case.
func TestCapability_CapabilitiesOf_PlainTextModel_Good(t *testing.T) {
	report, ok := CapabilitiesOf(&stubTextModel{backend: "plain"})
	if !ok || !report.Available {
		t.Fatalf("CapabilitiesOf(plain TextModel) = (%+v, %v), want an inferred available report", report, ok)
	}
	if !report.Supports(CapabilityGenerate) {
		t.Fatal("inferred TextModel report must support generate")
	}
}

// TestCapability_BackendCapabilities_Nil_Bad pins the nil-backend guard: an empty
// report, no panic.
func TestCapability_BackendCapabilities_Nil_Bad(t *testing.T) {
	if report := BackendCapabilities(nil); report.Available || len(report.Capabilities) != 0 {
		t.Fatalf("BackendCapabilities(nil) = %+v, want an empty report", report)
	}
}

// fitBackend is a Backend that also plans model fit, so BackendCapabilities
// reports the CapabilityModelFit branch.
type fitBackend struct{ stubBackend }

func (*fitBackend) PlanModelFit(context.Context, ModelIdentity, uint64) (*ModelFitReport, error) {
	return &ModelFitReport{Fits: true}, nil
}

// TestCapability_BackendCapabilities_ModelFit_Good pins the ModelFitPlanner
// branch: a backend that plans fit advertises model.fit alongside model.load.
func TestCapability_BackendCapabilities_ModelFit_Good(t *testing.T) {
	report := BackendCapabilities(&fitBackend{stubBackend{name: "gpu", available: true}})
	if !report.Supports(CapabilityModelLoad) || !report.Supports(CapabilityModelFit) {
		t.Fatalf("fit backend report = %+v, want model.load + model.fit", report.CapabilityIDs())
	}
}

// TestCapability_TextModelCapabilities_Nil_Bad pins the nil-model guard: a
// runtime-only report with no capabilities.
func TestCapability_TextModelCapabilities_Nil_Bad(t *testing.T) {
	report := TextModelCapabilities(RuntimeIdentity{Backend: "metal"}, nil)
	if report.Runtime.Backend != "metal" || report.Available || len(report.Capabilities) != 0 {
		t.Fatalf("TextModelCapabilities(nil model) = %+v, want a runtime-only empty report", report)
	}
}

// probeArchModel is a TextModel that implements AttentionInspector + Evaluator
// AND reports a NON-gemma architecture, so TextModelCapabilities exercises those
// two optional branches and the arch passes through the capability layer
// unaltered.
type probeArchModel struct {
	*stubTextModel
	arch string
}

func (m probeArchModel) Info() ModelInfo { return ModelInfo{Architecture: m.arch, NumLayers: 32} }
func (probeArchModel) InspectAttention(context.Context, string, ...GenerateOption) (*AttentionSnapshot, error) {
	return &AttentionSnapshot{}, nil
}
func (probeArchModel) Evaluate(context.Context, DatasetStream, EvalConfig) (*EvalReport, error) {
	return &EvalReport{}, nil
}

// TestCapability_TextModelCapabilities_ProbeSurface_Good pins the
// AttentionInspector + Evaluator branches and arch-neutrality together: a model
// declaring a non-gemma architecture surfaces probe.attention + evaluation, and
// the report's Model.Architecture is the model's own declaration, not a default.
func TestCapability_TextModelCapabilities_ProbeSurface_Good(t *testing.T) {
	model := probeArchModel{stubTextModel: &stubTextModel{}, arch: "qwen3-moe"}
	report := TextModelCapabilities(RuntimeIdentity{Backend: "metal"}, model)
	if report.Model.Architecture != "qwen3-moe" {
		t.Fatalf("report Model.Architecture = %q, want the model's declared qwen3-moe", report.Model.Architecture)
	}
	if !report.Supports(CapabilityAttentionProbe) {
		t.Fatal("AttentionInspector model must advertise probe.attention")
	}
	if !report.Supports(CapabilityEvaluation) {
		t.Fatal("Evaluator model must advertise evaluation")
	}
}
