// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the capability / report surface.
// Per AX-11 — every model load synthesises a CapabilityReport,
// every dispatcher does Supports(id) / Capability(id) lookups during
// routing decisions, and BackendCapabilities + TextModelCapabilities
// run once per Register() and once per LoadModel respectively. Even
// modest allocation cost compounds across the per-request cache check
// and the per-route capability scan.
//
// Run:    go test -bench=BenchmarkCapability -benchmem -run='^$' .

package inference

import (
	"testing"
)

// Sinks defeat compiler DCE.
var (
	capBenchSinkReport     CapabilityReport
	capBenchSinkCapability Capability
	capBenchSinkCapBool    bool
	capBenchSinkCapIDs     []CapabilityID
	capBenchSinkProfile    AlgorithmProfile
	capBenchSinkAnyOK      bool
)

// benchAlgorithmProfile builds a representative algorithm profile —
// the shape backends publish to expose their feature surface without
// leaking concrete runtime types.
func benchAlgorithmProfile() AlgorithmProfile {
	return AlgorithmProfile{
		ID:               CapabilityKVSnapshot,
		Group:            CapabilityGroupRuntime,
		CapabilityStatus: CapabilityStatusSupported,
		RuntimeStatus:    FeatureRuntimeNative,
		Algorithm:        "qwen3-paged-q8",
		Detail:           "native kv snapshot with paged q8 encoding",
		Architectures:    []string{"qwen3", "gemma3", "llama3"},
		Requires:         []CapabilityID{CapabilityModelLoad, CapabilityStateBundle},
		Provides:         []string{"snapshot", "resume", "fork"},
		Notes:            []string{"verified against gemma3-1b", "q8 only"},
	}
}

// benchCapabilityReport builds a CapabilityReport with the typical
// 8-12 capability entries a real text-model backend publishes. Used
// to exercise lookup + clone paths against realistic input shape.
func benchCapabilityReport() CapabilityReport {
	return CapabilityReport{
		Runtime:       RuntimeIdentity{Backend: "metal", Device: "M3 Ultra", NativeRuntime: true},
		Model:         ModelIdentity{Architecture: "qwen3", NumLayers: 28, QuantBits: 4},
		Tokenizer:     TokenizerIdentity{Kind: "sentencepiece", EOSID: 2},
		Adapter:       AdapterIdentity{Hash: "sha256:abc", Format: "lora", Rank: 16},
		Available:     true,
		Architectures: []string{"qwen3", "gemma3", "llama3"},
		Quantizations: []string{"q4_0", "q8_0", "f16"},
		CacheModes:    []string{"paged-q8", "paged-f16"},
		Capabilities: []Capability{
			SupportedCapability(CapabilityModelLoad, CapabilityGroupRuntime),
			SupportedCapability(CapabilityGenerate, CapabilityGroupModel),
			SupportedCapability(CapabilityChat, CapabilityGroupModel),
			SupportedCapability(CapabilityClassify, CapabilityGroupModel),
			SupportedCapability(CapabilityBatchGenerate, CapabilityGroupModel),
			SupportedCapability(CapabilityTokenizer, CapabilityGroupModel),
			SupportedCapability(CapabilityKVSnapshot, CapabilityGroupRuntime),
			ExperimentalCapability(CapabilityProbeEvents, CapabilityGroupProbe, "research telemetry"),
			PlannedCapability(CapabilityQuantization, CapabilityGroupRuntime, "future"),
			UnsupportedCapability(CapabilityGRPO, CapabilityGroupTraining, "no trainer"),
		},
		Labels: map[string]string{"profile": "qwen3-paged-q8"},
	}
}

// --- Constructors (per-Register / per-LoadModel cost) ---

func BenchmarkCapability_NewCapability(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapability = NewCapability(CapabilityGenerate, CapabilityGroupModel, CapabilityStatusSupported, "")
	}
}

func BenchmarkCapability_SupportedCapability(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapability = SupportedCapability(CapabilityGenerate, CapabilityGroupModel)
	}
}

func BenchmarkCapability_ExperimentalCapability(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapability = ExperimentalCapability(CapabilityProbeEvents, CapabilityGroupProbe, "telemetry")
	}
}

func BenchmarkCapability_PlannedCapability(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapability = PlannedCapability(CapabilityQuantization, CapabilityGroupRuntime, "future")
	}
}

func BenchmarkCapability_UnsupportedCapability(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapability = UnsupportedCapability(CapabilityGRPO, CapabilityGroupTraining, "no trainer")
	}
}

// --- Lookup hot path: Supports / Capability ---
// Dispatchers call these per request to decide which backend
// handles which surface. A 10-cap report scanned linearly is the
// floor we pay every routing decision.

func BenchmarkCapability_Supports_Hit(b *testing.B) {
	report := benchCapabilityReport()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapBool = report.Supports(CapabilityGenerate)
	}
}

func BenchmarkCapability_Supports_HitMiddle(b *testing.B) {
	// Middle of the 10-entry list — average linear-scan cost.
	report := benchCapabilityReport()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapBool = report.Supports(CapabilityKVSnapshot)
	}
}

func BenchmarkCapability_Supports_Miss(b *testing.B) {
	// Worst case — full scan with no match.
	report := benchCapabilityReport()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapBool = report.Supports(CapabilityMoELazyExperts)
	}
}

func BenchmarkCapability_Capability_Hit(b *testing.B) {
	report := benchCapabilityReport()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapability, capBenchSinkCapBool = report.Capability(CapabilityGenerate)
	}
}

func BenchmarkCapability_Capability_Miss(b *testing.B) {
	report := benchCapabilityReport()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapability, capBenchSinkCapBool = report.Capability(CapabilityMoELazyExperts)
	}
}

// --- ID-list helpers (typical request: "what does this backend do?") ---

func BenchmarkCapability_SupportedCapabilityIDs(b *testing.B) {
	report := benchCapabilityReport()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapIDs = report.SupportedCapabilityIDs()
	}
}

func BenchmarkCapability_CapabilityIDs(b *testing.B) {
	report := benchCapabilityReport()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapIDs = report.CapabilityIDs()
	}
}

// --- Usable (single-cap usability check, called per scan iteration) ---

func BenchmarkCapability_Usable_Supported(b *testing.B) {
	cap := SupportedCapability(CapabilityGenerate, CapabilityGroupModel)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapBool = cap.Usable()
	}
}

func BenchmarkCapability_Usable_Planned(b *testing.B) {
	cap := PlannedCapability(CapabilityQuantization, CapabilityGroupRuntime, "future")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapBool = cap.Usable()
	}
}

// --- AlgorithmProfile.Capability — profile → portable cap conversion ---
// Backends call this once per published algorithm during init.

func BenchmarkCapability_AlgorithmProfile_Capability(b *testing.B) {
	profile := benchAlgorithmProfile()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkCapability = profile.Capability()
	}
}

func BenchmarkCapability_CloneAlgorithmProfile(b *testing.B) {
	profile := benchAlgorithmProfile()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkProfile = CloneAlgorithmProfile(profile)
	}
}

// --- BackendCapabilities — per-Register inference floor ---

func BenchmarkCapability_BackendCapabilities_Plain(b *testing.B) {
	backend := &stubBackend{name: "stub", available: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkReport = BackendCapabilities(backend)
	}
}

func BenchmarkCapability_BackendCapabilities_Nil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkReport = BackendCapabilities(nil)
	}
}

// --- TextModelCapabilities — per-LoadModel inference floor ---
// The full optional-interface assertion ladder pays here.

func BenchmarkCapability_TextModelCapabilities_Plain(b *testing.B) {
	model := &stubTextModel{}
	runtime := RuntimeIdentity{Backend: "test"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkReport = TextModelCapabilities(runtime, model)
	}
}

func BenchmarkCapability_TextModelCapabilities_FullSurface(b *testing.B) {
	model := &capabilityModel{stubTextModel: &stubTextModel{}}
	runtime := RuntimeIdentity{Backend: "test"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkReport = TextModelCapabilities(runtime, model)
	}
}

func BenchmarkCapability_TextModelCapabilities_Nil(b *testing.B) {
	runtime := RuntimeIdentity{Backend: "test"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkReport = TextModelCapabilities(runtime, nil)
	}
}

// --- CapabilitiesOf — generic any-typed dispatch lookup ---

func BenchmarkCapability_CapabilitiesOf_Reporter(b *testing.B) {
	value := any(&capabilityModel{stubTextModel: &stubTextModel{}})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkReport, capBenchSinkAnyOK = CapabilitiesOf(value)
	}
}

func BenchmarkCapability_CapabilitiesOf_Backend(b *testing.B) {
	value := any(Backend(&stubBackend{name: "stub", available: true}))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkReport, capBenchSinkAnyOK = CapabilitiesOf(value)
	}
}

func BenchmarkCapability_CapabilitiesOf_TextModel(b *testing.B) {
	value := any(TextModel(&stubTextModel{}))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkReport, capBenchSinkAnyOK = CapabilitiesOf(value)
	}
}

func BenchmarkCapability_CapabilitiesOf_Unknown(b *testing.B) {
	value := any(struct{}{})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkReport, capBenchSinkAnyOK = CapabilitiesOf(value)
	}
}

func BenchmarkCapability_CapabilitiesOf_Nil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		capBenchSinkReport, capBenchSinkAnyOK = CapabilitiesOf(nil)
	}
}
