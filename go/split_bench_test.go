// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for split-inference plan primitives — preset expansion,
// custom-components compaction, plan validation, and the per-component
// HasComponent lookup. Per AX-11 — PlanModelSlice + ValidateSplitInferencePlan
// fire once per model load on a split-inference deployment; HasComponent
// runs in tight loops inside the planner and inside validation.
//
// Run:    go test -bench='BenchmarkSplit' -benchmem -run='^$' .

package inference

import (
	"testing"
)

// Sinks defeat compiler DCE.
var (
	splitBenchSinkPlan ModelSlicePlan
	splitBenchSinkErr  error
	splitBenchSinkBool bool
)

// benchSplitPlan returns a fully populated client-preset plan — reused
// across HasComponent + ValidateSplitInferencePlan benches.
func benchSplitPlan() ModelSlicePlan {
	plan, err := PlanModelSlice(ModelSliceRequest{
		Preset: ModelSlicePresetClient,
		Model: ModelIdentity{
			Path:         "/models/qwen3-4b",
			Architecture: "qwen3",
			QuantBits:    4,
			NumLayers:    28,
		},
		OutputPath: "/tmp/qwen3-client",
	})
	if err != nil {
		panic(err)
	}
	return plan
}

// --- PlanModelSlice — preset expansion (per-deployment plan path) ---

func BenchmarkSplit_PlanModelSlice_Full(b *testing.B) {
	req := ModelSliceRequest{Preset: ModelSlicePresetFull}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkPlan, splitBenchSinkErr = PlanModelSlice(req)
	}
}

func BenchmarkSplit_PlanModelSlice_Client(b *testing.B) {
	req := ModelSliceRequest{Preset: ModelSlicePresetClient}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkPlan, splitBenchSinkErr = PlanModelSlice(req)
	}
}

func BenchmarkSplit_PlanModelSlice_Server(b *testing.B) {
	req := ModelSliceRequest{Preset: ModelSlicePresetServer}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkPlan, splitBenchSinkErr = PlanModelSlice(req)
	}
}

func BenchmarkSplit_PlanModelSlice_Attention(b *testing.B) {
	req := ModelSliceRequest{Preset: ModelSlicePresetAttention}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkPlan, splitBenchSinkErr = PlanModelSlice(req)
	}
}

func BenchmarkSplit_PlanModelSlice_ExpertServer(b *testing.B) {
	req := ModelSliceRequest{Preset: ModelSlicePresetExpertServer}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkPlan, splitBenchSinkErr = PlanModelSlice(req)
	}
}

// Custom-components path — exercises compactModelComponents + labels clone.
func BenchmarkSplit_PlanModelSlice_Custom(b *testing.B) {
	req := ModelSliceRequest{
		Components: []ModelComponent{
			ModelComponentTokenizer,
			ModelComponentAttention,
			ModelComponentAttention, // duplicate — exercises seen-set
			ModelComponentEmbeddings,
			"", // empty — exercises skip branch
			ModelComponentLMHead,
		},
		Labels: map[string]string{
			"workload": "long_context",
			"profile":  "m3-ultra-96gb",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkPlan, splitBenchSinkErr = PlanModelSlice(req)
	}
}

// --- HasComponent — per-component lookup hot path ---

func BenchmarkSplit_HasComponent_FullPlan_Hit(b *testing.B) {
	plan, err := PlanModelSlice(ModelSliceRequest{Preset: ModelSlicePresetFull})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkBool = plan.HasComponent(ModelComponentExperts)
	}
}

func BenchmarkSplit_HasComponent_FullPlan_Miss(b *testing.B) {
	plan, err := PlanModelSlice(ModelSliceRequest{Preset: ModelSlicePresetServer})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkBool = plan.HasComponent(ModelComponentAttention)
	}
}

// --- ValidateSplitInferencePlan — pre-load validation pass ---

func BenchmarkSplit_ValidatePlan_Local(b *testing.B) {
	plan := SplitInferencePlan{
		Mode:       SplitInferenceModeLocal,
		LocalSlice: benchSplitPlan(),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkErr = ValidateSplitInferencePlan(plan)
	}
}

func BenchmarkSplit_ValidatePlan_RemoteFFN(b *testing.B) {
	plan := SplitInferencePlan{
		Mode:       SplitInferenceModeRemoteFFN,
		LocalSlice: benchSplitPlan(),
		Endpoints: []SplitEndpoint{
			{ID: "ffn-0", Role: SplitEndpointRoleFFN, URL: "http://127.0.0.1:8765", LayerStart: 0, LayerEnd: 28},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkErr = ValidateSplitInferencePlan(plan)
	}
}

func BenchmarkSplit_ValidatePlan_RemoteEmbedFFN(b *testing.B) {
	plan := SplitInferencePlan{
		Mode:       SplitInferenceModeRemoteEmbedFFN,
		LocalSlice: benchSplitPlan(),
		Endpoints: []SplitEndpoint{
			{ID: "embed-0", Role: SplitEndpointRoleEmbeddings, URL: "http://127.0.0.1:8761"},
			{ID: "ffn-0", Role: SplitEndpointRoleFFN, URL: "http://127.0.0.1:8765", LayerStart: 0, LayerEnd: 28},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkErr = ValidateSplitInferencePlan(plan)
	}
}

func BenchmarkSplit_ValidatePlan_RemoteExperts(b *testing.B) {
	plan := SplitInferencePlan{
		Mode:       SplitInferenceModeRemoteExperts,
		LocalSlice: benchSplitPlan(),
		Endpoints: []SplitEndpoint{
			{ID: "expert-0", Role: SplitEndpointRoleExpert, URL: "http://127.0.0.1:8770", ExpertStart: 0, ExpertEnd: 32},
			{ID: "expert-1", Role: SplitEndpointRoleExpert, URL: "http://127.0.0.1:8771", ExpertStart: 32, ExpertEnd: 64},
			{ID: "expert-2", Role: SplitEndpointRoleExpert, URL: "http://127.0.0.1:8772", ExpertStart: 64, ExpertEnd: 96},
			{ID: "expert-3", Role: SplitEndpointRoleExpert, URL: "http://127.0.0.1:8773", ExpertStart: 96, ExpertEnd: 128},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkErr = ValidateSplitInferencePlan(plan)
	}
}

// Negative path — missing required endpoint. Exercises the error-return
// fast path so it can be compared against the success cost.
func BenchmarkSplit_ValidatePlan_MissingEndpoint(b *testing.B) {
	plan := SplitInferencePlan{
		Mode:       SplitInferenceModeRemoteFFN,
		LocalSlice: benchSplitPlan(),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitBenchSinkErr = ValidateSplitInferencePlan(plan)
	}
}
