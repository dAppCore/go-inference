// SPDX-Licence-Identifier: EUPL-1.2

// Deeper benchmarks for the tuning contract shapes.
// Per AX-11 — the existing tuning_bench_test.go covers main paths.
// These benches drill into the CandidateID variants (workload + cache
// mode + context length combinations), sameModelIdentity / sameRuntime
// / sameAdapter shape variants (hash vs path vs identity-only), and
// PlanModelReplace edge cases (runtime-only change, adapter-only
// change, all-empty). All of these fire in tight loops during autotune.
//
// Run:    go test -bench='BenchmarkTuningDeep' -benchmem -run='^$' .

package inference

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE. Distinct names from the other bench files.
var (
	tuneDeepSinkID     string
	tuneDeepSinkPlan   ModelReplacePlan
	tuneDeepSinkScore  TuningScore
	tuneDeepSinkString string
)

// --- CandidateID variants ---
// CandidateID builds a deterministic ID from workload + cache mode +
// context length + batch size. The existing bench covers a single
// combination; these cover the surface area.

func BenchmarkTuningDeep_CandidateID_ShortFields(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkID = CandidateID(TuningWorkloadChat, "p", 256, 1)
	}
}

func BenchmarkTuningDeep_CandidateID_LongFields(b *testing.B) {
	// Long cache mode + large context — exercises strconv.AppendInt
	// on 6-digit numbers.
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkID = CandidateID(TuningWorkloadLongContext, "paged-q8-experimental", 131072, 32)
	}
}

func BenchmarkTuningDeep_CandidateID_AgentState(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkID = CandidateID(TuningWorkloadAgentState, "paged", 8192, 1)
	}
}

func BenchmarkTuningDeep_CandidateID_Throughput(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkID = CandidateID(TuningWorkloadThroughput, "paged-q4", 4096, 16)
	}
}

func BenchmarkTuningDeep_CandidateID_EmptyMode(b *testing.B) {
	// Empty cache mode — minimum-length string path.
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkID = CandidateID(TuningWorkloadLowLatency, "", 1024, 1)
	}
}

// --- PlanModelReplace edge cases ---
// The existing benches cover ReuseState / CheckpointState / SummaryWindow
// at the top of the matrix. These cover the inner shapes.

func BenchmarkTuningDeep_PlanModelReplace_RuntimeOnly(b *testing.B) {
	// Same model + same adapter, runtime differs only in cache mode.
	model := ModelIdentity{Hash: "abc", Architecture: "qwen3", QuantBits: 4}
	adapter := AdapterIdentity{Hash: "lora1"}
	req := ModelReplaceRequest{
		CurrentModel:   model,
		NextModel:      model,
		CurrentRuntime: RuntimeIdentity{Backend: "metal", CacheMode: "paged"},
		NextRuntime:    RuntimeIdentity{Backend: "metal", CacheMode: "paged-q4"},
		CurrentAdapter: adapter,
		NextAdapter:    adapter,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkPlan = PlanModelReplace(req)
	}
}

func BenchmarkTuningDeep_PlanModelReplace_AdapterOnly(b *testing.B) {
	// Same model + same runtime, adapter changed.
	model := ModelIdentity{Hash: "abc", Architecture: "qwen3", QuantBits: 4}
	runtime := RuntimeIdentity{Backend: "metal", CacheMode: "paged"}
	req := ModelReplaceRequest{
		CurrentModel:   model,
		NextModel:      model,
		CurrentRuntime: runtime,
		NextRuntime:    runtime,
		CurrentAdapter: AdapterIdentity{Hash: "lora1"},
		NextAdapter:    AdapterIdentity{Hash: "lora2"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkPlan = PlanModelReplace(req)
	}
}

func BenchmarkTuningDeep_PlanModelReplace_PathBasedModel(b *testing.B) {
	// Model identity by path (no hash). Exercises sameModelIdentity's
	// path-based branch — the Path+QuantBits+QuantType check.
	req := ModelReplaceRequest{
		CurrentModel:   ModelIdentity{Path: "/m/qwen", QuantBits: 4, QuantType: "q4_k_m"},
		NextModel:      ModelIdentity{Path: "/m/qwen", QuantBits: 4, QuantType: "q4_k_m"},
		CurrentRuntime: RuntimeIdentity{Backend: "metal"},
		NextRuntime:    RuntimeIdentity{Backend: "metal"},
		CurrentAdapter: AdapterIdentity{Path: "/a/lora1"},
		NextAdapter:    AdapterIdentity{Path: "/a/lora1"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkPlan = PlanModelReplace(req)
	}
}

func BenchmarkTuningDeep_PlanModelReplace_ArchitectureOnly(b *testing.B) {
	// No hash, no path — falls to architecture+quant+context comparison.
	req := ModelReplaceRequest{
		CurrentModel:   ModelIdentity{Architecture: "qwen3", QuantBits: 4, ContextLength: 4096},
		NextModel:      ModelIdentity{Architecture: "qwen3", QuantBits: 4, ContextLength: 4096},
		CurrentRuntime: RuntimeIdentity{Backend: "metal"},
		NextRuntime:    RuntimeIdentity{Backend: "metal"},
		CurrentAdapter: AdapterIdentity{},
		NextAdapter:    AdapterIdentity{},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkPlan = PlanModelReplace(req)
	}
}

func BenchmarkTuningDeep_PlanModelReplace_AllEmpty(b *testing.B) {
	// Empty identities — both sides "match" trivially (everything zero).
	req := ModelReplaceRequest{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkPlan = PlanModelReplace(req)
	}
}

// --- ScoreTuningMeasurements edge cases ---

func BenchmarkTuningDeep_Score_ZeroMeasurements(b *testing.B) {
	// All-zero measurements — the score should be 0 with no labels.
	m := TuningMeasurements{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkScore = ScoreTuningMeasurements(TuningWorkloadChat, m)
	}
}

func BenchmarkTuningDeep_Score_LongContext_NoCache(b *testing.B) {
	// PromptCacheHitRate = 0 — the cache-enabled-label branch is
	// skipped.
	m := TuningMeasurements{
		PrefillTokensPerSec: 800,
		DecodeTokensPerSec:  100,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkScore = ScoreTuningMeasurements(TuningWorkloadLongContext, m)
	}
}

func BenchmarkTuningDeep_Score_LowLatency_FirstTokenOnly(b *testing.B) {
	// FirstTokenMilliseconds set, TotalMilliseconds zero — only the
	// first-token branch fires.
	m := TuningMeasurements{
		FirstTokenMilliseconds: 25,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkScore = ScoreTuningMeasurements(TuningWorkloadLowLatency, m)
	}
}

func BenchmarkTuningDeep_Score_AgentState_NoStateBundle(b *testing.B) {
	// Only KVRestore set; StateBundle zero. Exercises the partial
	// state-restore branch without the bundle branch.
	m := TuningMeasurements{
		PrefillTokensPerSec:   800,
		DecodeTokensPerSec:    100,
		PromptCacheHitRate:    0.6,
		KVRestoreMilliseconds: 3,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkScore = ScoreTuningMeasurements(TuningWorkloadAgentState, m)
	}
}

// --- DefaultTuningWorkloads slice clone ---
// The existing bench measures the default constructor; this confirms
// the slice copy is cheap relative to other slice ops.

func BenchmarkTuningDeep_DefaultWorkloads_Append(b *testing.B) {
	base := DefaultTuningWorkloads()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Append one workload to the default — common shape for a
		// UI building a "+custom" list.
		clone := append([]TuningWorkload(nil), base...)
		clone = append(clone, TuningWorkload("custom"))
		_ = clone
	}
}

// --- MachineDeviceInfo JSON marshal ---
// Bench-light surface. Fires on every UI report refresh.

func BenchmarkTuningDeep_MachineDeviceInfo_Marshal(b *testing.B) {
	info := MachineDeviceInfo{
		Name:                         "Apple M3 Ultra",
		Architecture:                 "arm64",
		MaxBufferLength:              64 << 30,
		MaxRecommendedWorkingSetSize: 80 << 30,
		MemorySize:                   96 << 30,
		Labels: map[string]string{
			"chip":    "m3-ultra",
			"variant": "studio",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkString = core.JSONMarshalString(info)
	}
}

// --- TuningPlanRequest marshal ---

func BenchmarkTuningDeep_TuningPlanRequest_Marshal(b *testing.B) {
	req := TuningPlanRequest{
		Runtime: RuntimeIdentity{Backend: "metal", Device: "m3-ultra"},
		Device: MachineDeviceInfo{
			Name:                         "Apple M3 Ultra",
			Architecture:                 "arm64",
			MaxRecommendedWorkingSetSize: 80 << 30,
		},
		Model: ModelIdentity{Architecture: "qwen3", QuantBits: 4},
		Workloads: []TuningWorkload{
			TuningWorkloadChat,
			TuningWorkloadLongContext,
			TuningWorkloadAgentState,
		},
		Budget: TuningBudget{
			MaxCandidates:   8,
			SmokeTokens:     128,
			Runs:            3,
			AllowStateBench: true,
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkString = core.JSONMarshalString(req)
	}
}

// --- TuningProfileKey marshal ---
// Per-profile lookup key — fires on every cache hit during a model load.

func BenchmarkTuningDeep_TuningProfileKey_Marshal(b *testing.B) {
	key := TuningProfileKey{
		MachineHash: "sha256-abcd-1234-5678",
		Runtime:     RuntimeIdentity{Backend: "metal", Device: "m3-ultra"},
		Model:       ModelIdentity{Architecture: "qwen3", QuantBits: 4, ContextLength: 32768},
		Adapter:     AdapterIdentity{Hash: "lora1"},
		Workload:    TuningWorkloadAgentState,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuneDeepSinkString = core.JSONMarshalString(key)
	}
}
