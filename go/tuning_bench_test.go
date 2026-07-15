// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the tuning contract shapes — DefaultTuningWorkloads
// constructor, ScoreTuningMeasurements (per-result scoring), PlanModelReplace
// (per-model-swap state-reuse decision), CandidateID (per-candidate ID
// builder), and JSON marshal for the larger MachineDiscoveryReport / TuningPlan
// envelopes that the local-tuning UI fetches on every refresh. Per AX-11 —
// ScoreTuningMeasurements + CandidateID fire in tight loops during autotune;
// PlanModelReplace runs on every model swap; the report marshals are the
// wire format on every UI refresh.
//
// Run:    go test -bench='BenchmarkTuning' -benchmem -run='^$' .

package inference

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE. Distinct names from the other bench files.
var (
	tuningBenchSinkWorkloads []TuningWorkload
	tuningBenchSinkScore     TuningScore
	tuningBenchSinkPlan      ModelReplacePlan
	tuningBenchSinkID        string
	tuningBenchSinkString    string
)

// --- DefaultTuningWorkloads (constructor allocation cost) ---

func BenchmarkTuning_DefaultTuningWorkloads(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkWorkloads = DefaultTuningWorkloads()
	}
}

// --- ScoreTuningMeasurements — per-workload scoring switch ---

func BenchmarkTuning_ScoreMeasurements_Chat(b *testing.B) {
	m := TuningMeasurements{
		PrefillTokensPerSec: 900,
		DecodeTokensPerSec:  120,
		PeakMemoryBytes:     8 << 30,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkScore = ScoreTuningMeasurements(TuningWorkloadChat, m)
	}
}

func BenchmarkTuning_ScoreMeasurements_LongContext(b *testing.B) {
	m := TuningMeasurements{
		PrefillTokensPerSec: 1200,
		DecodeTokensPerSec:  45,
		PromptCacheHitRate:  0.8,
		PeakMemoryBytes:     12 << 30,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkScore = ScoreTuningMeasurements(TuningWorkloadLongContext, m)
	}
}

func BenchmarkTuning_ScoreMeasurements_AgentState(b *testing.B) {
	m := TuningMeasurements{
		PrefillTokensPerSec:     900,
		DecodeTokensPerSec:      120,
		PromptCacheHitRate:      0.75,
		KVRestoreMilliseconds:   4,
		StateBundleMilliseconds: 2,
		PeakMemoryBytes:         8 << 30,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkScore = ScoreTuningMeasurements(TuningWorkloadAgentState, m)
	}
}

func BenchmarkTuning_ScoreMeasurements_Throughput(b *testing.B) {
	m := TuningMeasurements{
		PrefillTokensPerSec: 2400,
		DecodeTokensPerSec:  220,
		PeakMemoryBytes:     16 << 30,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkScore = ScoreTuningMeasurements(TuningWorkloadThroughput, m)
	}
}

func BenchmarkTuning_ScoreMeasurements_LowLatency(b *testing.B) {
	m := TuningMeasurements{
		DecodeTokensPerSec:     80,
		FirstTokenMilliseconds: 20,
		TotalMilliseconds:      120,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkScore = ScoreTuningMeasurements(TuningWorkloadLowLatency, m)
	}
}

func BenchmarkTuning_ScoreMeasurements_Default(b *testing.B) {
	m := TuningMeasurements{
		PrefillTokensPerSec: 1100,
		DecodeTokensPerSec:  90,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Empty workload string falls to the default branch.
		tuningBenchSinkScore = ScoreTuningMeasurements(TuningWorkload(""), m)
	}
}

// --- PlanModelReplace — per-swap state-reuse decision ---

func BenchmarkTuning_PlanModelReplace_ReuseState(b *testing.B) {
	model := ModelIdentity{Path: "/models/qwen", Hash: "abc", Architecture: "qwen3", QuantBits: 4}
	runtime := RuntimeIdentity{Backend: "metal", CacheMode: "paged"}
	adapter := AdapterIdentity{Hash: "lora1"}
	req := ModelReplaceRequest{
		CurrentModel:   model,
		NextModel:      model,
		CurrentRuntime: runtime,
		NextRuntime:    runtime,
		CurrentAdapter: adapter,
		NextAdapter:    adapter,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkPlan = PlanModelReplace(req)
	}
}

func BenchmarkTuning_PlanModelReplace_CheckpointState(b *testing.B) {
	model := ModelIdentity{Path: "/models/qwen", Hash: "abc", Architecture: "qwen3", QuantBits: 4}
	adapter := AdapterIdentity{Hash: "lora1"}
	req := ModelReplaceRequest{
		CurrentModel:   model,
		NextModel:      model,
		CurrentRuntime: RuntimeIdentity{Backend: "metal", CacheMode: "paged"},
		NextRuntime:    RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"},
		CurrentAdapter: adapter,
		NextAdapter:    adapter,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkPlan = PlanModelReplace(req)
	}
}

func BenchmarkTuning_PlanModelReplace_SummaryWindow(b *testing.B) {
	current := ModelIdentity{Path: "/models/qwen", Hash: "abc", Architecture: "qwen3", QuantBits: 4}
	next := ModelIdentity{Path: "/models/gemma", Hash: "def", Architecture: "gemma4", QuantBits: 4}
	req := ModelReplaceRequest{
		CurrentModel:   current,
		NextModel:      next,
		CurrentRuntime: RuntimeIdentity{Backend: "metal", CacheMode: "paged"},
		NextRuntime:    RuntimeIdentity{Backend: "metal", CacheMode: "paged"},
		CurrentAdapter: AdapterIdentity{Hash: "lora1"},
		NextAdapter:    AdapterIdentity{Hash: "lora2"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkPlan = PlanModelReplace(req)
	}
}

// --- CandidateID — per-candidate stable ID builder ---

func BenchmarkTuning_CandidateID(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkID = CandidateID(TuningWorkloadLongContext, "paged-q8", 32768, 4)
	}
}

// --- JSON marshal — UI-facing report envelopes ---

func BenchmarkTuning_TuningCandidate_Marshal(b *testing.B) {
	candidate := TuningCandidate{
		ID:                   "long_context:paged-q8:ctx32768:batch4",
		Workload:             TuningWorkloadLongContext,
		Model:                ModelIdentity{Architecture: "qwen3", QuantBits: 4, ContextLength: 32768},
		Runtime:              RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"},
		ContextLength:        32768,
		ParallelSlots:        2,
		PromptCache:          true,
		PromptCacheMinTokens: 512,
		CachePolicy:          "lru",
		CacheMode:            "paged-q8",
		BatchSize:            4,
		PrefillChunkSize:     512,
		ExpectedQuantization: 4,
		MemoryLimitBytes:     16 << 30,
		CacheLimitBytes:      8 << 30,
		WiredLimitBytes:      4 << 30,
		Reasons:              []string{"context fits", "cache hit > 0.8"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkString = core.JSONMarshalString(candidate)
	}
}

func BenchmarkTuning_TuningResult_Marshal(b *testing.B) {
	result := TuningResult{
		Candidate: TuningCandidate{
			ID:            "long_context:paged-q8:ctx32768:batch4",
			Workload:      TuningWorkloadLongContext,
			Model:         ModelIdentity{Architecture: "qwen3", QuantBits: 4},
			ContextLength: 32768,
			BatchSize:     4,
		},
		Measurements: TuningMeasurements{
			PromptTokens:           2048,
			GeneratedTokens:        128,
			LoadMilliseconds:       1240,
			FirstTokenMilliseconds: 35,
			PrefillTokensPerSec:    1200,
			DecodeTokensPerSec:     45,
			PromptCacheHitRate:     0.81,
			KVRestoreMilliseconds:  12,
			TotalMilliseconds:      4200,
			PeakMemoryBytes:        12 << 30,
			ActiveMemoryBytes:      8 << 30,
		},
		Score: TuningScore{
			Workload:            TuningWorkloadLongContext,
			Score:               125.4,
			PrefillTokensPerSec: 1200,
			DecodeTokensPerSec:  45,
			PromptCacheHitRate:  0.81,
			PeakMemoryBytes:     12 << 30,
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkString = core.JSONMarshalString(result)
	}
}

func BenchmarkTuning_MachineDiscoveryReport_Marshal(b *testing.B) {
	report := MachineDiscoveryReport{
		Runtime: RuntimeIdentity{Backend: "metal", Device: "m3-ultra", Version: "0.10"},
		Device: MachineDeviceInfo{
			Name:                         "Apple M3 Ultra",
			Architecture:                 "arm64",
			MaxBufferLength:              64 << 30,
			MaxRecommendedWorkingSetSize: 80 << 30,
			MemorySize:                   96 << 30,
		},
		Available:  true,
		CacheModes: []string{"paged", "paged-q8", "paged-q4"},
		Models: []DiscoveredModel{
			{Path: "/models/qwen3-4b", ModelType: "qwen3", QuantBits: 4, NumFiles: 4, Format: "safetensors"},
			{Path: "/models/gemma3-1b", ModelType: "gemma3", QuantBits: 4, NumFiles: 1, Format: "safetensors"},
			{Path: "/models/llama3-8b", ModelType: "llama", QuantBits: 4, NumFiles: 4, Format: "safetensors"},
		},
		Workloads: DefaultTuningWorkloads(),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkString = core.JSONMarshalString(report)
	}
}

func BenchmarkTuning_TuningPlan_Marshal(b *testing.B) {
	plan := TuningPlan{
		Runtime: RuntimeIdentity{Backend: "metal", Device: "m3-ultra"},
		Model:   ModelIdentity{Architecture: "qwen3", QuantBits: 4},
		Workloads: []TuningWorkload{
			TuningWorkloadChat,
			TuningWorkloadLongContext,
			TuningWorkloadAgentState,
		},
		Candidates: []TuningCandidate{
			{ID: "chat:paged:ctx4096:batch1", Workload: TuningWorkloadChat, ContextLength: 4096, BatchSize: 1, CacheMode: "paged"},
			{ID: "long_context:paged-q8:ctx32768:batch4", Workload: TuningWorkloadLongContext, ContextLength: 32768, BatchSize: 4, CacheMode: "paged-q8"},
			{ID: "agent_state:paged:ctx8192:batch1", Workload: TuningWorkloadAgentState, ContextLength: 8192, BatchSize: 1, CacheMode: "paged"},
		},
		Recommended: map[TuningWorkload]string{
			TuningWorkloadChat:        "chat:paged:ctx4096:batch1",
			TuningWorkloadLongContext: "long_context:paged-q8:ctx32768:batch4",
			TuningWorkloadAgentState:  "agent_state:paged:ctx8192:batch1",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkString = core.JSONMarshalString(plan)
	}
}

func BenchmarkTuning_TuningEvent_Marshal(b *testing.B) {
	event := TuningEvent{
		Kind: TuningEventResult,
		Candidate: TuningCandidate{
			ID:       "long_context:paged-q8:ctx32768:batch4",
			Workload: TuningWorkloadLongContext,
			Model:    ModelIdentity{Architecture: "qwen3", QuantBits: 4},
		},
		Result: &TuningResult{
			Measurements: TuningMeasurements{
				PrefillTokensPerSec: 1200,
				DecodeTokensPerSec:  45,
			},
			Score: TuningScore{Workload: TuningWorkloadLongContext, Score: 125.4},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkString = core.JSONMarshalString(event)
	}
}

func BenchmarkTuning_TuningProfile_Marshal(b *testing.B) {
	profile := TuningProfile{
		Key: TuningProfileKey{
			MachineHash: "sha256-abcd-1234",
			Runtime:     RuntimeIdentity{Backend: "metal", Device: "m3-ultra"},
			Model:       ModelIdentity{Architecture: "qwen3", QuantBits: 4},
			Workload:    TuningWorkloadLongContext,
		},
		Candidate: TuningCandidate{
			ID:            "long_context:paged-q8:ctx32768:batch4",
			Workload:      TuningWorkloadLongContext,
			ContextLength: 32768,
			BatchSize:     4,
			CacheMode:     "paged-q8",
		},
		Measurements: TuningMeasurements{
			PrefillTokensPerSec: 1200,
			DecodeTokensPerSec:  45,
			PromptCacheHitRate:  0.81,
		},
		Score:         TuningScore{Workload: TuningWorkloadLongContext, Score: 125.4},
		CreatedAtUnix: 1700000000,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tuningBenchSinkString = core.JSONMarshalString(profile)
	}
}
