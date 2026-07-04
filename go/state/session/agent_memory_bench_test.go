// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for agent_memory.go — the session-side agent-memory wake/
// sleep lifecycle adapters. Per AX-11 — these helpers fire per turn
// (every Sleep and every WakeState/SleepState request goes through the
// metadata + label adapter path), so their alloc shape sets the per-turn
// floor for the inference contract layer.
//
// Run:    go test -bench='BenchmarkSessionAgent' -benchmem -run='^$' ./session

package session

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/state/agent"
)

// Sinks defeat compiler DCE.
var (
	sessionAgentBenchSinkBool      bool
	sessionAgentBenchSinkLabels    []string
	sessionAgentBenchSinkSleepOpts agent.SleepOptions
	sessionAgentBenchSinkWakeOpts  agent.WakeOptions
	sessionAgentBenchSinkInfMeta   map[string]string
	sessionAgentBenchSinkInfWake   inference.AgentMemorySleepResult
)

// --- shouldPrefillFoldedAgentMemory ---

// No folded marker — the dominant case. Token count makes PrefixTokens
// positive so we actually exercise the meta + label scans.
func BenchmarkSessionAgent_ShouldPrefill_NoMarker(b *testing.B) {
	entry := agent.StateIndexEntry{
		TokenCount: 4096,
		Meta:       map[string]string{"adapter_hash": "abc"},
		Labels:     []string{"env=prod", "agent=cladius"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkBool = shouldPrefillFoldedAgentMemory(entry)
	}
}

// Has folded_state=true marker — meta branch taken via canonical fast path.
func BenchmarkSessionAgent_ShouldPrefill_MetaTrue(b *testing.B) {
	entry := agent.StateIndexEntry{
		TokenCount: 4096,
		Meta:       map[string]string{"folded_state": "true"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkBool = shouldPrefillFoldedAgentMemory(entry)
	}
}

// Has folded-state label only — exercises the labels-loop fast path.
func BenchmarkSessionAgent_ShouldPrefill_LabelHit(b *testing.B) {
	entry := agent.StateIndexEntry{
		TokenCount: 4096,
		Labels:     []string{"env=prod", "folded-state"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkBool = shouldPrefillFoldedAgentMemory(entry)
	}
}

// --- agentMemoryLabelsFromInference ---

// Nil labels — fast path returns nil.
func BenchmarkSessionAgent_LabelsFromInf_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkLabels = agentMemoryLabelsFromInference(nil)
	}
}

// Three labels — common case.
func BenchmarkSessionAgent_LabelsFromInf_Three(b *testing.B) {
	in := map[string]string{
		"env":        "prod",
		"agent":      "cladius",
		"experiment": "",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkLabels = agentMemoryLabelsFromInference(in)
	}
}

// --- agentMemoryMetadataFromInference ---

// Empty req — all empty-fast-path branches.
func BenchmarkSessionAgent_MetadataFromInf_Empty(b *testing.B) {
	req := inference.AgentMemorySleepRequest{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkInfMeta = agentMemoryMetadataFromInference(req)
	}
}

// Realistic req with adapter + runtime — drives 9 addAgentMemoryMetadata.
// Worst-case all-fields-set; hint=9 forces the swissmap 4-alloc bucket
// layout. Common-case 8-or-fewer fields hits the 2-alloc compact layout
// (see BenchmarkSessionAgent_MetadataFromInf_Typical).
func BenchmarkSessionAgent_MetadataFromInf_Full(b *testing.B) {
	req := inference.AgentMemorySleepRequest{
		Adapter: inference.AdapterIdentity{
			Hash:   "abc123",
			Path:   "/models/lora.safetensors",
			Format: "safetensors",
			Rank:   16,
			Alpha:  32.0,
		},
		Runtime: inference.RuntimeIdentity{
			Backend:   "metal",
			Device:    "Apple M3 Ultra",
			CacheMode: "page",
			Version:   "0.42",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkInfMeta = agentMemoryMetadataFromInference(req)
	}
}

// Caller-supplied Metadata (3 custom keys) plus 7 standard fields —
// exercises the metadata-merge path which combines req.Metadata into
// the pre-sized destination map.
func BenchmarkSessionAgent_MetadataFromInf_WithMetadata(b *testing.B) {
	req := inference.AgentMemorySleepRequest{
		Adapter: inference.AdapterIdentity{
			Hash: "abc", Format: "safetensors", Rank: 16, Alpha: 32.0,
		},
		Runtime: inference.RuntimeIdentity{
			Backend: "metal", Device: "Apple M3 Ultra", Version: "0.42",
		},
		Metadata: map[string]string{
			"custom_a": "value-a",
			"custom_b": "value-b",
			"custom_c": "value-c",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkInfMeta = agentMemoryMetadataFromInference(req)
	}
}

// Typical req — most fields set, but CacheMode commonly empty (e.g. the
// metal backend uses its single default). 8 entries fit in the swissmap
// 2-alloc compact layout.
func BenchmarkSessionAgent_MetadataFromInf_Typical(b *testing.B) {
	req := inference.AgentMemorySleepRequest{
		Adapter: inference.AdapterIdentity{
			Hash:   "abc123",
			Path:   "/models/lora.safetensors",
			Format: "safetensors",
			Rank:   16,
			Alpha:  32.0,
		},
		Runtime: inference.RuntimeIdentity{
			Backend: "metal",
			Device:  "Apple M3 Ultra",
			Version: "0.42",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkInfMeta = agentMemoryMetadataFromInference(req)
	}
}

// --- agentMemorySleepOptionsFromInference ---

// Full req — drives both the metadata builder and the labels-from-inf
// path together; this is the per-turn cost.
func BenchmarkSessionAgent_SleepOptsFromInf(b *testing.B) {
	req := inference.AgentMemorySleepRequest{
		EntryURI: "state://entry",
		Adapter: inference.AdapterIdentity{
			Hash: "abc", Format: "safetensors", Rank: 16, Alpha: 32.0,
		},
		Runtime: inference.RuntimeIdentity{
			Backend: "metal", Device: "Apple M3 Ultra", Version: "0.42",
		},
		Labels: map[string]string{"agent": "cladius"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkSleepOpts = agentMemorySleepOptionsFromInference(req)
	}
}

// --- agentMemoryWakeOptionsFromInference ---

// Per-wake req-to-opts conversion. Mostly struct assembly + the
// NormaliseTokenizer call inside stateBundleTokenizerFromInference.
func BenchmarkSessionAgent_WakeOptsFromInf(b *testing.B) {
	req := inference.AgentMemoryWakeRequest{
		IndexURI:  "state://index",
		EntryURI:  "state://entry",
		Tokenizer: inference.TokenizerIdentity{Kind: "sentencepiece", Path: "/tokenizer.json"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkWakeOpts = WakeOptionsFromInference(req)
	}
}

// --- toInferenceAgentMemorySleepResult ---

// Hot-path result formatter — Sleep returns this on every call.
func BenchmarkSessionAgent_ToInfSleepResult(b *testing.B) {
	report := &agent.SleepReport{
		EntryURI:      "state://entry",
		BundleURI:     "state://bundle",
		IndexURI:      "state://index",
		Title:         "session-42",
		SnapshotHash:  "abc",
		IndexHash:     "def",
		TokenCount:    4096,
		BlockSize:     128,
		BlocksWritten: 32,
		BlocksReused:  4,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = toInferenceAgentMemorySleepResult(report)
	}
}
