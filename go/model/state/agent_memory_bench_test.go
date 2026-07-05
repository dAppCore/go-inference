// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the agent-memory durable-state contracts.
// Per AX-11 — Ref / WakeRequest / SleepRequest fire on every session
// hand-off (wake at start, sleep at end, fork per branch). The struct
// surface itself is small but the Labels/StateRefs slices and maps
// are the per-call allocation floor; benching the construction path
// keeps the cost visible while the contracts are stable.
//
// Run:    go test -bench='Benchmark' -benchmem -run='^$' ./state

package state

import (
	"context"
	"testing"
)

// Sinks defeat compiler DCE. Distinct names per state-package bench file.
var (
	agentMemorySinkRef     Ref
	agentMemorySinkWake    WakeRequest
	agentMemorySinkSleep   SleepRequest
	agentMemorySinkSession Session
	agentMemorySinkWakeR   *WakeResult
	agentMemorySinkSleepR  *SleepResult
	agentMemorySinkErr     error
)

// --- Ref construction (the per-chunk envelope) ---

func BenchmarkAgentMemory_Ref_Construct_Minimal(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agentMemorySinkRef = Ref{
			URI:        "state://agents/cladius/seed",
			Kind:       "agent_memory",
			TokenStart: 0,
			TokenCount: 4096,
		}
	}
}

func BenchmarkAgentMemory_Ref_Construct_Labels_10(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		labels := make(map[string]string, 10)
		for j := 0; j < 10; j++ {
			labels[benchKey(j)] = benchValue(j)
		}
		agentMemorySinkRef = Ref{
			URI:    "state://agents/cladius/seed",
			Kind:   "agent_memory",
			Labels: labels,
		}
	}
}

func BenchmarkAgentMemory_Ref_Construct_Labels_100(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		labels := make(map[string]string, 100)
		for j := 0; j < 100; j++ {
			labels[benchKey(j)] = benchValue(j)
		}
		agentMemorySinkRef = Ref{
			URI:    "state://agents/cladius/seed",
			Kind:   "agent_memory",
			Labels: labels,
		}
	}
}

func BenchmarkAgentMemory_Ref_Construct_Labels_1000(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		labels := make(map[string]string, 1000)
		for j := 0; j < 1000; j++ {
			labels[benchKey(j)] = benchValue(j)
		}
		agentMemorySinkRef = Ref{
			URI:    "state://agents/cladius/seed",
			Kind:   "agent_memory",
			Labels: labels,
		}
	}
}

// --- StateRefs slice growth (per-bundle pointer list) ---

func BenchmarkAgentMemory_Ref_StateRefs_10(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refs := make([]StateRef, 0, 10)
		for j := 0; j < 10; j++ {
			refs = append(refs, StateRef{
				Kind:      "kv",
				URI:       "state://kv/block",
				SizeBytes: uint64(j * 1024),
			})
		}
		agentMemorySinkRef = Ref{StateRefs: refs}
	}
}

func BenchmarkAgentMemory_Ref_StateRefs_100(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refs := make([]StateRef, 0, 100)
		for j := 0; j < 100; j++ {
			refs = append(refs, StateRef{
				Kind:      "kv",
				URI:       "state://kv/block",
				SizeBytes: uint64(j * 1024),
			})
		}
		agentMemorySinkRef = Ref{StateRefs: refs}
	}
}

func BenchmarkAgentMemory_Ref_StateRefs_1000(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refs := make([]StateRef, 0, 1000)
		for j := 0; j < 1000; j++ {
			refs = append(refs, StateRef{
				Kind:      "kv",
				URI:       "state://kv/block",
				SizeBytes: uint64(j * 1024),
			})
		}
		agentMemorySinkRef = Ref{StateRefs: refs}
	}
}

// --- WakeRequest / SleepRequest construction (every session boundary) ---

func BenchmarkAgentMemory_WakeRequest_Build(b *testing.B) {
	model := ModelIdentity{ID: "gemma4", Hash: "model-a", NumLayers: 28}
	tok := TokenizerIdentity{Hash: "tok-a"}
	adapter := AdapterIdentity{Hash: "adapter-a", Rank: 8}
	runtime := RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agentMemorySinkWake = WakeRequest{
			IndexURI:  "state://lthn/projects/core/go-mlx/seed/index",
			EntryURI:  "state://lthn/projects/core/go-mlx/seed",
			Model:     model,
			Tokenizer: tok,
			Adapter:   adapter,
			Runtime:   runtime,
		}
	}
}

func BenchmarkAgentMemory_SleepRequest_Build(b *testing.B) {
	model := ModelIdentity{ID: "gemma4", Hash: "model-a", NumLayers: 28}
	tok := TokenizerIdentity{Hash: "tok-a"}
	adapter := AdapterIdentity{Hash: "adapter-a", Rank: 8}
	runtime := RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agentMemorySinkSleep = SleepRequest{
			EntryURI:          "state://lthn/projects/core/go-mlx/checkpoints/latest",
			BundleURI:         "state://lthn/projects/core/go-mlx/checkpoints/latest/bundle",
			IndexURI:          "state://lthn/projects/core/go-mlx/checkpoints/latest/index",
			ParentEntryURI:    "state://lthn/projects/core/go-mlx/seed",
			Model:             model,
			Tokenizer:         tok,
			Adapter:           adapter,
			Runtime:           runtime,
			ReuseParentPrefix: true,
			BlockSize:         512,
		}
	}
}

// --- Type-alias indirection (AgentMemory* = parent type) ---
// Confirms the alias adds zero cost vs the canonical type.

func BenchmarkAgentMemory_AliasRef_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agentMemorySinkRef = AgentMemoryRef{
			URI:        "state://agents/cladius/seed",
			Kind:       "agent_memory",
			TokenCount: 4096,
		}
	}
}

// --- Session/Forker invocation through the interface (per-fork cost) ---

func BenchmarkAgentMemory_Forker_ForkState(b *testing.B) {
	var forker Forker = benchForker{}
	req := WakeRequest{
		IndexURI: "state://index",
		Model:    ModelIdentity{ID: "tiny"},
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agentMemorySinkSession, agentMemorySinkWakeR, agentMemorySinkErr = forker.ForkState(ctx, req)
	}
}

func BenchmarkAgentMemory_Session_SleepState(b *testing.B) {
	var session Session = benchSession{}
	req := SleepRequest{EntryURI: "state://entry"}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		agentMemorySinkSleepR, agentMemorySinkErr = session.SleepState(ctx, req)
	}
}

// --- Bench helpers (kept local to this file to avoid cross-file overlap) ---

func benchKey(i int) string {
	// Fixed-shape keys keep the bench deterministic without touching
	// the production path; %d format is the same one core.Sprintf hits.
	switch i % 4 {
	case 0:
		return "scope"
	case 1:
		return "operator"
	case 2:
		return "branch"
	default:
		return "project_id"
	}
}

func benchValue(i int) string {
	switch i % 4 {
	case 0:
		return "repo"
	case 1:
		return "snider"
	case 2:
		return "dev"
	default:
		return "core/go-mlx"
	}
}

type benchForker struct{}

func (benchForker) ForkState(_ context.Context, req WakeRequest) (Session, *WakeResult, error) {
	return benchSession{}, &WakeResult{Entry: Ref{URI: req.IndexURI + "/entry"}, PrefixTokens: 12}, nil
}

type benchSession struct{}

func (benchSession) WakeState(_ context.Context, req WakeRequest) (*WakeResult, error) {
	return &WakeResult{Entry: Ref{URI: req.EntryURI}, PrefixTokens: 12}, nil
}

func (benchSession) SleepState(_ context.Context, req SleepRequest) (*SleepResult, error) {
	return &SleepResult{Entry: Ref{URI: req.EntryURI}, TokenCount: 12}, nil
}
