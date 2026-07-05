// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for artifact.Export — the .train file primitive.
// Per AX-11 — Export fires once per session-state snapshot we want to
// archive (every "save trace" call). The cost scales with the KV
// snapshot size: kv.Analyze + SAMIFromKV + JSON marshal + state.Put
// all run on every call. Multiple input sizes reveal whether the
// per-record overhead dominates or the analysis loop does.
//
// Run:    go test -bench=Benchmark -benchmem -run='^$' ./go/artifact

package artifact

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	state "dappco.re/go/inference/model/state"
)

// Sinks defeat compiler DCE.
var (
	artifactSinkRecord *Record
	artifactSinkErr    error
)

// benchSnapshot builds a representative kv.Snapshot — token count and
// layer/head shape sized to the qwen3-class range.
func benchSnapshot(tokenCount int) *kv.Snapshot {
	tokens := make([]int32, tokenCount)
	headKey := make([]float32, tokenCount)
	headValue := make([]float32, tokenCount)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
		headKey[i] = float32(i)
		headValue[i] = float32(i + 1000)
	}
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "qwen3",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        tokenCount,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []kv.LayerSnapshot{
			{Layer: 0, CacheIndex: 0, Heads: []kv.HeadSnapshot{{Key: headKey, Value: headValue}}},
			{Layer: 1, CacheIndex: 1, Heads: []kv.HeadSnapshot{{Key: headKey, Value: headValue}}},
		},
	}
}

// --- Export — analysis only (no Store, no KVPath) ---

func BenchmarkExport_AnalysisOnly_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		artifactSinkRecord, artifactSinkErr = Export(ctx, snap, Options{
			Model:  "lem-gemma",
			Prompt: "trace me",
		})
	}
}

func BenchmarkExport_AnalysisOnly_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		artifactSinkRecord, artifactSinkErr = Export(ctx, snap, Options{
			Model:  "lem-gemma",
			Prompt: "trace me",
		})
	}
}

// --- Export with precomputed analysis (skip the Analyze call) ---

func BenchmarkExport_PrecomputedAnalysis_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	analysis := kv.Analyze(snap)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		artifactSinkRecord, artifactSinkErr = Export(ctx, snap, Options{
			Model:    "lem-gemma",
			Prompt:   "trace me",
			Analysis: analysis,
		})
	}
}

// --- Export with KVPath (disk-write side effect) ---

func BenchmarkExport_KVPath_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	dir := b.TempDir()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		artifactSinkRecord, artifactSinkErr = Export(ctx, snap, Options{
			Model:  "lem-gemma",
			Prompt: "trace me",
			KVPath: core.JoinPath(dir, "state.kvbin"),
		})
	}
}

// --- Export with in-memory Store (the JSON-marshal + Put hot path) ---

func BenchmarkExport_StorePut_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		artifactSinkRecord, artifactSinkErr = Export(ctx, snap, Options{
			Model:  "lem-gemma",
			Prompt: "trace me",
			Store:  store,
			URI:    "mlx://session/trace",
			Tags:   map[string]string{"arch": "qwen3"},
		})
	}
}

func BenchmarkExport_StorePut_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		artifactSinkRecord, artifactSinkErr = Export(ctx, snap, Options{
			Model:  "lem-gemma",
			Prompt: "trace me",
			Store:  store,
			URI:    "mlx://session/trace",
		})
	}
}

// --- Full Export — KVPath + Store + Analysis (the canonical trace-save call) ---

func BenchmarkExport_Full_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	ctx := context.Background()
	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		artifactSinkRecord, artifactSinkErr = Export(ctx, snap, Options{
			Model:  "lem-gemma",
			Prompt: "full trace",
			KVPath: core.JoinPath(dir, "state.kvbin"),
			Store:  store,
			URI:    "mlx://session/trace",
			Title:  "trace",
			Tags:   map[string]string{"arch": "qwen3"},
			Labels: []string{"bench"},
		})
	}
}
