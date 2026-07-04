// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for KV snapshot save/load + analysis primitives.
// Per AX-11 — Snapshot.Save fires per generation step (checkpointing);
// LoadWithOptions fires per session resume; Analyze runs on every
// resumed snapshot. The binary encoder (bytes / writeWithOptions)
// is the inner loop both Save and SaveStateBlocks hit.
//
// Run:    go test -bench='BenchmarkSnapshot|BenchmarkAnalyze|BenchmarkHash' -benchmem -run='^$' ./go/kv

package kv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/state"
)

// Sinks defeat compiler DCE.
var (
	benchSinkSnapshot *Snapshot
	benchSinkBytes    []byte
	benchSinkErr      error
	benchSinkString   string
	benchSinkAnalysis *Analysis
	benchSinkRef      state.ChunkRef
)

// benchSnapshot builds a representative snapshot — token count and
// layer/head shape sized to the qwen3-class range. Same fixture
// helper as the existing block-loading benches but exposed at file
// scope so the new save/load benches can share it.
func benchSnapshot(tokenCount int) *Snapshot {
	tokens := make([]int32, tokenCount)
	fullKey := make([]float32, tokenCount)
	fullValue := make([]float32, tokenCount)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
		fullKey[i] = float32(i)
		fullValue[i] = float32(i + 1000)
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "qwen3",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        tokenCount,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{
			{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{{Key: fullKey, Value: fullValue}}},
			{Layer: 1, CacheIndex: 1, Heads: []HeadSnapshot{{Key: fullKey, Value: fullValue}}},
		},
	}
}

// --- Save / SaveWithOptions ---

func BenchmarkAnalyze_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkAnalysis = Analyze(snap)
	}
}

func BenchmarkAnalyze_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkAnalysis = Analyze(snap)
	}
}

// benchGQAHeadDimSnapshot builds a GQA (numHeads≤4) snapshot with
// headDim > 1 so the analyzeKVGQA → kvAnalysisPositionDifferentiation
// general path (not the headDim=1 specialisation) gets exercised.
// Real qwen3 GQA layers carry headDim 64-128; the headDim=1 fixture
// the suite ships with skips the inner-k-loop entirely. seqLen is
// kept modest because the path is O(seqLen²·headDim).
func benchGQAHeadDimSnapshot(seqLen, headDim int) *Snapshot {
	tokens := make([]int32, seqLen)
	key := make([]float32, seqLen*headDim)
	value := make([]float32, seqLen*headDim)
	for pos := range seqLen {
		tokens[pos] = int32(pos + 1)
		for k := range headDim {
			// Vary across both position and dim so the inner dot is
			// non-trivial (not orthogonal, not identical).
			key[pos*headDim+k] = float32(pos+1) * float32(k+1) * 0.01
			value[pos*headDim+k] = float32(pos+2) * float32(k+1) * 0.01
		}
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "qwen3",
		Tokens:        tokens,
		TokenOffset:   seqLen,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        seqLen,
		HeadDim:       headDim,
		NumQueryHeads: 8,
		Layers: []LayerSnapshot{
			{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{{Key: key, Value: value}}},
			{Layer: 1, CacheIndex: 1, Heads: []HeadSnapshot{{Key: key, Value: value}}},
		},
	}
}

func BenchmarkAnalyze_GQA_256Tokens_64HeadDim(b *testing.B) {
	snap := benchGQAHeadDimSnapshot(256, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkAnalysis = Analyze(snap)
	}
}

func BenchmarkAnalyze_GQA_512Tokens_64HeadDim(b *testing.B) {
	snap := benchGQAHeadDimSnapshot(512, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkAnalysis = Analyze(snap)
	}
}

// benchMultiHeadSnapshot builds a numHeads>4 snapshot so Analyze
// routes through analyzeKVMultiHead → kvAnalysisPairCoherence instead
// of the GQA path. Shape mirrors a qwen3-class layer slice with 8
// heads × 64 headDim — the per-pair inner dot is realistic, not the
// headDim=1 degenerate the GQA benches use.
func benchMultiHeadSnapshot(tokenCount, numHeads, headDim int) *Snapshot {
	tokens := make([]int32, tokenCount)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
	}
	layers := make([]LayerSnapshot, 2)
	for layer := range layers {
		heads := make([]HeadSnapshot, numHeads)
		for h := range heads {
			key := make([]float32, tokenCount*headDim)
			value := make([]float32, tokenCount*headDim)
			for pos := range tokenCount {
				key[pos*headDim+h%headDim] = 1
				value[pos*headDim+(numHeads-h-1)%headDim] = 1
			}
			heads[h] = HeadSnapshot{Key: key, Value: value}
		}
		layers[layer] = LayerSnapshot{Layer: layer, CacheIndex: layer, Heads: heads}
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "qwen3",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     2,
		NumHeads:      numHeads,
		SeqLen:        tokenCount,
		HeadDim:       headDim,
		NumQueryHeads: numHeads,
		Layers:        layers,
	}
}

func BenchmarkAnalyze_MultiHead_512Tokens_8Heads_64HeadDim(b *testing.B) {
	snap := benchMultiHeadSnapshot(512, 8, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkAnalysis = Analyze(snap)
	}
}

func BenchmarkAnalyze_MultiHead_2048Tokens_8Heads_64HeadDim(b *testing.B) {
	snap := benchMultiHeadSnapshot(2048, 8, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkAnalysis = Analyze(snap)
	}
}

// --- HashSnapshot ---

func BenchmarkHashSnapshot_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString, benchSinkErr = HashSnapshot(snap)
	}
}

// --- SaveStateBlocks (the chunked-write path the existing
// block-load benches resolve from) ---

func BenchmarkSnapshot_SaveStateBlocks_3Blocks(b *testing.B) {
	store := state.NewInMemoryStore(nil)
	snap := benchSnapshot(1536) // 3 × 512-block
	opts := StateBlockOptions{BlockSize: 512, KVEncoding: EncodingNative}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundle, err := snap.SaveStateBlocks(ctx, store, opts)
		benchSinkErr = err
		if bundle != nil && len(bundle.Blocks) > 0 {
			benchSinkRef = bundle.Blocks[0].State
		}
	}
}
