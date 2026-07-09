// SPDX-Licence-Identifier: EUPL-1.2

// StateBlockOptions / PutOptions variation benches.
//
// W7-A landed two optimisations on this surface: a shared default
// Labels slice when opts.Labels is empty (saved a per-block alloc) and
// a Tags map pre-sized for the 6 deterministic bookkeeping tags
// SaveStateBlocks writes after cloning. This file widens coverage so
// future changes to the Labels / Tags / Track / URI surface have a
// regression baseline.
//
// Coverage map (W7-F deepening pass):
//
//   - SaveStateBlocks with empty Labels (default-shared-slice path)
//   - SaveStateBlocks with one user Label (the +2-pad pre-size path)
//   - SaveStateBlocks with five user Labels (geometric-grow protection
//     guard)
//   - SaveStateBlocks with empty Tags / one Tag / many Tags
//   - SaveStateBlocks with custom URI / Title / Kind / Track
//   - kvSnapshotStateBlockPutOptions helper isolated (no IO) so future
//     allocs in the helper surface against the bench.
//
// Run: go test -bench='BenchmarkPutoptions' -benchmem -run='^$' ./go/kv

package kv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/model/state"
)

// --- Labels variations ---

func BenchmarkPutoptions_SaveBlocks_EmptyLabels(b *testing.B) {
	benchSaveBlocksWithOpts(b, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
		Labels:     nil,
	})
}

func BenchmarkPutoptions_SaveBlocks_OneLabel(b *testing.B) {
	benchSaveBlocksWithOpts(b, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
		Labels:     []string{"benchsite"},
	})
}

func BenchmarkPutoptions_SaveBlocks_ManyLabels(b *testing.B) {
	benchSaveBlocksWithOpts(b, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
		Labels:     []string{"benchsite", "session", "warm", "qwen3", "raw"},
	})
}

// --- Tags variations ---

func BenchmarkPutoptions_SaveBlocks_EmptyTags(b *testing.B) {
	benchSaveBlocksWithOpts(b, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
		Tags:       nil,
	})
}

func BenchmarkPutoptions_SaveBlocks_OneTag(b *testing.B) {
	benchSaveBlocksWithOpts(b, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
		Tags:       map[string]string{"session_id": "abc"},
	})
}

func BenchmarkPutoptions_SaveBlocks_ManyTags(b *testing.B) {
	benchSaveBlocksWithOpts(b, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
		Tags: map[string]string{
			"session_id":   "abc",
			"model":        "qwen3",
			"context_size": "2048",
			"variant":      "raw",
			"warm":         "true",
		},
	})
}

// --- URI / Title / Kind / Track custom ---

func BenchmarkPutoptions_SaveBlocks_CustomURIAndTitle(b *testing.B) {
	benchSaveBlocksWithOpts(b, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
		URI:        "state://benchsite/turn-001",
		Title:      "warm bench block",
		Kind:       "bench/kv-block",
		Track:      "bench-track",
	})
}

func benchSaveBlocksWithOpts(b *testing.B, opts StateBlockOptions) {
	b.Helper()
	snap := benchSnapshot(1536) // 3 × 512 blocks
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		bundle, err := snap.SaveStateBlocks(ctx, store, opts)
		if err != nil {
			b.Fatal(err)
		}
		if bundle != nil && len(bundle.Blocks) > 0 {
			benchSinkRef = bundle.Blocks[0].State
		}
	}
}

// --- Helper-only — kvSnapshotStateBlockPutOptions in isolation.
// The IO-free path that fires once per block during SaveStateBlocks.
// Pegging the helper against the no-options baseline catches regressions
// in the labels / tags / URI build path without IO noise. ---

func BenchmarkPutoptions_HelperOnly_EmptyOptions(b *testing.B) {
	block := Block{Index: 0, TokenStart: 0, TokenCount: 512}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPutOptions = kvSnapshotStateBlockPutOptions(block, StateBlockOptions{}, "deadbeef", "native", kvSnapshotStatePayloadRaw)
	}
}

func BenchmarkPutoptions_HelperOnly_ManyLabelsAndTags(b *testing.B) {
	block := Block{Index: 0, TokenStart: 0, TokenCount: 512}
	opts := StateBlockOptions{
		Labels: []string{"benchsite", "session", "warm", "qwen3", "raw"},
		Tags: map[string]string{
			"session_id":   "abc",
			"model":        "qwen3",
			"context_size": "2048",
			"variant":      "raw",
			"warm":         "true",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPutOptions = kvSnapshotStateBlockPutOptions(block, opts, "deadbeef", "native", kvSnapshotStatePayloadRaw)
	}
}

// Sink for the helper benches — keeps the PutOptions alive past DCE.
var benchSinkPutOptions state.PutOptions
