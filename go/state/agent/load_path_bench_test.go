// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the State-index LOAD / wake-read path — the complement
// to the index BUILD benches in index_bench_test.go. The build side
// (NewStateIndex) fires once per Sleep; this side fires on every Wake:
// resolve the index JSON from the store, resolve the referenced block
// bundle, and restore the prefix snapshot the chosen entry requires.
//
// Two fixture tiers, by what each function actually touches:
//
//   - Parse-only (LoadStateIndex, LoadMemvidIndex): never read blocks,
//     only ResolveURI + JSON-unmarshal the index payload + Validate. A
//     synthetic byte-span bundle is enough, and a 10/100/1000-entry
//     sweep surfaces any per-entry parse scaling (mirrors the
//     NewStateIndex sweep).
//   - Block-reading (LoadPrefixFromStateIndex, LoadPrefixFromMemvidIndex,
//     LoadWakeSnapshot): walk all the way through
//     kv.LoadStateBlockBundle + kv.LoadPrefixFromStateBlocksWithOptions,
//     which physically read block chunks. These need a REAL snapshot
//     saved via SaveStateBlocks so the blocks resolve — the synthetic
//     byte-span bundle would fail the chunk read.
//
// LoadWakeSnapshot is benched through opts.IndexURI with opts.Index nil
// on purpose: that is the LOAD path (PlanWake -> loadIndex ->
// LoadStateIndex). Passing opts.Index would short-circuit the index
// load and measure plan-only shaping, which BenchmarkWakeSleep_PlanWake
// already covers.
//
// The dominant allocators on every path here (JSON unmarshal of the
// index/bundle, the kv block-restore) live in the core + kv packages,
// not in agent — read the FLAT pprof column for lines physically in
// agent/*.go to find any agent-resident cost.
//
// Run:    go test -bench='BenchmarkLoad' -benchmem -run='^$' ./go/agent

package agent

import (
	"context"
	"testing"

	"dappco.re/go/inference/bundle"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/memory"
	state "dappco.re/go/inference/state"
)

// Sinks defeat compiler DCE for the load-path results.
var (
	loadBenchSinkIndex    *StateIndex
	loadBenchSinkSnapshot *kv.Snapshot
	loadBenchSinkEntry    StateIndexEntry
	loadBenchSinkReport   *WakeReport
	loadBenchSinkErr      error
)

// loadBenchModelInfo mirrors the synthetic bundle's identity so the
// compatibility check inside PlanWake passes without a Metal model.
func loadBenchModelInfo() memory.ModelInfo {
	return memory.ModelInfo{
		Architecture:  "qwen3",
		NumLayers:     28,
		QuantBits:     4,
		ContextLength: 40960,
	}
}

// loadBenchParseStore seeds an in-memory store with a saved StateIndex
// (entryCount entries) over a synthetic byte-span bundle and returns the
// store + the index URI. The bundle's blocks are NOT stored — fine for
// the parse-only Load*Index functions which never read a block. Mirrors
// BenchmarkIndex_SaveStateIndex's seeding so the parse benches reuse the
// existing fixture helpers.
//
//	store, uri := loadBenchParseStore(b, 100)
func loadBenchParseStore(b *testing.B, entryCount int) (state.Store, string) {
	b.Helper()
	blk := benchIndexBundle(b, entryCount)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(entryCount)))
	if err != nil {
		b.Fatalf("NewStateIndex(%d): %v", entryCount, err)
	}
	store := state.NewInMemoryStore(nil)
	const uri = "mlx://bench/index"
	if _, err := SaveStateIndex(context.Background(), store, idx, uri); err != nil {
		b.Fatalf("SaveStateIndex: %v", err)
	}
	return store, uri
}

// loadBenchPrefixFixture seeds an in-memory store with a REAL 4-token
// snapshot (blocks stored so they resolve), its bundle manifest, and a
// StateIndex with a single entry covering a 2-token prefix. Returns the
// store, the in-memory index, the entry URI, and the index URI. This is
// the only fixture whose blocks physically read back, so it backs every
// function that walks kv.LoadPrefixFromStateBlocksWithOptions.
//
//	store, idx, entryURI, indexURI := loadBenchPrefixFixture(b)
func loadBenchPrefixFixture(b *testing.B) (store state.Store, idx *StateIndex, entryURI, indexURI string) {
	b.Helper()
	ctx := context.Background()
	const bundleURI = "mlx://bench/prefix/bundle"
	const idxURI = "mlx://bench/prefix/index"
	const entURI = "mlx://bench/prefix/entry"

	mem := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot() // 4 tokens, blocks resolve
	blk, err := snapshot.SaveStateBlocks(ctx, mem, kv.StateBlockOptions{BlockSize: 2, KVEncoding: kv.EncodingNative})
	if err != nil {
		b.Fatalf("SaveStateBlocks: %v", err)
	}
	if _, err := kv.SaveStateBlockBundle(ctx, mem, blk, bundleURI); err != nil {
		b.Fatalf("SaveStateBlockBundle: %v", err)
	}
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: bundleURI,
		Title:     "bench-prefix",
		// Model + ModelPath populated to match the benchIndexOptions
		// convention: a named model means the compatibility check
		// skips the bare-model-hash recompute branch (indexModel), so
		// the wake path carries zero agent-resident allocations and the
		// floor is airtight — every alloc is downstream json/kv or an
		// escape-by-return struct.
		Model:     "qwen3-7b",
		ModelPath: "/models/qwen3-7b",
		ModelInfo: loadBenchModelInfo(),
		Tokenizer: bundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		Entries:   []StateIndexEntry{{URI: entURI, TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		b.Fatalf("NewStateIndex(prefix): %v", err)
	}
	if _, err := SaveStateIndex(ctx, mem, index, idxURI); err != nil {
		b.Fatalf("SaveStateIndex(prefix): %v", err)
	}
	return mem, index, entURI, idxURI
}

// --- LoadStateIndex / LoadMemvidIndex — parse-only path. ResolveURI +
// JSON unmarshal of the index payload + Validate. Sweep entry counts to
// surface per-entry parse scaling. LoadMemvidIndex is the deprecated
// pass-through alias to LoadStateIndex (identical profile) — benched to
// close its 0%-coverage flag, not for an independent win.

func benchLoadStateIndex(b *testing.B, entryCount int) {
	store, uri := loadBenchParseStore(b, entryCount)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loadBenchSinkIndex, loadBenchSinkErr = LoadStateIndex(ctx, store, uri)
	}
}

func BenchmarkLoad_LoadStateIndex_10Entries(b *testing.B)   { benchLoadStateIndex(b, 10) }
func BenchmarkLoad_LoadStateIndex_100Entries(b *testing.B)  { benchLoadStateIndex(b, 100) }
func BenchmarkLoad_LoadStateIndex_1000Entries(b *testing.B) { benchLoadStateIndex(b, 1000) }

func BenchmarkLoad_LoadMemvidIndex_100Entries(b *testing.B) {
	store, uri := loadBenchParseStore(b, 100)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loadBenchSinkIndex, loadBenchSinkErr = LoadMemvidIndex(ctx, store, uri) //nolint:staticcheck // benching the deprecated alias to close its coverage gap
	}
}

// --- LoadPrefixFromStateIndex / LoadPrefixFromMemvidIndex — the full
// block-reading restore: Validate, Entry lookup, LoadStateBlockBundle,
// LoadPrefixFromStateBlocksWithOptions. The index is in-memory so the
// bench isolates the bundle-resolve + block-restore cost (the index JSON
// parse is covered separately above). RawKVOnly trims the snapshot
// post-processing to the raw KV restore.

func BenchmarkLoad_LoadPrefixFromStateIndex(b *testing.B) {
	store, idx, entryURI, _ := loadBenchPrefixFixture(b)
	ctx := context.Background()
	opts := kv.LoadOptions{RawKVOnly: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loadBenchSinkSnapshot, loadBenchSinkEntry, loadBenchSinkErr = LoadPrefixFromStateIndex(ctx, store, idx, entryURI, opts)
	}
}

func BenchmarkLoad_LoadPrefixFromMemvidIndex(b *testing.B) {
	store, idx, entryURI, _ := loadBenchPrefixFixture(b)
	ctx := context.Background()
	opts := kv.LoadOptions{RawKVOnly: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loadBenchSinkSnapshot, loadBenchSinkEntry, loadBenchSinkErr = LoadPrefixFromMemvidIndex(ctx, store, idx, entryURI, opts) //nolint:staticcheck // benching the deprecated alias to close its coverage gap
	}
}

// --- LoadWakeSnapshot — the whole wake-read path from a stored index
// URI: PlanWake (loadIndex -> LoadStateIndex, compatibility check,
// LoadStateBlockBundle, plan + report build) then the prefix block
// restore. Index is referenced by URI with opts.Index nil so the index
// JSON load is INCLUDED — this is the genuine cold-resume cost. (The
// in-memory-index plan-only shape is BenchmarkWakeSleep_PlanWake.)

func BenchmarkLoad_LoadWakeSnapshot_FromURI(b *testing.B) {
	store, _, entryURI, indexURI := loadBenchPrefixFixture(b)
	ctx := context.Background()
	opts := WakeOptions{
		IndexURI:    indexURI,
		EntryURI:    entryURI,
		Tokenizer:   bundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		LoadOptions: kv.LoadOptions{RawKVOnly: true},
	}
	info := loadBenchModelInfo()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loadBenchSinkSnapshot, loadBenchSinkReport, loadBenchSinkErr = LoadWakeSnapshot(ctx, store, opts, info)
	}
}
