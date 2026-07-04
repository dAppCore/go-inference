// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the State index primitives. Per AX-11 — NewStateIndex
// fires per sleep round, Validate fires per load + per save, and
// indexHash + indexEntryHash run inside both. The hash builder concat
// chain (NewBuilder + N WriteString calls) is the dominant cost as
// entry count grows; 10/100/1000 entry sweeps map onto realistic
// chapter-marker counts (single chapter, a book, a 1000-checkpoint
// session log).
//
// Run:    go test -bench='BenchmarkIndex' -benchmem -run='^$' ./go/agent

package agent

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/bundle"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/memory"
	state "dappco.re/go/inference/state"
)

// Sinks defeat compiler DCE.
var (
	indexBenchSinkIndex   *StateIndex
	indexBenchSinkEntry   StateIndexEntry
	indexBenchSinkErr     error
	indexBenchSinkOK      bool
	indexBenchSinkInt     int
	indexBenchSinkString  string
	indexBenchSinkEntries []StateIndexEntry
	indexBenchSinkRef     state.ChunkRef
)

// benchIndexBundle returns a StateBlockBundle sized for the requested
// entry count (1 block per entry pair so the synthetic byte-span
// resolver has something to compute). Keep distinct from the
// test-side kvSnapshotIndexTestBundle so tests + benches can coexist.
//
//	bundle := benchIndexBundle(b, entryCount)
func benchIndexBundle(b *testing.B, entryCount int) *kv.StateBlockBundle {
	b.Helper()
	tokenCount := entryCount * 2
	blocks := make([]kv.StateBlockRef, entryCount)
	for i := range entryCount {
		blocks[i] = kv.StateBlockRef{
			Index:            i,
			TokenStart:       i * 2,
			TokenCount:       2,
			PayloadByteCount: 128,
			State:            state.ChunkRef{ChunkID: i + 1, FrameOffset: uint64(64 + i*128), HasFrameOffset: true},
		}
	}
	return &kv.StateBlockBundle{
		Version:      kv.MemvidBlockVersion,
		Kind:         kv.MemvidBlockBundleKind,
		SnapshotHash: "bench-snapshot-hash",
		KVEncoding:   kv.EncodingNative,
		Architecture: "qwen3",
		TokenCount:   tokenCount,
		TokenOffset:  tokenCount,
		BlockSize:    2,
		NumLayers:    28,
		NumHeads:     16,
		SeqLen:       tokenCount,
		HeadDim:      64,
		Blocks:       blocks,
	}
}

// benchIndexEntries generates a fresh entry slice. The slice is
// re-allocated on every call so each benchmark iteration sees fixed
// fixture cost — useful when timing NewStateIndex which mutates its
// inputs via cloneIndexEntries.
//
//	entries := benchIndexEntries(count)
func benchIndexEntries(count int) []StateIndexEntry {
	entries := make([]StateIndexEntry, count)
	for i := range count {
		entries[i] = StateIndexEntry{
			URI:        "mlx://book/chapter-" + benchItoa(i),
			Title:      "Chapter " + benchItoa(i),
			TokenStart: i * 2,
			TokenCount: 2,
			Labels:     []string{"chapter", "agent-state"},
			Meta:       map[string]string{"ordinal": benchItoa(i)},
		}
	}
	return entries
}

// benchItoa — small inline integer-to-string helper. Kept local to
// avoid importing strconv at the top of the bench file.
func benchItoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	neg := n < 0
	if neg {
		n = -n
	}
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// benchIndexOptions returns a populated StateIndexOptions struct used by
// every NewStateIndex bench.
func benchIndexOptions(bundleURI string, entries []StateIndexEntry) StateIndexOptions {
	return StateIndexOptions{
		BundleURI: bundleURI,
		Title:     "bench-book",
		Model:     "qwen3-7b",
		ModelPath: "/models/qwen3-7b",
		ModelInfo: memory.ModelInfo{
			Architecture:  "qwen3",
			NumLayers:     28,
			QuantBits:     4,
			ContextLength: 40960,
		},
		Tokenizer: bundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		Entries:   entries,
	}
}

// --- NewStateIndex — full construction path: validate bundle, clone
// entries, fill byte spans, hash each entry, hash the index. ---

func BenchmarkIndex_NewStateIndex_10Entries(b *testing.B) {
	blk := benchIndexBundle(b, 10)
	opts := benchIndexOptions("mlx://bench/bundle", benchIndexEntries(10))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkIndex, indexBenchSinkErr = NewStateIndex(blk, opts)
	}
}

func BenchmarkIndex_NewStateIndex_100Entries(b *testing.B) {
	blk := benchIndexBundle(b, 100)
	opts := benchIndexOptions("mlx://bench/bundle", benchIndexEntries(100))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkIndex, indexBenchSinkErr = NewStateIndex(blk, opts)
	}
}

func BenchmarkIndex_NewStateIndex_1000Entries(b *testing.B) {
	blk := benchIndexBundle(b, 1000)
	opts := benchIndexOptions("mlx://bench/bundle", benchIndexEntries(1000))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkIndex, indexBenchSinkErr = NewStateIndex(blk, opts)
	}
}

// Default full-bundle entry path — exercises the branch in
// NewStateIndex that synthesises a single entry covering the
// whole bundle when caller supplies no entries.
func BenchmarkIndex_NewStateIndex_DefaultFullEntry(b *testing.B) {
	blk := benchIndexBundle(b, 10)
	opts := benchIndexOptions("mlx://bench/bundle", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkIndex, indexBenchSinkErr = NewStateIndex(blk, opts)
	}
}

// --- Validate — schema + bounds + duplicate-URI + hash check. Hit on
// every load and at the tail of every NewStateIndex.

func BenchmarkIndex_Validate_10Entries(b *testing.B) {
	blk := benchIndexBundle(b, 10)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(10)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkErr = idx.Validate()
	}
}

func BenchmarkIndex_Validate_1000Entries(b *testing.B) {
	blk := benchIndexBundle(b, 1000)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(1000)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkErr = idx.Validate()
	}
}

// --- indexHash / indexEntryHash — inner hash chain. These are the
// expensive primitives both NewStateIndex and Validate hit. Worth
// benching standalone so codex can see the per-entry SHA cost.

func BenchmarkIndex_IndexHash_10Entries(b *testing.B) {
	blk := benchIndexBundle(b, 10)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(10)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkString = indexHash(idx)
	}
}

func BenchmarkIndex_IndexHash_1000Entries(b *testing.B) {
	blk := benchIndexBundle(b, 1000)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(1000)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkString = indexHash(idx)
	}
}

func BenchmarkIndex_IndexEntryHash_RichEntry(b *testing.B) {
	entry := StateIndexEntry{
		URI:        "mlx://book/chapter-7",
		BundleURI:  "mlx://book/bundle",
		Title:      "Chapter 7",
		TokenStart: 1024,
		TokenCount: 2048,
		ByteStart:  131072,
		ByteCount:  524288,
		Labels:     []string{"chapter", "agent-state", "checkpoint"},
		Meta:       map[string]string{"ordinal": "7", "author": "cladius", "model": "qwen3-7b"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkString = indexEntryHash(&entry)
	}
}

// --- Entry — linear lookup by URI. Hit per LoadPrefixFromStateIndex
// + per CheckStateIndexCompatibility. O(n) entries.

func BenchmarkIndex_Entry_FirstHit_1000(b *testing.B) {
	blk := benchIndexBundle(b, 1000)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(1000)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	uri := "mlx://book/chapter-0"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkEntry, indexBenchSinkOK = idx.Entry(uri)
	}
}

func BenchmarkIndex_Entry_LastHit_1000(b *testing.B) {
	blk := benchIndexBundle(b, 1000)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(1000)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	uri := "mlx://book/chapter-999"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkEntry, indexBenchSinkOK = idx.Entry(uri)
	}
}

func BenchmarkIndex_Entry_Miss_1000(b *testing.B) {
	blk := benchIndexBundle(b, 1000)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(1000)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	uri := "mlx://book/missing"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkEntry, indexBenchSinkOK = idx.Entry(uri)
	}
}

// --- RequiredContextLength — sweeps all entries. Hit during
// CheckStateIndexCompatibility.

func BenchmarkIndex_RequiredContextLength_100Entries(b *testing.B) {
	blk := benchIndexBundle(b, 100)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(100)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkInt = idx.RequiredContextLength()
	}
}

func BenchmarkIndex_RequiredContextLength_1000Entries(b *testing.B) {
	blk := benchIndexBundle(b, 1000)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(1000)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkInt = idx.RequiredContextLength()
	}
}

// --- cloneIndexEntries — defensive copy with label + meta clone.
// Hit inside NewStateIndex on every call.

func BenchmarkIndex_CloneIndexEntries_100(b *testing.B) {
	entries := benchIndexEntries(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkEntries = cloneIndexEntries(entries)
	}
}

func BenchmarkIndex_CloneIndexEntries_1000(b *testing.B) {
	entries := benchIndexEntries(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkEntries = cloneIndexEntries(entries)
	}
}

// --- CheckStateIndexCompatibility — hot path when waking from a
// resumed session, fires once per load.

func BenchmarkIndex_CheckStateIndexCompatibility_Matching(b *testing.B) {
	blk := benchIndexBundle(b, 10)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(10)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	info := memory.ModelInfo{Architecture: "qwen3", NumLayers: 28, QuantBits: 4, ContextLength: 40960}
	tok := bundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkErr = CheckStateIndexCompatibility(info, tok, idx)
	}
}

// --- SaveStateIndex + LoadStateIndex — full roundtrip through an
// in-memory state store. Captures the JSON marshal + Put + Resolve +
// Unmarshal + Validate chain per wake/sleep round.

func BenchmarkIndex_SaveStateIndex_10Entries(b *testing.B) {
	blk := benchIndexBundle(b, 10)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(10)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	ctx := context.Background()
	uri := "mlx://bench/index"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		indexBenchSinkRef, indexBenchSinkErr = SaveStateIndex(ctx, store, idx, uri)
	}
}

func BenchmarkIndex_LoadStateIndex_10Entries(b *testing.B) {
	blk := benchIndexBundle(b, 10)
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(10)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	uri := "mlx://bench/index"
	if _, err := SaveStateIndex(ctx, store, idx, uri); err != nil {
		b.Fatalf("SaveStateIndex: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkIndex, indexBenchSinkErr = LoadStateIndex(ctx, store, uri)
	}
}

// --- PrefixTokens — trivial accessor but hit during every
// LoadPrefixFromStateIndex + blocksNeededForPrefix walk.

func BenchmarkIndex_PrefixTokens(b *testing.B) {
	entry := StateIndexEntry{TokenStart: 1024, TokenCount: 2048}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		indexBenchSinkInt = entry.PrefixTokens()
	}
}

// Avoid unused-import warnings from helpers that may not be referenced
// directly by every bench (e.g. core, when fixtures are nilable).
var _ = core.Trim
