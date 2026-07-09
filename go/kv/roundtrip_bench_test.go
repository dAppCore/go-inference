// SPDX-Licence-Identifier: EUPL-1.2

// Round-trip benches for KV snapshot persistence — capture-equivalent
// fixtures pushed through the full Save → Load → Restore cycle, and
// the in-memory MarshalBinary → UnmarshalBinary parity path.
//
// Coverage map (W7-F deepening pass, additive to snapshot_bench_test.go
// + blocks_benchmark_test.go):
//
//   - Single-snapshot full disk round-trip at 512 / 2048 / 8192 tokens —
//     measures the encode + write + read + parse path together. Existing
//     benches isolate each leg; this one captures the cumulative cost,
//     which is what callers (session resume) actually pay.
//   - MarshalBinary → UnmarshalBinary in-memory round-trip — isolates
//     the encoder + decoder against disk-IO noise.
//   - SaveStateBlocks → LoadFromStateBlocks full cycle through a
//     state.InMemoryStore at 3 blocks (1536 tokens) — the persisted
//     state substrate round-trip Virgil exercises per session resume.
//   - Save → Load → SliceBlock prefix restore — the warm-resume path.
//
// Run: go test -bench='BenchmarkRoundtrip' -benchmem -run='^$' ./go/kv

package kv

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
)

// --- Single-snapshot full disk round-trip ---

func BenchmarkRoundtrip_SaveLoad_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := snap.Save(path); err != nil {
			b.Fatal(err)
		}
		out, err := Load(path)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

func BenchmarkRoundtrip_SaveLoad_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := snap.Save(path); err != nil {
			b.Fatal(err)
		}
		out, err := Load(path)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

func BenchmarkRoundtrip_SaveLoad_8192Tokens(b *testing.B) {
	snap := benchSnapshot(8192)
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := snap.Save(path); err != nil {
			b.Fatal(err)
		}
		out, err := Load(path)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

// --- In-memory MarshalBinary → UnmarshalBinary round-trip ---

func BenchmarkRoundtrip_MarshalUnmarshal_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := snap.MarshalBinary()
		if err != nil {
			b.Fatal(err)
		}
		var out Snapshot
		if err := out.UnmarshalBinary(data); err != nil {
			b.Fatal(err)
		}
		benchSinkBytes = data
	}
}

func BenchmarkRoundtrip_MarshalUnmarshal_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := snap.MarshalBinary()
		if err != nil {
			b.Fatal(err)
		}
		var out Snapshot
		if err := out.UnmarshalBinary(data); err != nil {
			b.Fatal(err)
		}
		benchSinkBytes = data
	}
}

// --- State-block persisted round-trip — the Virgil cold-store path ---

func BenchmarkRoundtrip_StateBlocks_SaveLoad_3Blocks(b *testing.B) {
	snap := benchSnapshot(1536)
	opts := StateBlockOptions{BlockSize: 512, KVEncoding: EncodingNative}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		bundle, err := snap.SaveStateBlocks(ctx, store, opts)
		if err != nil {
			b.Fatal(err)
		}
		restored, err := LoadFromStateBlocks(ctx, store, bundle)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = restored
	}
}

// --- Resume path: Save → Load → SliceBlock prefix carve-out ---

func BenchmarkRoundtrip_LoadAndSlicePrefix_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	if err := snap.Save(path); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loaded, err := Load(path)
		if err != nil {
			b.Fatal(err)
		}
		// Slice the first 1024-token prefix — the prompt-restart shape
		// where the resumed session re-warms half the previous window.
		out, err := loaded.SliceBlock(0, 1024, 0, false)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

// --- Multi-step round-trip — captures cumulative ns + total allocs across
// the SaveStateBlocks → LoadPrefixTokens → LoadPrefixFromStateBlocks chain
// (the Virgil per-turn warm path: token-only prefix wake before full KV
// hydrate). ---

func BenchmarkRoundtrip_MultiStep_StateBlocks_3Blocks(b *testing.B) {
	snap := benchSnapshot(1536)
	opts := StateBlockOptions{BlockSize: 512, KVEncoding: EncodingNative}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		bundle, err := snap.SaveStateBlocks(ctx, store, opts)
		if err != nil {
			b.Fatal(err)
		}
		toks, err := LoadPrefixTokensFromStateBlocks(ctx, store, bundle, bundle.TokenCount)
		if err != nil {
			b.Fatal(err)
		}
		full, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, bundle.TokenCount, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		stateBlocksBenchmarkTokens = toks
		benchSinkSnapshot = full
	}
}
