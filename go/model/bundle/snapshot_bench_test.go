// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the Snapshot read-back path — the inverse of the
// New/Save build path benched in bundle_bench_test.go. These three
// functions are how a migrated bundle hands its KV state back to a
// runtime for replay:
//
//   - Snapshot()          — defensive clone of the embedded kv.Snapshot
//     (or a KVPath disk-load); fires once per "restore this session".
//   - SnapshotFromState()  — the State-backed variant. When the bundle
//     carries an inline KV it short-circuits to Snapshot (same clone);
//     when the KV lives in a State cold store it resolves + decodes the
//     chunk via kv.LoadFromState. The decode branch is the one that does
//     real per-restore work (base64 → binary → snapshot → hash-verify).
//   - SnapshotFromMemvid() — deprecated alias of SnapshotFromState.
//
// Per AX-11: measure allocs/op + B/op on the in-memory paths so the
// restore surface has the same alloc-floor evidence the build surface
// already has. The KVPath branch is deliberately not benched — it is
// disk-I/O-bound (kv.Load reads a file), outside the pure-Go scope.
//
// Run:    go test -bench=Benchmark -benchmem -run='^$' ./go/bundle
//
// reuses benchBundleSnapshot from bundle_bench_test.go (same package).

package bundle

import (
	"context"
	"testing"

	"dappco.re/go/inference/kv"
	state "dappco.re/go/inference/model/state"
)

// Sinks defeat compiler DCE for the snapshot read-back benches.
var (
	bundleSinkSnapshot *kv.Snapshot
	bundleSinkRef      state.ChunkRef
)

// --- Snapshot — defensive clone of the embedded KV (the common restore) ---

func BenchmarkBundle_Snapshot_Small(b *testing.B) {
	snap := benchBundleSnapshot(64, 2)
	bundle, err := New(snap, Options{Model: "qwen3-0.6b", Source: ModelInfo{Architecture: "qwen3", NumLayers: 2}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSnapshot, bundleSinkErr = bundle.Snapshot()
	}
}

func BenchmarkBundle_Snapshot_Typical(b *testing.B) {
	snap := benchBundleSnapshot(512, 8)
	bundle, err := New(snap, Options{Model: "qwen3", Source: ModelInfo{Architecture: "qwen3", NumLayers: 8}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSnapshot, bundleSinkErr = bundle.Snapshot()
	}
}

func BenchmarkBundle_Snapshot_Large(b *testing.B) {
	snap := benchBundleSnapshot(2048, 28)
	bundle, err := New(snap, Options{Model: "qwen3", Source: ModelInfo{Architecture: "qwen3", NumLayers: 28}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSnapshot, bundleSinkErr = bundle.Snapshot()
	}
}

// --- SnapshotFromState — inline-KV short-circuit (delegates to Snapshot) ---

// Bundle carries an embedded KV, so SnapshotFromState takes the
// `b.KV != nil` fast path and clones rather than touching the store. The
// alloc profile must match Snapshot_Typical — this bench guards that the
// delegation adds no per-call overhead of its own.
func BenchmarkBundle_SnapshotFromState_InlineKV(b *testing.B) {
	snap := benchBundleSnapshot(512, 8)
	bundle, err := New(snap, Options{Model: "qwen3", Source: ModelInfo{Architecture: "qwen3", NumLayers: 8}})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSnapshot, bundleSinkErr = bundle.SnapshotFromState(ctx, store)
	}
}

// --- SnapshotFromState — State cold-store resolve + decode (the real work) ---

// No inline KV: the bundle holds only a State Ref, so SnapshotFromState
// must resolve the chunk and run the full kv.LoadFromState decode
// (envelope JSON → base64 → binary → parse → hash-verify) on every call.
// This is the per-restore cost when the KV was offloaded to cold storage.
// The store + ref are built once, outside the timed loop.
func benchStateBackedBundle(b *testing.B, tokenCount, numLayers int) (*Bundle, context.Context, state.Store) {
	b.Helper()
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	snap := benchBundleSnapshot(tokenCount, numLayers)
	ref, err := snap.SaveState(ctx, store, kv.StateOptions{})
	if err != nil {
		b.Fatalf("SaveState: %v", err)
	}
	hash, err := kv.HashSnapshot(snap)
	if err != nil {
		b.Fatalf("HashSnapshot: %v", err)
	}
	bundle := &Bundle{
		Version: Version, Kind: Kind, KVHash: hash,
		Refs: []Ref{{Kind: RefState, URI: StateURI(ref), State: ref}},
	}
	return bundle, ctx, store
}

func BenchmarkBundle_SnapshotFromState_Decode_Small(b *testing.B) {
	bundle, ctx, store := benchStateBackedBundle(b, 64, 2)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSnapshot, bundleSinkErr = bundle.SnapshotFromState(ctx, store)
	}
}

func BenchmarkBundle_SnapshotFromState_Decode_Typical(b *testing.B) {
	bundle, ctx, store := benchStateBackedBundle(b, 512, 8)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSnapshot, bundleSinkErr = bundle.SnapshotFromState(ctx, store)
	}
}

func BenchmarkBundle_SnapshotFromState_Decode_Large(b *testing.B) {
	bundle, ctx, store := benchStateBackedBundle(b, 2048, 28)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSnapshot, bundleSinkErr = bundle.SnapshotFromState(ctx, store)
	}
}

// --- SnapshotFromMemvid — deprecated alias, documents identical cost ---

// SnapshotFromMemvid is a thin forward to SnapshotFromState; this bench
// exists so the deprecated entry point carries the same alloc evidence
// and any future divergence shows up in the bench diff.
func BenchmarkBundle_SnapshotFromMemvid_Decode_Typical(b *testing.B) {
	bundle, ctx, store := benchStateBackedBundle(b, 512, 8)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundleSinkSnapshot, bundleSinkErr = bundle.SnapshotFromMemvid(ctx, store)
	}
}
