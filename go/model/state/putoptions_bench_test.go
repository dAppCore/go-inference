// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the PutOptions input shape across the Writer surface.
// Per AX-11 — PutOptions is the per-call envelope every Put/PutBytes
// hits. The Tags map is the dominant allocator under heavy metadata
// loads (memvid bundle saves carry 4-12 tags per chunk). The URI string
// length matters because the Memory backend mirrors URIs into a lookup
// table — long URIs compound into the uri map.
//
// Run:    go test -bench='BenchmarkPutOptions' -benchmem -run='^$' ./state

package state

import (
	"context"
	"testing"
)

// Sinks defeat compiler DCE. Distinct names per state-package bench file.
var (
	putOptsSinkRef ChunkRef
	putOptsSinkErr error
)

// --- Tags map size sweep ---
// Memvid bundle saves typically carry 0-8 tags per record (kind, track,
// epoch, source-tool, env, etc.). The Put path doesn't clone the map
// today but the structural shape benches confirm the read cost.

func BenchmarkPutOptions_NoTags(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	opts := PutOptions{Kind: "bench"}
	data := make([]byte, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef, putOptsSinkErr = store.PutBytes(ctx, data, opts)
	}
}

func BenchmarkPutOptions_Tags_1(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	opts := PutOptions{
		Kind: "bench",
		Tags: map[string]string{"epoch": "3"},
	}
	data := make([]byte, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef, putOptsSinkErr = store.PutBytes(ctx, data, opts)
	}
}

func BenchmarkPutOptions_Tags_4(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	opts := PutOptions{
		Kind: "bench",
		Tags: map[string]string{
			"epoch":  "3",
			"track":  "primary",
			"source": "memvid",
			"env":    "bench",
		},
	}
	data := make([]byte, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef, putOptsSinkErr = store.PutBytes(ctx, data, opts)
	}
}

func BenchmarkPutOptions_Tags_8(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	opts := PutOptions{
		Kind: "bench",
		Tags: map[string]string{
			"epoch":   "3",
			"track":   "primary",
			"source":  "memvid",
			"env":     "bench",
			"branch":  "dev",
			"runner":  "homelab",
			"adapter": "lora-1",
			"model":   "qwen3",
		},
	}
	data := make([]byte, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef, putOptsSinkErr = store.PutBytes(ctx, data, opts)
	}
}

// --- Labels slice size ---
// Per Lethean convention, Labels is the unordered string-list of
// arbitrary classifiers (e.g. "kind:training", "source:hypnos"). The
// slice header is shared by reference but indexes any persistence
// hashing.

func BenchmarkPutOptions_Labels_0(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	opts := PutOptions{Kind: "bench"}
	data := make([]byte, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef, putOptsSinkErr = store.PutBytes(ctx, data, opts)
	}
}

func BenchmarkPutOptions_Labels_4(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	opts := PutOptions{
		Kind:   "bench",
		Labels: []string{"k0:v0", "k1:v1", "k2:v2", "k3:v3"},
	}
	data := make([]byte, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef, putOptsSinkErr = store.PutBytes(ctx, data, opts)
	}
}

// --- URI variants ---
// Empty URI bypasses the uri[] index write. Typical URI is a normal
// state:// path. Very-long URI tests the map-write of a 256-char key
// (e.g. fully-qualified bundle URI with epoch+layer suffixes).

func BenchmarkPutOptions_URI_Empty(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	opts := PutOptions{Kind: "bench"}
	data := make([]byte, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef, putOptsSinkErr = store.PutBytes(ctx, data, opts)
	}
}

func BenchmarkPutOptions_URI_Typical(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	opts := PutOptions{
		Kind: "bench",
		URI:  "state://lthn/projects/core/go-mlx/seed/v1/bundle",
	}
	data := make([]byte, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef, putOptsSinkErr = store.PutBytes(ctx, data, opts)
	}
}

func BenchmarkPutOptions_URI_Long(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	// 256-char URI — realistic for a fully-qualified bundle/segment/epoch
	// path that includes runtime + model identity in the leaf.
	uri := "state://lthn/projects/core/go-mlx/snapshots/2026-05-22T12:00:00Z/" +
		"runtime/metal/m3-ultra/model/qwen3-27b-4bit/adapter/lora-1/" +
		"workload/long-context/segment/chunk-00000042/epoch-3/layer/all"
	opts := PutOptions{Kind: "bench", URI: uri}
	data := make([]byte, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef, putOptsSinkErr = store.PutBytes(ctx, data, opts)
	}
}

// --- HasFrameOffset variants ---
// PutBytes always sets HasFrameOffset on the returned ref. The shape
// is asserted at the ref layer below; this bench exercises the
// observable cost of constructing the ref with explicit defaults.

func BenchmarkPutOptions_Construct_HasFrameOffset(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef = ChunkRef{
			ChunkID:        i,
			FrameOffset:    uint64(i),
			HasFrameOffset: true,
			Codec:          CodecMemory,
		}
	}
}

func BenchmarkPutOptions_Construct_NoFrameOffset(b *testing.B) {
	// Some adapters omit the frame offset (e.g. opaque-blob stores).
	// Confirms the "small" ref shape costs the same to construct.
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef = ChunkRef{
			ChunkID: i,
			Codec:   CodecMemory,
		}
	}
}

// --- Title / Track / Kind string variants ---
// Same shape but with all metadata strings populated — the per-call
// cost should be ~constant since the map writes dominate, but the
// bench tracks regressions in the metadata-rich path.

func BenchmarkPutOptions_FullMetadata(b *testing.B) {
	store := NewInMemoryStore(nil)
	ctx := context.Background()
	opts := PutOptions{
		URI:    "state://bench/full",
		Title:  "bench-chunk-with-long-title-for-realistic-meta",
		Kind:   "training-checkpoint",
		Track:  "primary-train",
		Tags:   map[string]string{"epoch": "3", "branch": "dev"},
		Labels: []string{"kind:training", "source:hypnos"},
	}
	data := make([]byte, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		putOptsSinkRef, putOptsSinkErr = store.PutBytes(ctx, data, opts)
	}
}
