// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the filestore PutOptions surface.
// Per AX-11 — filestore writes the PutOptions metadata as JSON inline
// in the record (recordMeta). Tag-map size dominates because the JSON
// marshal walks every entry. Title / URI lengths show up in the meta
// blob size + the per-record on-disk write.
//
// Run:    go test -bench='BenchmarkFilestorePutOpts' -benchmem -run='^$' ./state/filestore

package filestore

import (
	"context"
	"testing"

	state "dappco.re/go/inference/state"
)

// Sinks defeat compiler DCE.
var (
	fpoSinkRef state.ChunkRef
	fpoSinkErr error
)

// --- Empty meta fast path ---
// Many code paths (KV snapshots, sentinel records, internal-only
// blobs) write a record with no PutOptions content. The hand-rolled
// fast path skips core.JSONMarshal entirely — its alloc shape is the
// floor for what PutBytesStream can deliver on a streaming write.

func BenchmarkFilestorePutOpts_Empty(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/empty.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fpoSinkRef, fpoSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

// --- Tag map size sweep ---
// Memvid-style bundle saves carry 4-12 tags per chunk. The JSON
// marshal walks every entry; the on-disk record carries the marshalled
// bytes. Bench tracks the size-scaling cost.

func BenchmarkFilestorePutOpts_NoTags(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/tags0.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fpoSinkRef, fpoSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

func BenchmarkFilestorePutOpts_Tags_1(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/tags1.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{
		Kind: "bench",
		Tags: map[string]string{"epoch": "3"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fpoSinkRef, fpoSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

func BenchmarkFilestorePutOpts_Tags_4(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/tags4.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{
		Kind: "bench",
		Tags: map[string]string{
			"epoch":  "3",
			"track":  "primary",
			"source": "memvid",
			"env":    "bench",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fpoSinkRef, fpoSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

func BenchmarkFilestorePutOpts_Tags_8(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/tags8.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{
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
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fpoSinkRef, fpoSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

// --- Labels slice size sweep ---

func BenchmarkFilestorePutOpts_Labels_4(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/labels4.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{
		Kind:   "bench",
		Labels: []string{"k0:v0", "k1:v1", "k2:v2", "k3:v3"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fpoSinkRef, fpoSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

func BenchmarkFilestorePutOpts_Labels_8(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/labels8.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{
		Kind:   "bench",
		Labels: []string{"k0:v0", "k1:v1", "k2:v2", "k3:v3", "k4:v4", "k5:v5", "k6:v6", "k7:v7"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fpoSinkRef, fpoSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

// --- URI length sensitivity ---

func BenchmarkFilestorePutOpts_URI_Long(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/uri-long.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 64)
	uri := "mlx://lthn/projects/core/go-mlx/snapshots/2026-05-22T12:00:00Z/" +
		"runtime/metal/m3-ultra/model/qwen3-27b-4bit/adapter/lora-1/" +
		"workload/long-context/segment/chunk-00000042/epoch-3/layer/all"
	opts := state.PutOptions{Kind: "bench", URI: uri}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fpoSinkRef, fpoSinkErr = store.PutBytes(ctx, payload, opts)
	}
}

// --- FullMetadata (all fields populated) ---
// Stress shape — every PutOptions field has content. Real-world saves
// of training-checkpoint records carry full metadata.

func BenchmarkFilestorePutOpts_FullMetadata(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/full.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 64)
	opts := state.PutOptions{
		URI:    "mlx://bench/full",
		Title:  "bench-chunk-with-long-title-for-realistic-meta",
		Kind:   "training-checkpoint",
		Track:  "primary-train",
		Tags:   map[string]string{"epoch": "3", "branch": "dev"},
		Labels: []string{"kind:training", "source:hypnos"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fpoSinkRef, fpoSinkErr = store.PutBytes(ctx, payload, opts)
	}
}
