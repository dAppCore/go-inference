// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the local model-cache resolution surface. These run on every
// local model load — LocalModelFiles scans the snapshot directory once,
// WeightFormatAndBytes and LocalModelID then walk the result / the cache path.

package hf

import (
	"testing"

	core "dappco.re/go"
)

// benchModelFiles is a real-shaped weight manifest: a four-shard safetensors
// pack plus the two tokeniser sidecars LocalModelFiles surfaces.
var benchModelFiles = []ModelFile{
	{Name: "model-00001-of-00004.safetensors", Size: 4_294_967_296},
	{Name: "model-00002-of-00004.safetensors", Size: 4_294_967_296},
	{Name: "model-00003-of-00004.safetensors", Size: 4_294_967_296},
	{Name: "model-00004-of-00004.safetensors", Size: 1_073_741_824},
	{Name: "tokenizer.json", Size: 17_000_000},
	{Name: "tokenizer_config.json", Size: 48_000},
}

// BenchmarkHfCache_WeightFormatAndBytes measures the per-load weight-format
// scan over a four-shard pack — pure, no I/O.
func BenchmarkHfCache_WeightFormatAndBytes(b *testing.B) {
	b.ReportAllocs()
	var format string
	var total uint64
	for i := 0; i < b.N; i++ {
		format, total = WeightFormatAndBytes(benchModelFiles)
	}
	_, _ = format, total
}

// BenchmarkHfCache_LocalModelID measures the per-load "org/name" derivation
// walking up a HuggingFace `models--org--name` cache path — pure, no I/O.
func BenchmarkHfCache_LocalModelID(b *testing.B) {
	const cacheRoot = "/home/user/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit"
	snapshot := cacheRoot + "/snapshots/abcdef0123456789abcdef0123456789abcdef01"
	b.ReportAllocs()
	var id string
	for i := 0; i < b.N; i++ {
		id = LocalModelID(snapshot, cacheRoot)
	}
	_ = id
}

// BenchmarkHfCache_LocalModelFiles measures one snapshot-directory scan — the
// single ReadDir plus per-entry filter that replaced five filepath.Glob passes.
func BenchmarkHfCache_LocalModelFiles(b *testing.B) {
	root := b.TempDir()
	for _, name := range []string{
		"model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
		"model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors",
		"tokenizer.json", "tokenizer_config.json", "config.json", "README.md",
	} {
		if result := core.WriteFile(core.PathJoin(root, name), []byte("x"), 0o644); !result.OK {
			b.Fatalf("write fixture %s: %v", name, result.Value)
		}
	}
	b.ReportAllocs()
	var files []ModelFile
	for i := 0; i < b.N; i++ {
		files = LocalModelFiles(root)
	}
	_ = files
}
