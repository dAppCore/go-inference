// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the model-directory discovery walk + path helpers.
// Per AX-11 — Discover walks every subdirectory of the user's model
// root, parses config.json for each candidate, and counts .safetensors
// shards. With dozens of fine-tunes per root the per-directory cost
// compounds. joinPath / cleanPath / absolutePath sit in the per-walk
// hot loop.
//
// Run:    go test -bench='BenchmarkDiscover' -benchmem -run='^$' .

package inference

import (
	"slices"
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE. Distinct names from other bench files.
var (
	discoverBenchSinkModels []DiscoveredModel
	discoverBenchSinkPath   string
	discoverBenchSinkCount  int
)

// makeBenchModelDir is a file-scope helper so the bench fixture build
// stays out of the timed loop. Same shape as createModelDir in the test
// suite but with no t.Helper bookkeeping.
func makeBenchModelDir(b *testing.B, dir string, config map[string]any, shards int) {
	b.Helper()
	if r := core.MkdirAll(dir, 0o755); !r.OK {
		b.Fatal(r.Value)
	}
	if config != nil {
		data := []byte(core.JSONMarshalString(config))
		if r := core.WriteFile(core.JoinPath(dir, "config.json"), data, 0o644); !r.OK {
			b.Fatal(r.Value)
		}
	}
	for i := range shards {
		name := core.Sprintf("model-%05d-of-%05d.safetensors", i+1, shards)
		if r := core.WriteFile(core.JoinPath(dir, name), []byte("weights"), 0o644); !r.OK {
			b.Fatal(r.Value)
		}
	}
}

// --- Discover end-to-end (per-call walk floor) ---

func BenchmarkDiscover_SingleModel_TwoShards(b *testing.B) {
	base := b.TempDir()
	makeBenchModelDir(b, core.JoinPath(base, "qwen3-4b"), map[string]any{
		"model_type": "qwen3",
		"quantization": map[string]any{
			"bits":       4,
			"group_size": 64,
		},
	}, 2)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		discoverBenchSinkModels = slices.Collect(Discover(base))
	}
}

// Three sibling models — the common "models/" layout where a user has a
// handful of checkpoints under one root.
func BenchmarkDiscover_ThreeSiblings(b *testing.B) {
	base := b.TempDir()
	makeBenchModelDir(b, core.JoinPath(base, "gemma3-1b"), map[string]any{"model_type": "gemma3"}, 1)
	makeBenchModelDir(b, core.JoinPath(base, "qwen3-4b"), map[string]any{"model_type": "qwen3"}, 4)
	makeBenchModelDir(b, core.JoinPath(base, "llama3-8b"), map[string]any{"model_type": "llama"}, 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		discoverBenchSinkModels = slices.Collect(Discover(base))
	}
}

// Nested directory tree — exercises the recursive descent path.
func BenchmarkDiscover_NestedTree(b *testing.B) {
	base := b.TempDir()
	makeBenchModelDir(b, core.JoinPath(base, "base"), map[string]any{"model_type": "base"}, 1)
	makeBenchModelDir(b, core.JoinPath(base, "base", "ft-a"), map[string]any{"model_type": "ft-a"}, 1)
	makeBenchModelDir(b, core.JoinPath(base, "base", "ft-b"), map[string]any{"model_type": "ft-b"}, 1)
	makeBenchModelDir(b, core.JoinPath(base, "base", "ft-b", "v2"), map[string]any{"model_type": "ft-b-v2"}, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		discoverBenchSinkModels = slices.Collect(Discover(base))
	}
}

// Miss path — no config.json anywhere, just non-model files. Discover
// must still stat every entry.
func BenchmarkDiscover_NoModels_TenJunkDirs(b *testing.B) {
	base := b.TempDir()
	for i := range 10 {
		dir := core.JoinPath(base, core.Sprintf("junk-%d", i))
		if r := core.MkdirAll(dir, 0o755); !r.OK {
			b.Fatal(r.Value)
		}
		if r := core.WriteFile(core.JoinPath(dir, "README.md"), []byte("not a model"), 0o644); !r.OK {
			b.Fatal(r.Value)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		discoverBenchSinkModels = slices.Collect(Discover(base))
	}
}

// Early-exit path — caller takes the first match. Proxy for the common
// "pick by architecture" pattern in interactive UIs.
func BenchmarkDiscover_EarlyBreak_TwoSiblings(b *testing.B) {
	base := b.TempDir()
	makeBenchModelDir(b, core.JoinPath(base, "model-a"), map[string]any{"model_type": "a"}, 1)
	makeBenchModelDir(b, core.JoinPath(base, "model-b"), map[string]any{"model_type": "b"}, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range Discover(base) {
			count++
			break
		}
		discoverBenchSinkCount = count
	}
}

// --- Path helpers used in the inner walk loop ---

func BenchmarkDiscover_JoinPath_ThreeParts(b *testing.B) {
	a, c, d := "/models", "qwen3-4b", "config.json"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		discoverBenchSinkPath = joinPath(a, c, d)
	}
}

func BenchmarkDiscover_AbsolutePath_AlreadyAbsolute(b *testing.B) {
	in := "/Volumes/Data/models/qwen3-4b"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		discoverBenchSinkPath = absolutePath(in)
	}
}

func BenchmarkDiscover_AbsolutePath_Relative(b *testing.B) {
	in := "models/qwen3-4b"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		discoverBenchSinkPath = absolutePath(in)
	}
}
