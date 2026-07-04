// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"runtime/debug"
	"testing"

	"dappco.re/go/inference/model"
)

// The gemma4 declaration package was whole un-benched (no _bench_test.go). These are its AX-11
// alloc baselines: the pure-Go arch derivation (Config.Arch → model.DeriveLayers, the cache-topology
// lift the metal forward bakes in) and the weight assembler (Assemble — the per-weight
// quant-agnostic model.Linear build). Both are loader-side, run once per model load, NOT per token;
// the figure of merit is allocs/op as a one-time-cost floor, measured here so a later change to the
// derivation or the assembler is held to it.
//
// Config.Arch and Assemble-over-synthetic are CPU-only (no GPU, no model) and run in core go qa.
// BenchmarkLoad_RealE2B is the real-checkpoint mmap-metadata path — it mmaps the e2b shards and
// assembles the byte views WITHOUT uploading to the GPU (LoadDirMmap + Assemble, exactly what
// load_test.go's TestLoad_EFamily_QuantAgnostic exercises): no compute, no device buffers, so it
// stays within the AX-11 model-loads gate (metadata only). It is OPT-IN: it skips unless the e2b
// snapshot is cached (gemma4Snapshot, shared with load_test.go), so CI without the cache skips it.

// benchE2BArch is a realistic gemma4-E2B-shaped config for the arch-derivation bench — the per-layer
// layer_types pattern (sliding/global interleave) + KV-share that make model.DeriveLayers do real work.
func benchE2BArch() Config {
	const layers = 30
	lt := make([]string, layers)
	for i := range lt {
		if (i+1)%5 == 0 { // every 5th layer global, the gemma4 interleave shape
			lt[i] = "full_attention"
		} else {
			lt[i] = "sliding_attention"
		}
	}
	return Config{
		HiddenSize: 2048, NumHiddenLayers: layers, IntermediateSize: 8192,
		NumAttentionHeads: 8, NumKeyValueHeads: 2, HeadDim: 256, GlobalHeadDim: 256,
		VocabSize: 262144, RMSNormEps: 1e-6, RopeTheta: 1_000_000,
		SlidingWindow: 1024, NumKVSharedLayers: 10, LayerTypes: lt,
	}
}

// BenchmarkConfigArch measures the arch derivation (Config.Arch → model.DeriveLayers): the per-layer
// attention-type + KV-cache-sharing resolution, allocated once per model load. CPU-only.
func BenchmarkConfigArch(b *testing.B) {
	cfg := benchE2BArch()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := cfg.Arch(); err != nil {
			b.Fatalf("Arch: %v", err)
		}
	}
}

// BenchmarkAssemble_Synthetic measures the weight assembler over a complete synthetic tensor set —
// the per-weight model.Linear / norm build (normalizeNames + the per-layer walk), one-time per load,
// no GPU. A 30-layer E2B-shaped arch so the per-layer loop cost is realistic.
func BenchmarkAssemble_Synthetic(b *testing.B) {
	arch, err := benchE2BArch().Arch()
	if err != nil {
		b.Fatalf("Arch: %v", err)
	}
	ts := minimalGemma4Tensors(arch)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := gemma4Assemble(ts, arch); err != nil {
			b.Fatalf("Assemble: %v", err)
		}
	}
}

// BenchmarkLoad_RealE2B measures the real-checkpoint load (config.json → Arch, mmap the shards →
// Assemble the byte views) of a cached gemma-4-E2B-it-4bit checkpoint — metadata only, no GPU
// upload, no compute (within the AX-11 model-loads gate). Each op mmaps + assembles + Close; the
// figure is the one-time per-model load alloc cost. Opt-in: skips when the snapshot isn't cached.
func BenchmarkLoad_RealE2B(b *testing.B) {
	dir := gemma4Snapshot("models--mlx-community--gemma-4-E2B-it-4bit")
	if dir == "" {
		b.Skip("gemma-4-E2B-it-4bit not cached (opt-in real-checkpoint metadata bench)")
	}
	defer debug.SetMemoryLimit(debug.SetMemoryLimit(60 << 30)) // GC backstop; mmap is lazy, RSS stays tiny

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, dm, err := model.Load(dir)
		if err != nil {
			b.Fatalf("model.Load(%s): %v", dir, err)
		}
		_ = dm.Close() // release the mmap each op so the bench measures one load's cost, flat RSS
	}
}
