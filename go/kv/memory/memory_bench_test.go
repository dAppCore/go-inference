// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the local-inference memory planner. Per AX-11 —
// NewPlan fires per session/runtime/restart per loaded model (rare
// but on the cold-start path), classForBytes + percentBytes + the
// architecture/quantization hint functions run on every plan. NewPlan +
// ancillary helpers are CPU-only — no Metal, no cgo — and are the slow
// part of any cold-start path where the memory planner is consulted
// before model load. (Architecture-name normalisation now lives in
// profile.NormalizeArchitecture and is benched there.)
//
// Run:    go test -bench='BenchmarkMemory|BenchmarkClassForBytes|BenchmarkPercentBytes|BenchmarkMinPositive' -benchmem -run='^$' ./go/memory

package memory

import (
	"testing"

	mp "dappco.re/go/inference/model/modelpack"
)

// Sinks defeat compiler DCE.
var (
	benchMemoryPlan  Plan
	benchMemoryClass Class
	benchMemoryStr   string
	benchMemoryInt   int
	benchMemoryU64   uint64
)

// --- NewPlan — cold-start memory plan derivation ---

// 16GB-class — the smallest tier, cheapest plan.
func BenchmarkMemory_NewPlan_16GB_NoPack(b *testing.B) {
	in := Input{
		Device: DeviceInfo{
			Architecture:                 "apple7",
			MemorySize:                   16 * GiB,
			MaxRecommendedWorkingSetSize: 14 * GiB,
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryPlan = NewPlan(in)
	}
}

// 96GB-class — the typical M3 Ultra topology measured against
// project_local_inference_topology.
func BenchmarkMemory_NewPlan_96GB_NoPack(b *testing.B) {
	in := Input{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * GiB,
			MaxRecommendedWorkingSetSize: 90 * GiB,
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryPlan = NewPlan(in)
	}
}

// MoE pack adds architecture hints + expert residency + KV estimation
// work to the plan.
func BenchmarkMemory_NewPlan_96GB_Qwen3MoEPack(b *testing.B) {
	pack := mp.ModelPack{
		Architecture:  "qwen3_moe",
		ContextLength: 32768,
		NumLayers:     48,
		HiddenSize:    4096,
		QuantBits:     4,
		QuantType:     "q4_0",
		QuantFamily:   "gguf",
		WeightBytes:   20 * 1024 * 1024 * 1024,
	}
	in := Input{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * GiB,
			MaxRecommendedWorkingSetSize: 90 * GiB,
		},
		Pack: &pack,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryPlan = NewPlan(in)
	}
}

// Gemma 4 small-model packs apply the q6/q8/q4 product quantisation
// policy before model-quant warnings and KV estimation.
func BenchmarkMemory_NewPlan_96GB_Gemma4SmallPack(b *testing.B) {
	pack := mp.ModelPack{
		Architecture:  "gemma4_text",
		ContextLength: 32768,
		NumLayers:     34,
		HiddenSize:    2304,
		QuantBits:     6,
		QuantType:     "affine",
		QuantFamily:   "mlx",
		WeightBytes:   5 * 1024 * 1024 * 1024,
	}
	in := Input{
		Device: DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * GiB,
			MaxRecommendedWorkingSetSize: 90 * GiB,
		},
		Pack: &pack,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryPlan = NewPlan(in)
	}
}

// MiniMax M2 triggers the heaviest hint branch (context cap, batch
// floor, cache-mode override).
func BenchmarkMemory_NewPlan_96GB_MiniMaxM2Pack(b *testing.B) {
	pack := mp.ModelPack{
		Architecture:  "minimax_m2",
		ContextLength: 196608,
		NumLayers:     62,
		HiddenSize:    3072,
	}
	in := Input{
		Device: DeviceInfo{MemorySize: 96 * GiB, MaxRecommendedWorkingSetSize: 90 * GiB},
		Pack:   &pack,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryPlan = NewPlan(in)
	}
}

// BERT encoder bypasses generation KV cache estimation — exercises
// the early-return path of usesGenerationKVCache.
func BenchmarkMemory_NewPlan_16GB_BertEmbeddingPack(b *testing.B) {
	pack := mp.ModelPack{
		Architecture:  "bert",
		ContextLength: 512,
		NumLayers:     12,
		HiddenSize:    768,
		Embedding:     &mp.ModelEmbeddingProfile{Dimension: 768, Pooling: "mean", MaxSequenceLength: 512},
		WeightBytes:   420 * 1024 * 1024,
		QuantBits:     16,
		QuantType:     "fp16",
		QuantFamily:   "dense",
	}
	in := Input{
		Device: DeviceInfo{MemorySize: 16 * GiB, MaxRecommendedWorkingSetSize: 13 * GiB},
		Pack:   &pack,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryPlan = NewPlan(in)
	}
}

// ModelInfo without Pack — the simpler hint path with architecture
// cap only.
func BenchmarkMemory_NewPlan_24GB_ModelInfo(b *testing.B) {
	info := ModelInfo{
		Architecture:  "qwen3_6",
		VocabSize:     151936,
		NumLayers:     28,
		HiddenSize:    2048,
		QuantBits:     4,
		ContextLength: 40960,
	}
	in := Input{
		Device:    DeviceInfo{MemorySize: 24 * GiB, MaxRecommendedWorkingSetSize: 21 * GiB},
		ModelInfo: &info,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryPlan = NewPlan(in)
	}
}

// --- ClassForBytes — the exported per-byte tier classifier ---

func BenchmarkClassForBytes_16GB(b *testing.B) {
	bytes := uint64(16 * GiB)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryClass = ClassForBytes(bytes)
	}
}

func BenchmarkClassForBytes_96GB(b *testing.B) {
	bytes := uint64(96 * GiB)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryClass = ClassForBytes(bytes)
	}
}

func BenchmarkClassForBytes_Zero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryClass = ClassForBytes(0)
	}
}

// --- percentBytes / minPositive — fires on every NewPlan ---

func BenchmarkPercentBytes_Typical(b *testing.B) {
	value := uint64(90 * GiB)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryU64 = percentBytes(value, 85)
	}
}

func BenchmarkMinPositive_BothPositive(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryInt = minPositive(8192, 32768)
	}
}

func BenchmarkMinPositive_FirstZero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMemoryInt = minPositive(0, 32768)
	}
}
