// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the option-builder surface.
// Per AX-11 — ApplyGenerateOpts fires per Generate/Chat/Classify/Batch
// call (per request), and ApplyLoadOpts fires per LoadModel (per model
// load). Option builders are tiny closures, but the slices.Clone in
// WithStopTokens IS allocation, and the per-request loop runs O(n)
// in option count, so the construction floor is a real cost surface
// for high-fanout request paths.
//
// Run:    go test -bench=BenchmarkOptions -benchmem -run='^$' .

package inference

import (
	"testing"
)

// Sinks defeat compiler DCE.
var (
	optionsBenchSinkGenerateCfg GenerateConfig
	optionsBenchSinkLoadCfg     LoadConfig
	optionsBenchSinkGenerateOpt GenerateOption
	optionsBenchSinkLoadOpt     LoadOption
)

// --- DefaultGenerateConfig (per-call floor when no opts supplied) ---

func BenchmarkOptions_DefaultGenerateConfig(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateCfg = DefaultGenerateConfig()
	}
}

// --- Individual GenerateOption builders ---

func BenchmarkOptions_WithMaxTokens(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateOpt = WithMaxTokens(256)
	}
}

func BenchmarkOptions_WithTemperature(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateOpt = WithTemperature(0.7)
	}
}

func BenchmarkOptions_WithTopK(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateOpt = WithTopK(40)
	}
}

func BenchmarkOptions_WithTopP(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateOpt = WithTopP(0.9)
	}
}

// WithStopTokens with a single stop token (most common — just EOS).
func BenchmarkOptions_WithStopTokens_One(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateOpt = WithStopTokens(2)
	}
}

// WithStopTokens with EOS + pad — the clone-the-slice cost surfaces here.
func BenchmarkOptions_WithStopTokens_Three(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateOpt = WithStopTokens(2, 1, 0)
	}
}

// 16 stop tokens — heavy stop-token sets (custom EOS variants for some models).
func BenchmarkOptions_WithStopTokens_Sixteen(b *testing.B) {
	ids := []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateOpt = WithStopTokens(ids...)
	}
}

func BenchmarkOptions_WithRepeatPenalty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateOpt = WithRepeatPenalty(1.1)
	}
}

func BenchmarkOptions_WithLogits(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateOpt = WithLogits()
	}
}

// --- ApplyGenerateOpts — the per-request hot path ---

func BenchmarkOptions_ApplyGenerateOpts_Nil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateCfg = ApplyGenerateOpts(nil)
	}
}

func BenchmarkOptions_ApplyGenerateOpts_Empty(b *testing.B) {
	opts := []GenerateOption{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateCfg = ApplyGenerateOpts(opts)
	}
}

// Minimal — single option (just MaxTokens, the most common knob).
func BenchmarkOptions_ApplyGenerateOpts_Minimal(b *testing.B) {
	opts := []GenerateOption{WithMaxTokens(128)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateCfg = ApplyGenerateOpts(opts)
	}
}

// Typical chat-time option set — caps + sampling.
func BenchmarkOptions_ApplyGenerateOpts_Typical(b *testing.B) {
	opts := []GenerateOption{
		WithMaxTokens(256),
		WithTemperature(0.7),
		WithTopP(0.9),
		WithRepeatPenalty(1.1),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateCfg = ApplyGenerateOpts(opts)
	}
}

// Heavy — every knob set, including stop-token clone cost.
func BenchmarkOptions_ApplyGenerateOpts_Heavy(b *testing.B) {
	opts := []GenerateOption{
		WithMaxTokens(2048),
		WithTemperature(0.8),
		WithTopK(50),
		WithTopP(0.95),
		WithStopTokens(0, 1, 2, 3),
		WithRepeatPenalty(1.15),
		WithLogits(),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateCfg = ApplyGenerateOpts(opts)
	}
}

// nil-option slot in the slice — common when callers conditionally
// append options. Tests the nil-skip branch cost.
func BenchmarkOptions_ApplyGenerateOpts_WithNilOptions(b *testing.B) {
	opts := []GenerateOption{
		WithMaxTokens(128),
		nil,
		WithTemperature(0.7),
		nil,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkGenerateCfg = ApplyGenerateOpts(opts)
	}
}

// --- LoadOption builders ---

func BenchmarkOptions_WithBackend(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkLoadOpt = WithBackend("metal")
	}
}

func BenchmarkOptions_WithContextLen(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkLoadOpt = WithContextLen(4096)
	}
}

func BenchmarkOptions_WithGPULayers(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkLoadOpt = WithGPULayers(-1)
	}
}

func BenchmarkOptions_WithParallelSlots(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkLoadOpt = WithParallelSlots(4)
	}
}

func BenchmarkOptions_WithAdapterPath(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkLoadOpt = WithAdapterPath("/models/lora/v1")
	}
}

// --- ApplyLoadOpts — the per-LoadModel hot path ---

func BenchmarkOptions_ApplyLoadOpts_Nil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkLoadCfg = ApplyLoadOpts(nil)
	}
}

func BenchmarkOptions_ApplyLoadOpts_Minimal(b *testing.B) {
	opts := []LoadOption{WithBackend("metal")}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkLoadCfg = ApplyLoadOpts(opts)
	}
}

func BenchmarkOptions_ApplyLoadOpts_Typical(b *testing.B) {
	opts := []LoadOption{
		WithBackend("metal"),
		WithContextLen(4096),
		WithGPULayers(-1),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkLoadCfg = ApplyLoadOpts(opts)
	}
}

func BenchmarkOptions_ApplyLoadOpts_Heavy(b *testing.B) {
	opts := []LoadOption{
		WithBackend("rocm"),
		WithContextLen(32768),
		WithGPULayers(40),
		WithParallelSlots(8),
		WithAdapterPath("/models/lora/domain-v2"),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkLoadCfg = ApplyLoadOpts(opts)
	}
}

func BenchmarkOptions_ApplyLoadOpts_WithNilOptions(b *testing.B) {
	opts := []LoadOption{
		WithBackend("metal"),
		nil,
		WithContextLen(4096),
		nil,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		optionsBenchSinkLoadCfg = ApplyLoadOpts(opts)
	}
}
