// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for parser registry construction + lookup. Per AX-11 —
// Default() rebuilds the entire registry (10 architectures × marker
// fan-out) every call, NewRegistry() + Register() are the per-consumer
// build paths, Lookup is the per-dispatch hot path, and ForHint is the
// per-request convenience wrapper that hits Default() + LookupHint on
// every call when the consumer doesn't cache a Registry. HintFromInference
// is the inline-allocation cost paid per generation request.
//
// Run:    go test -bench='Benchmark_Registry' -benchmem -run='^$' ./go/parser

package parser

import (
	"testing"

	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	registryBenchRegistry *Registry
	registryBenchParser   OutputParser
	registryBenchOK       bool
	registryBenchHint     Hint
)

// --- Default + NewRegistry (per-build floor) ---

func Benchmark_Registry_NewRegistry(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchRegistry = NewRegistry()
	}
}

func Benchmark_Registry_Default(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchRegistry = Default()
	}
}

// --- Register (per-alias insert) ---

func Benchmark_Registry_RegisterSingleAlias(b *testing.B) {
	registry := NewRegistry()
	parser := newBuiltinOutputParser("custom", genericMarkers())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registry.Register(parser, "alias")
	}
}

func Benchmark_Registry_RegisterMultiAlias(b *testing.B) {
	registry := NewRegistry()
	parser := newBuiltinOutputParser("custom", genericMarkers())
	aliases := []string{"a1", "a2", "a3", "a4", "a5"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registry.Register(parser, aliases...)
	}
}

// --- Lookup: per-dispatch hot path ---

func Benchmark_Registry_Lookup_Hit_Qwen(b *testing.B) {
	registry := Default()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser, registryBenchOK = registry.Lookup("qwen3")
	}
}

func Benchmark_Registry_Lookup_Hit_Gemma(b *testing.B) {
	registry := Default()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser, registryBenchOK = registry.Lookup("gemma4_text")
	}
}

// Miss path forces a full map probe + key normalisation.
func Benchmark_Registry_Lookup_Miss(b *testing.B) {
	registry := Default()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser, registryBenchOK = registry.Lookup("not-a-real-arch")
	}
}

// Lookup pays NormaliseKey on every call — exercise the
// normalisation cost separately by feeding mixed-case input.
func Benchmark_Registry_Lookup_Hit_Normalise(b *testing.B) {
	registry := Default()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser, registryBenchOK = registry.Lookup("Qwen-3.5")
	}
}

func Benchmark_Registry_Lookup_NilReceiver(b *testing.B) {
	var registry *Registry
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser, registryBenchOK = registry.Lookup("qwen3")
	}
}

// --- LookupHint: Family() + Lookup() + fallback ---

func Benchmark_Registry_LookupHint_Qwen(b *testing.B) {
	registry := Default()
	hint := Hint{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser = registry.LookupHint(hint)
	}
}

func Benchmark_Registry_LookupHint_Gemma(b *testing.B) {
	registry := Default()
	hint := Hint{Architecture: "gemma4_text"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser = registry.LookupHint(hint)
	}
}

func Benchmark_Registry_LookupHint_Unknown(b *testing.B) {
	registry := Default()
	hint := Hint{Architecture: "not-a-real-arch"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser = registry.LookupHint(hint)
	}
}

func Benchmark_Registry_LookupHint_NilReceiver(b *testing.B) {
	var registry *Registry
	hint := Hint{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser = registry.LookupHint(hint)
	}
}

// --- ForHint: the convenience wrapper that hits Default() + LookupHint ---

func Benchmark_Registry_ForHint_Qwen(b *testing.B) {
	hint := Hint{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser = ForHint(hint)
	}
}

func Benchmark_Registry_ForHint_Gemma(b *testing.B) {
	hint := Hint{Architecture: "gemma4_text"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser = ForHint(hint)
	}
}

func Benchmark_Registry_ForHint_Unknown(b *testing.B) {
	hint := Hint{Architecture: "not-a-real-arch"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchParser = ForHint(hint)
	}
}

// --- HintFromInference: per-request inline alloc ---

func Benchmark_Registry_HintFromInference(b *testing.B) {
	info := inference.ModelInfo{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registryBenchHint = HintFromInference(info)
	}
}
