// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the parser selection layer — NormaliseKey + Family. Per
// AX-11 — both fire on every Registry.Lookup / LookupHint call, which
// itself fires per generation request when callers don't cache. The
// helpers replaceAll and indexString are also exercised because they
// are the inner string-scan loop the entire package depends on
// (parseReasoningText, parseToolText, processor.findStart, et al.).
//
// Run:    go test -bench='Benchmark_Selector' -benchmem -run='^$' ./go/parser

package parser

import "testing"

// Sinks defeat compiler DCE.
var (
	selectorBenchKey   string
	selectorBenchFam   string
	selectorBenchIdx   int
)

// --- NormaliseKey: per-Lookup hot path ---
// NormaliseKey runs core.Lower + core.Trim + two replaceAll passes.
// The replaceAll pass is the unique cost — it allocates a Builder
// on every call regardless of whether substitution actually happens.

func Benchmark_Selector_NormaliseKey_AlreadyClean(b *testing.B) {
	value := "qwen3"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchKey = NormaliseKey(value)
	}
}

func Benchmark_Selector_NormaliseKey_MixedCase(b *testing.B) {
	value := "Qwen3"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchKey = NormaliseKey(value)
	}
}

func Benchmark_Selector_NormaliseKey_NeedsReplace(b *testing.B) {
	value := "Qwen-3.5"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchKey = NormaliseKey(value)
	}
}

func Benchmark_Selector_NormaliseKey_Empty(b *testing.B) {
	value := ""
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchKey = NormaliseKey(value)
	}
}

// --- Family: branch-heavy classifier called per LookupHint ---

func Benchmark_Selector_Family_Qwen(b *testing.B) {
	hint := Hint{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchFam = Family(hint)
	}
}

func Benchmark_Selector_Family_Gemma(b *testing.B) {
	hint := Hint{Architecture: "gemma4_text"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchFam = Family(hint)
	}
}

// Granite hits the LAST switch arm before generic — worst-case for
// the chained Contains() probe.
func Benchmark_Selector_Family_Granite(b *testing.B) {
	hint := Hint{Architecture: "granite"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchFam = Family(hint)
	}
}

// Unknown architecture falls all the way through every switch arm.
func Benchmark_Selector_Family_Unknown(b *testing.B) {
	hint := Hint{Architecture: "not-a-real-arch"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchFam = Family(hint)
	}
}

// With AdapterName the combined string is longer + scanned twice.
func Benchmark_Selector_Family_QwenWithAdapter(b *testing.B) {
	hint := Hint{Architecture: "qwen3", AdapterName: "lora-coder"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchFam = Family(hint)
	}
}

// --- replaceAll: NormaliseKey inner loop ---

func Benchmark_Selector_ReplaceAll_NoMatch(b *testing.B) {
	text := "qwen3"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchKey = replaceAll(text, "-", "_")
	}
}

func Benchmark_Selector_ReplaceAll_SingleMatch(b *testing.B) {
	text := "qwen-3"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchKey = replaceAll(text, "-", "_")
	}
}

func Benchmark_Selector_ReplaceAll_ManyMatches(b *testing.B) {
	text := "a-b-c-d-e-f-g-h"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchKey = replaceAll(text, "-", "_")
	}
}

// Empty `old` short-circuits at the function head.
func Benchmark_Selector_ReplaceAll_EmptyOld(b *testing.B) {
	text := "qwen-3"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchKey = replaceAll(text, "", "_")
	}
}

// --- indexString: the inner scan loop everything else resolves to ---

func Benchmark_Selector_IndexString_HitEarly(b *testing.B) {
	text := "<think>plan</think>answer with a tail of fluff to scan past"
	substr := "<think>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchIdx = indexString(text, substr)
	}
}

func Benchmark_Selector_IndexString_HitLate(b *testing.B) {
	// 256 bytes of filler + the substring at the tail.
	filler := ""
	for i := 0; i < 64; i++ {
		filler += "word"
	}
	text := filler + "<think>"
	substr := "<think>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchIdx = indexString(text, substr)
	}
}

func Benchmark_Selector_IndexString_Miss(b *testing.B) {
	filler := ""
	for i := 0; i < 64; i++ {
		filler += "word"
	}
	text := filler
	substr := "<think>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchIdx = indexString(text, substr)
	}
}

func Benchmark_Selector_IndexString_EmptySubstr(b *testing.B) {
	text := "some text"
	substr := ""
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchIdx = indexString(text, substr)
	}
}

func Benchmark_Selector_IndexString_SubstrLongerThanText(b *testing.B) {
	text := "hi"
	substr := "<long_marker_name>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchIdx = indexString(text, substr)
	}
}

// 2048-byte miss — proxy for scanning a full generation stream looking
// for a marker that never appears.
func Benchmark_Selector_IndexString_Miss_2048bytes(b *testing.B) {
	filler := ""
	for i := 0; i < 512; i++ {
		filler += "word"
	}
	text := filler
	substr := "<think>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selectorBenchIdx = indexString(text, substr)
	}
}
