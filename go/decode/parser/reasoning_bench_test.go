// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the unexported reasoning state machine —
// parseReasoningText, findReasoningStart, firstReasoningEnd,
// trimReasoningText. Per AX-11 — parseReasoningText is the per-flush
// hot loop ParseReasoning resolves to; findReasoningStart and
// firstReasoningEnd are the per-marker-candidate inner scans driven
// by indexString. With qwen3-class generation flushes hundreds of
// times per response, the per-call cost compounds.
//
// Run:    go test -bench='Benchmark_Reasoning' -benchmem -run='^$' ./go/parser
//
// Stream sizes mirror realistic generation outputs:
//   - 32-token  ≈ very short answer
//   - 256-token ≈ typical chat-response length
//   - 2048-token ≈ long-form generation (the loop pays N times here)

package parser

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	reasoningBenchResult  inference.ReasoningParseResult
	reasoningBenchIdx     int
	reasoningBenchMarker  reasoningMarker
	reasoningBenchOK      bool
	reasoningBenchEndIdx  int
	reasoningBenchEndSize int
	reasoningBenchText    string
)

// reasoningBenchWords builds a synthetic prose stream of approx
// `tokens` words — cheap proxy for byte cost the scanner pays.
func reasoningBenchWords(tokens int) string {
	out := core.NewBuilder()
	for i := 0; i < tokens; i++ {
		out.WriteString("word ")
	}
	return out.String()
}

// reasoningBenchStream wraps a span of words inside the requested
// marker pair, with the span covering `spanFraction` of the total.
func reasoningBenchStream(tokens int, spanFraction float64, startMarker, endMarker string) string {
	span := int(float64(tokens) * spanFraction)
	if span < 1 {
		span = 1
	}
	if span > tokens {
		span = tokens
	}
	pre := (tokens - span) / 2
	post := tokens - span - pre
	out := core.NewBuilder()
	out.WriteString(reasoningBenchWords(pre))
	out.WriteString(startMarker)
	out.WriteString(reasoningBenchWords(span))
	out.WriteString(endMarker)
	out.WriteString(reasoningBenchWords(post))
	return out.String()
}

// --- parseReasoningText: per-flush hot loop ---

var reasoningBenchArchitectures = []struct {
	id      string
	markers []reasoningMarker
	start   string
	end     string
}{
	{"Qwen", qwenMarkers(), "<think>", "</think>"},
	{"Gemma", gemmaMarkers(), "<start_of_turn>thinking\n", "<end_of_turn>"},
	{"GPTOSS", gptOSSMarkers(), "<|channel>analysis\n", "<|channel>final\n"},
	{"Generic", genericMarkers(), "<thinking>", "</thinking>"},
}

var reasoningBenchStreamSizes = []int{32, 256, 2048}

var reasoningBenchSpanFractions = []struct {
	id   string
	frac float64
}{
	{"Span10pct", 0.10},
	{"Span50pct", 0.50},
	{"Span90pct", 0.90},
}

func Benchmark_Reasoning_ParseText(b *testing.B) {
	for _, arch := range reasoningBenchArchitectures {
		for _, size := range reasoningBenchStreamSizes {
			for _, span := range reasoningBenchSpanFractions {
				text := reasoningBenchStream(size, span.frac, arch.start, arch.end)
				markers := arch.markers
				b.Run(arch.id+"/"+span.id+"/"+core.Sprintf("Tokens%d", size), func(b *testing.B) {
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						reasoningBenchResult = parseReasoningText(text, markers)
					}
				})
			}
		}
	}
}

// Edge case: no reasoning span at all (every marker misses).
// The visible-only short-circuit path is the most common per-response
// shape for non-reasoning models.
func Benchmark_Reasoning_ParseText_NoSpan_Qwen(b *testing.B) {
	text := reasoningBenchWords(256)
	markers := qwenMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchResult = parseReasoningText(text, markers)
	}
}

// Edge case: unclosed reasoning span — exercises the
// firstReasoningEnd < 0 branch. The first-marker-unclosed path
// short-circuits the builder (visible == text[:idx] slice, no copy)
// — pinned by Test_Reasoning_ParseText_Unclosed_OneAlloc.
func Benchmark_Reasoning_ParseText_Unclosed_Qwen(b *testing.B) {
	text := "preamble <think>" + reasoningBenchWords(200)
	markers := qwenMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchResult = parseReasoningText(text, markers)
	}
}

// Test_Reasoning_ParseText_Unclosed_OneAlloc locks the unclosed-first-
// marker short-circuit: the visible text is a direct slice of the
// input (no builder, no String() copy) and the single reasoning
// segment is the only allocation. Adapter sites that see partial
// flushes with an open `<think>` tag hit this branch on every flush.
func Test_Reasoning_ParseText_Unclosed_OneAlloc(t *testing.T) {
	text := "preamble <think>" + reasoningBenchWords(200)
	markers := qwenMarkers()
	allocs := testing.AllocsPerRun(50, func() {
		reasoningBenchResult = parseReasoningText(text, markers)
	})
	if allocs > 1 {
		t.Fatalf("expected <=1 alloc/op on unclosed-first-marker short-circuit, got %.2f", allocs)
	}
	if reasoningBenchResult.VisibleText != "preamble " {
		t.Fatalf("expected VisibleText=='preamble ', got %q", reasoningBenchResult.VisibleText)
	}
	if len(reasoningBenchResult.Reasoning) != 1 {
		t.Fatalf("expected exactly 1 reasoning segment, got %d", len(reasoningBenchResult.Reasoning))
	}
}

// --- findReasoningStart: per-marker fan-out, dominated by indexString ---

func Benchmark_Reasoning_FindStart_HitEarly_Qwen(b *testing.B) {
	text := "<think>plan</think>" + reasoningBenchWords(256)
	markers := qwenMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchIdx, reasoningBenchMarker, reasoningBenchOK = findReasoningStart(text, markers)
	}
}

func Benchmark_Reasoning_FindStart_HitMid_Qwen(b *testing.B) {
	text := reasoningBenchStream(256, 0.50, "<think>", "</think>")
	markers := qwenMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchIdx, reasoningBenchMarker, reasoningBenchOK = findReasoningStart(text, markers)
	}
}

func Benchmark_Reasoning_FindStart_HitLate_Qwen(b *testing.B) {
	text := reasoningBenchWords(256) + "<think>plan</think>tail"
	markers := qwenMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchIdx, reasoningBenchMarker, reasoningBenchOK = findReasoningStart(text, markers)
	}
}

func Benchmark_Reasoning_FindStart_Miss_Qwen(b *testing.B) {
	text := reasoningBenchWords(256)
	markers := qwenMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchIdx, reasoningBenchMarker, reasoningBenchOK = findReasoningStart(text, markers)
	}
}

// Gemma + gpt-oss carry the worst-case marker fan-out — every miss
// forces every candidate to be scanned.
func Benchmark_Reasoning_FindStart_Miss_Gemma(b *testing.B) {
	text := reasoningBenchWords(256)
	markers := gemmaMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchIdx, reasoningBenchMarker, reasoningBenchOK = findReasoningStart(text, markers)
	}
}

func Benchmark_Reasoning_FindStart_Miss_GPTOSS(b *testing.B) {
	text := reasoningBenchWords(256)
	markers := gptOSSMarkers()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchIdx, reasoningBenchMarker, reasoningBenchOK = findReasoningStart(text, markers)
	}
}

// --- firstReasoningEnd: per-end-marker scan inside an open span ---

func Benchmark_Reasoning_FirstEnd_HitEarly(b *testing.B) {
	text := "</think>" + reasoningBenchWords(256)
	ends := []string{"</think>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchEndIdx, reasoningBenchEndSize = firstReasoningEnd(text, ends)
	}
}

func Benchmark_Reasoning_FirstEnd_HitLate(b *testing.B) {
	text := reasoningBenchWords(256) + "</think>"
	ends := []string{"</think>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchEndIdx, reasoningBenchEndSize = firstReasoningEnd(text, ends)
	}
}

func Benchmark_Reasoning_FirstEnd_Miss(b *testing.B) {
	text := reasoningBenchWords(256)
	ends := []string{"</think>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchEndIdx, reasoningBenchEndSize = firstReasoningEnd(text, ends)
	}
}

// gpt-oss carries 3 end-marker candidates — every miss pays for all 3.
func Benchmark_Reasoning_FirstEnd_Miss_GPTOSS(b *testing.B) {
	text := reasoningBenchWords(256)
	ends := []string{"<|channel>final\n", "<|channel>assistant\n", "<|channel>assistant"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchEndIdx, reasoningBenchEndSize = firstReasoningEnd(text, ends)
	}
}

// --- trimReasoningText: thin core.Trim wrapper, but called per segment ---

func Benchmark_Reasoning_Trim_Short(b *testing.B) {
	text := "  plan with leading and trailing whitespace  "
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchText = trimReasoningText(text)
	}
}

func Benchmark_Reasoning_Trim_Long(b *testing.B) {
	text := "  " + reasoningBenchWords(256) + "  "
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reasoningBenchText = trimReasoningText(text)
	}
}

// AX-11: zero-alloc budget for parseReasoningText on no-span responses.
// Every assistant response from a non-reasoning model (or a reasoning
// model that didn't emit a marker this turn) hits this path; the
// previous shape unconditionally allocated a strings.Builder + paid
// a full text copy. Regression here scales per-response.
func TestAllocBudget_Reasoning_ParseText_NoSpan(t *testing.T) {
	cases := []struct {
		name   string
		tokens int
	}{
		{"Short", 32},
		{"Mid", 256},
		{"Long", 2048},
	}
	markers := qwenMarkers()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			text := reasoningBenchWords(tc.tokens)
			avg := testing.AllocsPerRun(5, func() {
				reasoningBenchResult = parseReasoningText(text, markers)
			})
			const budget = 0.0
			if avg > budget {
				t.Fatalf("parseReasoningText no-span %s alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
					"This is the per-response common path. A regression here scales per response —\n"+
					"every assistant turn from a non-reasoning model pays this.\n"+
					"Profile: go test -bench=Benchmark_Reasoning_ParseText_NoSpan_Qwen -benchmem -memprofile=/tmp/r.mem",
					tc.name, avg, budget)
			}
		})
	}
}
